#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Visualize a ROS bag containing an event camera stream and a force sensor.

Output layout (1920x1080, 30 fps):
  left  : event image, rotated 90 deg clockwise, red = ON, blue = OFF
  right : top    - (fx, fy) vector in the XY plane
          bottom - fz vs time with a moving cursor and value label

Supports the two force message layouts found in the data directory:
  geometry_msgs/WrenchStamped  -> wrench.force.x/y/z
  force_data_process/Force     -> fx/fy/fz

Run with the system Python that has ROS Noetic and OpenCV installed:
    /usr/bin/python3 bag_vis.py
    /usr/bin/python3 bag_vis.py /path/to/bags -j 4
    /usr/bin/python3 bag_vis.py --preview 30
"""

import argparse
import glob
import json
import math
import os
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import rosbag
import rospy


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

EVENT_TOPIC_CANDIDATES = ("/capture_node/events",)
EVENT_MSG_TYPES = ("dv_ros_msgs/EventArray",)
FORCE_TOPIC_CANDIDATES = ("/force_sensor/data", "/force_sensor/array")
FORCE_MSG_TYPES = ("force_data_process/Force", "geometry_msgs/WrenchStamped")

# Serialized dv_ros_msgs/Event (no padding): x, y, stamp.sec, stamp.nsec, polarity
EVENT_DT = np.dtype(
    [
        ("x", "<u2"),
        ("y", "<u2"),
        ("sec", "<u4"),
        ("nsec", "<u4"),
        ("pol", "u1"),
    ]
)

CANVAS_W, CANVAS_H = 1920, 1080
FPS_DEFAULT = 30

# BGR palette (dark theme)
COL_BG = (18, 16, 15)
COL_PANEL = (32, 28, 25)
COL_PANEL2 = (40, 35, 31)
COL_GRID = (58, 52, 47)
COL_AXIS = (92, 86, 80)
COL_TEXT = (236, 236, 236)
COL_MUTED = (150, 148, 144)
COL_RED = (78, 78, 255)
COL_BLUE = (255, 168, 72)
COL_GREEN = (118, 220, 130)
COL_YELLOW = (80, 212, 255)
COL_CYAN = (238, 214, 90)
COL_TX = (80, 120, 255)
COL_TY = (255, 170, 80)
COL_TZ = (118, 220, 130)
COL_TX_DIM = (70, 70, 130)
COL_TY_DIM = (130, 100, 60)
COL_TZ_DIM = (60, 100, 65)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


_LOG_LOCK = threading.Lock()


def log(msg):
    with _LOG_LOCK:
        print(msg, flush=True)


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class ProgressBar:
    """Thread-safe console progress bar, one instance per bag."""

    def __init__(self, label, total, single_line=False, min_interval=1.0):
        self.label = label
        self.total = max(1, int(total))
        self.single_line = bool(single_line) and sys.stdout.isatty()
        self.min_interval = 0.2 if self.single_line else min_interval
        self.n = 0
        self.start = time.monotonic()
        self.last = 0.0
        self.closed = False

    def update(self, n=None, force=False):
        if self.closed:
            return
        if n is None:
            self.n += 1
        else:
            self.n = int(n)
        now = time.monotonic()
        if not force and now - self.last < self.min_interval:
            return
        self.last = now
        frac = min(max(self.n / self.total, 0.0), 1.0)
        width = 28
        filled = int(round(frac * width))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(now - self.start, 1e-9)
        fps = self.n / elapsed
        eta = (self.total - self.n) / fps if fps > 0 else 0.0
        msg = (
            f"{self.label} [{bar}] {frac*100:5.1f}% "
            f"{self.n}/{self.total} {fps:5.1f} fps ETA {format_duration(eta)}"
        )
        with _LOG_LOCK:
            if self.single_line:
                print("\r" + msg + "\033[K", end="", flush=True)
            else:
                print(msg, flush=True)

    def close(self):
        if self.closed:
            return
        self.update(self.n, force=True)
        self.closed = True
        if self.single_line:
            with _LOG_LOCK:
                print()


def text(img, s, org, scale=0.55, color=COL_TEXT, thickness=1):
    cv2.putText(
        img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def text_right(img, s, right_x, y, scale=0.55, color=COL_TEXT, thickness=1):
    (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    text(img, s, (int(right_x - w), int(y)), scale, color, thickness)


def put_box(img, x, y, w, h, color, alpha=0.72):
    """Blend a filled rectangle, touching only the rectangle ROI."""
    H, W = img.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(W, int(x) + int(w))
    y1 = min(H, int(y) + int(h))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x1 - x0 - 1, y1 - y0 - 1), color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def blend_vline(img, x, y0, y1, color, alpha=0.85, thickness=2):
    """Blend a vertical cursor line, touching only a narrow ROI."""
    H, W = img.shape[:2]
    x0 = max(0, int(x) - thickness)
    x1 = min(W, int(x) + thickness + 1)
    y0 = max(0, int(y0))
    y1 = min(H, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    lx = int(x) - x0
    cv2.line(overlay, (lx, 0), (lx, y1 - y0 - 1), color, thickness)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def nice_step(span, target=5.0):
    """A 1/2/2.5/5 x 10^k step giving roughly `target` divisions."""
    if span <= 0 or not np.isfinite(span):
        return 1.0
    raw = span / target
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def format_n(v):
    av = abs(v)
    if av >= 100:
        return f"{v:+.1f}"
    if av >= 10:
        return f"{v:+.2f}"
    return f"{v:+.3f}"


# ---------------------------------------------------------------------------
# rosbag reading
# ---------------------------------------------------------------------------


def get_raw_event(item):
    """Extract raw bytes from BagMessage returned by read_messages(raw=True)."""
    data = item[1]
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, (tuple, list)):
        # (typename, raw, md5sum, (secs, nsecs), class)
        for entry in data:
            if isinstance(entry, (bytes, bytearray)):
                return bytes(entry)
    raise TypeError("cannot find raw bytes in rosbag raw message")


def detect_topics(bag):
    info = bag.get_type_and_topic_info().topics

    event_topic = None
    for cand in EVENT_TOPIC_CANDIDATES:
        if cand in info:
            event_topic = cand
            break
    if event_topic is None:
        for name, meta in info.items():
            if meta.msg_type in EVENT_MSG_TYPES:
                event_topic = name
                break
    if event_topic is None:
        raise RuntimeError(f"no dv_ros_msgs/EventArray topic found in {bag.filename}")

    force_topic = None
    for cand in FORCE_TOPIC_CANDIDATES:
        if cand in info:
            force_topic = cand
            break
    if force_topic is None:
        for name, meta in info.items():
            if meta.msg_type in FORCE_MSG_TYPES:
                force_topic = name
                break
    if force_topic is None:
        raise RuntimeError(f"no supported force topic found in {bag.filename}")

    return event_topic, info[event_topic].msg_type, force_topic, info[force_topic].msg_type


def read_force(bag, topic, msg_type, start_time=None, end_time=None):
    """Return (t, fx, fy, fz, tx, ty, tz) as float64 arrays, sorted and deduped."""
    kwargs = {}
    if start_time is not None:
        kwargs["start_time"] = rospy.Time.from_sec(start_time)
    if end_time is not None:
        kwargs["end_time"] = rospy.Time.from_sec(end_time)
    rows = []
    for _, msg, rec in bag.read_messages(topics=[topic], **kwargs):
        try:
            if msg_type == "geometry_msgs/WrenchStamped":
                f = msg.wrench.force
                q = msg.wrench.torque
                fx, fy, fz = f.x, f.y, f.z
                tx, ty, tz = q.x, q.y, q.z
            elif msg_type == "force_data_process/Force":
                fx, fy, fz = msg.fx, msg.fy, msg.fz
                tx, ty, tz = msg.tx, msg.ty, msg.tz
            else:
                raise RuntimeError(f"unsupported force message type: {msg_type}")

            stamp = msg.header.stamp.to_sec()
            if not np.isfinite(stamp) or stamp <= 0:
                stamp = rec.to_sec()
            rows.append((stamp, fx, fy, fz, tx, ty, tz))
        except AttributeError as exc:
            raise RuntimeError(
                f"cannot read force message of type {msg_type}: {exc}"
            ) from exc

    if not rows:
        raise RuntimeError(f"no force samples on {topic}")

    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr = arr[np.argsort(arr[:, 0], kind="mergesort")]

    # Keep the last sample if several share an identical timestamp.
    keep = np.ones(len(arr), dtype=bool)
    keep[1:] = np.diff(arr[:, 0]) > 0
    arr = arr[keep]

    if len(arr) < 2:
        raise RuntimeError(f"not enough force samples on {topic}")
    return (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3],
            arr[:, 4], arr[:, 5], arr[:, 6])


def load_sync_calibration(path):
    """Load a sync_calibrate.py JSON file, or return None."""
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"cannot read sync calibration {path}: {exc}") from exc
    if not isinstance(data, dict) or "bags" not in data:
        raise RuntimeError(f"invalid sync calibration file: {path}")
    return data


def resolve_force_shift(stem, args, calibration):
    """Return (shift_seconds, source_label) for this bag."""
    if getattr(args, "no_sync", False):
        return 0.0, "disabled"
    manual = getattr(args, "force_shift", None)
    if manual is not None:
        return float(manual), "manual"
    if calibration is None:
        return 0.0, "none"

    entry = calibration.get("bags", {}).get(stem)
    if entry:
        shift_ms = entry.get("force_time_shift_ms")
        if shift_ms is not None:
            return float(shift_ms) / 1000.0, "bag"
    global_shift = calibration.get("global", {}).get("force_time_shift_ms", 0.0)
    return float(global_shift) / 1000.0, "global"


def parse_event_raw(raw):
    """Parse one serialized dv_ros_msgs/EventArray.

    Returns (x, y, t, polarity, height, width) or None for empty arrays.
    """
    off = 0
    # std_msgs/Header: seq, stamp.sec, stamp.nsec, frame_id string
    _seq, _sec, _nsec = struct.unpack_from("<III", raw, off)
    off += 12
    frame_len = struct.unpack_from("<I", raw, off)[0]
    off += 4 + frame_len

    height, width, n_events = struct.unpack_from("<III", raw, off)
    off += 12

    if n_events == 0:
        return None

    expected = off + n_events * EVENT_DT.itemsize
    if expected != len(raw):
        raise RuntimeError(
            "unexpected EventArray layout: "
            f"{len(raw)} bytes, {n_events} events, header offset {off}, "
            f"expected {expected} bytes"
        )

    ev = np.frombuffer(raw, dtype=EVENT_DT, count=n_events, offset=off)
    t = ev["sec"].astype(np.float64) + ev["nsec"].astype(np.float64) * 1e-9
    return ev["x"], ev["y"], t, ev["pol"] != 0, int(height), int(width)


def estimate_event_rate(bag, topic, t0, window=2.0):
    """Count events in the first `window` seconds to choose a display decay."""
    t_end = min(t0 + window, bag.get_end_time())
    if t_end <= t0:
        return 0.0
    n = 0
    for item in bag.read_messages(
        topics=[topic],
        raw=True,
        start_time=rospy.Time.from_sec(t0),
        end_time=rospy.Time.from_sec(t_end),
    ):
        raw = get_raw_event(item)
        off = 12
        frame_len = struct.unpack_from("<I", raw, off)[0]
        off += 4 + frame_len
        n += struct.unpack_from("<I", raw, off + 8)[0]
    return n / max(t_end - t0, 1e-9)


# ---------------------------------------------------------------------------
# event surface
# ---------------------------------------------------------------------------


class EventSurface:
    """Last-event-time surface with exponential decay and red/blue polarity."""

    mode = "surface"

    def __init__(self, height, width, tau, fps=30, t0=0.0):
        self.h = height
        self.w = width
        self.tau = np.float32(tau)
        self.t0 = float(t0)
        # Relative timestamps keep the age computation in float32 and avoid
        # precision loss from large absolute ROS times.
        self.last_t = np.full((height, width), -1e9, dtype=np.float32)
        self.last_pol = np.zeros((height, width), dtype=np.int8)

        self._age = np.empty((height, width), dtype=np.float32)
        self._inten = np.empty((height, width), dtype=np.float32)
        self._pos = np.empty((height, width), dtype=np.float32)
        self._neg = np.empty((height, width), dtype=np.float32)
        self._u8pos = np.empty((height, width), dtype=np.uint8)
        self._u8neg = np.empty((height, width), dtype=np.uint8)
        self._img = np.zeros((height, width, 3), dtype=np.uint8)

    def add(self, x, y, t, pol):
        if len(x) == 0:
            return
        # x/y are uint16; cast once for fast fancy indexing
        xi = x.astype(np.int32, copy=False)
        yi = y.astype(np.int32, copy=False)
        self.last_t[yi, xi] = (t - self.t0).astype(np.float32)
        self.last_pol[yi, xi] = np.where(pol, 1, -1).astype(np.int8, copy=False)

    def render(self, t_now, gamma=0.72):
        t_rel = np.float32(t_now - self.t0)
        np.subtract(t_rel, self.last_t, out=self._age)
        np.maximum(self._age, 0.0, out=self._age)
        np.divide(self._age, -self.tau, out=self._age)
        np.exp(self._age, out=self._inten)

        valid = self.last_t > -1e8
        self._inten *= valid
        if gamma != 1.0:
            np.power(self._inten, gamma, out=self._inten)

        pos_mask = self.last_pol > 0
        neg_mask = self.last_pol < 0
        np.multiply(self._inten, pos_mask, out=self._pos)
        np.multiply(self._inten, neg_mask, out=self._neg)
        if pos_mask.any():
            cv2.GaussianBlur(self._pos, (3, 3), 0, dst=self._pos)
        if neg_mask.any():
            cv2.GaussianBlur(self._neg, (3, 3), 0, dst=self._neg)

        cv2.convertScaleAbs(self._pos, dst=self._u8pos, alpha=255.0)
        cv2.convertScaleAbs(self._neg, dst=self._u8neg, alpha=255.0)
        img = self._img
        img[..., 2] = self._u8pos  # red
        img[..., 0] = self._u8neg  # blue
        return img


class DensitySurface:
    """Exponentially decaying event-count image, normalized per frame.

    Better than a time surface when the sensor produces a high event rate and
    a large amount of background activity (session06).
    """

    mode = "density"

    def __init__(self, height, width, tau, fps=30, t0=0.0):
        self.h = height
        self.w = width
        self.tau = float(tau)
        self.t0 = float(t0)
        self.decay = float(math.exp(-1.0 / max(fps * tau, 1e-6)))
        self.acc_pos = np.zeros((height, width), dtype=np.float32)
        self.acc_neg = np.zeros((height, width), dtype=np.float32)
        self._ch = (
            np.empty((height, width), dtype=np.float32),
            np.empty((height, width), dtype=np.float32),
        )
        self._u8 = (
            np.empty((height, width), dtype=np.uint8),
            np.empty((height, width), dtype=np.uint8),
        )
        self._img = np.zeros((height, width, 3), dtype=np.uint8)

    def add(self, x, y, t, pol):
        if len(x) == 0:
            return
        xi = x.astype(np.int32, copy=False)
        yi = y.astype(np.int32, copy=False)
        xmin, xmax = int(xi.min()), int(xi.max())
        ymin, ymax = int(yi.min()), int(yi.max())
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        if xmax >= self.w or ymax >= self.h:
            xmax = min(xmax, self.w - 1)
            ymax = min(ymax, self.h - 1)
            m = (xi >= xmin) & (xi <= xmax) & (yi >= ymin) & (yi <= ymax)
            xi, yi, pol = xi[m], yi[m], pol[m]
            if len(xi) == 0:
                return
        if xmin > xmax or ymin > ymax:
            return

        rw = xmax - xmin + 1
        rh = ymax - ymin + 1
        n = rw * rh
        idx = (yi - ymin).astype(np.int64) * rw + (xi - xmin)
        if pol.any():
            idx += pol.astype(np.int64) * n
        hist = np.bincount(idx, minlength=2 * n)
        self.acc_pos[ymin : ymax + 1, xmin : xmax + 1] += (
            hist[:n].reshape(rh, rw).astype(np.float32)
        )
        self.acc_neg[ymin : ymax + 1, xmin : xmax + 1] += (
            hist[n:].reshape(rh, rw).astype(np.float32)
        )

    def render(self, t_now, gamma=1.40):
        for acc, out, u8 in (
            (self.acc_pos, self._ch[0], self._u8[0]),
            (self.acc_neg, self._ch[1], self._u8[1]),
        ):
            np.log1p(acc, out=out)
            # A strided subsample is plenty for a robust normalization scale.
            mx = float(np.percentile(out[::4, ::4], 99.5)) if out.size else 0.0
            if mx > 1e-9:
                np.divide(out, mx, out=out)
            np.multiply(out, 0.60, out=out)
            np.clip(out, 0.0, 1.0, out=out)
            if gamma != 1.0:
                np.power(out, gamma, out=out)
            cv2.convertScaleAbs(out, dst=u8, alpha=255.0)

        img = self._img
        img[..., 2] = self._u8[0]
        img[..., 0] = self._u8[1]

        # Decay after rendering so this frame's events are fully visible.
        self.acc_pos *= self.decay
        self.acc_neg *= self.decay
        return img


# ---------------------------------------------------------------------------
# frame renderer
# ---------------------------------------------------------------------------


class Renderer:
    def __init__(
        self,
        bag_name,
        duration,
        f_t,
        f_fx,
        f_fy,
        f_fz,
        f_tx,
        f_ty,
        f_tz,
        event_h,
        event_w,
        force_orientation="new",
    ):
        self.bag_name = bag_name
        self.duration = float(duration)
        self.force_orientation = force_orientation
        self.f_t = f_t  # already relative to the rendered time window
        self.f_fx = f_fx
        self.f_fy = f_fy
        self.f_fz = f_fz
        self.f_tx = f_tx
        self.f_ty = f_ty
        self.f_tz = f_tz
        self.event_h = event_h
        self.event_w = event_w

        # --- event panel geometry (portrait after 90 deg rotation) ---
        self.ew = 750
        self.eh = 1000
        self.ex = 24
        self.ey = 56

        # --- semi-transparent force HUD in the middle of the event image ---
        self.hud_cx = self.ex + self.ew // 2
        self.hud_cy = self.ey + self.eh // 2
        self.hud_r = 210
        max_f = float(np.max(np.abs(np.concatenate([f_fx, f_fy]))))
        max_f = max(max_f, 0.1)
        step_f = nice_step(max_f * 2.16, target=6.0)
        self.force_step = step_f
        self.force_limit = max(
            step_f * math.ceil((max_f * 1.08) / step_f), step_f
        )
        self.force_scale = self.hud_r / self.force_limit

        # --- torque time-series panel (replaces the old FX-FY plane) ---
        self.tx0, self.ty0, self.tx1, self.ty1 = 852, 76, 1896, 586
        max_t = float(np.max(np.abs(np.concatenate([f_tx, f_ty, f_tz]))))
        max_t = max(max_t, 1e-6)
        step_t = nice_step(max_t * 2.16, target=6.0)
        self.torque_step = step_t
        self.torque_limit = max(
            step_t * math.ceil((max_t * 1.08) / step_t), step_t
        )

        # --- fz plot geometry ---
        self.gx0, self.gy0, self.gx1, self.gy1 = 900, 662, 1880, 1020
        fz_min = float(np.min(f_fz))
        fz_max = float(np.max(f_fz))
        fz_span = max(fz_max - fz_min, 1e-6)
        self.fz_step = nice_step(fz_span, target=8.0)
        if fz_min > -0.05 * fz_span:
            self.fz_min = 0.0
        else:
            self.fz_min = math.floor((fz_min * 1.05) / self.fz_step) * self.fz_step
        if fz_max < 0.05 * fz_span:
            self.fz_max = 0.0
        else:
            self.fz_max = math.ceil((fz_max * 1.05) / self.fz_step) * self.fz_step
        if self.fz_max <= self.fz_min:
            self.fz_max = self.fz_min + self.fz_step

        # Downsampled polylines for efficient time-series rendering.
        n = len(self.f_t)
        if n > 4000:
            idx = np.linspace(0, n - 1, 4000).astype(np.int64)
            idx = np.unique(idx)
        else:
            idx = np.arange(n)
        self.fz_ds_t = self.f_t[idx]
        self.fz_ds_v = f_fz[idx]
        self.fz_full_pts = self._fz_points(self.fz_ds_t, self.fz_ds_v)

        self.torque_ds_t = self.f_t[idx]
        self.torque_pts = {
            "tx": self._torque_points(self.torque_ds_t, f_tx[idx]),
            "ty": self._torque_points(self.torque_ds_t, f_ty[idx]),
            "tz": self._torque_points(self.torque_ds_t, f_tz[idx]),
        }
        self._torque_series = (
            ("tx", COL_TX, COL_TX_DIM),
            ("ty", COL_TY, COL_TY_DIM),
            ("tz", COL_TZ, COL_TZ_DIM),
        )

        # Static parts of the layout are drawn once and copied per frame.
        self.static = np.empty((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        self.canvas = np.empty_like(self.static)
        self._build_static()

    # ------------------------------------------------------------------
    # coordinate maps
    # ------------------------------------------------------------------

    def _hud_xy(self, fx, fy):
        fx = np.asarray(fx, dtype=np.float64)
        fy = np.asarray(fy, dtype=np.float64)
        if self.force_orientation == "old":
            # Old bags: screen +X = -fy, screen +Y = +fx
            px = self.hud_cx - fy * self.force_scale
            py = self.hud_cy + fx * self.force_scale
        else:
            # New bags: screen +X = -fx, screen +Y = -fy
            # so fx negative points right and fy negative points down.
            px = self.hud_cx - fx * self.force_scale
            py = self.hud_cy - fy * self.force_scale
        return px.astype(np.int32), py.astype(np.int32)

    def _fz_xy(self, t, v):
        t = np.asarray(t, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        px = self.gx0 + (t / self.duration) * (self.gx1 - self.gx0)
        py = self.gy1 - ((v - self.fz_min) / (self.fz_max - self.fz_min)) * (
            self.gy1 - self.gy0
        )
        return px.astype(np.int32), py.astype(np.int32)

    def _fz_points(self, t, v):
        px, py = self._fz_xy(np.asarray(t), np.asarray(v))
        return np.column_stack([px, py]).astype(np.int32)

    def _torque_xy(self, t, v):
        t = np.asarray(t, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        px = self.tx0 + (t / self.duration) * (self.tx1 - self.tx0)
        py = (self.ty0 + self.ty1) / 2.0 - (v / self.torque_limit) * (
            (self.ty1 - self.ty0) / 2.0
        )
        return px.astype(np.int32), py.astype(np.int32)

    def _torque_points(self, t, v):
        px, py = self._torque_xy(np.asarray(t), np.asarray(v))
        return np.column_stack([px, py]).astype(np.int32)

    # ------------------------------------------------------------------
    # drawing helpers
    # ------------------------------------------------------------------

    def _draw_event_static(self, canvas):
        x, y, w, h = self.ex, self.ey, self.ew, self.eh
        cv2.rectangle(canvas, (x - 1, y - 1), (x + w, y + h), COL_GRID, 1)
        text(canvas, "EVENT CAMERA  |  90 deg CW", (x, 32), 0.58, COL_TEXT, 1)

    def _draw_event_panel(self, canvas, event_img, t):
        x, y, w, h = self.ex, self.ey, self.ew, self.eh
        rot = cv2.rotate(event_img, cv2.ROTATE_90_CLOCKWISE)
        if rot.shape[1] != w or rot.shape[0] != h:
            rot = cv2.resize(rot, (w, h), interpolation=cv2.INTER_LINEAR)
        canvas[y : y + h, x : x + w] = rot

        text(canvas, "red = ON   blue = OFF", (x, y + h - 16), 0.48, COL_MUTED, 1)

        label = f"t = {t:6.2f} s / {self.duration:6.2f} s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        put_box(canvas, x + 10, y + 10, tw + 20, th + 14, (0, 0, 0), 0.55)
        text(canvas, label, (x + 20, y + 10 + th + 4), 0.62, COL_TEXT, 1)

    def _draw_force_hud(self, canvas, t, fx, fy):
        """Semi-transparent fx-fy vector overlaid on the event image."""
        ex, ey, ew, eh = self.ex, self.ey, self.ew, self.eh
        roi = canvas[ey : ey + eh, ex : ex + ew]
        hud = roi.copy()

        cx = self.hud_cx - ex
        cy = self.hud_cy - ey
        r = self.hud_r
        sc = self.force_scale
        lim = self.force_limit
        step = self.force_step

        cv2.circle(hud, (cx, cy), r, COL_GRID, 1, cv2.LINE_AA)
        cv2.line(hud, (cx - r, cy), (cx + r, cy), COL_AXIS, 1, cv2.LINE_AA)
        cv2.line(hud, (cx, cy - r), (cx, cy + r), COL_AXIS, 1, cv2.LINE_AA)

        # grid chords clipped to the circle
        v = -lim
        while v <= lim + 1e-9:
            if abs(v) > 1e-9:
                gx = cx - v * sc
                dx = gx - cx
                if abs(dx) < r - 3:
                    half = math.sqrt(max(r * r - dx * dx, 0.0))
                    cv2.line(
                        hud,
                        (int(round(gx)), int(round(cy - half))),
                        (int(round(gx)), int(round(cy + half))),
                        COL_GRID,
                        1,
                    )
                gy = cy + v * sc
                dy = gy - cy
                if abs(dy) < r - 3:
                    half = math.sqrt(max(r * r - dy * dy, 0.0))
                    cv2.line(
                        hud,
                        (int(round(cx - half)), int(round(gy))),
                        (int(round(cx + half)), int(round(gy))),
                        COL_GRID,
                        1,
                    )
            v += step

        # recent trail
        lo = int(np.searchsorted(self.f_t, max(t - 2.5, 0.0), side="left"))
        hi = int(np.searchsorted(self.f_t, t, side="right"))
        if hi > lo + 1:
            stride = max(1, (hi - lo) // 400)
            pts = np.column_stack(
                self._hud_xy(
                    self.f_fx[lo:hi:stride], self.f_fy[lo:hi:stride]
                )
            ).astype(np.int32)
            pts[:, 0] -= ex
            pts[:, 1] -= ey
            cv2.polylines(hud, [pts], False, COL_CYAN, 1, cv2.LINE_AA)

            lo2 = int(np.searchsorted(self.f_t, max(t - 0.5, 0.0), side="left"))
            if hi > lo2 + 1:
                stride2 = max(1, (hi - lo2) // 150)
                pts2 = np.column_stack(
                    self._hud_xy(
                        self.f_fx[lo2:hi:stride2], self.f_fy[lo2:hi:stride2]
                    )
                ).astype(np.int32)
                pts2[:, 0] -= ex
                pts2[:, 1] -= ey
                cv2.polylines(hud, [pts2], False, COL_CYAN, 2, cv2.LINE_AA)

        # current vector
        if self.force_orientation == "old":
            tip_x = int(round(cx - fy * sc))
            tip_y = int(round(cy + fx * sc))
        else:
            tip_x = int(round(cx - fx * sc))
            tip_y = int(round(cy - fy * sc))
        cv2.arrowedLine(
            hud,
            (cx, cy),
            (tip_x, tip_y),
            COL_YELLOW,
            3,
            cv2.LINE_AA,
            tipLength=0.12,
        )
        cv2.circle(hud, (tip_x, tip_y), 5, COL_YELLOW, -1, cv2.LINE_AA)
        cv2.circle(hud, (cx, cy), 3, COL_TEXT, -1, cv2.LINE_AA)

        cv2.addWeighted(hud, 0.62, roi, 0.38, 0, roi)

        # labels stay fully opaque for readability
        if self.force_orientation == "old":
            title = "FX-FY  (N)   |   FX+ down, FY- right"
            right_label = "FY-"
            bottom_label = "FX+"
        else:
            title = "FX-FY  (N)   |   FX- right, FY- down"
            right_label = "FX-"
            bottom_label = "FY-"
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        tx = ex + (ew - tw) // 2
        ty = cy - r - 10
        put_box(canvas, tx - 6, ty - th - 4, tw + 12, th + 10, (0, 0, 0), 0.55)
        text(canvas, title, (tx, ty), 0.48, COL_TEXT, 1)

        mag = float(math.hypot(fx, fy))
        label = f"fx {fx:+.2f}   fy {fy:+.2f}   |f| {mag:.2f} N"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tx = ex + (ew - tw) // 2
        ty = cy + r + 28
        put_box(canvas, tx - 6, ty - th - 4, tw + 12, th + 10, (0, 0, 0), 0.55)
        text(canvas, label, (tx, ty), 0.55, COL_YELLOW, 1)

        text(canvas, bottom_label, (cx + 8, cy + r - 8), 0.40, COL_MUTED, 1)
        text(canvas, right_label, (cx + r - 42, cy - 8), 0.40, COL_MUTED, 1)

    def _draw_torque_static(self, canvas):
        x0, y0, x1, y1 = self.tx0, self.ty0, self.tx1, self.ty1
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_PANEL, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_GRID, 1)
        text(
            canvas,
            "TORQUE / TIME (Nm)  |  sign inverted",
            (x0, y0 - 14),
            0.55,
            COL_TEXT,
            1,
        )

        # y grid / labels
        v = -self.torque_limit
        while v <= self.torque_limit + 1e-9:
            py = int(
                round(
                    (y0 + y1) / 2.0
                    - (v / self.torque_limit) * ((y1 - y0) / 2.0)
                )
            )
            cv2.line(canvas, (x0 + 1, py), (x1 - 1, py), COL_GRID, 1)
            label = "0" if abs(v) < self.torque_step * 1e-6 else f"{v:g}"
            text_right(canvas, label, x0 - 10, py + 5, 0.42, COL_MUTED, 1)
            v += self.torque_step

        # x grid / time labels
        tstep = nice_step(self.duration, target=10.0)
        tt = 0.0
        while tt <= self.duration + 1e-9:
            px = int(round(x0 + (tt / self.duration) * (x1 - x0)))
            cv2.line(canvas, (px, y0 + 1), (px, y1 - 1), COL_GRID, 1)
            text(canvas, f"{tt:g}", (px - 12, y1 + 22), 0.42, COL_MUTED, 1)
            tt += tstep

        cv2.line(
            canvas,
            (x0 + 1, (y0 + y1) // 2),
            (x1 - 1, (y0 + y1) // 2),
            COL_AXIS,
            1,
        )

        # full dim curves, drawn once
        for name, _col, dim in self._torque_series:
            pts = self.torque_pts[name]
            if len(pts) > 1:
                cv2.polylines(canvas, [pts], False, dim, 1, cv2.LINE_AA)

        # legend box and color keys, drawn once
        lx = x1 - 320
        ly = y0 + 14
        put_box(canvas, lx - 8, ly - 6, 320, 92, (0, 0, 0), 0.62)
        for i, (_name, col, _dim) in enumerate(self._torque_series):
            yy = ly + 20 + i * 24
            cv2.line(canvas, (lx, yy - 4), (lx + 22, yy - 4), col, 3, cv2.LINE_AA)
        text(canvas, "Nm", (lx + 30, ly + 86), 0.40, COL_MUTED, 1)

    def _draw_torque(self, canvas, t, tx, ty, tz):
        x0, y0, x1, y1 = self.tx0, self.ty0, self.tx1, self.ty1
        vals = {"tx": tx, "ty": ty, "tz": tz}

        j = int(np.searchsorted(self.torque_ds_t, t, side="right"))
        for name, col, _dim in self._torque_series:
            pts = self.torque_pts[name]
            if j > 1:
                cv2.polylines(canvas, [pts[:j]], False, col, 2, cv2.LINE_AA)

        cxp = int(round(x0 + (t / self.duration) * (x1 - x0)))
        blend_vline(canvas, cxp, y0 + 1, y1 - 1, COL_YELLOW, 0.85, 2)
        for name, col, _dim in self._torque_series:
            _, py = self._torque_xy(np.array([t]), np.array([vals[name]]))
            cv2.circle(canvas, (cxp, int(py[0])), 4, col, -1, cv2.LINE_AA)

        # current values in the static legend box
        lx = x1 - 320
        ly = y0 + 14
        for i, (name, col, _dim) in enumerate(self._torque_series):
            yy = ly + 20 + i * 24
            text(
                canvas,
                f"{name}  {format_n(vals[name])}",
                (lx + 30, yy),
                0.50,
                col,
                1,
            )

    def _draw_fz_static(self, canvas):
        x0, y0, x1, y1 = self.gx0, self.gy0, self.gx1, self.gy1
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_PANEL, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_GRID, 1)
        text(canvas, "FZ / TIME (N)", (x0, y0 - 14), 0.55, COL_TEXT, 1)

        # horizontal grid / y labels
        v = self.fz_min
        while v <= self.fz_max + 1e-9:
            _, py = self._fz_xy(0.0, np.array([v]))
            py = int(py[0])
            cv2.line(canvas, (x0 + 1, py), (x1 - 1, py), COL_GRID, 1)
            text_right(canvas, f"{v:g}", x0 - 10, py + 5, 0.42, COL_MUTED, 1)
            v += self.fz_step

        # vertical grid / time labels
        tstep = nice_step(self.duration, target=7.0)
        tt = 0.0
        while tt <= self.duration + 1e-9:
            px, _ = self._fz_xy(np.array([tt]), self.fz_min)
            px = int(px[0])
            cv2.line(canvas, (px, y0 + 1), (px, y1 - 1), COL_GRID, 1)
            text(canvas, f"{tt:g}", (px - 12, y1 + 22), 0.42, COL_MUTED, 1)
            tt += tstep

        # full fz curve, dim, drawn once
        if len(self.fz_full_pts) > 1:
            cv2.polylines(
                canvas, [self.fz_full_pts], False, (92, 104, 88), 1, cv2.LINE_AA
            )

    def _draw_fz(self, canvas, t, fz):
        x0, y0, x1, y1 = self.gx0, self.gy0, self.gx1, self.gy1
        text_right(canvas, f"t = {t:.2f} s", x1, y0 - 14, 0.50, COL_MUTED, 1)

        # played part, bright
        j = int(np.searchsorted(self.fz_ds_t, t, side="right"))
        if j > 1:
            cv2.polylines(
                canvas, [self.fz_full_pts[:j]], False, COL_GREEN, 2, cv2.LINE_AA
            )

        # cursor
        cxp = float(self._fz_xy(np.array([t]), np.array([fz]))[0][0])
        cyp = float(self._fz_xy(np.array([t]), np.array([fz]))[1][0])
        cxp_i, cyp_i = int(round(cxp)), int(round(cyp))
        blend_vline(canvas, cxp_i, y0 + 1, y1 - 1, COL_YELLOW, 0.85, 2)
        cv2.circle(canvas, (cxp_i, cyp_i), 5, COL_YELLOW, -1, cv2.LINE_AA)

        # value badge
        label = f"fz = {fz:+.3f} N"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        bx = cxp_i + 12
        if bx + tw + 16 > x1:
            bx = cxp_i - tw - 28
        bx = max(bx, x0 + 8)
        by = y0 + 14
        put_box(canvas, bx, by, tw + 16, th + 12, (0, 0, 0), 0.72)
        text(canvas, label, (bx + 8, by + th + 4), 0.62, COL_YELLOW, 1)

    # ------------------------------------------------------------------

    def _build_static(self):
        self.static[:] = COL_BG
        self._draw_event_static(self.static)
        self._draw_torque_static(self.static)
        self._draw_fz_static(self.static)
        text(self.static, self.bag_name, (848, 40), 0.62, COL_TEXT, 1)
        text_right(
            self.static,
            "EVENT + FORCE + TORQUE VISUALIZATION",
            CANVAS_W - 24,
            40,
            0.48,
            COL_MUTED,
            1,
        )

    def render(self, t, event_img, fx, fy, fz, tx, ty, tz):
        canvas = self.canvas
        np.copyto(canvas, self.static)

        self._draw_event_panel(canvas, event_img, t)
        self._draw_force_hud(canvas, t, fx, fy)
        self._draw_torque(canvas, t, tx, ty, tz)
        self._draw_fz(canvas, t, fz)
        return canvas


# ---------------------------------------------------------------------------
# bag processing
# ---------------------------------------------------------------------------


def force_at(t, f_t, f_fx, f_fy, f_fz, f_tx, f_ty, f_tz):
    return (
        float(np.interp(t, f_t, f_fx)),
        float(np.interp(t, f_t, f_fy)),
        float(np.interp(t, f_t, f_fz)),
        float(np.interp(t, f_t, f_tx)),
        float(np.interp(t, f_t, f_ty)),
        float(np.interp(t, f_t, f_tz)),
    )


def make_ffmpeg(out_path, fps, crf, preset="veryfast", encoder="libx264"):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{CANVAS_W}x{CANVAS_H}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
    ]
    if encoder == "h264_nvenc":
        nv_preset = {
            "ultrafast": "fast",
            "veryfast": "fast",
            "fast": "fast",
            "medium": "medium",
            "slow": "slow",
        }.get(preset, "medium")
        cmd += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            nv_preset,
            "-cq",
            str(crf),
            "-b:v",
            "0",
        ]
    else:
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def process_bag(bag_path, args):
    bag = rosbag.Bag(bag_path, "r")
    try:
        event_topic, event_type, force_topic, force_type = detect_topics(bag)
        stem = os.path.splitext(os.path.basename(bag_path))[0]
        tag = f"[{stem}]"
        log(f"{tag} events : {event_topic} ({event_type})")
        log(f"{tag} force  : {force_topic} ({force_type})")

        t0 = bag.get_start_time()
        t1 = bag.get_end_time()
        if args.start is not None:
            t0 = max(t0, bag.get_start_time() + args.start)
        if args.end is not None:
            t1 = min(t1, bag.get_start_time() + args.end)
        if t1 <= t0:
            raise RuntimeError("empty time window; check --start/--end")
        duration = t1 - t0

        shift_s, shift_source = resolve_force_shift(
            stem, args, getattr(args, "sync_calib_data", None)
        )
        if shift_source != "none":
            log(
                f"{tag} sync   : force shift {shift_s*1000:+.1f} ms "
                f"({shift_source})"
            )

        # Read enough force data for interpolation after the time shift.
        force_margin = max(1.0, abs(shift_s) + 0.5)
        (
            f_t,
            f_fx,
            f_fy,
            f_fz,
            f_tx,
            f_ty,
            f_tz,
        ) = read_force(
            bag,
            force_topic,
            force_type,
            start_time=max(bag.get_start_time(), t0 - force_margin),
            end_time=min(bag.get_end_time(), t1 + force_margin),
        )
        # The sensor is mounted on the contact side and reports the reaction
        # force/torque. Flip the signs so the plots show contact values.
        f_fx = -f_fx
        f_fy = -f_fy
        f_fz = -f_fz
        f_tx = -f_tx
        f_ty = -f_ty
        f_tz = -f_tz
        log(
            f"{tag} force  : (sign inverted) "
            f"{len(f_t)} samples, fx/fy=[{f_fx.min():.2f}, {f_fx.max():.2f}]/"
            f"[{f_fy.min():.2f}, {f_fy.max():.2f}], fz=[{f_fz.min():.2f}, {f_fz.max():.2f}]"
        )
        log(
            f"{tag} torque : "
            f"tx/ty=[{f_tx.min():.3f}, {f_tx.max():.3f}]/"
            f"[{f_ty.min():.3f}, {f_ty.max():.3f}], "
            f"tz=[{f_tz.min():.3f}, {f_tz.max():.3f}] Nm"
        )

        # Restrict force samples to the shifted render window so the shifted
        # force curve still spans exactly [0, duration].
        sel_lo = t0 - shift_s - 1e-6
        sel_hi = t1 - shift_s + 1e-6
        sel = (f_t >= sel_lo) & (f_t <= sel_hi)
        if sel.sum() < 2:
            sel = np.ones(len(f_t), dtype=bool)
        f_t = f_t[sel]
        f_fx = f_fx[sel]
        f_fy = f_fy[sel]
        f_fz = f_fz[sel]
        f_tx = f_tx[sel]
        f_ty = f_ty[sel]
        f_tz = f_tz[sel]
        f_t_rel = f_t - t0 + shift_s

        rate = estimate_event_rate(bag, event_topic, t0, window=min(2.0, duration))
        rate_per_px = rate / (480.0 * 640.0)

        mode = args.event_mode
        if mode == "auto":
            mode = "density" if rate > 2.0e6 else "surface"

        if args.event_tau is not None:
            tau = args.event_tau
        elif mode == "density":
            tau = 0.20
        else:
            tau = float(np.clip(0.35 / max(rate_per_px, 1e-6), 0.035, 0.18))
        log(
            f"{tag} events : ~{rate/1e6:.2f} M ev/s, "
            f"mode = {mode}, display tau = {tau*1000:.1f} ms"
        )

        # Find the actual event resolution from the first non-empty message.
        first = None
        for item in bag.read_messages(topics=[event_topic], raw=True):
            parsed = parse_event_raw(get_raw_event(item))
            if parsed is not None:
                first = parsed
                break
        if first is None:
            raise RuntimeError("event stream is empty")
        event_h, event_w = first[4], first[5]
        if (event_h, event_w) != (480, 640):
            log(f"{tag} warning: event resolution is {event_w}x{event_h}, not 640x480")

        surface = (
            DensitySurface(event_h, event_w, tau, args.fps, t0=t0)
            if mode == "density"
            else EventSurface(event_h, event_w, tau, args.fps, t0=t0)
        )
        renderer = Renderer(
            stem,
            duration,
            f_t_rel,
            f_fx,
            f_fy,
            f_fz,
            f_tx,
            f_ty,
            f_tz,
            event_h,
            event_w,
            force_orientation=getattr(args, "force_orientation", "new"),
        )

        if args.preview is not None:
            return render_preview(
                bag, event_topic, surface, renderer, args, stem, t0, t1
            )

        n_frames = max(1, int(math.ceil(duration * args.fps)))
        out_path = os.path.join(args.out_dir, stem + ".mp4")
        log(f"{tag} video  : {out_path} ({n_frames} frames @ {args.fps} fps)")
        proc = make_ffmpeg(
            out_path, args.fps, args.crf, args.preset, args.encoder
        )

        t_start = time.time()
        frame_idx = 0
        total_events = 0
        pend_x, pend_y, pend_t, pend_p = [], [], [], []
        bar = ProgressBar(tag, n_frames, single_line=(getattr(args, "jobs", 1) == 1))

        def flush_pending():
            nonlocal pend_x, pend_y, pend_t, pend_p
            if not pend_x:
                return None
            if len(pend_x) == 1:
                arrs = (pend_x[0], pend_y[0], pend_t[0], pend_p[0])
            else:
                arrs = (
                    np.concatenate(pend_x),
                    np.concatenate(pend_y),
                    np.concatenate(pend_t),
                    np.concatenate(pend_p),
                )
            pend_x, pend_y, pend_t, pend_p = [], [], [], []
            return arrs

        def emit_frame(idx):
            nonlocal frame_idx
            t = min(t0 + (idx + 1) / args.fps, t1)
            t_rel = t - t0
            ev_img = surface.render(t)
            fx, fy, fz, tx, ty, tz = force_at(
                t_rel, f_t_rel, f_fx, f_fy, f_fz, f_tx, f_ty, f_tz
            )
            canvas = renderer.render(t_rel, ev_img, fx, fy, fz, tx, ty, tz)
            proc.stdin.write(canvas.tobytes())
            frame_idx = idx + 1
            bar.update(frame_idx)

        try:
            read_start = rospy.Time.from_sec(max(bag.get_start_time(), t0 - 0.05))
            read_end = rospy.Time.from_sec(min(bag.get_end_time(), t1 + 0.05))
            for item in bag.read_messages(
                topics=[event_topic],
                raw=True,
                start_time=read_start,
                end_time=read_end,
            ):
                parsed = parse_event_raw(get_raw_event(item))
                if parsed is None:
                    continue
                ex, ey, et, ep, eh, ew = parsed
                if (eh, ew) != (event_h, event_w):
                    raise RuntimeError("event resolution changed within the bag")
                if len(et) == 0:
                    continue
                keep = (et >= t0) & (et <= t1)
                if not keep.all():
                    ex, ey, et, ep = ex[keep], ey[keep], et[keep], ep[keep]
                if len(et) == 0:
                    continue

                total_events += len(et)
                pend_x.append(ex)
                pend_y.append(ey)
                pend_t.append(et)
                pend_p.append(ep)
                last_event_t = float(et[-1])

                if frame_idx >= n_frames:
                    break
                if last_event_t < t0 + (frame_idx + 1) / args.fps:
                    continue

                arrs = flush_pending()
                if arrs is None:
                    continue
                px, py, pt, pp = arrs
                while frame_idx < n_frames:
                    tf = t0 + (frame_idx + 1) / args.fps
                    if tf > last_event_t:
                        break
                    n = int(np.searchsorted(pt, tf, side="right"))
                    if n:
                        surface.add(px[:n], py[:n], pt[:n], pp[:n])
                        px, py, pt, pp = px[n:], py[n:], pt[n:], pp[n:]
                    emit_frame(frame_idx)
                    if frame_idx >= n_frames:
                        break
                if len(px):
                    pend_x, pend_y, pend_t, pend_p = [px], [py], [pt], [pp]

            if frame_idx < n_frames:
                arrs = flush_pending()
                if arrs is not None:
                    px, py, pt, pp = arrs
                    final_t = min(t0 + n_frames / args.fps, t1)
                    keep = pt <= final_t
                    if keep.any():
                        surface.add(px[keep], py[keep], pt[keep], pp[keep])
                while frame_idx < n_frames:
                    emit_frame(frame_idx)
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            rc = proc.wait()
            bar.close()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {rc}")

        elapsed = time.time() - t_start
        log(
            f"{tag} done   : {n_frames} frames, {total_events/1e6:.1f}M events, "
            f"{elapsed:.1f}s, {os.path.getsize(out_path)/1e6:.1f} MB"
        )
        return out_path
    finally:
        bag.close()


def render_preview(bag, event_topic, surface, renderer, args, stem, t0, t1):
    target = args.preview
    target_abs = t0 + target
    if target_abs > t1:
        target_abs = t1
        target = t1 - t0

    end_time = rospy.Time.from_sec(target_abs)
    # Only the recent past matters for either rendering mode.
    if surface.mode == "density":
        window = surface.tau
    else:
        window = max(0.5, surface.tau * 3.0)
    start_time = rospy.Time.from_sec(max(bag.get_start_time(), target_abs - window - 0.05))
    for item in bag.read_messages(
        topics=[event_topic], raw=True, start_time=start_time, end_time=end_time
    ):
        parsed = parse_event_raw(get_raw_event(item))
        if parsed is None:
            continue
        ex, ey, et, ep, _eh, _ew = parsed
        mask = (et <= target_abs) & (et > target_abs - window)
        if mask.any():
            surface.add(ex[mask], ey[mask], et[mask], ep[mask])

    ev_img = surface.render(target_abs)
    fx, fy, fz, tx, ty, tz = force_at(
        target,
        renderer.f_t,
        renderer.f_fx,
        renderer.f_fy,
        renderer.f_fz,
        renderer.f_tx,
        renderer.f_ty,
        renderer.f_tz,
    )
    canvas = renderer.render(target, ev_img, fx, fy, fz, tx, ty, tz)
    out = os.path.join(args.out_dir, f"preview_{stem}_{target:.1f}s.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cv2.imwrite(out, canvas)
    log(
        f"[{stem}] preview: {out} "
        f"(fx={fx:+.3f}, fy={fy:+.3f}, fz={fz:+.3f}, "
        f"tx={tx:+.3f}, ty={ty:+.3f}, tz={tz:+.3f})"
    )
    return out


# ---------------------------------------------------------------------------


def collect_bags(args):
    """Resolve positional paths, --bags, or --data-dir into a bag file list."""
    entries = []
    if args.paths:
        entries.extend(args.paths)
    elif args.bags:
        entries.extend(args.bags)
    else:
        entries.append(args.data_dir)

    bags = []
    seen = set()
    for entry in entries:
        entry = os.path.expanduser(entry)
        if os.path.isdir(entry):
            pattern = (
                os.path.join(entry, "**", "*.bag")
                if args.recursive
                else os.path.join(entry, "*.bag")
            )
            found = sorted(glob.glob(pattern, recursive=args.recursive))
        elif os.path.isfile(entry):
            found = [entry]
        else:
            matches = sorted(glob.glob(entry))
            found = [m for m in matches if os.path.isfile(m)]
            if not found:
                log(f"error: not found: {entry}")
                continue
        if not found:
            log(f"warning: no .bag files in {entry}")
        for path in found:
            absolute = os.path.abspath(path)
            if absolute not in seen:
                seen.add(absolute)
                bags.append(path)
    return bags


def main():
    ap = argparse.ArgumentParser(
        description="Render event-camera + force videos from ROS bags."
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="bag files or folders; overrides --bags/--data-dir",
    )
    ap.add_argument(
        "--data-dir",
        "--folder",
        dest="data_dir",
        default="data",
        help="directory containing .bag files (default: data)",
    )
    ap.add_argument(
        "--out-dir", default="output", help="output directory (default: output)"
    )
    ap.add_argument(
        "--bags", nargs="*", default=None, help="explicit bag path(s); overrides --data-dir"
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="search subdirectories when a folder is given",
    )
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="number of bags to process in parallel (0 = auto, default)",
    )
    ap.add_argument(
        "--preset",
        default="veryfast",
        help="x264 preset (default: veryfast; try medium for smaller files)",
    )
    ap.add_argument(
        "--encoder",
        choices=("libx264", "h264_nvenc"),
        default="libx264",
        help="video encoder (default: libx264; h264_nvenc uses the GPU)",
    )
    ap.add_argument("--fps", type=int, default=FPS_DEFAULT)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument(
        "--event-tau",
        type=float,
        default=None,
        help="event display decay constant in seconds (default: auto)",
    )
    ap.add_argument(
        "--event-mode",
        choices=("auto", "surface", "density"),
        default="auto",
        help="event rendering: last-event time surface, count density, or auto",
    )
    ap.add_argument(
        "--force-orientation",
        choices=("new", "old"),
        default="new",
        help="force HUD axes: new = fx- right / fy- down (default), "
        "old = fx+ down / fy- right",
    )
    ap.add_argument(
        "--sync-calib",
        default=None,
        metavar="JSON",
        help="event/force sync calibration JSON from sync_calibrate.py",
    )
    ap.add_argument(
        "--force-shift",
        type=float,
        default=None,
        metavar="SECONDS",
        help="manual force time shift; overrides --sync-calib",
    )
    ap.add_argument(
        "--no-sync",
        action="store_true",
        help="disable event/force time alignment",
    )
    ap.add_argument(
        "--preview",
        type=float,
        default=None,
        metavar="SECONDS",
        help="render one PNG at this time instead of encoding a video",
    )
    ap.add_argument("--start", type=float, default=None, help="start offset in seconds")
    ap.add_argument("--end", type=float, default=None, help="end offset in seconds")
    args = ap.parse_args()

    args.sync_calib_data = None
    if args.sync_calib:
        args.sync_calib_data = load_sync_calibration(args.sync_calib)
        log(f"sync calibration: {args.sync_calib}")

    bags = collect_bags(args)
    if not bags:
        log("error: no .bag files found")
        return 2

    auto_jobs = min(len(bags), os.cpu_count() or 1, 8)
    jobs = args.jobs if args.jobs > 0 else auto_jobs
    jobs = max(1, min(jobs, len(bags), os.cpu_count() or 1))
    args.jobs = jobs
    if jobs > 1:
        # Avoid oversubscribing OpenCV's internal thread pool when several
        # bags are rendered at the same time.
        cv2.setNumThreads(max(1, (os.cpu_count() or 1) // jobs))

    log(f"processing {len(bags)} bag(s) with {jobs} worker(s)")
    ok = True
    if jobs == 1:
        for bag_path in bags:
            try:
                process_bag(bag_path, args)
            except Exception as exc:
                log(f"error while processing {bag_path}: {exc}")
                ok = False
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(process_bag, bag_path, args): bag_path
                for bag_path in bags
            }
            for future in as_completed(futures):
                bag_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    log(f"error while processing {bag_path}: {exc}")
                    ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
