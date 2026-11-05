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
    /usr/bin/python3 bag_vis.py --preview 30
"""

import argparse
import glob
import math
import os
import struct
import subprocess
import sys
import time

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


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def log(msg):
    print(msg, flush=True)


def text(img, s, org, scale=0.55, color=COL_TEXT, thickness=1):
    cv2.putText(
        img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def text_right(img, s, right_x, y, scale=0.55, color=COL_TEXT, thickness=1):
    (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    text(img, s, (int(right_x - w), int(y)), scale, color, thickness)


def put_box(img, x, y, w, h, color, alpha=0.72):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


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


def read_force(bag, topic, msg_type):
    """Return (t, fx, fy, fz) as float64 arrays, sorted and deduplicated."""
    rows = []
    for _, msg, rec in bag.read_messages(topics=[topic]):
        try:
            if msg_type == "geometry_msgs/WrenchStamped":
                f = msg.wrench.force
                fx, fy, fz = f.x, f.y, f.z
            elif msg_type == "force_data_process/Force":
                fx, fy, fz = msg.fx, msg.fy, msg.fz
            else:
                raise RuntimeError(f"unsupported force message type: {msg_type}")

            stamp = msg.header.stamp.to_sec()
            if not np.isfinite(stamp) or stamp <= 0:
                stamp = rec.to_sec()
            rows.append((stamp, fx, fy, fz))
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
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


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

    def __init__(self, height, width, tau, fps=30):
        self.h = height
        self.w = width
        self.tau = float(tau)
        self.last_t = np.full((height, width), -1e18, dtype=np.float64)
        self.last_pol = np.zeros((height, width), dtype=np.int8)

    def add(self, x, y, t, pol):
        if len(x) == 0:
            return
        # x/y are uint16; cast once for fast fancy indexing
        xi = x.astype(np.int32, copy=False)
        yi = y.astype(np.int32, copy=False)
        self.last_t[yi, xi] = t
        self.last_pol[yi, xi] = np.where(pol, 1, -1).astype(np.int8, copy=False)

    def render(self, t_now, gamma=0.72):
        age = t_now - self.last_t
        np.maximum(age, 0.0, out=age)
        inten = np.exp(-age / self.tau).astype(np.float32)
        valid = self.last_t > -1e17
        inten *= valid

        if gamma != 1.0:
            np.power(inten, gamma, out=inten)

        pos = np.where(self.last_pol > 0, inten, 0.0).astype(np.float32)
        neg = np.where(self.last_pol < 0, inten, 0.0).astype(np.float32)
        if pos.any():
            pos = cv2.GaussianBlur(pos, (3, 3), 0)
        if neg.any():
            neg = cv2.GaussianBlur(neg, (3, 3), 0)

        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[..., 2] = np.clip(pos * 255.0, 0, 255).astype(np.uint8)  # red
        img[..., 0] = np.clip(neg * 255.0, 0, 255).astype(np.uint8)  # blue
        return img


class DensitySurface:
    """Exponentially decaying event-count image, normalized per frame.

    Better than a time surface when the sensor produces a high event rate and
    a large amount of background activity (session06).
    """

    mode = "density"

    def __init__(self, height, width, tau, fps=30):
        self.h = height
        self.w = width
        self.tau = float(tau)
        self.decay = float(math.exp(-1.0 / max(fps * tau, 1e-6)))
        self.acc_pos = np.zeros((height, width), dtype=np.float32)
        self.acc_neg = np.zeros((height, width), dtype=np.float32)

    def add(self, x, y, t, pol):
        if len(x) == 0:
            return
        idx = y.astype(np.int64) * self.w + x.astype(np.int64)
        n = self.h * self.w
        if pol.any():
            hist = np.bincount(idx[pol], minlength=n).reshape(self.h, self.w)
            self.acc_pos += hist.astype(np.float32)
        neg = ~pol
        if neg.any():
            hist = np.bincount(idx[neg], minlength=n).reshape(self.h, self.w)
            self.acc_neg += hist.astype(np.float32)

    def render(self, t_now, gamma=1.40):
        channels = []
        for acc in (self.acc_pos, self.acc_neg):
            a = np.log1p(acc)
            mx = float(np.percentile(a, 99.5)) if a.size else 0.0
            if mx > 1e-9:
                a /= mx
            np.clip(a * 0.60, 0.0, 1.0, out=a)
            if gamma != 1.0:
                np.power(a, gamma, out=a)
            channels.append(a)

        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[..., 2] = np.clip(channels[0] * 255.0, 0, 255).astype(np.uint8)
        img[..., 0] = np.clip(channels[1] * 255.0, 0, 255).astype(np.uint8)

        # Decay after rendering so this frame's events are fully visible.
        self.acc_pos *= self.decay
        self.acc_neg *= self.decay
        return img


# ---------------------------------------------------------------------------
# frame renderer
# ---------------------------------------------------------------------------


class Renderer:
    def __init__(self, bag_name, duration, f_t, f_fx, f_fy, f_fz, event_h, event_w):
        self.bag_name = bag_name
        self.duration = float(duration)
        self.f_t = f_t  # already relative to the rendered time window
        self.f_fx = f_fx
        self.f_fy = f_fy
        self.f_fz = f_fz
        self.event_h = event_h
        self.event_w = event_w

        # --- event panel geometry (portrait after 90 deg rotation) ---
        self.ew = 750
        self.eh = 1000
        self.ex = 24
        self.ey = 56

        # --- 2D force plane geometry ---
        self.px0, self.py0, self.px1, self.py1 = 852, 76, 1362, 586
        max_abs = float(np.max(np.abs(np.concatenate([f_fx, f_fy]))))
        max_abs = max(max_abs, 0.1)
        step = nice_step(max_abs * 2.16, target=6.0)
        limit = step * math.ceil((max_abs * 1.08) / step)
        self.plane_limit = max(limit, step)
        self.plane_step = step
        self.plane_cx = (self.px0 + self.px1) // 2
        self.plane_cy = (self.py0 + self.py1) // 2
        self.plane_scale = min(self.px1 - self.px0, self.py1 - self.py0) / (
            2.0 * self.plane_limit
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
        self.fz_max = math.ceil((fz_max * 1.05) / self.fz_step) * self.fz_step
        if self.fz_max <= self.fz_min:
            self.fz_max = self.fz_min + self.fz_step

        # Downsampled polyline for efficient fz rendering.
        n = len(self.f_t)
        if n > 4000:
            idx = np.linspace(0, n - 1, 4000).astype(np.int64)
            idx = np.unique(idx)
        else:
            idx = np.arange(n)
        self.fz_ds_t = self.f_t[idx]
        self.fz_ds_v = f_fz[idx]
        self.fz_full_pts = self._fz_points(self.fz_ds_t, self.fz_ds_v)

    # ------------------------------------------------------------------
    # coordinate maps
    # ------------------------------------------------------------------

    def _plane_xy(self, fx, fy):
        fx = np.asarray(fx, dtype=np.float64)
        fy = np.asarray(fy, dtype=np.float64)
        px = self.plane_cx + fx * self.plane_scale
        py = self.plane_cy - fy * self.plane_scale
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

    # ------------------------------------------------------------------
    # drawing helpers
    # ------------------------------------------------------------------

    def _draw_event_panel(self, canvas, event_img, t):
        x, y, w, h = self.ex, self.ey, self.ew, self.eh
        rot = cv2.rotate(event_img, cv2.ROTATE_90_CLOCKWISE)
        if rot.shape[1] != w or rot.shape[0] != h:
            rot = cv2.resize(rot, (w, h), interpolation=cv2.INTER_LINEAR)
        canvas[y : y + h, x : x + w] = rot
        cv2.rectangle(canvas, (x - 1, y - 1), (x + w, y + h), COL_GRID, 1)

        text(canvas, "EVENT CAMERA  |  90 deg CW", (x, 32), 0.58, COL_TEXT, 1)
        text(canvas, "red = ON   blue = OFF", (x, y + h - 16), 0.48, COL_MUTED, 1)

        label = f"t = {t:6.2f} s / {self.duration:6.2f} s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        put_box(canvas, x + 10, y + 10, tw + 20, th + 14, (0, 0, 0), 0.55)
        text(canvas, label, (x + 20, y + 10 + th + 4), 0.62, COL_TEXT, 1)

    def _draw_plane(self, canvas, t, fx, fy):
        x0, y0, x1, y1 = self.px0, self.py0, self.px1, self.py1
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_PANEL, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_GRID, 1)
        text(canvas, "FX-FY FORCE VECTOR (N)", (x0, y0 - 14), 0.55, COL_TEXT, 1)

        lim = self.plane_limit
        step = self.plane_step
        cx, cy, sc = self.plane_cx, self.plane_cy, self.plane_scale

        # grid + tick labels
        k = int(math.floor(-lim / step))
        while k * step <= lim + 1e-9:
            v = k * step
            if abs(v) > 1e-9:
                px = int(round(cx + v * sc))
                py = int(round(cy - v * sc))
                if x0 < px < x1:
                    cv2.line(canvas, (px, y0 + 1), (px, y1 - 1), COL_GRID, 1)
                if y0 < py < y1:
                    cv2.line(canvas, (x0 + 1, py), (x1 - 1, py), COL_GRID, 1)

                label = f"{v:g}"
                if x0 + 20 < px < x1 - 20:
                    text(canvas, label, (px - 12, y1 - 8), 0.40, COL_MUTED, 1)
                if y0 + 14 < py < y1 - 8:
                    text(canvas, label, (x0 + 5, py - 4), 0.40, COL_MUTED, 1)
            k += 1

        cv2.line(canvas, (x0 + 1, cy), (x1 - 1, cy), COL_AXIS, 1)
        cv2.line(canvas, (cx, y0 + 1), (cx, y1 - 1), COL_AXIS, 1)

        # recent trail (last 2.5 s)
        lo = int(np.searchsorted(self.f_t, max(t - 2.5, 0.0), side="left"))
        hi = int(np.searchsorted(self.f_t, t, side="right"))
        if hi > lo + 1:
            stride = max(1, (hi - lo) // 500)
            tt = self.f_t[lo:hi:stride]
            vx = self.f_fx[lo:hi:stride]
            vy = self.f_fy[lo:hi:stride]
            pts = np.column_stack(self._plane_xy(vx, vy)).astype(np.int32)
            cv2.polylines(canvas, [pts], False, COL_CYAN, 1, cv2.LINE_AA)

            # brighter last 0.5 s
            lo2 = int(np.searchsorted(self.f_t, max(t - 0.5, 0.0), side="left"))
            if hi > lo2 + 1:
                stride2 = max(1, (hi - lo2) // 200)
                pts2 = np.column_stack(
                    self._plane_xy(
                        self.f_fx[lo2:hi:stride2], self.f_fy[lo2:hi:stride2]
                    )
                ).astype(np.int32)
                cv2.polylines(canvas, [pts2], False, COL_CYAN, 2, cv2.LINE_AA)

        # current vector
        tip = (int(round(cx + fx * sc)), int(round(cy - fy * sc)))
        cv2.arrowedLine(
            canvas, (cx, cy), tip, COL_YELLOW, 3, cv2.LINE_AA, tipLength=0.12
        )
        cv2.circle(canvas, tip, 5, COL_YELLOW, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, COL_TEXT, -1, cv2.LINE_AA)

        if hi > 0:
            cur_label = f"({fx:+.2f}, {fy:+.2f})"
            (tw, th), _ = cv2.getTextSize(
                cur_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
            )
            tx = min(max(tip[0] + 10, x0 + 4), x1 - tw - 10)
            ty = min(max(tip[1] - 10, y0 + th + 6), y1 - 6)
            put_box(canvas, tx - 4, ty - th - 4, tw + 8, th + 8, (0, 0, 0), 0.55)
            text(canvas, cur_label, (tx, ty), 0.52, COL_YELLOW, 1)

        # numeric readout
        mag = float(math.hypot(fx, fy))
        rx = x1 + 20
        rows = (
            ("FX", f"{format_n(fx)} N", COL_RED),
            ("FY", f"{format_n(fy)} N", COL_BLUE),
            ("|FXY|", f"{mag:.3f} N", COL_CYAN),
        )
        yy = y0 + 52
        for name, val, col in rows:
            text(canvas, name, (rx, yy), 0.55, COL_MUTED, 1)
            text(canvas, val, (rx, yy + 42), 0.95, col, 2)
            yy += 118

    def _draw_fz(self, canvas, t, fz):
        x0, y0, x1, y1 = self.gx0, self.gy0, self.gx1, self.gy1
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_PANEL, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_GRID, 1)
        text(canvas, "FZ / TIME (N)", (x0, y0 - 14), 0.55, COL_TEXT, 1)
        text_right(canvas, f"t = {t:.2f} s", x1, y0 - 14, 0.50, COL_MUTED, 1)

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

        # full fz curve, dim
        if len(self.fz_full_pts) > 1:
            cv2.polylines(canvas, [self.fz_full_pts], False, (92, 104, 88), 1, cv2.LINE_AA)

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
        overlay = canvas.copy()
        cv2.line(overlay, (cxp_i, y0 + 1), (cxp_i, y1 - 1), COL_YELLOW, 2)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
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

    def render(self, t, event_img, fx, fy, fz):
        canvas = np.empty((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = COL_BG

        self._draw_event_panel(canvas, event_img, t)
        self._draw_plane(canvas, t, fx, fy)
        self._draw_fz(canvas, t, fz)

        text(canvas, self.bag_name, (848, 40), 0.62, COL_TEXT, 1)
        text_right(canvas, "EVENT + FORCE VISUALIZATION", CANVAS_W - 24, 40, 0.48, COL_MUTED, 1)
        return canvas


# ---------------------------------------------------------------------------
# bag processing
# ---------------------------------------------------------------------------


def force_at(t, f_t, f_fx, f_fy, f_fz):
    return (
        float(np.interp(t, f_t, f_fx)),
        float(np.interp(t, f_t, f_fy)),
        float(np.interp(t, f_t, f_fz)),
    )


def make_ffmpeg(out_path, fps, crf):
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
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def process_bag(bag_path, args):
    bag = rosbag.Bag(bag_path, "r")
    try:
        event_topic, event_type, force_topic, force_type = detect_topics(bag)
        stem = os.path.splitext(os.path.basename(bag_path))[0]
        log(f"\n[{stem}]")
        log(f"  events : {event_topic} ({event_type})")
        log(f"  force  : {force_topic} ({force_type})")

        t0 = bag.get_start_time()
        t1 = bag.get_end_time()
        duration = t1 - t0
        if args.start is not None:
            t0 = max(t0, bag.get_start_time() + args.start)
        if args.end is not None:
            t1 = min(t1, bag.get_start_time() + args.end)
        duration = max(t1 - t0, 1e-6)

        f_t, f_fx, f_fy, f_fz = read_force(bag, force_topic, force_type)
        log(
            "  force  : "
            f"{len(f_t)} samples, fx/fy=[{f_fx.min():.2f}, {f_fx.max():.2f}]/"
            f"[{f_fy.min():.2f}, {f_fy.max():.2f}], fz=[{f_fz.min():.2f}, {f_fz.max():.2f}]"
        )

        # Restrict force samples to the rendered time window (important for --start/--end).
        sel = (f_t >= t0 - 1e-6) & (f_t <= t1 + 1e-6)
        if sel.sum() < 2:
            sel = np.ones(len(f_t), dtype=bool)
        f_t = f_t[sel]
        f_fx = f_fx[sel]
        f_fy = f_fy[sel]
        f_fz = f_fz[sel]
        f_t_rel = f_t - t0

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
            f"  events : ~{rate/1e6:.2f} M ev/s, "
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
            log(f"  warning: event resolution is {event_w}x{event_h}, not 640x480")

        surface = (
            DensitySurface(event_h, event_w, tau, args.fps)
            if mode == "density"
            else EventSurface(event_h, event_w, tau, args.fps)
        )
        renderer = Renderer(stem, duration, f_t_rel, f_fx, f_fy, f_fz, event_h, event_w)

        if args.preview is not None:
            return render_preview(
                bag, event_topic, surface, renderer, args, stem, t0, t1
            )

        n_frames = max(1, int(math.ceil(duration * args.fps)))
        out_path = os.path.join(args.out_dir, stem + ".mp4")
        log(f"  video  : {out_path} ({n_frames} frames @ {args.fps} fps)")
        proc = make_ffmpeg(out_path, args.fps, args.crf)

        t_start = time.time()
        frame_idx = 0
        total_events = 0
        px = np.empty(0, np.uint16)
        py = np.empty(0, np.uint16)
        pt = np.empty(0, np.float64)
        pp = np.empty(0, bool)

        def emit_frame(idx):
            nonlocal frame_idx
            t = min(t0 + (idx + 1) / args.fps, t1)
            t_rel = t - t0
            ev_img = surface.render(t)
            fx, fy, fz = force_at(t_rel, f_t_rel, f_fx, f_fy, f_fz)
            canvas = renderer.render(t_rel, ev_img, fx, fy, fz)
            proc.stdin.write(canvas.tobytes())
            frame_idx = idx + 1
            if frame_idx % 150 == 0 or frame_idx == n_frames:
                elapsed = time.time() - t_start
                log(
                    f"    frame {frame_idx}/{n_frames} "
                    f"({100.0*frame_idx/n_frames:5.1f}%) {elapsed:6.1f}s"
                )

        try:
            read_start = rospy.Time.from_sec(max(bag.get_start_time(), t0 - 0.05))
            for item in bag.read_messages(
                topics=[event_topic], raw=True, start_time=read_start
            ):
                parsed = parse_event_raw(get_raw_event(item))
                if parsed is None:
                    continue
                ex, ey, et, ep, eh, ew = parsed
                if (eh, ew) != (event_h, event_w):
                    raise RuntimeError("event resolution changed within the bag")
                if len(et) and et[-1] < t0:
                    continue
                if len(et) and et[0] < t0:
                    keep = et >= t0
                    ex, ey, et, ep = ex[keep], ey[keep], et[keep], ep[keep]
                total_events += len(ex)

                if len(px):
                    px = np.concatenate([px, ex])
                    py = np.concatenate([py, ey])
                    pt = np.concatenate([pt, et])
                    pp = np.concatenate([pp, ep])
                else:
                    px, py, pt, pp = ex, ey, et, ep

                last_event_t = float(et[-1]) if len(et) else t0
                while frame_idx < n_frames:
                    tf = t0 + (frame_idx + 1) / args.fps
                    if tf > last_event_t:
                        break
                    n = int(np.searchsorted(pt, tf, side="right"))
                    if n == 0:
                        break
                    surface.add(px[:n], py[:n], pt[:n], pp[:n])
                    emit_frame(frame_idx)
                    px, py, pt, pp = px[n:], py[n:], pt[n:], pp[n:]
                    if frame_idx >= n_frames:
                        break
                if frame_idx >= n_frames:
                    break

            while frame_idx < n_frames:
                emit_frame(frame_idx)
        finally:
            proc.stdin.close()
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {rc}")

        elapsed = time.time() - t_start
        log(
            f"  done   : {n_frames} frames, {total_events/1e6:.1f}M events, "
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
    fx, fy, fz = force_at(
        target,
        renderer.f_t,
        renderer.f_fx,
        renderer.f_fy,
        renderer.f_fz,
    )
    canvas = renderer.render(target, ev_img, fx, fy, fz)
    out = os.path.join(args.out_dir, f"preview_{stem}_{target:.1f}s.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cv2.imwrite(out, canvas)
    log(f"  preview: {out} (fx={fx:+.3f}, fy={fy:+.3f}, fz={fz:+.3f})")
    return out


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Render event-camera + force videos from ROS bags."
    )
    ap.add_argument(
        "--data-dir", default="data", help="directory containing .bag files (default: data)"
    )
    ap.add_argument(
        "--out-dir", default="output", help="output directory (default: output)"
    )
    ap.add_argument(
        "--bags", nargs="*", default=None, help="explicit bag path(s); overrides --data-dir"
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
        "--preview",
        type=float,
        default=None,
        metavar="SECONDS",
        help="render one PNG at this time instead of encoding a video",
    )
    ap.add_argument("--start", type=float, default=None, help="start offset in seconds")
    ap.add_argument("--end", type=float, default=None, help="end offset in seconds")
    args = ap.parse_args()

    if args.bags:
        bags = list(args.bags)
    else:
        bags = sorted(glob.glob(os.path.join(args.data_dir, "*.bag")))
    if not bags:
        log(f"error: no .bag files found in {args.data_dir!r}")
        return 2

    ok = True
    for bag_path in bags:
        if not os.path.isfile(bag_path):
            log(f"error: not a file: {bag_path}")
            ok = False
            continue
        try:
            process_bag(bag_path, args)
        except Exception as exc:
            log(f"error while processing {bag_path}: {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
