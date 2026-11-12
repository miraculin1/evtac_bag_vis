#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Dedicated visualization for the periodic global flicker in session06.

The event stream contains a very short (~1 ms) global burst every ~66.5 ms
(~15 Hz), followed by a low-rate decay tail.  This script detects the burst
train, builds a phase-averaged event image over one flicker cycle, and renders:

  output/session06_flicker_analysis.png  - static diagnostic figure
  output/session06_flicker.mp4           - animated flicker analysis

Run with the system Python:
    /usr/bin/python3 flicker_vis.py
"""

import argparse
import math
import os
import struct
import subprocess
import sys
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import rospy

EVENT_TOPIC = "/capture_node/events"
EVENT_DT = np.dtype(
    [
        ("x", "<u2"),
        ("y", "<u2"),
        ("sec", "<u4"),
        ("nsec", "<u4"),
        ("pol", "u1"),
    ]
)

SENSOR_H, SENSOR_W = 480, 640
BIN01 = 1e-4  # 0.1 ms
BIN1 = 1e-3  # 1 ms
PHASES = 128  # phase bins per flicker cycle

# BGR palette
COL_BG = (18, 16, 15)
COL_PANEL = (32, 28, 25)
COL_GRID = (58, 52, 47)
COL_AXIS = (95, 88, 82)
COL_TEXT = (236, 236, 236)
COL_MUTED = (150, 148, 144)
COL_FLASH = (80, 100, 255)
COL_GREEN = (118, 220, 130)
COL_YELLOW = (80, 212, 255)
COL_CYAN = (238, 214, 90)


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


# ---------------------------------------------------------------------------
# rosbag reading
# ---------------------------------------------------------------------------


def get_raw_event(item):
    data = item[1]
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, (tuple, list)):
        for entry in data:
            if isinstance(entry, (bytes, bytearray)):
                return bytes(entry)
    raise TypeError("cannot find raw bytes in rosbag raw message")


def parse_event_raw(raw):
    off = 0
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
            f"unexpected EventArray layout: {len(raw)} bytes, {n_events} events"
        )

    ev = np.frombuffer(raw, dtype=EVENT_DT, count=n_events, offset=off)
    t = ev["sec"].astype(np.float64) + ev["nsec"].astype(np.float64) * 1e-9
    return ev["x"], ev["y"], t, ev["pol"] != 0, int(height), int(width)


# ---------------------------------------------------------------------------
# flicker analysis
# ---------------------------------------------------------------------------


def scan_event_rate(bag, topic):
    """Build global 0.1 ms and 1 ms event-count histograms."""
    t0 = bag.get_start_time()
    t1 = bag.get_end_time()
    duration = t1 - t0

    n01 = int(duration / BIN01) + 2
    cnt01 = np.zeros(n01, dtype=np.int64)
    total = 0
    positive = 0

    for item in bag.read_messages(topics=[topic], raw=True):
        parsed = parse_event_raw(get_raw_event(item))
        if parsed is None:
            continue
        x, y, t, pol, _h, _w = parsed
        ts = t - t0
        idx = np.clip((ts / BIN01).astype(np.int64), 0, n01 - 1)
        cnt01 += np.bincount(idx, minlength=n01)[:n01]
        total += len(ts)
        positive += int(pol.sum())

    n1 = int(duration / BIN1) + 2
    usable = n1 * 10
    if len(cnt01) < usable:
        cnt01 = np.pad(cnt01, (0, usable - len(cnt01)))
    cnt1 = cnt01[:usable].reshape(n1, 10).sum(axis=1)

    return duration, cnt01, cnt1, total, positive


def detect_peaks(cnt1, threshold=50000.0):
    """Group 1 ms bins above threshold and return peak times in seconds."""
    idx = np.where(cnt1 > threshold)[0]
    if len(idx) == 0:
        return np.empty(0, dtype=np.float64)

    groups = []
    start = idx[0]
    prev = idx[0]
    for q in idx[1:]:
        if q - prev > 2:
            groups.append((start, prev))
            start = q
        prev = q
    groups.append((start, prev))

    peaks = np.empty(len(groups), dtype=np.float64)
    for i, (a, b) in enumerate(groups):
        k = a + int(np.argmax(cnt1[a : b + 1]))
        peaks[i] = (k + 0.5) * BIN1
    return peaks


def refine_peaks(peaks, cnt01):
    """Use the 0.1 ms histogram to locate each burst peak precisely."""
    refined = np.empty_like(peaks)
    for i, p in enumerate(peaks):
        k = int(round(p / BIN01))
        lo = max(0, k - 12)
        hi = min(len(cnt01), k + 13)
        if hi <= lo:
            refined[i] = p
            continue
        refined[i] = (lo + int(np.argmax(cnt01[lo:hi]))) * BIN01
    return refined


def estimate_period(peaks):
    d = np.diff(peaks)
    d = d[(d > 0.04) & (d < 0.10)]
    if len(d) == 0:
        return 0.0665
    return float(np.median(d))


def compute_recovery(cnt1, peaks, period):
    """Measure the post-burst recovery time to the P95 quiet baseline.

    Recovery is the first 1 ms bin after the burst where two consecutive
    bins fall below the 95th percentile of the quiet-period event rate.
    """
    peak_bins = np.array([int(round(p / BIN1)) for p in peaks], dtype=np.int64)

    quiet_mask = np.ones(len(cnt1), dtype=bool)
    for p in peak_bins:
        quiet_mask[max(0, p) : min(len(cnt1), p + 25)] = False
    quiet = cnt1[quiet_mask]
    baseline = float(np.median(quiet))
    threshold = float(np.percentile(quiet, 95))

    max_len = max(1, int(round(period / BIN1)))
    recs = []
    for p in peak_bins:
        seg = cnt1[p + 1 : p + 1 + max_len]
        if len(seg) < 3:
            continue
        found = None
        for t in range(len(seg) - 1):
            if seg[t] <= threshold and seg[t + 1] <= threshold:
                found = float(t + 1)
                break
        if found is not None:
            recs.append(found)
    recs = np.asarray(recs, dtype=np.float64)
    recs = recs[np.isfinite(recs)]
    if len(recs) == 0:
        return baseline, threshold, recs, 0.0, 0.0
    return (
        baseline,
        threshold,
        recs,
        float(np.median(recs)),
        float(np.percentile(recs, 90)),
    )


def triggered_average(cnt01, peaks, period):
    nph = int(round(period / BIN01))
    acc = np.zeros(nph, dtype=np.float64)
    n = 0
    for p in peaks:
        i = int(round(p / BIN01))
        if i + nph >= len(cnt01):
            continue
        acc += cnt01[i : i + nph]
        n += 1
    if n == 0:
        raise RuntimeError("no complete flicker cycles available")
    return acc / n, n


def compute_fft(cnt1, start_s, duration_s):
    i0 = max(0, int(start_s / BIN1))
    i1 = min(len(cnt1), i0 + int(duration_s / BIN1))
    y = cnt1[i0:i1].astype(np.float64)
    y -= y.mean()
    if len(y) < 16:
        return None
    f = np.fft.rfftfreq(len(y), d=BIN1)
    sp = np.abs(np.fft.rfft(y))
    return f, sp


def build_phase_images(
    bag, topic, phase_start, phase_duration, t_ref, period, n_phases=PHASES
):
    """Average event images into phase bins over one flicker cycle."""
    acc = np.zeros((n_phases, SENSOR_H, SENSOR_W), dtype=np.int64)
    t_end = phase_start + phase_duration
    t0 = bag.get_start_time()

    for item in bag.read_messages(
        topics=[topic],
        raw=True,
        start_time=rospy.Time.from_sec(t0 + phase_start),
        end_time=rospy.Time.from_sec(t0 + t_end),
    ):
        parsed = parse_event_raw(get_raw_event(item))
        if parsed is None:
            continue
        x, y, t, pol, _h, _w = parsed
        t_rel = t - t0
        mask = (t_rel >= phase_start) & (t_rel < t_end)
        if not mask.any():
            continue
        x = x[mask]
        y = y[mask]
        t_rel = t_rel[mask]

        phase = ((t_rel - t_ref) % period) / period
        pi = np.clip((phase * n_phases).astype(np.int32), 0, n_phases - 1)
        pix = y.astype(np.int64) * SENSOR_W + x.astype(np.int64)

        for p in np.unique(pi):
            m = pi == p
            hist = np.bincount(pix[m], minlength=SENSOR_H * SENSOR_W)
            acc[p] += hist.reshape(SENSOR_H, SENSOR_W).astype(np.int64)

    log_acc = np.log1p(acc.astype(np.float32))
    vals = log_acc[log_acc > 0]
    vmax = float(np.percentile(vals, 99.9)) if len(vals) else 1.0
    imgs = np.clip(log_acc / max(vmax, 1e-6), 0.0, 1.0)
    phase_totals = acc.sum(axis=(1, 2))
    return imgs, phase_totals, acc


# ---------------------------------------------------------------------------
# static figure
# ---------------------------------------------------------------------------


def make_static_figure(
    out_path,
    cnt1,
    cnt01,
    peaks,
    period,
    avg,
    fft,
    phase_imgs,
    phase_totals,
    total_events,
    positive,
    threshold,
    start_s,
    recovery,
):
    plt.rcParams.update(
        {
            "figure.facecolor": "#12100f",
            "savefig.facecolor": "#12100f",
            "axes.facecolor": "#1e1a17",
            "axes.edgecolor": "#4a443e",
            "text.color": "#ececec",
            "axes.labelcolor": "#cfcfcf",
            "xtick.color": "#b0aca6",
            "ytick.color": "#b0aca6",
            "grid.color": "#3a3530",
            "font.size": 10,
        }
    )

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    gs = fig.add_gridspec(
        3, 12, height_ratios=[2.0, 2.0, 4.7], hspace=0.48, wspace=0.85
    )
    rec_baseline, rec_threshold, recs, rec_median, rec_p90 = recovery

    # ---- 1 ms rate trace -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0:8])
    i0 = max(0, int(start_s / BIN1))
    i1 = min(len(cnt1), i0 + 1000)
    seg = cnt1[i0:i1]
    tt = np.arange(i0, i1) * BIN1
    for p in peaks:
        if tt[0] <= p <= tt[-1]:
            ax.axvspan(
                p,
                p + rec_median / 1000.0,
                color="#3f6f8f",
                alpha=0.16,
                lw=0,
            )
    ax.plot(tt, np.maximum(seg, 1), lw=0.8, color="#ff6b4a")
    ax.axhline(rec_baseline, color="#7aa6b8", lw=0.7, ls="--", alpha=0.8)
    ax.axhline(rec_threshold, color="#ffd76a", lw=0.8, ls="--", alpha=0.9)
    ax.set_xlim(tt[0], tt[-1])
    ax.set_yscale("log")
    ax.set_ylim(1, max(cnt1.max() * 1.5, 100))
    ax.set_title(
        "global event rate, 1 ms bins (log scale; shaded = recovery)", loc="left"
    )
    ax.set_ylabel("events / ms")
    ax.grid(True, alpha=0.35)
    ax.text(
        0.99,
        0.93,
        f"peak {cnt1.max()/1e3:.0f}k ev/ms\n"
        f"baseline {rec_baseline:.0f} / threshold {rec_threshold:.0f} ev/ms\n"
        f"median recovery {rec_median:.0f} ms",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#ffd76a",
        fontsize=9,
        bbox=dict(facecolor="#12100f", edgecolor="#5a5048", alpha=0.8),
    )

    # ---- FFT -------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 8:12])
    if fft is not None:
        f, sp = fft
        norm = sp / max(sp.max(), 1e-9)
        sel = f <= 200
        ax.plot(f[sel], norm[sel], lw=0.9, color="#ffb347")
        for h in range(1, 14):
            fh = h / period
            if fh <= 200:
                ax.axvline(fh, color="#5a5048", lw=0.5, alpha=0.5)
        for fh, label in [(1 / period, "15"), (2 / period, "30"), (4 / period, "60")]:
            ax.annotate(
                f"{label} Hz",
                xy=(fh, 0.55),
                xytext=(fh + 3, 0.72),
                color="#ffd76a",
                fontsize=8,
                arrowprops=dict(arrowstyle="-", color="#8a8078", lw=0.7),
            )
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1.08)
    ax.set_title("FFT of 1 ms event rate", loc="left")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalized amplitude")
    ax.grid(True, alpha=0.35)

    # ---- triggered average ----------------------------------------------
    ax = fig.add_subplot(gs[1, 0:6])
    ph_ms = np.arange(len(avg)) * BIN01 * 1000.0
    ax.axvspan(0, rec_median, color="#7ee787", alpha=0.08, lw=0)
    ax.plot(ph_ms, avg, lw=1.0, color="#ff6b4a")
    ax.fill_between(ph_ms, 0, avg, color="#ff6b4a", alpha=0.18)
    ax.axvline(0, color="#ffd76a", lw=0.7, alpha=0.7)
    ax.axhline(
        rec_threshold / 10.0, color="#ffd76a", lw=0.8, ls="--", alpha=0.9
    )
    ax.axvline(rec_median, color="#7ee787", lw=1.4)
    ax.set_xlim(0, period * 1000)
    ax.set_ylim(0, avg.max() * 1.12)
    ax.set_title("burst-triggered average (0.1 ms bins)", loc="left")
    ax.set_xlabel("phase in flicker cycle (ms)")
    ax.set_ylabel("events / 0.1 ms")
    ax.grid(True, alpha=0.35)
    ax.text(
        0.99,
        0.93,
        f"period {period*1000:.2f} ms   |   {1/period:.2f} Hz\n"
        f"threshold {rec_threshold:.0f} ev/ms   |   recovery {rec_median:.0f} ms (median)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#ffd76a",
        fontsize=9,
    )

    # ---- burst onset zoom ------------------------------------------------
    ax = fig.add_subplot(gs[1, 6:9])
    z = avg[: int(0.010 / BIN01)]
    tz = np.arange(len(z)) * BIN01 * 1000.0
    ax.plot(tz, z, lw=1.2, color="#ff6b4a")
    ax.fill_between(tz, 0, z, color="#ff6b4a", alpha=0.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, z.max() * 1.12)
    ax.set_title("burst onset, 0-10 ms", loc="left")
    ax.set_xlabel("time after burst peak (ms)")
    ax.set_ylabel("events / 0.1 ms")
    ax.grid(True, alpha=0.35)

    # ---- recovery time histogram ----------------------------------------
    ax = fig.add_subplot(gs[1, 9:12])
    if len(recs):
        hi = max(40.0, rec_p90 + 5.0)
        bins = np.arange(0, hi + 2.0, 2.0)
        ax.hist(recs, bins=bins, color="#ff6b4a", alpha=0.75, edgecolor="#12100f")
        ax.axvline(rec_median, color="#7ee787", lw=1.4)
        ax.axvline(rec_p90, color="#ffd76a", lw=1.0, ls="--")
        ax.text(
            0.97,
            0.95,
            f"median {rec_median:.0f} ms\nP90 {rec_p90:.0f} ms",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#ffd76a",
            fontsize=9,
        )
        ax.set_xlim(0, hi)
    ax.set_title("recovery time distribution", loc="left")
    ax.set_xlabel("time to normal rate (ms)")
    ax.set_ylabel("bursts")
    ax.grid(True, alpha=0.35)

    # ---- stats -----------------------------------------------------------
    # Key numbers are placed in the subtitle to avoid covering any panel.

    # ---- phase montage ---------------------------------------------------
    sub = gs[2, :].subgridspec(1, 8, wspace=0.05)
    picks = [0, 1, 2, 3, 4, 8, 16, 64]
    for k, pi in enumerate(picks):
        ax = fig.add_subplot(sub[0, k])
        img = cv2.rotate(
            (phase_imgs[pi] * 255).astype(np.uint8), cv2.ROTATE_90_CLOCKWISE
        )
        ax.imshow(img, cmap="inferno", vmin=0, vmax=255)
        ax.set_title(f"phase {pi*period/PHASES*1000:5.2f} ms", fontsize=9)
        ax.axis("off")

    fig.suptitle(
        "session06 - periodic global flicker in the event stream",
        fontsize=16,
        x=0.01,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.01,
        0.953,
        "A global flash (~1-2 ms) repeats every 66.7 ms; "
        "the current 200 ms event-density view averages it away.",
        fontsize=10,
        color="#bdb6ad",
        ha="left",
    )
    fig.text(
        0.01,
        0.928,
        f"{total_events/1e6:.1f} M events  |  {len(peaks)} bursts detected  |  "
        f"period {period*1000:.2f} ms ({1/period:.2f} Hz)  |  "
        f"peak 1 ms bin {cnt1.max()/1e3:.0f}k events  |  "
        f"peak 0.1 ms bin {cnt01.max()/1e3:.0f}k events  |  "
        f"baseline {rec_baseline:.0f} events/ms  |  "
        f"recovery median {rec_median:.0f} ms / P90 {rec_p90:.0f} ms  |  "
        f"positive {100*positive/max(total_events,1):.1f}%",
        fontsize=9,
        color="#8f8880",
        ha="left",
    )

    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    log(f"  static : {out_path}")


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------


def make_ffmpeg(out_path, fps, crf, w=1920, h=1080):
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
        f"{w}x{h}",
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


def draw_rate_panel(canvas, cnt1, peaks, recovery, t_now, rect, rate_max):
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_PANEL, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_GRID, 1)
    text(
        canvas,
        "GLOBAL EVENT RATE (1 ms, log)  |  yellow = normal threshold, shaded = recovery",
        (x0, y0 - 14),
        0.50,
        COL_TEXT,
        1,
    )

    i1 = int(t_now / BIN1)
    i0 = max(0, i1 - 1000)
    seg = cnt1[i0:i1]
    if len(seg) < 2:
        return

    rec_baseline, rec_threshold, recs, rec_median, rec_p90 = recovery
    plot_x0 = x0 + 70
    plot_y0 = y0 + 20
    plot_x1 = x1 - 16
    plot_y1 = y1 - 36
    xmax = max(t_now, 1e-6)
    xmin = max(0.0, t_now - 1.0)
    log_den = math.log1p(rate_max)

    def sx(t):
        return int(plot_x0 + (t - xmin) / max(xmax - xmin, 1e-9) * (plot_x1 - plot_x0))

    def sy(v):
        return int(plot_y1 - math.log1p(max(v, 0.0)) / log_den * (plot_y1 - plot_y0))

    # recovery bands
    for p in peaks:
        if p < xmin or p > xmax:
            continue
        xa = sx(max(p, xmin))
        xb = sx(min(p + rec_median / 1000.0, xmax))
        cv2.rectangle(canvas, (xa, plot_y0), (xb, plot_y1), (40, 58, 72), -1)

    # grid + y ticks (log decades)
    for v in (1, 10, 100, 1000, 10000, 100000):
        if v > rate_max:
            continue
        yy = sy(v)
        cv2.line(canvas, (plot_x0, yy), (plot_x1, yy), COL_GRID, 1)
        label = f"{v/1000:.0f}k" if v >= 1000 else str(v)
        text_right(canvas, label, plot_x0 - 8, yy + 5, 0.42, COL_MUTED, 1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        tt = xmin + (xmax - xmin) * frac
        xx = sx(tt)
        cv2.line(canvas, (xx, plot_y0), (xx, plot_y1), COL_GRID, 1)
        text(canvas, f"{tt:.2f}", (xx - 14, plot_y1 + 22), 0.42, COL_MUTED, 1)

    # baseline / recovery threshold
    for v, col in ((rec_baseline, (120, 150, 170)), (rec_threshold, COL_YELLOW)):
        yy = sy(v)
        cv2.line(canvas, (plot_x0, yy), (plot_x1, yy), col, 1, cv2.LINE_AA)

    pts = []
    for j, v in enumerate(seg):
        tt = (i0 + j) * BIN1
        pts.append((sx(tt), sy(float(v))))
    if len(pts) > 1:
        cv2.polylines(canvas, [np.asarray(pts, np.int32)], False, COL_FLASH, 1, cv2.LINE_AA)
    cv2.line(canvas, (plot_x1, plot_y0), (plot_x1, plot_y1), COL_YELLOW, 1)


def draw_average_panel(canvas, avg, period, phase_ms, rect, recovery):
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_PANEL, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_GRID, 1)
    text(canvas, "BURST-TRIGGERED AVERAGE  (0.1 ms bins)", (x0, y0 - 14), 0.55, COL_TEXT, 1)

    rec_baseline, rec_threshold, recs, rec_median, rec_p90 = recovery
    plot_x0 = x0 + 70
    plot_y0 = y0 + 20
    plot_x1 = x1 - 16
    plot_y1 = y1 - 34
    ymax = float(avg.max()) * 1.12

    def sx(ms):
        return int(plot_x0 + ms / (period * 1000.0) * (plot_x1 - plot_x0))

    def sy(v):
        return int(plot_y1 - v / max(ymax, 1e-9) * (plot_y1 - plot_y0))

    for frac in (0.0, 0.5, 1.0):
        v = ymax * frac
        yy = sy(v)
        cv2.line(canvas, (plot_x0, yy), (plot_x1, yy), COL_GRID, 1)
        text_right(canvas, f"{v/1e3:.1f}k", plot_x0 - 8, yy + 5, 0.42, COL_MUTED, 1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        ms = frac * period * 1000.0
        xx = sx(ms)
        cv2.line(canvas, (xx, plot_y0), (xx, plot_y1), COL_GRID, 1)
        text(canvas, f"{ms:.1f}", (xx - 12, plot_y1 + 20), 0.42, COL_MUTED, 1)

    # shaded recovery interval
    cv2.rectangle(
        canvas,
        (sx(0), plot_y0),
        (sx(rec_median), plot_y1),
        (44, 66, 48),
        -1,
    )
    pts = [
        (sx(i * BIN01 * 1000.0), sy(float(v))) for i, v in enumerate(avg)
    ]
    if len(pts) > 1:
        cv2.polylines(canvas, [np.asarray(pts, np.int32)], False, COL_FLASH, 2, cv2.LINE_AA)
    thr_y = sy(rec_threshold / 10.0)
    cv2.line(canvas, (plot_x0, thr_y), (plot_x1, thr_y), COL_YELLOW, 1, cv2.LINE_AA)
    rec_x = sx(rec_median)
    cv2.line(canvas, (rec_x, plot_y0), (rec_x, plot_y1), COL_GREEN, 2)
    text(
        canvas,
        f"recovery {rec_median:.0f} ms",
        (rec_x + 6, plot_y0 + 18),
        0.42,
        COL_GREEN,
        1,
    )
    cxp = sx(phase_ms)
    cv2.line(canvas, (cxp, plot_y0), (cxp, plot_y1), COL_YELLOW, 2)


def draw_fft_panel(canvas, fft, period, rect):
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_PANEL, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_GRID, 1)
    text(canvas, "FFT OF EVENT RATE", (x0, y0 - 14), 0.55, COL_TEXT, 1)

    if fft is None:
        return
    f, sp = fft
    norm = sp / max(sp.max(), 1e-9)
    sel = f <= 200
    f = f[sel]
    norm = norm[sel]

    plot_x0 = x0 + 70
    plot_y0 = y0 + 20
    plot_x1 = x1 - 16
    plot_y1 = y1 - 34

    def sx(freq):
        return int(plot_x0 + freq / 200.0 * (plot_x1 - plot_x0))

    def sy(v):
        return int(plot_y1 - min(v, 1.05) / 1.08 * (plot_y1 - plot_y0))

    for frac in (0.0, 0.5, 1.0):
        yy = sy(frac * 1.08)
        cv2.line(canvas, (plot_x0, yy), (plot_x1, yy), COL_GRID, 1)
    for freq in (0, 50, 100, 150, 200):
        xx = sx(freq)
        cv2.line(canvas, (xx, plot_y0), (xx, plot_y1), COL_GRID, 1)
        text(canvas, str(freq), (xx - 12, plot_y1 + 20), 0.42, COL_MUTED, 1)

    pts = [(sx(float(fi)), sy(float(v))) for fi, v in zip(f, norm)]
    if len(pts) > 1:
        cv2.polylines(canvas, [np.asarray(pts, np.int32)], False, (80, 180, 255), 1, cv2.LINE_AA)
    for h in range(1, 14):
        fh = h / period
        if fh <= 200:
            xx = sx(fh)
            yv = sy(norm[np.argmin(np.abs(f - fh))])
            cv2.circle(canvas, (xx, yv), 3, COL_YELLOW, -1, cv2.LINE_AA)


def draw_recovery_hist_panel(canvas, recovery, rect):
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_PANEL, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_GRID, 1)

    rec_baseline, rec_threshold, recs, rec_median, rec_p90 = recovery
    text(
        canvas,
        "RECOVERY TIME DISTRIBUTION",
        (x0, y0 - 14),
        0.55,
        COL_TEXT,
        1,
    )
    if len(recs) == 0:
        return

    hi = max(40.0, rec_p90 + 5.0)
    bins = np.arange(0.0, hi + 2.0, 2.0)
    hist, edges = np.histogram(recs, bins=bins)
    max_count = max(int(hist.max()), 1)

    plot_x0 = x0 + 60
    plot_y0 = y0 + 18
    plot_x1 = x1 - 16
    plot_y1 = y1 - 34

    def sx(ms):
        return int(plot_x0 + ms / hi * (plot_x1 - plot_x0))

    def sy(cnt):
        return int(plot_y1 - cnt / max_count * (plot_y1 - plot_y0))

    for frac in (0.0, 0.5, 1.0):
        yy = sy(max_count * frac)
        cv2.line(canvas, (plot_x0, yy), (plot_x1, yy), COL_GRID, 1)
        text_right(canvas, f"{max_count*frac:.0f}", plot_x0 - 8, yy + 5, 0.42, COL_MUTED, 1)
    for ms in np.arange(0, hi + 0.1, 10):
        xx = sx(ms)
        cv2.line(canvas, (xx, plot_y0), (xx, plot_y1), COL_GRID, 1)
        text(canvas, f"{ms:.0f}", (xx - 8, plot_y1 + 20), 0.42, COL_MUTED, 1)

    for i, h in enumerate(hist):
        xa = sx(edges[i])
        xb = sx(edges[i + 1]) - 2
        cv2.rectangle(canvas, (xa, sy(h)), (xb, plot_y1), (80, 110, 220), -1)

    xm = sx(rec_median)
    xp = sx(rec_p90)
    cv2.line(canvas, (xm, plot_y0), (xm, plot_y1), COL_GREEN, 2)
    cv2.line(canvas, (xp, plot_y0), (xp, plot_y1), COL_YELLOW, 1, cv2.LINE_AA)
    text_right(canvas, f"median {rec_median:.0f} ms", x1 - 16, y0 + 22, 0.42, COL_GREEN, 1)
    text_right(canvas, f"P90 {rec_p90:.0f} ms", x1 - 16, y0 + 42, 0.42, COL_YELLOW, 1)


def make_video(
    out_path,
    cnt1,
    cnt01,
    peaks,
    avg,
    period,
    fft,
    phase_imgs,
    phase_totals,
    start_s,
    duration_s,
    fps,
    crf,
    recovery,
):
    n_frames = max(1, int(round(duration_s * fps)))
    proc = make_ffmpeg(out_path, fps, crf)

    # Pre-render the phase images to the final portrait sizes.
    picks = [0, 1, 2, 3, 4, 8, 16, 64]
    phase_large = []
    phase_small = {}
    for i in range(PHASES):
        img = (phase_imgs[i] * 255.0).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        phase_large.append(cv2.resize(img, (480, 640), interpolation=cv2.INTER_LINEAR))
        if i in picks:
            phase_small[i] = cv2.resize(img, (60, 80), interpolation=cv2.INTER_LINEAR)

    rate_max = max(float(cnt1.max()) * 1.05, 1.0)
    w, h = 1920, 1080
    t_start = time.time()

    try:
        for fi in range(n_frames):
            t_now = start_s + (fi + 1) / fps
            phase_idx = fi % PHASES
            phase_ms = phase_idx * period / PHASES * 1000.0

            canvas = np.empty((h, w, 3), dtype=np.uint8)
            canvas[:] = COL_BG

            # ---- title ---------------------------------------------------
            text(canvas, "SESSION06  -  FLICKER ANALYSIS", (690, 42), 0.78, COL_TEXT, 2)
            text(
                canvas,
                f"{1/period:.2f} Hz burst train   |   period {period*1000:.2f} ms   |   "
                f"peak {cnt01.max()/1e3:.0f}k events / 0.1 ms   |   "
                f"recovery {recovery[3]:.0f} ms (median)",
                (690, 72),
                0.48,
                COL_MUTED,
                1,
            )

            # ---- phase image --------------------------------------------
            px, py, pw, ph = 84, 110, 480, 640
            cv2.putText(
                canvas,
                "PHASE-AVERAGED EVENT IMAGE  (slow-motion loop)",
                (24, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COL_TEXT,
                1,
                cv2.LINE_AA,
            )
            canvas[py : py + ph, px : px + pw] = phase_large[phase_idx]
            cv2.rectangle(canvas, (px - 1, py - 1), (px + pw, py + ph), COL_GRID, 1)

            # phase filmstrip: one representative image per phase
            text(canvas, "PHASE FILMSTRIP", (64, 772), 0.46, COL_MUTED, 1)
            film_y = 782
            thumb_w, thumb_h, gap = 60, 80, 8
            film_x = 64
            active_pick = picks[int(np.argmin(np.abs(np.array(picks) - phase_idx)))]
            for k, pi in enumerate(picks):
                xx = film_x + k * (thumb_w + gap)
                canvas[film_y : film_y + thumb_h, xx : xx + thumb_w] = phase_small[pi]
                col = COL_YELLOW if pi == active_pick else COL_GRID
                cv2.rectangle(
                    canvas,
                    (xx - 1, film_y - 1),
                    (xx + thumb_w, film_y + thumb_h),
                    col,
                    2 if pi == active_pick else 1,
                )

            # phase progress bar
            bx0, bx1, by = 64, 600, 884
            cv2.rectangle(canvas, (bx0, by), (bx1, by + 8), COL_GRID, -1)
            frac = (phase_idx + 1) / PHASES
            cv2.rectangle(
                canvas, (bx0, by), (bx0 + int(frac * (bx1 - bx0)), by + 8), COL_FLASH, -1
            )
            text(
                canvas,
                f"phase {phase_ms:5.2f} / {period*1000:.2f} ms   "
                f"({phase_idx+1}/{PHASES})",
                (64, by + 38),
                0.56,
                COL_YELLOW,
                1,
            )
            text(
                canvas,
                "one cycle = 2.13 s at 30 fps",
                (64, by + 66),
                0.46,
                COL_MUTED,
                1,
            )

            # ---- right panels -------------------------------------------
            draw_rate_panel(
                canvas, cnt1, peaks, recovery, t_now, (690, 130, 1896, 500), rate_max
            )
            draw_average_panel(
                canvas, avg, period, phase_ms, (690, 555, 1896, 770), recovery
            )
            draw_fft_panel(canvas, fft, period, (690, 835, 1290, 1035))
            draw_recovery_hist_panel(canvas, recovery, (1310, 835, 1896, 1035))

            # current burst count
            i1 = int(t_now / BIN1)
            if 0 <= i1 < len(cnt1):
                text_right(
                    canvas,
                    f"t = {t_now:6.2f} s    current rate {cnt1[i1]:.0f} ev/ms",
                    1896,
                    42,
                    0.52,
                    COL_MUTED,
                    1,
                )

            proc.stdin.write(canvas.tobytes())
            if (fi + 1) % 150 == 0 or fi + 1 == n_frames:
                log(
                    f"    frame {fi+1}/{n_frames} "
                    f"({100.0*(fi+1)/n_frames:5.1f}%) {time.time()-t_start:6.1f}s"
                )
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")
    log(f"  video  : {out_path} ({n_frames} frames, {os.path.getsize(out_path)/1e6:.1f} MB)")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Visualize the periodic global flicker in an event-camera bag."
    )
    ap.add_argument(
        "--bag",
        default="data/session06_slow_press.bag",
        help="input bag (default: data/session06_slow_press.bag)",
    )
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--event-topic", default=EVENT_TOPIC)
    ap.add_argument("--start", type=float, default=20.0, help="video start time (s)")
    ap.add_argument("--duration", type=float, default=20.0, help="video duration (s)")
    ap.add_argument("--phase-duration", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.bag):
        log(f"error: bag not found: {args.bag}")
        return 2

    t0 = time.time()
    bag = rosbag.Bag(args.bag, "r")
    try:
        log(f"[{os.path.basename(args.bag)}]")
        log("  scanning event rate ...")
        duration, cnt01, cnt1, total, positive = scan_event_rate(
            bag, args.event_topic
        )
        log(
            f"  events : {total/1e6:.1f} M, "
            f"max {cnt01.max()/1e3:.0f}k/ms, "
            f"median {np.median(cnt1):.0f}/ms"
        )

        peaks = detect_peaks(cnt1)
        peaks = refine_peaks(peaks, cnt01)
        period = estimate_period(peaks)
        avg, n_cycles = triggered_average(cnt01, peaks, period)
        recovery = compute_recovery(cnt1, peaks, period)
        rec_baseline, rec_threshold, recs, rec_median, rec_p90 = recovery
        log(
            f"  flicker: {len(peaks)} bursts, period {period*1000:.2f} ms "
            f"({1/period:.2f} Hz), {n_cycles} cycles averaged"
        )
        log(
            f"  recover: baseline {rec_baseline:.0f}/ms, threshold "
            f"{rec_threshold:.0f}/ms, median {rec_median:.1f} ms, "
            f"P90 {rec_p90:.1f} ms ({len(recs)} bursts)"
        )

        fft = compute_fft(cnt1, args.start, 10.0)

        # Align the phase images to a burst in the video window.
        target = args.start + 0.5
        ref = float(peaks[np.argmin(np.abs(peaks - target))])
        log(f"  phase  : {PHASES} bins, reference burst at {ref:.3f} s")
        phase_imgs, phase_totals, _acc = build_phase_images(
            bag,
            args.event_topic,
            args.start,
            min(args.phase_duration, duration - args.start),
            ref,
            period,
        )

        os.makedirs(args.out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.bag))[0]
        static_path = os.path.join(args.out_dir, f"{stem}_flicker_analysis.png")
        make_static_figure(
            static_path,
            cnt1,
            cnt01,
            peaks,
            period,
            avg,
            fft,
            phase_imgs,
            phase_totals,
            total,
            positive,
            50000.0,
            args.start,
            recovery,
        )

        if not args.no_video:
            video_path = os.path.join(args.out_dir, f"{stem}_flicker.mp4")
            make_video(
                video_path,
                cnt1,
                cnt01,
                peaks,
                avg,
                period,
                fft,
                phase_imgs,
                phase_totals,
                args.start,
                min(args.duration, duration - args.start),
                args.fps,
                args.crf,
                recovery,
            )

        log(f"  done   : {time.time()-t0:.1f}s")
    finally:
        bag.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
