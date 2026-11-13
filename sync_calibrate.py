#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Estimate the event-to-force time lag for event-camera / force-sensor bags.

The event camera sees the gel deformation caused by force changes, so the
event count is correlated with the force-change magnitude but is delayed by
the sensor/transport/viscoelastic response.  This script estimates that delay
with a windowed cross-correlation and writes a calibration JSON that
``bag_vis.py`` can consume.

Typical usage:
    /usr/bin/python3 sync_calibrate.py tac_bag
    /usr/bin/python3 sync_calibrate.py tac_bag -o calibration/event_force_sync.json
    /usr/bin/python3 sync_calibrate.py tac_bag -j 4

Sign convention:
    force_time_shift_ms > 0  ->  events lag force by this amount.
    bag_vis.py shifts force timestamps forward by this amount so the force
    curve and the event stream line up.
"""

import argparse
import datetime
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np
import rosbag

import bag_vis as bv


DEFAULT_OUT = os.path.join("calibration", "event_force_sync.json")


# ---------------------------------------------------------------------------
# signal helpers
# ---------------------------------------------------------------------------


def moving_average(a, k):
    """Edge-padded moving average.  k is in bins."""
    k = max(1, int(k))
    if k <= 1:
        return np.asarray(a, dtype=np.float64).copy()
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    padded = np.pad(
        np.asarray(a, dtype=np.float64), (pad_left, pad_right), mode="edge"
    )
    kernel = np.ones(k, dtype=np.float64) / float(k)
    return np.convolve(padded, kernel, mode="valid")


def read_event_counts(bag, topic, t0, duration, bin_s):
    """Return (counts_per_bin, total_events)."""
    n_bins = int(math.ceil(duration / bin_s)) + 1
    counts = np.zeros(n_bins, dtype=np.int64)
    total = 0
    for item in bag.read_messages(topics=[topic], raw=True):
        parsed = bv.parse_event_raw(bv.get_raw_event(item))
        if parsed is None:
            continue
        _x, _y, t, _pol, _h, _w = parsed
        if len(t) == 0:
            continue
        idx = ((t - t0) / bin_s).astype(np.int64)
        np.clip(idx, 0, n_bins - 1, out=idx)
        counts += np.bincount(idx, minlength=n_bins)
        total += len(t)
    return counts, total


def force_signals(bag, force_topic, force_type, centers):
    """Interpolate force magnitude / change-rate signals onto bin centers."""
    f_t, fx, fy, fz, _tx, _ty, _tz = bv.read_force(bag, force_topic, force_type)
    mag = np.sqrt(fx * fx + fy * fy + fz * fz)
    dvec = np.sqrt(
        np.gradient(fx, f_t) ** 2
        + np.gradient(fy, f_t) ** 2
        + np.gradient(fz, f_t) ** 2
    )
    dmag = np.abs(np.gradient(mag, f_t))
    return {
        "t": f_t,
        "mag": mag,
        "dvec": dvec,
        "dmag": dmag,
        "mag_b": np.interp(centers, f_t, mag),
        "dvec_b": np.interp(centers, f_t, dvec),
        "dmag_b": np.interp(centers, f_t, dmag),
        "force_range": float(mag.max() - mag.min()),
        "force_max": float(mag.max()),
    }


# ---------------------------------------------------------------------------
# lag estimation
# ---------------------------------------------------------------------------


def xcorr_lag(a, b, active, max_lag_bins, min_overlap=20):
    """Cross-correlate a and b on active samples.

    Returns (best_lag_bins, best_corr, corr_by_lag) or None.

    lag > 0 compares a[i] with b[i + lag].  With a = event count and
    b = force-change rate, a negative best lag means events lag force.
    """
    idx = np.flatnonzero(active)
    if len(idx) < min_overlap:
        return None

    a_act = a[idx]
    b_act = b[idx]
    a_mean = float(a_act.mean())
    b_mean = float(b_act.mean())
    a_std = float(a_act.std())
    b_std = float(b_act.std())
    if a_std < 1e-12 or b_std < 1e-12:
        return None

    az = (a - a_mean) / a_std
    bz = (b - b_mean) / b_std
    n = len(a)
    best_lag = 0
    best_corr = -2.0
    corr_by_lag = {}

    for lag in range(-max_lag_bins, max_lag_bins + 1):
        if lag >= 0:
            aa = az[: n - lag]
            bb = bz[lag:]
            mask = active[: n - lag] & active[lag:]
        else:
            aa = az[-lag:]
            bb = bz[: n + lag]
            mask = active[-lag:] & active[: n + lag]
        count = int(mask.sum())
        if count < min_overlap:
            corr_by_lag[lag] = np.nan
            continue
        corr = float(np.mean(aa[mask] * bb[mask]))
        corr_by_lag[lag] = corr
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    if best_corr <= -1.5:
        return None
    return best_lag, best_corr, corr_by_lag


def parabolic_lag(lag, corr_by_lag):
    """Sub-bin peak interpolation using the three samples around the peak."""
    y0 = corr_by_lag.get(lag - 1, np.nan)
    y1 = corr_by_lag.get(lag, np.nan)
    y2 = corr_by_lag.get(lag + 1, np.nan)
    if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2)):
        return float(lag)
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(lag)
    offset = 0.5 * (y0 - y2) / denom
    if abs(offset) > 1.0:
        return float(lag)
    return float(lag + offset)


# ---------------------------------------------------------------------------
# per-bag calibration
# ---------------------------------------------------------------------------


def calibrate_bag(bag_path, args):
    bag = rosbag.Bag(bag_path, "r")
    try:
        event_topic, _event_type, force_topic, force_type = bv.detect_topics(bag)
        t0 = bag.get_start_time()
        t1 = bag.get_end_time()
        duration = max(t1 - t0, 1e-6)

        bin_s = args.bin_ms / 1000.0
        counts, total_events = read_event_counts(
            bag, event_topic, t0, duration, bin_s
        )
        n_bins = len(counts)
        centers = t0 + (np.arange(n_bins) + 0.5) * bin_s
        fs = force_signals(bag, force_topic, force_type, centers)

        k = max(1, int(round(args.smooth_ms / args.bin_ms)))
        event_s = moving_average(counts.astype(np.float64), k)
        force_s = moving_average(fs["dvec_b" if args.signal == "dvec" else "dmag_b"], k)
        mag_s = moving_average(fs["mag_b"], k)
        dvec_s = moving_average(fs["dvec_b"], k)

        edge = k // 2
        active = (mag_s > args.min_force_mag) | (
            dvec_s > float(np.percentile(dvec_s, 90.0))
        )
        active[:edge] = False
        if edge:
            active[-edge:] = False

        win = max(10, int(round(args.window_s / bin_s)))
        step = max(1, int(round(args.step_s / bin_s)))
        max_lag_bins = max(1, int(round(args.max_lag_ms / 1000.0 / bin_s)))
        min_active_window = max(20, int(round(0.5 / bin_s)))

        window_lags = []
        window_corrs = []
        for start in range(0, max(1, n_bins - win + 1), step):
            end = min(start + win, n_bins)
            if end - start < max(10, win // 2):
                continue
            active_window = active[start:end]
            if int(active_window.sum()) < min_active_window:
                continue
            res = xcorr_lag(
                event_s[start:end],
                force_s[start:end],
                active_window,
                max_lag_bins,
            )
            if res is None:
                continue
            lag, corr, corr_by_lag = res
            if corr < args.min_window_corr:
                continue
            if abs(lag) >= max_lag_bins:
                continue
            lag_f = parabolic_lag(lag, corr_by_lag)
            window_lags.append(-lag_f * args.bin_ms)  # positive = events lag force
            window_corrs.append(corr)

        full_lag_ms = None
        full_corr = None
        if int(active.sum()) >= min_active_window:
            res = xcorr_lag(event_s, force_s, active, max_lag_bins)
            if res is not None:
                lag, corr, corr_by_lag = res
                if abs(lag) < max_lag_bins:
                    full_lag_ms = -parabolic_lag(lag, corr_by_lag) * args.bin_ms
                    full_corr = corr

        event_rate = total_events / duration
        result = {
            "bag": os.path.basename(bag_path),
            "event_rate": float(event_rate),
            "force_range": float(fs["force_range"]),
            "force_max": float(fs["force_max"]),
            "n_windows": len(window_lags),
            "corr_median": float(np.median(window_corrs)) if window_corrs else None,
            "full_lag_ms": full_lag_ms,
            "full_corr": full_corr,
            "estimated_lag_ms": None,
            "lag_mad_ms": None,
            "force_time_shift_ms": None,
            "quality": "insufficient_signal",
            "reason": "",
        }

        if not window_lags:
            result["reason"] = "no valid active windows"
            return result

        lag_ms = float(np.median(window_lags))
        lag_mad = float(np.median(np.abs(np.asarray(window_lags) - lag_ms)))
        result["estimated_lag_ms"] = lag_ms
        result["lag_mad_ms"] = lag_mad

        reasons = []
        if event_rate < args.min_event_rate:
            reasons.append(
                f"event rate {event_rate/1e3:.1f}k/s < {args.min_event_rate/1e3:.1f}k/s"
            )
        if fs["force_range"] < args.min_force_range:
            reasons.append(
                f"force range {fs['force_range']:.2f} N < {args.min_force_range:.2f} N"
            )
        if len(window_lags) < args.min_windows:
            reasons.append(
                f"only {len(window_lags)} valid windows < {args.min_windows}"
            )
        if result["corr_median"] is None or result["corr_median"] < args.min_corr:
            reasons.append(
                f"median corr {result['corr_median']} < {args.min_corr}"
            )
        if lag_mad > args.max_mad_ms:
            reasons.append(f"lag MAD {lag_mad:.0f} ms > {args.max_mad_ms:.0f} ms")
        if abs(lag_ms) > args.max_lag_ms - args.bin_ms:
            reasons.append("peak at search boundary")
        if (
            full_lag_ms is not None
            and abs(lag_ms - full_lag_ms) > args.max_full_diff_ms
        ):
            reasons.append(
                f"window/full lag differ by {abs(lag_ms-full_lag_ms):.0f} ms"
            )

        if not reasons:
            result["quality"] = "good"
            result["force_time_shift_ms"] = lag_ms
        else:
            low_signal = (
                event_rate < args.min_event_rate
                or fs["force_range"] < args.min_force_range
            )
            result["quality"] = "insufficient_signal" if low_signal else "unstable"
            result["reason"] = "; ".join(reasons)
        return result
    finally:
        bag.close()


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def write_calibration(path, args, results):
    good = [
        r
        for r in results
        if r["quality"] == "good" and r["force_time_shift_ms"] is not None
    ]
    if good:
        values = np.asarray([r["force_time_shift_ms"] for r in good], dtype=np.float64)
        global_shift = float(np.median(values))
        global_mad = float(np.median(np.abs(values - global_shift)))
    else:
        global_shift = 0.0
        global_mad = 0.0

    data = {
        "version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": "windowed_xcorr_event_count_vs_force_change",
        "sign_convention": (
            "positive force_time_shift_ms means events lag force; "
            "bag_vis shifts force timestamps forward by this amount"
        ),
        "params": {
            "bin_ms": args.bin_ms,
            "smooth_ms": args.smooth_ms,
            "window_s": args.window_s,
            "step_s": args.step_s,
            "max_lag_ms": args.max_lag_ms,
            "signal": args.signal,
            "active_mask": (
                f"force_mag > {args.min_force_mag} N or dvec > p90"
            ),
        },
        "global": {
            "force_time_shift_ms": global_shift,
            "event_lag_ms": global_shift,
            "n_good_bags": len(good),
            "n_total_bags": len(results),
            "mad_ms": global_mad,
        },
        "bags": {
            r["bag"].rsplit(".bag", 1)[0]: r
            for r in sorted(results, key=lambda item: item["bag"])
        },
    }

    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    return data


def print_summary(results, data):
    bv.log("")
    bv.log(
        f"{'bag':38s} {'rate':>9s} {'range':>8s} {'lag':>9s} "
        f"{'MAD':>7s} {'corr':>6s}  quality"
    )
    for r in sorted(results, key=lambda x: x["bag"]):
        rate = f"{r['event_rate']/1e3:.1f}k/s"
        force_range = f"{r['force_range']:.1f}N"
        lag = (
            f"{r['force_time_shift_ms']:+.0f}ms"
            if r["force_time_shift_ms"] is not None
            else "-"
        )
        mad = f"{r['lag_mad_ms']:.0f}ms" if r["lag_mad_ms"] is not None else "-"
        corr = (
            f"{r['corr_median']:.3f}"
            if r["corr_median"] is not None
            else "-"
        )
        bv.log(
            f"{r['bag']:38s} {rate:>9s} {force_range:>8s} {lag:>9s} "
            f"{mad:>7s} {corr:>6s}  {r['quality']}"
        )
        if r["reason"]:
            bv.log(f"    reason: {r['reason']}")

    g = data["global"]
    bv.log("")
    bv.log(
        f"global force shift: {g['force_time_shift_ms']:+.1f} ms "
        f"({g['n_good_bags']}/{g['n_total_bags']} good bags, "
        f"MAD {g['mad_ms']:.1f} ms)"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Calibrate event-to-force time lag and write a JSON file."
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
        default="tac_bag",
        help="directory containing .bag files (default: tac_bag)",
    )
    ap.add_argument(
        "--bags", nargs="*", default=None, help="explicit bag path(s)"
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="search subdirectories when a folder is given",
    )
    ap.add_argument(
        "-o",
        "--out",
        default=DEFAULT_OUT,
        help=f"output JSON path (default: {DEFAULT_OUT})",
    )
    ap.add_argument("--bin-ms", type=float, default=10.0)
    ap.add_argument("--smooth-ms", type=float, default=50.0)
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--step-s", type=float, default=2.5)
    ap.add_argument("--max-lag-ms", type=float, default=800.0)
    ap.add_argument("--min-corr", type=float, default=0.15)
    ap.add_argument("--min-window-corr", type=float, default=0.10)
    ap.add_argument("--min-event-rate", type=float, default=5000.0)
    ap.add_argument("--min-force-range", type=float, default=1.0)
    ap.add_argument("--min-force-mag", type=float, default=1.0)
    ap.add_argument("--min-windows", type=int, default=3)
    ap.add_argument("--max-mad-ms", type=float, default=150.0)
    ap.add_argument("--max-full-diff-ms", type=float, default=150.0)
    ap.add_argument(
        "--signal",
        choices=("dvec", "dmag"),
        default="dvec",
        help="force-change signal: vector derivative (default) or magnitude derivative",
    )
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="number of bags to process in parallel (0 = auto)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # Reuse bag_vis's path resolver.
    ns = SimpleNamespace(
        paths=args.paths,
        bags=args.bags,
        data_dir=args.data_dir,
        recursive=args.recursive,
    )
    bags = bv.collect_bags(ns)
    if not bags:
        bv.log("error: no .bag files found")
        return 2

    auto_jobs = min(len(bags), os.cpu_count() or 1, 8)
    jobs = args.jobs if args.jobs > 0 else auto_jobs
    jobs = max(1, min(jobs, len(bags), os.cpu_count() or 1))

    if not args.quiet:
        bv.log(f"calibrating {len(bags)} bag(s) with {jobs} worker(s)")

    results = []
    if jobs == 1:
        for path in bags:
            if not args.quiet:
                bv.log(f"[{os.path.basename(path)}] analyzing ...")
            try:
                result = calibrate_bag(path, args)
                results.append(result)
                if not args.quiet:
                    lag = result["force_time_shift_ms"]
                    lag_text = f"{lag:+.0f} ms" if lag is not None else "n/a"
                    bv.log(
                        f"[{os.path.basename(path)}] lag={lag_text} "
                        f"quality={result['quality']}"
                    )
            except Exception as exc:
                bv.log(f"error while calibrating {path}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(calibrate_bag, path, args): path for path in bags
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if not args.quiet:
                        lag = result["force_time_shift_ms"]
                        lag_text = f"{lag:+.0f} ms" if lag is not None else "n/a"
                        bv.log(
                            f"[{os.path.basename(path)}] lag={lag_text} "
                            f"quality={result['quality']}"
                        )
                except Exception as exc:
                    bv.log(f"error while calibrating {path}: {exc}")

    if not results:
        bv.log("error: no bag could be calibrated")
        return 1

    data = write_calibration(args.out, args, results)
    if not args.quiet:
        print_summary(results, data)
        bv.log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
