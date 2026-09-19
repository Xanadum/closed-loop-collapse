"""z-offset ABSORPTION analysis (read-only; NO new rollouts).

Uses the existing per-episode action traces (7 offsets x 6 in-dist eids {2,9,11,12,14,23} rng0)
in eval_logs_zoffset_ztrace/. For each episode, post-grasp segment = steps at/after the first
gripper-close command (env_gripper < 0.5), matching analyze_zgap.py.

Definitions (per step, within post-grasp segment):
  net_z = trace world_z          (the ACTUAL executed z delta, which INCLUDES the injected offset,
                                   because the offset is added before the trace log)
  raw_z = trace world_z - offset (the model's OWN commanded z delta, with the injection removed)

Aggregation matches analyze_zgap: per-episode mean over post-grasp steps, then mean across grasped
episodes. Reports:
  (a) per offset: mean_raw and mean_net, for full post-grasp segment AND late half.
  (b) marginal slope d(mean_net)/d(offset) over the grasp-preserved window {0.002,0.004,0.006}
      (net-slope; the raw-slope is net-slope minus 1).
  (c) per-point deviation from the linear (zero-absorption) expectation net = baseline_raw + offset,
      where baseline_raw = mean at offset 0.0. deviation = mean_net(offset) - (baseline_raw + offset).

Data only. No conclusion is drawn.
"""
import csv
import glob
import os
from statistics import mean

TRDIR = "eval_logs_zoffset_ztrace"
OUT = "z_offset_absorption.csv"
OFFSETS = [
    ("z000", 0.0), ("z002", 0.002), ("z004", 0.004), ("z006", 0.006),
    ("z008", 0.008), ("z012", 0.012), ("z020", 0.020),
]
WINDOW = {0.002, 0.004, 0.006}  # grasp-preserved window for the marginal slope


def read_trace(path):
    steps = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                steps.append((float(r["env_gripper"]), float(r["world_z"])))
            except (KeyError, ValueError):
                pass
    return steps


def post_grasp_segment(steps):
    """Return (full_seg_z, late_half_z) lists of net world_z, or None if never gripper-closed."""
    close_step = None
    for i, (g, _z) in enumerate(steps):
        if g < 0.5:
            close_step = i
            break
    if close_step is None:
        return None
    seg = [z for (_g, z) in steps[close_step:]]
    if not seg:
        return None
    late = seg[len(seg) // 2:] or seg
    return seg, late


def per_offset(label, off):
    """Return dict with grasped count and per-episode-averaged mean net/raw (full & late)."""
    files = sorted(glob.glob(os.path.join(TRDIR, f"{label}_eggplant_trace_ep*_action_trace.csv")))
    ep_net_full, ep_net_late = [], []
    for fp in files:
        res = post_grasp_segment(read_trace(fp))
        if res is None:
            continue
        seg, late = res
        ep_net_full.append(mean(seg))
        ep_net_late.append(mean(late))
    g = len(ep_net_full)
    if g == 0:
        return dict(label=label, off=off, grasped=0,
                    net_full=float("nan"), net_late=float("nan"),
                    raw_full=float("nan"), raw_late=float("nan"))
    net_full = mean(ep_net_full)
    net_late = mean(ep_net_late)
    return dict(label=label, off=off, grasped=g,
                net_full=net_full, net_late=net_late,
                raw_full=net_full - off, raw_late=net_late - off)


def slope(xs, ys):
    """Least-squares slope of ys vs xs; None if <2 points or degenerate."""
    pts = [(x, y) for x, y in zip(xs, ys) if y == y]  # drop NaN
    if len(pts) < 2:
        return None
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


def fmt(v):
    return "" if v != v else f"{v:+.5f}"


def main():
    rows = [per_offset(lbl, off) for lbl, off in OFFSETS]
    base = next(r for r in rows if r["off"] == 0.0)
    base_raw_full = base["raw_full"]   # == net at 0.0
    base_raw_late = base["raw_late"]

    # (a) table
    print("=== (a) per-offset mean raw & net post-grasp z (per-episode mean, avg over grasped eps) ===")
    print(f"{'offset':7s} {'grsp':>4s} {'net_full':>10s} {'raw_full':>10s} "
          f"{'net_late':>10s} {'raw_late':>10s}")
    for r in rows:
        print(f"{r['off']:<7.3f} {r['grasped']:4d} {fmt(r['net_full']):>10s} {fmt(r['raw_full']):>10s} "
              f"{fmt(r['net_late']):>10s} {fmt(r['raw_late']):>10s}")

    # (b) marginal slope of net vs offset over window {0.002,0.004,0.006}
    win = [r for r in rows if r["off"] in WINDOW]
    xs = [r["off"] for r in win]
    s_full = slope(xs, [r["net_full"] for r in win])
    s_late = slope(xs, [r["net_late"] for r in win])
    print("\n=== (b) marginal slope d(mean_net)/d(offset) over {0.002,0.004,0.006} ===")
    print(f"  net_full slope = {('%.3f' % s_full) if s_full is not None else 'n/a'}   "
          f"(raw_full slope = {('%.3f' % (s_full - 1)) if s_full is not None else 'n/a'})")
    print(f"  net_late slope = {('%.3f' % s_late) if s_late is not None else 'n/a'}   "
          f"(raw_late slope = {('%.3f' % (s_late - 1)) if s_late is not None else 'n/a'})")
    print("  reference: zero-absorption -> net slope 1.0 (raw 0.0); full-absorption -> net slope 0.0 (raw -1.0)")

    # (c) deviation from linear zero-absorption expectation: net = baseline_raw + offset
    print("\n=== (c) deviation from linear expectation net = baseline_raw + offset ===")
    print(f"  baseline_raw (offset=0.0): full={fmt(base_raw_full)}  late={fmt(base_raw_late)}")
    print(f"{'offset':7s} {'exp_full':>10s} {'net_full':>10s} {'dev_full':>10s} "
          f"{'exp_late':>10s} {'net_late':>10s} {'dev_late':>10s}")
    csv_rows = []
    for r in rows:
        exp_full = base_raw_full + r["off"]
        exp_late = base_raw_late + r["off"]
        dev_full = r["net_full"] - exp_full if r["net_full"] == r["net_full"] else float("nan")
        dev_late = r["net_late"] - exp_late if r["net_late"] == r["net_late"] else float("nan")
        print(f"{r['off']:<7.3f} {fmt(exp_full):>10s} {fmt(r['net_full']):>10s} {fmt(dev_full):>10s} "
              f"{fmt(exp_late):>10s} {fmt(r['net_late']):>10s} {fmt(dev_late):>10s}")
        csv_rows.append([f"{r['off']:.3f}", r["grasped"],
                         fmt(r["net_full"]), fmt(r["raw_full"]), fmt(r["net_late"]), fmt(r["raw_late"]),
                         fmt(dev_full), fmt(dev_late)])

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["offset", "grasped", "net_full", "raw_full", "net_late", "raw_late",
                    "dev_full", "dev_late"])
        w.writerows(csv_rows)
    print(f"\nwrote {OUT}")
    print("DATA ONLY — no conclusion drawn.")


if __name__ == "__main__":
    main()
