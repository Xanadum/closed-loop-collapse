"""JOB 2 (E1) absorption analysis for one checkpoint. Reuses the ORIGINAL formula by importing
read_trace / post_grasp_segment / slope from analyze_zoffset_absorption (read-only import; the module's
main() only runs under __main__, so import is side-effect-free). Identical to the V1 analysis:
  net_z = world_z (includes injected offset); raw_z = net - offset; per-episode mean over post-grasp
  segment (first env_gripper<0.5), then mean over grasped episodes; baseline = net at offset 0.0;
  dev = net - (baseline_raw + offset); absorption_frac = -dev/offset (= D2 definition).

Writes (paper_exports/):
  e1_absorption_detail_<CK>.csv   full net/raw/dev per offset x segment (provenance; == z_offset_absorption.csv schema)
  e1_absorption_d2rows_<CK>.csv   D2-schema rows for this checkpoint
Then REBUILDS E1_absorption_suite.csv = D2_original_absorption.csv (v1_300k) + every e1_absorption_d2rows_*.csv
(idempotent; safe to call mid-sweep and per-checkpoint).

Usage: python e1_absorption_analyze.py <trace_dir> <CK_LABEL>
"""
import csv
import glob
import os
import sys
from statistics import mean

from analyze_zoffset_absorption import read_trace, post_grasp_segment  # identical formula, read-only

EXPORT = "~/SimplerEnv_stable/paper_exports"
OFFSETS = [("z000", 0.0), ("z002", 0.002), ("z004", 0.004), ("z006", 0.006),
           ("z008", 0.008), ("z012", 0.012), ("z020", 0.020)]
CK_NAME = {"rep8000": "repaired_8000", "zc8000": "zerocontrol_8000", "tea15": "teacher15"}


def per_offset(trdir, zlabel, off):
    files = sorted(glob.glob(os.path.join(trdir, f"{zlabel}_eggplant_trace_ep*_action_trace.csv")))
    nf, nl = [], []
    for fp in files:
        res = post_grasp_segment(read_trace(fp))
        if res is None:
            continue
        seg, late = res
        nf.append(mean(seg)); nl.append(mean(late))
    g = len(nf)
    if g == 0:
        return dict(off=off, grasped=0, net_full=float("nan"), net_late=float("nan"),
                    raw_full=float("nan"), raw_late=float("nan"))
    return dict(off=off, grasped=g, net_full=mean(nf), net_late=mean(nl),
                raw_full=mean(nf) - off, raw_late=mean(nl) - off)


def fmt(v):
    return "" if v != v else f"{v:+.5f}"


def frac(dev, off):
    if off == 0.0 or dev != dev:
        return ""
    return f"{-dev / off:+.4f}"


def main():
    trdir, ck = sys.argv[1], sys.argv[2]
    ckname = CK_NAME.get(ck, ck)
    rows = [per_offset(trdir, zl, off) for zl, off in OFFSETS]
    base = next(r for r in rows if r["off"] == 0.0)
    braw_f, braw_l = base["raw_full"], base["raw_late"]

    # detail CSV
    det = os.path.join(EXPORT, f"e1_absorption_detail_{ck}.csv")
    with open(det, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["offset", "grasped", "net_full", "raw_full", "net_late", "raw_late", "dev_full", "dev_late"])
        for r in rows:
            df = r["net_full"] - (braw_f + r["off"]) if r["net_full"] == r["net_full"] else float("nan")
            dl = r["net_late"] - (braw_l + r["off"]) if r["net_late"] == r["net_late"] else float("nan")
            r["_devf"], r["_devl"] = df, dl
            w.writerow([f"{r['off']:.3f}", r["grasped"], fmt(r["net_full"]), fmt(r["raw_full"]),
                        fmt(r["net_late"]), fmt(r["raw_late"]), fmt(df), fmt(dl)])

    # D2-schema rows for this checkpoint
    src = f"{trdir}|e1_absorption_detail_{ck}.csv"
    d2 = os.path.join(EXPORT, f"e1_absorption_d2rows_{ck}.csv")
    with open(d2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "offset", "segment", "absorption_frac", "n_episodes", "source_path"])
        for r in rows:
            n = r["grasped"]
            for seg, dev in (("full", r["_devf"]), ("late", r["_devl"])):
                af = frac(dev, r["off"]) if n > 0 else ""
                w.writerow([ckname, f"{r['off']:.3f}", seg, af, n, src])

    # rebuild the suite = D2 (v1) + all per-checkpoint d2rows
    suite = os.path.join(EXPORT, "E1_absorption_suite.csv")
    hdr = ["checkpoint", "offset", "segment", "absorption_frac", "n_episodes", "source_path"]
    out = []
    d2orig = os.path.join(EXPORT, "D2_original_absorption.csv")
    if os.path.exists(d2orig):
        out += list(csv.reader(open(d2orig)))[1:]
    for fp in sorted(glob.glob(os.path.join(EXPORT, "e1_absorption_d2rows_*.csv"))):
        out += list(csv.reader(open(fp)))[1:]
    with open(suite, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(out)

    print(f"[{ckname}] wrote {det}, {d2}; suite now {len(out)} rows")
    print(f"{'offset':7s} {'grsp':>4s} {'net_full':>10s} {'net_late':>10s} {'absorb_f(full)':>14s} {'absorb_f(late)':>14s}")
    for r in rows:
        print(f"{r['off']:<7.3f} {r['grasped']:4d} {fmt(r['net_full']):>10s} {fmt(r['net_late']):>10s} "
              f"{str(frac(r['_devf'], r['off'])):>14s} {str(frac(r['_devl'], r['off'])):>14s}")


if __name__ == "__main__":
    main()
