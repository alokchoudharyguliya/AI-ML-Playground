"""Plot the CSV printed by roofline_probe.cu as a measured roofline.

usage:  python plot_roofline.py roofline.csv [--peak-gflops 19500 --peak-gbs 2039]
The optional --peak-* flags draw the datasheet roof for comparison.
"""
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--peak-gflops", type=float, default=None, help="datasheet FP32 peak (GFLOP/s)")
ap.add_argument("--peak-gbs", type=float, default=None, help="datasheet bandwidth (GB/s)")
args = ap.parse_args()

with open(args.csv) as f:
    rows = [r for r in csv.DictReader(f)]
ai = np.array([float(r["flop_per_byte"]) for r in rows])
gf = np.array([float(r["gflops"]) for r in rows])
gb = np.array([float(r["gbs"]) for r in rows])

# measured roofs: bandwidth from the lowest-intensity points, compute from the highest
bw = gb[:3].max()          # GB/s sustained while memory-bound
peak = gf.max()            # GFLOP/s sustained when compute-bound
ridge = peak / bw          # FLOP/B

x = np.logspace(np.log10(ai.min() / 2), np.log10(ai.max() * 2), 200)
roof = np.minimum(peak, bw * x)

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.loglog(x, roof, "-", lw=2.5, label=f"measured roof: {bw:.0f} GB/s, {peak / 1000:.1f} TFLOP/s")
ax.loglog(ai, gf, "o", ms=6, label="probe kernel (K = 1 ... 1024)")
if args.peak_gflops and args.peak_gbs:
    ax.loglog(x, np.minimum(args.peak_gflops, args.peak_gbs * x), "--", lw=1.2, color="gray", label="datasheet roof")
ax.axvline(ridge, ls=":", color="k", lw=1)
ax.annotate(f"measured ridge ~ {ridge:.1f} FLOP/B", (ridge, peak / 3), rotation=90, va="center", ha="right", fontsize=8)
ax.set_xlabel("arithmetic intensity (FLOP / byte)")
ax.set_ylabel("performance (GFLOP/s)")
ax.set_title("Measured roofline")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("roofline.png", dpi=150)
print(f"memory roof {bw:.0f} GB/s | compute roof {peak:.0f} GFLOP/s | ridge {ridge:.2f} FLOP/B -> roofline.png")
