#!/usr/bin/env python3
"""Fig 3: per-axis substitution cost and headline contrast.

Bars of per-axis cost and the cost(antiHER2) - cost(endocrine) CI, for receptor-status
(primary) and PAM50 (secondary) routing side by side.
Reads existing results only (no new computation): patient_routing_cost{,_receptor}.json.

Per-axis cost for endocrine/chemo depends on the routing scheme; antiHER2 mis-route (1.00)
and the headline contrast are stable across both schemes.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})

HERE = Path(__file__).resolve().parent
PAM = json.loads((HERE / "patient_routing_cost.json").read_text())
REC = json.loads((HERE / "patient_routing_cost_receptor.json").read_text())

AXES = ["antiHER2", "endocrine", "chemo"]
LABELS = ["Anti-HER2\n(ERBB2-amp)", "Endocrine\n(ER+)", "Chemo\n(TNBC/basal)"]

def per_axis(d, key):
    return [d["per_axis"][a][key] for a in AXES]

cost_rec = per_axis(REC, "mean_cost")
cost_pam = per_axis(PAM, "mean_cost")
mis_rec = [v * 100 for v in per_axis(REC, "misroute_rate")]
mis_pam = [v * 100 for v in per_axis(PAM, "misroute_rate")]

# Contrast key paths differ between the two JSONs: receptor = per_axis.headline_contrast,
# PAM50 = top-level headline_contrast_antiHER2_minus_endocrine. Both have value, ci95, excludes_zero.
hc_rec = REC["per_axis"]["headline_contrast"]
hc_pam = PAM.get("headline_contrast_antiHER2_minus_endocrine", {})
assert hc_rec.get("ci95") and hc_pam.get("ci95"), "both routings need a CI"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.5, 1.0]})

# ===== Panel A: per-axis cost, both routings side by side =====
x = np.arange(len(AXES)); w = 0.36
b1 = ax1.bar(x - w/2, cost_pam, w, label="PAM50 routing", color="#8aa9c9",
             edgecolor="black", linewidth=0.6, zorder=3)
b2 = ax1.bar(x + w/2, cost_rec, w, label="Receptor routing (primary)", color="#37527a",
             edgecolor="black", linewidth=0.6, zorder=3)
ax1.set_ylim(0, 1.06)   # top margin for annotations and bar labels
ax1.set_ylabel("Substitution cost  (mis-route rate × therapeutic distance)")
ax1.set_xticks(x); ax1.set_xticklabels(LABELS)
ax1.set_title("A  Per-axis substitution cost — routing-dependent except anti-HER2",
              loc="left", fontweight="bold", fontsize=10)
ax1.grid(axis="y", ls=":", lw=0.6, alpha=0.5, zorder=0)
ax1.legend(loc="upper right", frameon=False, fontsize=9)
for xi, v, m in zip(x - w/2, cost_pam, mis_pam):
    ax1.text(xi, v + 0.015, f"{v:.3f}\n({m:.0f}%)", ha="center", va="bottom", fontsize=8.2)
for xi, v, m in zip(x + w/2, cost_rec, mis_rec):
    ax1.text(xi, v + 0.015, f"{v:.3f}\n({m:.0f}%)", ha="center", va="bottom", fontsize=8.2,
             fontweight="bold")
# anti-HER2 = routing-invariant 100% mis-route
ax1.axvspan(-0.5, 0.5, color="#d1495b", alpha=0.07, zorder=0)
ax1.text(0, 0.99, "mis-route 100% under BOTH routings\n→ molecular test mandatory",
         ha="center", va="top", fontsize=8.2, color="#a13045", fontweight="bold",
         linespacing=1.25, transform=ax1.get_xaxis_transform())

# ===== Panel B: headline contrast (antiHER2 − endocrine) =====
rows = [("Receptor routing\n(primary)", hc_rec), ("PAM50 routing\n(secondary)", hc_pam)]
ys = np.arange(len(rows))[::-1]
for y, (name, hc) in zip(ys, rows):
    if not hc:
        continue
    v = hc["value"]; lo, hi = hc["ci95"]
    ax2.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color="#37527a",
                 ecolor="#37527a", elinewidth=2, capsize=5, markersize=8, zorder=3)
    ax2.text(v, y + 0.16, f"{v:.3f}  [{lo:.3f}, {hi:.3f}]", ha="center", va="bottom",
             fontsize=9, fontweight="bold")
ax2.axvline(0, color="#d1495b", ls="--", lw=1.2, zorder=2)
ax2.text(0, -0.62, "null (no contrast)", color="#a13045", fontsize=8.2, ha="center", va="top")
ax2.set_yticks(ys); ax2.set_yticklabels([r[0] for r in rows])
ax2.set_ylim(-0.75, len(rows) - 0.25)
ax2.set_xlabel("cost(anti-HER2) − cost(endocrine)")
ax2.set_title("B  Headline contrast — CI excludes 0 under both routings",
              loc="left", fontweight="bold", fontsize=10)
ax2.grid(axis="x", ls=":", lw=0.6, alpha=0.5)

fig.suptitle("Substitution cost is routing-dependent per axis, but the anti-HER2 − endocrine "
             "contrast is robust  (CPTAC external, hypothesis-only)",
             fontsize=11.0, fontweight="bold")
fig.text(0.5, -0.03,
         "Bars = mean cost (mis-route % in parens). Panel B = bootstrap 95% CI; excludes 0 → contrast robust to routing scheme.\n"
         "Honest reading: endocrine · chemo per-axis cost FLIPS between routings (0.378↔0.035, 0.105↔0.510) → NOT a safety claim. "
         "Robust = anti-HER2 mis-route 1.00 under both + the contrast above.",
         ha="center", va="top", fontsize=8.3, color="#444", linespacing=1.4)

out_png = HERE / "fig3_axis_cost.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(HERE / "fig3_axis_cost.pdf", bbox_inches="tight")

meta = {
    "figure": "Fig 3: per-axis cost + headline contrast",
    "relation_to_fig2": "Fig 2 = mechanism (confusion x distance); Fig 3 = magnitude and uncertainty",
    "sources": ["patient_routing_cost.json (PAM50 routing)",
                "patient_routing_cost_receptor.json (receptor routing, primary)"],
    "claim_level": "hypothesis_only",
    "per_axis_cost": {"receptor": dict(zip(AXES, cost_rec)), "pam50": dict(zip(AXES, cost_pam))},
    "misroute_pct": {"receptor": dict(zip(AXES, mis_rec)), "pam50": dict(zip(AXES, mis_pam))},
    "headline_contrast": {"receptor": hc_rec, "pam50": hc_pam},
    "honest_reading": REC.get("honest_reading", ""),
    "note": "Per-axis cost depends on the routing scheme (endocrine/chemo flip). "
            "Stable across schemes: antiHER2 misroute 1.00 and headline contrast CI excluding 0.",
}
(HERE / "fig3_axis_cost.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print("wrote fig3_axis_cost.{png,pdf,json}")
print("receptor cost:", dict(zip(AXES, cost_rec)))
print("pam50    cost:", dict(zip(AXES, cost_pam)))
print("contrast receptor:", hc_rec, "| pam50:", hc_pam)
