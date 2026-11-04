#!/usr/bin/env python3
"""
C1 cost-of-substitution, step 2: cost metric.

Two parts:
  (A) Therapeutic distance between therapy axes (no predictions needed).
      Uses frozen_map.json only: per-axis drug rankings on discriminating drugs,
      compared pairwise by Kendall tau and top-k overlap.
  (B) Per-patient mis-routing cost.
      Each patient is assigned an axis from the measured and from the H&E-predicted
      subtype; cost = therapeutic distance between the two axes.
      Reported per true axis with an antiHER2 - endocrine contrast and bootstrap CI.

cost = confusion (mis-routing frequency) x therapeutic distance (value of one mis-route).
"""
import os
from pathlib import Path
import json, itertools, numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
MANIFEST = DATA_ROOT / "embedding_manifest_cptac_uni.csv"

fm = json.loads((HERE / "frozen_map.json").read_text())
axes = list(fm["frozen_map_axis_to_drug_z"].keys())          # endocrine, antiHER2, chemo
zmap = fm["frozen_map_axis_to_drug_z"]

# Discriminating drugs: gap >= 0.5 (same threshold as frozen_map.json top_discriminating).
# Only drugs with values on all three axes are used.
drugs = [d for d in zmap[axes[0]] if all(d in zmap[a] for a in axes)]
def gap(d):
    vs=[zmap[a][d] for a in axes]; return max(vs)-min(vs)
disc = sorted([d for d in drugs if gap(d) >= 0.5], key=gap, reverse=True)
print(f"{len(disc)} discriminating (gap>=0.5) of {len(drugs)} common drugs")

def ranking(axis, drug_set):
    # Most sensitive (lowest z) first. drug -> rank (0 = most sensitive)
    order = sorted(drug_set, key=lambda d: zmap[axis][d])
    return {d: i for i, d in enumerate(order)}

def kendall_tau(r1, r2, items):
    n = len(items); c = d = 0
    for a, b in itertools.combinations(items, 2):
        s = np.sign((r1[a]-r1[b]) * (r2[a]-r2[b]))
        if s > 0: c += 1
        elif s < 0: d += 1
    return (c-d)/(c+d) if (c+d) else float("nan")

def topk_overlap(axisA, axisB, drug_set, k=10):
    a = sorted(drug_set, key=lambda d: zmap[axisA][d])[:k]
    b = sorted(drug_set, key=lambda d: zmap[axisB][d])[:k]
    return len(set(a) & set(b)) / k

print("\n=== (A) Therapy-axis distance matrix (discriminating drugs) ===")
print("Lower Kendall tau / top10 overlap = more distant rankings = costlier mis-route\n")
rk = {a: ranking(a, disc) for a in axes}
print(f"{'axis pair':28} {'Kendall-tau':>12} {'top10-overlap':>14}")
dist = {}
for a, b in itertools.combinations(axes, 2):
    tau = kendall_tau(rk[a], rk[b], disc)
    ov = topk_overlap(a, b, disc, 10)
    dist[(a, b)] = {"kendall_tau": round(tau, 3), "top10_overlap": round(ov, 3),
                    "therapeutic_distance": round(1-tau, 3)}
    print(f"{a+' <-> '+b:28} {tau:12.3f} {ov:14.3f}")

out = {
    "exp": "C1_cost_step2A_therapeutic_distance",
    "note": ("cost = mis-routing frequency (confusion) x therapeutic distance. "
             "Therapeutic distance = 1 - Kendall tau on discriminating drugs; higher = costlier confusion of that axis pair."),
    "n_discriminating_drugs": len(disc),
    "axis_pair_distance": {f"{a}__{b}": v for (a, b), v in dist.items()},
}
(HERE / "therapeutic_distance.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\nwrote therapeutic_distance.json")

# ============================================================
# (B) Patient mis-routing cost from PAM50 predictions (predictions_ext.npz).
#     cost = therapeutic distance between measured axis and predicted axis.
#     PAM50 class: LumA=0, LumB=1, Basal=2, HER2=3, Normal=4
#     (pipeline/modeling/configs/baseline_pam50_clam.yaml)
# ============================================================
PAM50_AXIS = {0: "endocrine", 1: "endocrine", 2: "chemo", 3: "antiHER2", 4: None}  # Normal=None (excluded)
D = {(a, b): dist.get((a, b), dist.get((b, a)))["therapeutic_distance"]
     for a in axes for b in axes if a != b}
def tdist(a, b):
    if a == b: return 0.0
    return D[(a, b)] if (a, b) in D else D[(b, a)]

import csv as _csv

def measured_pam50(n_expected):
    """Measured PAM50 labels in npz row order.

    npz row i = i-th row of the CPTAC embedding manifest with a PAM50 label.
    """
    pam = {'LumA': 0, 'LumB': 1, 'Basal': 2, 'HER2': 3, 'Normal-like': 4}
    rows = [r for r in _csv.DictReader(open(MANIFEST)) if r.get('pam50', '').strip() in pam]
    lab = np.array([pam[r['pam50'].strip()] for r in rows])
    assert len(lab) == n_expected, f"manifest PAM50 rows ({len(lab)}) != npz rows ({n_expected})"
    return lab

pz = np.load(REPO / "breast/models/pam50_clam_mb_uni_v1/predictions_ext.npz", allow_pickle=True)
pred = pz["pred"]
meas = pz["label"] if "label" in pz.files else measured_pam50(len(pred))
rng = np.random.default_rng(42)

recs = []
for m, p in zip(meas, pred):
    ma, pa = PAM50_AXIS[int(m)], PAM50_AXIS[int(p)]
    if ma is None:            # measured Normal-like: not routed (excluded)
        continue
    if pa is None:            # predicted Normal: routed to no treatment, max-distance penalty
        cost, dropped = max(D.values()), True
    else:
        cost, dropped = tdist(ma, pa), False
    recs.append({"true_axis": ma, "pred_axis": pa or "none(Normal)", "cost": cost, "dropped": dropped})

import collections
by = collections.defaultdict(list)
for r in recs: by[r["true_axis"]].append(r)

print("\n=== (B) Patient mis-routing cost (PAM50 predictions, n=%d, Normal-like excluded) ===" % len(recs))
print("cost = therapeutic distance on mis-route (0 = correct). dropped = predicted Normal (no treatment)")
print(f"{'true axis':12} {'n':>4} {'mean_cost':>10} {'mis-route%':>11} {'dropped%':>9}")
axis_stat = {}
for a in ["endocrine", "antiHER2", "chemo"]:
    rs = by[a]; n = len(rs)
    mc = float(np.mean([r["cost"] for r in rs])) if n else float("nan")
    mis = float(np.mean([r["cost"] > 0 for r in rs])) if n else float("nan")
    drp = float(np.mean([r["dropped"] for r in rs])) if n else float("nan")
    axis_stat[a] = {"n": n, "mean_cost": round(mc, 3), "misroute_rate": round(mis, 3), "dropped_rate": round(drp, 3)}
    print(f"{a:12} {n:4d} {mc:10.3f} {mis*100:10.1f}% {drp*100:8.1f}%")

# Headline contrast: antiHER2 - endocrine, bootstrap 95% CI
he = np.array([r["cost"] for r in by["antiHER2"]]); en = np.array([r["cost"] for r in by["endocrine"]])
boots = [rng.choice(he, len(he)).mean() - rng.choice(en, len(en)).mean() for _ in range(2000)]
lo, hi = np.percentile(boots, [2.5, 97.5])
contrast = float(he.mean() - en.mean())
print(f"\n[headline] cost(antiHER2) - cost(endocrine) = {contrast:.3f}  95%CI [{lo:.3f}, {hi:.3f}]"
      f"  CI excludes 0: {'YES' if lo > 0 else 'NO'}")

outB = {
    "exp": "C1_cost_step2B_patient_routing_PAM50",
    "routing_source": "PAM50 CLAM-MB predictions_ext.npz (proba/pred) + measured PAM50 labels",
    "note": ("PAM50 prediction -> therapy-axis routing. cost = therapeutic distance on mis-route (step 2A). "
             "The model never predicts HER2-enriched, so all true HER2 patients are routed off the antiHER2 axis."),
    "n_routed": len(recs), "normal_like_excluded": int((meas == 4).sum()),
    "per_axis": axis_stat,
    "headline_contrast_antiHER2_minus_endocrine": {"value": round(contrast, 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)], "excludes_zero": bool(lo > 0)},
    "caveat": ["PAM50-to-axis mapping (HER2E ~ antiHER2) approximates receptor status",
               "Predicted Normal-like = no treatment, max-distance penalty"],
}
(HERE / "patient_routing_cost.json").write_text(json.dumps(outB, indent=2, ensure_ascii=False))
print("\nwrote patient_routing_cost.json")

# ============================================================
# (B') Receptor-status routing from slide_id-indexed predictions.
#   Input: breast/models/cptac_ext_predictions_indexed.csv
#     columns: slide_id, er_pred_prob, her2_pred_prob (probabilities in 0-1)
#   Measured labels: CPTAC manifest (er/her2 = "Positive"/"Negative", has_er/has_her2)
#   Routing (clinical hierarchy): HER2+ -> antiHER2; elif ER+ -> endocrine; else -> chemo
# ============================================================
PRED_CSV = REPO / "breast" / "models" / "cptac_ext_predictions_indexed.csv"

def route(er_pos, her2_pos):
    if her2_pos: return "antiHER2"
    if er_pos:   return "endocrine"
    return "chemo"

print("\n=== (B') Receptor-status routing (slide_id-indexed predictions) ===")
if not PRED_CSV.exists():
    print(f"Predictions not found: {PRED_CSV}")
    print("  Expected columns: slide_id, er_pred_prob, her2_pred_prob")
else:
    meas = {r["slide_id"]: r for r in _csv.DictReader(open(MANIFEST))}
    preds = {r["slide_id"]: r for r in _csv.DictReader(open(PRED_CSV))}
    THR = 0.5
    rr = []
    for sid, pr in preds.items():
        m = meas.get(sid)
        if not m or m.get("has_er") != "1" or m.get("has_her2") != "1":
            continue
        m_axis = route(m["er"] == "Positive", m["her2"] == "Positive")
        p_axis = route(float(pr["er_pred_prob"]) >= THR, float(pr["her2_pred_prob"]) >= THR)
        rr.append({"true_axis": m_axis, "pred_axis": p_axis, "cost": tdist(m_axis, p_axis)})
    byr = collections.defaultdict(list)
    for r in rr: byr[r["true_axis"]].append(r)
    print(f"n routed = {len(rr)} (has_er & has_her2)")
    stat = {}
    for a in ["endocrine", "antiHER2", "chemo"]:
        rs = byr[a]; n = len(rs)
        mc = float(np.mean([r["cost"] for r in rs])) if n else float("nan")
        mis = float(np.mean([r["cost"] > 0 for r in rs])) if n else float("nan")
        stat[a] = {"n": n, "mean_cost": round(mc, 3), "misroute_rate": round(mis, 3)}
        print(f"  {a:10} n={n:4d} mean_cost={mc:.3f} mis-route={mis*100:.1f}%")
    he = np.array([r["cost"] for r in byr["antiHER2"]]); en = np.array([r["cost"] for r in byr["endocrine"]])
    if len(he) and len(en):
        boots = [rng.choice(he, len(he)).mean() - rng.choice(en, len(en)).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5]); ct = float(he.mean() - en.mean())
        print(f"  [headline] cost(antiHER2)-cost(endocrine)={ct:.3f} 95%CI[{lo:.3f},{hi:.3f}] "
              f"excludes 0: {'YES' if lo>0 else 'NO'}")
        stat["headline_contrast"] = {"value": round(ct, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)], "excludes_zero": bool(lo > 0)}
    (HERE / "patient_routing_cost_receptor.json").write_text(
        json.dumps({"exp": "C1_step2B_receptor_routing", "routing": "HER2+>antiHER2; ER+>endocrine; else chemo",
                    "pred_source": str(PRED_CSV.relative_to(REPO)), "per_axis": stat}, indent=2, ensure_ascii=False))
    print("  wrote patient_routing_cost_receptor.json")
