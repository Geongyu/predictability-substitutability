#!/usr/bin/env python3
"""Receptor-status routing cost: patient-cluster bootstrap CIs.

Recomputes the headline contrast (antiHER2 - endocrine) and per-axis mean cost,
with 95% CIs from a patient-cluster bootstrap (slides from the same patient are
resampled together). Records the routing threshold and CI settings in the JSON.
"""
import csv, json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
PRED = "breast/models/cptac_ext_predictions_indexed.csv"           # repo-relative
MANIFEST = DATA_ROOT / "embedding_manifest_cptac_uni.csv"
JSON = HERE / "patient_routing_cost_receptor.json"
THRESHOLD = 0.5
B = 5000
SEED = 42

DIST = {("endocrine", "antiHER2"): 0.395, ("endocrine", "chemo"): 0.695, ("antiHER2", "chemo"): 0.765}

def dist(a, b):
    return 0.0 if a == b else (DIST.get((a, b)) or DIST.get((b, a)))

def route_true(er, her2):
    if her2 == "positive": return "antiHER2"
    return "endocrine" if er == "positive" else "chemo"

def route_pred(er_p, her2_p, thr=THRESHOLD):
    if her2_p > thr: return "antiHER2"
    return "endocrine" if er_p > thr else "chemo"

def main():
    pred = {r["slide_id"]: r for r in csv.DictReader(open(REPO / PRED))}
    rows = []
    for r in csv.DictReader(open(MANIFEST)):
        sid = r["slide_id"]
        if sid not in pred: continue
        er = (r.get("er", "") or "").strip().lower()
        her2 = (r.get("her2", "") or "").strip().lower()
        if her2 not in ("positive", "negative") or er not in ("positive", "negative"): continue
        p = pred[sid]
        t = route_true(er, her2)
        q = route_pred(float(p["er_pred_prob"]), float(p["her2_pred_prob"]))
        rows.append((r["case_id"], t, q, dist(t, q)))

    case = np.array([r[0] for r in rows]); axis = np.array([r[1] for r in rows])
    predr = np.array([r[2] for r in rows]); cost = np.array([r[3] for r in rows])
    n_slides, n_pat = len(rows), len(set(case))
    print(f"n_slides={n_slides}  n_patients={n_pat}  ({n_slides/n_pat:.2f} slides/pt)")

    axes = ["endocrine", "antiHER2", "chemo"]
    def mean_cost(ax, msk): return cost[msk][axis[msk] == ax].mean() if (axis[msk] == ax).any() else np.nan
    def headline(msk): return mean_cost("antiHER2", msk) - mean_cost("endocrine", msk)

    full = np.ones(n_slides, bool)
    per_axis = {ax: {"n": int((axis == ax).sum()),
                     "mean_cost": round(float(cost[axis == ax].mean()), 4),
                     "misroute_rate": round(float(1 - (predr[axis == ax] == ax).mean()), 3)} for ax in axes}
    ct = float(headline(full))

    # patient-cluster bootstrap: headline + per-axis mean_cost
    pats = sorted(set(case)); idx_by = {p: np.where(case == p)[0] for p in pats}
    rng = np.random.default_rng(SEED)
    bh, bax = [], {ax: [] for ax in axes}
    for _ in range(B):
        pick = rng.choice(pats, len(pats), replace=True)
        i = np.concatenate([idx_by[p] for p in pick])
        a2 = axis[i]
        if (a2 == "antiHER2").sum() == 0 or (a2 == "endocrine").sum() == 0: continue
        bh.append(cost[i][a2 == "antiHER2"].mean() - cost[i][a2 == "endocrine"].mean())
        for ax in axes:
            if (a2 == ax).any(): bax[ax].append(cost[i][a2 == ax].mean())
    lo_p, hi_p = np.percentile(bh, [2.5, 97.5])

    # Slide-level bootstrap, for comparison
    bs = []
    for _ in range(B):
        i = rng.integers(0, n_slides, n_slides); a2 = axis[i]
        if (a2 == "antiHER2").sum() == 0 or (a2 == "endocrine").sum() == 0: continue
        bs.append(cost[i][a2 == "antiHER2"].mean() - cost[i][a2 == "endocrine"].mean())
    lo_s, hi_s = np.percentile(bs, [2.5, 97.5])

    for ax in axes:
        lo, hi = np.percentile(bax[ax], [2.5, 97.5])
        per_axis[ax]["mean_cost_ci95_patient"] = [round(float(lo), 4), round(float(hi), 4)]
    per_axis["headline_contrast"] = {
        "value": round(ct, 4),
        "ci95": [round(float(lo_p), 4), round(float(hi_p), 4)],
        "ci95_method": "patient-cluster bootstrap",
        "excludes_zero": bool(lo_p > 0),
        "ci95_slide_level_deprecated": [round(float(lo_s), 4), round(float(hi_s), 4)],
    }
    print(f"headline={ct:.4f}  patient-CI=[{lo_p:.4f},{hi_p:.4f}]  slide-CI=[{lo_s:.4f},{hi_s:.4f}]")

    j = json.load(open(JSON))
    j["pred_source"] = PRED
    j["routing_threshold"] = THRESHOLD
    j["ci_method"] = {"resample_unit": "patient (cluster bootstrap)", "n_bootstrap": B, "seed": SEED,
                      "n_slides": n_slides, "n_patients": n_pat,
                      "note": "Slide-level bootstrap is pseudo-replicated (mean %.2f slides/patient) and underestimates the CI; patient-cluster bootstrap is used." % (n_slides/n_pat)}
    j["per_axis"] = per_axis
    JSON.write_text(json.dumps(j, ensure_ascii=False, indent=2))
    print(f"wrote {JSON}")

if __name__ == "__main__":
    main()
