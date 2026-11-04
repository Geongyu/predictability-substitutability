#!/usr/bin/env python3
"""PAM50 routing cost: patient-cluster bootstrap CI and slide/case-indexed prediction CSV.

predictions_ext.npz has no id arrays. Row i of the npz corresponds to the i-th row of
embedding_manifest_cptac_uni.csv that has a PAM50 label (verified row-by-row against the
label array, 395/395), which gives each row a slide_id and case_id.

Outputs:
  (1) predictions_ext_indexed.csv: slide_id, case_id, true/pred class, proba, axes, cost.
  (2) patient_routing_cost.json: headline and per-axis CIs from a patient-cluster bootstrap.
"""
import csv, json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
NPZ = REPO / "breast/models/pam50_clam_mb_uni_v1/predictions_ext.npz"
MANIFEST = DATA_ROOT / "embedding_manifest_cptac_uni.csv"
JSON = HERE / "patient_routing_cost.json"
OUT_CSV_REL = "outputs/pam50_predictions_ext_indexed.csv"
OUT_CSV = REPO / OUT_CSV_REL
B, SEED = 5000, 42

PAM = {'LumA': 0, 'LumB': 1, 'Basal': 2, 'HER2': 3, 'Normal-like': 4}
CLASS_NAME = {0: 'LumA', 1: 'LumB', 2: 'Basal', 3: 'HER2', 4: 'Normal'}
PAM50_AXIS = {0: "endocrine", 1: "endocrine", 2: "chemo", 3: "antiHER2", 4: None}
DIST = {("endocrine", "antiHER2"): 0.395, ("endocrine", "chemo"): 0.695, ("antiHER2", "chemo"): 0.765}
MAXD = max(DIST.values())  # 0.765 = penalty for predicted Normal (no treatment)

def tdist(a, b):
    if a == b: return 0.0
    return DIST.get((a, b)) or DIST.get((b, a))

def main():
    pz = np.load(NPZ, allow_pickle=True)
    pred, proba = pz['pred'], pz['proba']
    mrows = [r for r in csv.DictReader(open(MANIFEST)) if r.get('pam50', '').strip() in PAM]
    mlab = np.array([PAM[r['pam50'].strip()] for r in mrows])
    assert len(mrows) == len(pred), "manifest/npz row count mismatch"
    if 'label' in pz.files:
        assert (mlab == pz['label']).all(), "manifest/npz label mismatch"
    lab = mlab

    rows = []
    for i, r in enumerate(mrows):
        m, p = int(lab[i]), int(pred[i])
        ma, pa = PAM50_AXIS[m], PAM50_AXIS[p]
        if ma is None:   # measured Normal: not routed
            continue
        if pa is None:   # predicted Normal: no treatment, max penalty
            cost = MAXD
        else:
            cost = tdist(ma, pa)
        rows.append({"slide_id": r['slide_id'], "case_id": r['case_id'],
                     "pam50_true": CLASS_NAME[m], "pam50_pred": CLASS_NAME[p],
                     **{f"proba_{CLASS_NAME[k]}": round(float(proba[i][k]), 6) for k in range(5)},
                     "true_axis": ma, "pred_axis": pa or "none_Normal", "cost": round(cost, 4)})

    # (1) Indexed CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    n_pat = len(set(r['case_id'] for r in rows))
    print(f"[CSV] {OUT_CSV}: {len(rows)} rows / {n_pat} patients ({len(rows)/n_pat:.2f} slides/patient)")

    # (2) Patient-cluster bootstrap
    case = np.array([r['case_id'] for r in rows]); axis = np.array([r['true_axis'] for r in rows])
    cost = np.array([r['cost'] for r in rows])
    def mc(ax, i): return cost[i][axis[i] == ax].mean() if (axis[i] == ax).any() else np.nan
    pats = sorted(set(case)); idx_by = {p: np.where(case == p)[0] for p in pats}
    rng = np.random.default_rng(SEED)
    axes = ["endocrine", "antiHER2", "chemo"]
    bh, bax = [], {a: [] for a in axes}
    for _ in range(B):
        pick = rng.choice(pats, len(pats), replace=True)
        i = np.concatenate([idx_by[p] for p in pick]); a2 = axis[i]
        if (a2 == "antiHER2").sum() == 0 or (a2 == "endocrine").sum() == 0: continue
        bh.append(cost[i][a2 == "antiHER2"].mean() - cost[i][a2 == "endocrine"].mean())
        for a in axes:
            if (a2 == a).any(): bax[a].append(cost[i][a2 == a].mean())
    lo_p, hi_p = np.percentile(bh, [2.5, 97.5])
    # Slide-level bootstrap, for comparison
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(rows), len(rows)); a2 = axis[i]
        if (a2 == "antiHER2").sum() == 0 or (a2 == "endocrine").sum() == 0: continue
        bs.append(cost[i][a2 == "antiHER2"].mean() - cost[i][a2 == "endocrine"].mean())
    lo_s, hi_s = np.percentile(bs, [2.5, 97.5])
    full = np.ones(len(rows), bool)
    ct = float(cost[axis == "antiHER2"].mean() - cost[axis == "endocrine"].mean())
    print(f"headline={ct:.4f}  patient-CI=[{lo_p:.4f},{hi_p:.4f}]  slide-CI=[{lo_s:.4f},{hi_s:.4f}]  excludes0={lo_p>0}")

    j = json.load(open(JSON))
    for a in axes:
        lo, hi = np.percentile(bax[a], [2.5, 97.5])
        j["per_axis"][a]["mean_cost_ci95_patient"] = [round(float(lo), 4), round(float(hi), 4)]
    j["headline_contrast_antiHER2_minus_endocrine"] = {
        "value": round(ct, 4), "ci95": [round(float(lo_p), 4), round(float(hi_p), 4)],
        "ci95_method": "patient-cluster bootstrap", "excludes_zero": bool(lo_p > 0),
        "ci95_slide_level_deprecated": [round(float(lo_s), 4), round(float(hi_s), 4)]}
    j["ci_method"] = {"resample_unit": "patient (cluster bootstrap)", "n_bootstrap": B, "seed": SEED,
                      "n_slides": len(rows), "n_patients": n_pat,
                      "index_source": "predictions_ext_indexed.csv (case_id assigned by row-wise match of npz to manifest PAM50 rows)"}
    j["indexed_predictions"] = OUT_CSV_REL
    JSON.write_text(json.dumps(j, ensure_ascii=False, indent=2))
    print(f"[JSON] updated {JSON}")

if __name__ == "__main__":
    main()
