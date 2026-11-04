#!/usr/bin/env python3
"""
Subtype-only baseline: lung EGFR/KRAS predicted from histology (LUAD/LUSC) alone.

For most other cancer/endpoint pairs a subtype-only baseline is circular, so only lung EGFR/KRAS is computed:
- EGFR/KRAS are enriched in LUAD (EGFR LUAD 10% / LUSC 1%, KRAS LUAD 12% / LUSC 0%), so histology alone is predictive.
- Tests whether H&E MIL exceeds LUAD/LUSC recognition for these endpoints.
- Excluded as circular: lung histology itself, HNSC HPV (subtype axis), STAD MSI (molecular subtype),
  STAD/HNSC Lauren/grade, CRC BRAF (CMS handled in COLORECTAL/full/run_cms_and_routing.py).

Baseline (no images): holdout prediction P(mut) = mutation rate of that histology in **train** -> holdout AUROC.
holdout = val+test from split.csv (site-disjoint, same as MIL). Bootstrap CI (1000).

Output: LUNG_NSCLC/full/baseline_subtype.json
Usage: python sh_subtype_baseline.py
"""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
D = HERE / "LUNG_NSCLC" / "full"


def load():
    labels = {r["case_id"]: r for r in csv.DictReader(open(D / "patient_labels.csv"))}
    split = {r["case_id"]: r["split"] for r in csv.DictReader(open(D / "split.csv"))}
    return labels, split


def boot_ci(y, p, n=1000, seed=42):
    y = np.array(y); p = np.array(p); rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return (round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)) if aucs else (None, None)


def run(ep, labels, split):
    # train positive rate by histology
    tr = {"LUAD": [0, 0], "LUSC": [0, 0]}
    for cid, r in labels.items():
        if split.get(cid) != "train":
            continue
        h = r["histology_lusc"]; v = r.get(ep, "")
        if h == "" or v == "":
            continue
        hl = "LUSC" if h in ("1", 1) else "LUAD"
        tr[hl][0] += int(v in ("1", 1)); tr[hl][1] += 1
    rate = {k: (tr[k][0] / tr[k][1] if tr[k][1] else 0.0) for k in tr}
    # holdout predictions
    y, p = [], []
    for cid, r in labels.items():
        if split.get(cid) not in ("val", "test"):
            continue
        h = r["histology_lusc"]; v = r.get(ep, "")
        if h == "" or v == "":
            continue
        hl = "LUSC" if h in ("1", 1) else "LUAD"
        y.append(int(v in ("1", 1))); p.append(rate[hl])
    auc = round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None
    lo, hi = boot_ci(y, p)
    return {"endpoint": ep, "baseline": "histology-only(LUAD/LUSC train prevalence)",
            "auc": auc, "ci95": [lo, hi], "n_pos": int(sum(y)), "n_hold": len(y),
            "train_rate": {k: round(rate[k], 4) for k in rate}}


def main():
    labels, split = load()
    res = {"track": "LUNG_NSCLC", "purpose": "subtype-only baseline (lung EGFR/KRAS only; non-circular)",
           "claim_level": "hypothesis_only", "critic_status": "pending",
           "note": "Other cancers/endpoints not computed: subtype-only is circular there.",
           "endpoints": {}}
    for ep in ("egfr_activating", "kras_g12c"):
        res["endpoints"][ep] = run(ep, labels, split)
        r = res["endpoints"][ep]
        print(f"{ep}: histology-only AUC={r['auc']} CI{r['ci95']} n_pos={r['n_pos']}/{r['n_hold']} rate={r['train_rate']}")
    out = D / "baseline_subtype.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
