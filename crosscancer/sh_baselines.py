#!/usr/bin/env python3
"""Pixel-mean baseline for all cancers and endpoints.

Slide embeddings are mean-pooled (no attention) -> patient mean -> logistic regression.
Measures how much of the MIL AUROC a mean-embedding baseline explains. MIL ~ pixel-mean means no gain from attention.
CPU only. Same split, holdout (val+test pooled, site-disjoint) and patient-level aggregation as the MIL runs.

Usage: python sh_baselines.py --cancer LUNG_NSCLC
Output: <cancer>/full/baseline_pixelmean.json
"""
import json, sys, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import run_mil_cost as m  # load_meta, bootstrap_auc

ENDPOINTS = {
    "LUNG_NSCLC":    ["histology_lusc", "egfr_activating", "kras_g12c"],
    "GASTRIC_STAD":  ["lauren_diffuse", "msi_h", "erbb2_amp", "ebv"],
    "HEADNECK_HNSC": ["hpv_pos", "egfr_amp", "grade_high"],
    "COLORECTAL":    ["msi_high", "anti_egfr_eligible", "braf_v600"],
}
HERE = Path(__file__).resolve().parent

def slide_mean(path):
    a = np.load(path).astype("float32")
    return a.mean(axis=0)  # (1024,)

def run_ep(slides, labels, ep, mean_cache):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    def rows(split):
        out = []
        for s in slides:
            lv = labels.get(s["case_id"], {}).get(ep, "")
            if s["split"] == split and lv != "":
                out.append((s, int(lv)))
        return out
    tr = rows("train"); hold = rows("val") + rows("test")
    if not tr or not hold or len({l for _, l in tr}) < 2:
        return {"status": "skip(insufficient)"}
    # patient-level mean embedding
    def pat_X(recs):
        by = defaultdict(list); yv = {}
        for s, y in recs:
            by[s["case_id"]].append(mean_cache[s["slide_id"]]); yv[s["case_id"]] = y
        cs = sorted(by); X = np.vstack([np.mean(by[c], axis=0) for c in cs]); Y = np.array([yv[c] for c in cs])
        return cs, X, Y
    _, Xtr, Ytr = pat_X(tr); ch, Xh, Yh = pat_X(hold)
    if len(set(Ytr)) < 2 or len(set(Yh)) < 2:
        return {"status": "skip(single-class)"}
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(sc.transform(Xtr), Ytr)
    p = clf.predict_proba(sc.transform(Xh))[:, 1]
    auc, lo, hi = m.bootstrap_auc(list(Yh), list(p))
    return {"auc": auc, "ci95": [lo, hi], "n_pos": int(Yh.sum()), "n_hold": len(Yh),
            "method": "pixel-mean(slide mean-pool->patient mean)->StandardScaler->LogReg(balanced)"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=list(ENDPOINTS))
    ap.add_argument("--endpoints")
    a = ap.parse_args()
    labels, split, slides = m.load_meta(a.cancer)
    eps = a.endpoints.split(",") if a.endpoints else ENDPOINTS[a.cancer]
    print(f"[{a.cancer}] slides={len(slides)} eps={eps}: building mean-pool cache...", flush=True)
    t0 = time.time()
    mean_cache = {s["slide_id"]: slide_mean(s["path"]) for s in slides}
    print(f"  cached {len(mean_cache)} slides ({time.time()-t0:.0f}s)", flush=True)
    res = {"track": a.cancer, "purpose": "pixel-mean baseline",
           "claim_level": "hypothesis_only", "critic_status": "pending", "endpoints": {}}
    for ep in eps:
        t = time.time(); r = run_ep(slides, labels, ep, mean_cache)
        res["endpoints"][ep] = r
        print(f"  {ep}: {r.get('auc', r.get('status'))} n_pos={r.get('n_pos')} ({time.time()-t:.0f}s)", flush=True)
    out = HERE / a.cancer / "full" / "baseline_pixelmean.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"wrote {out}", flush=True)

if __name__ == "__main__":
    main()
