#!/usr/bin/env python3
"""5-seed shuffle-null robustness for all endpoints.

Calls run_mil_cost.train_eval unchanged (real seed 42 reproduces mil_cost_results).
Per endpoint: real (seed 42) + shuffle-null with seeds [42, 1, 2, 3, 4].
Criterion: real_auroc > null_mean + 2*null_sd (ddof=1); otherwise FAIL (weak signal is not zero signal).

Usage:
  python sh_robustness_5seed.py --cancer LUNG_NSCLC --device cuda
  python sh_robustness_5seed.py --cancer LUNG_NSCLC --endpoints histology_lusc --real-only   # quick check
Output: <cancer>/full/shuffle_null_robustness.json (UNI), shuffle_null_robustness_<fm>.json (other FMs)
"""
import json, sys, time, argparse
from pathlib import Path
import numpy as np
import run_mil_cost as m

HERE = Path(__file__).resolve().parent
SEEDS = [42, 1, 2, 3, 4]
ENDPOINTS = {
    "LUNG_NSCLC":    ["histology_lusc", "egfr_activating", "kras_g12c"],
    "GASTRIC_STAD":  ["lauren_diffuse", "msi_h", "erbb2_amp", "ebv"],
    "HEADNECK_HNSC": ["hpv_pos", "egfr_amp", "grade_high"],
    "UCEC":          ["histology_serous", "msi_h", "her2_amp", "pole"],  # exploratory extension
    # COLORECTAL: endpoint in run_mil_cost CANCER_CFG (used for the multi-FM comparison)
    "COLORECTAL":    ["braf_v600e"],
}

def one_auc(slides, labels, ep, device, shuffle, seed):
    recs, _ = m.train_eval(slides, labels, ep, device, shuffle=shuffle, seed=seed)
    if recs is None:
        return None
    pa = m.patient_agg(recs)
    y = [v[1] for v in pa.values()]; p = [v[0] for v in pa.values()]
    auc, _, _ = m.bootstrap_auc(y, p)
    return auc, int(sum(y)), len(pa)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=list(ENDPOINTS))
    ap.add_argument("--device", default=m.default_device())
    ap.add_argument("--endpoints", help="comma-separated; default: all endpoints of the cancer")
    ap.add_argument("--real-only", action="store_true", help="quick check: real (seed 42) only")
    ap.add_argument("--fm", default="uni", choices=list(m.FM_SPEC),
                    help="foundation model embeddings: uni (default), virchow2, uni2h")
    ap.add_argument("--out")
    a = ap.parse_args()
    # set CLAM-SB feature_dim in run_mil_cost to the FM embedding dim
    m.FEATURE_DIM = m.FM_SPEC[a.fm]["dim"]
    labels, split, slides = m.load_meta(a.cancer, a.fm)
    if a.fm != "uni":
        print(f"[multi-FM] 5-seed test on {a.fm} embeddings (dim={m.FEATURE_DIM})", flush=True)
    eps = a.endpoints.split(",") if a.endpoints else ENDPOINTS[a.cancer]
    print(f"[{a.cancer}] slides={len(slides)} labels={len(labels)} device={a.device} eps={eps}", flush=True)
    res = {"track": a.cancer, "purpose": "5-seed shuffle-null robustness",
           "fm": a.fm, "feature_dim": m.FEATURE_DIM,
           "claim_level": "hypothesis_only", "critic_status": "pending",
           "criterion": "real_auroc > null_mean + 2*null_sd (5 seed, ddof=1)",
           "shuffle_seeds": SEEDS, "endpoints": {}}
    for ep in eps:
        t = time.time()
        real = one_auc(slides, labels, ep, a.device, False, 42)
        if real is None:
            res["endpoints"][ep] = {"status": "skip(insufficient)"}; print(f"  {ep}: skip", flush=True); continue
        real_auc, n_pos, n_hold = real
        if a.real_only:
            print(f"  {ep}: real(seed42)={real_auc} n_pos={n_pos}/{n_hold} ({time.time()-t:.0f}s)", flush=True)
            res["endpoints"][ep] = {"real_auroc": round(real_auc, 4), "n_pos": n_pos, "n_hold": n_hold, "real_only": True}
            continue
        nulls = []
        for s in SEEDS:
            r = one_auc(slides, labels, ep, a.device, True, s)
            nulls.append(round(r[0], 4) if r else None)
        nn = [x for x in nulls if x is not None]
        nm = float(np.mean(nn)); nsd = float(np.std(nn, ddof=1)); thr = nm + 2 * nsd
        passed = bool(real_auc > thr)
        res["endpoints"][ep] = {"real_auroc": round(real_auc, 4), "n_pos": n_pos, "n_hold": n_hold,
                                "null_mean": round(nm, 4), "null_sd": round(nsd, 4),
                                "threshold": round(thr, 4), "pass": passed, "null_seeds": nulls}
        print(f"  {ep}: real {real_auc} vs thr {thr:.4f} -> {'PASS' if passed else 'FAIL'}  null={nulls} ({time.time()-t:.0f}s)", flush=True)
    # output name: shuffle_null_robustness.json for UNI, FM suffix otherwise
    _fn = "shuffle_null_robustness.json" if a.fm == "uni" else f"shuffle_null_robustness_{a.fm}.json"
    out = Path(a.out) if a.out else HERE / a.cancer / "full" / _fn
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"wrote {out}", flush=True)

if __name__ == "__main__":
    main()
