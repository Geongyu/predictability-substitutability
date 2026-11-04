#!/usr/bin/env python3
"""
MIL training + substitutability test for cohorts without a frozen_map (STAD, HNSC, UCEC).
Reuses the run_mil_cost helpers.

Per endpoint {real, shuffle-null, prevalence}: CLAM-SB, AUROC + CI on val+test pooled holdout (site-disjoint).
- No frozen_map (therapeutic distance): mean_cost = null; per-endpoint misroute_rate is the primary metric.
- n_pos < 25 -> flagged exploratory (wide CI).
- Positive control: STAD = lauren_diffuse (strong morphology, expected >= 0.85),
  HNSC = grade_high (soft; may fall below 0.75 in SCC; HPV >= 0.80 serves as the sanity check).

Output: <cancer>/full/mil_cost_results.json
Usage: python sh_mil_cost.py --cancer GASTRIC_STAD --device cuda
"""
import json, sys, argparse, time
from pathlib import Path
from collections import defaultdict
import run_mil_cost as m

HERE = Path(__file__).resolve().parent
EXPLORATORY_NPOS = 25

CANCER_CFG = {
 "GASTRIC_STAD": {
   "endpoints": ["lauren_diffuse", "msi_h", "erbb2_amp", "ebv"],
   "positive_control": "lauren_diffuse",   # strong morphology (positive control)
   "route_axis": {"erbb2_amp": "antiHER2", "msi_h": "antiPD1"},  # routing axes (priority order)
   "route_default": "chemo",
 },
 "HEADNECK_HNSC": {
   "endpoints": ["hpv_pos", "egfr_amp", "grade_high"],
   "positive_control": "grade_high",       # soft (differentiation); HPV >= 0.80 is the sanity check
   "route_axis": {"hpv_pos": "deescalate", "egfr_amp": "antiEGFR"},
   "route_default": "standard",
 },
 # UCEC: exploratory extension, outside the preregistered cancers and the law aggregate
 "UCEC": {
   "endpoints": ["histology_serous", "msi_h", "her2_amp", "pole"],
   "positive_control": "histology_serous", # serous vs endometrioid: strong morphology (positive control)
   "route_axis": {"msi_h": "antiPD1", "her2_amp": "antiHER2"},  # dMMR -> IO (dostarlimab), serous HER2-amp -> antiHER2
   "route_default": "standard",
 },
}

def misroute_per_endpoint(results):
    """Per-endpoint misclassification rate (proba >= 0.5 vs true)."""
    mr={}
    for ep, r in results["endpoints"].items():
        pp=r.get("patient_proba",{}); pt=r.get("patient_true",{})
        if not pp: continue
        n=mis=0
        for c,proba in pp.items():
            true=pt.get(c)
            if true is None: continue
            n+=1; mis+=int(int(proba>=0.5)!=true)
        mr[ep]={"n":n,"misroute_rate":round(mis/n,3) if n else None,
                "auc":r.get("real",{}).get("auc"),
                "mean_cost":None,  # no frozen_map -> null
                }
    return mr

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=list(CANCER_CFG))
    ap.add_argument("--device", default=m.default_device())
    ap.add_argument("--fm", default="uni", choices=list(m.FM_SPEC),
                    help="foundation model embeddings: uni (default), virchow2, uni2h")
    a=ap.parse_args()
    # set CLAM-SB feature_dim in run_mil_cost to the FM embedding dim
    m.FEATURE_DIM = m.FM_SPEC[a.fm]["dim"]
    cfg=CANCER_CFG[a.cancer]
    labels, split, slides = m.load_meta(a.cancer, a.fm)
    if a.fm != "uni":
        print(f"[multi-FM] training on {a.fm} embeddings (dim={m.FEATURE_DIM})")
    print(f"{a.cancer}: {len(slides)} slides with embeddings, {len(labels)} labelled patients")
    results={"cancer":a.cancer,"n_slides":len(slides),"claim_level":"hypothesis_only",
             "critic_status":"pending","exploratory_npos_threshold":EXPLORATORY_NPOS,
             "fm":a.fm,"feature_dim":m.FEATURE_DIM,"endpoints":{}}
    for ep in cfg["endpoints"]:
        t=time.time(); r=m.run_endpoint(slides, labels, ep, a.device)
        # exploratory flag
        npos=r.get("real",{}).get("n_pos")
        if npos is not None:
            r["exploratory"]= bool(npos < EXPLORATORY_NPOS)
        results["endpoints"][ep]=r
        real=r.get("real",{}); sh=r.get("shuffle_null",{})
        flag=" [EXPLORATORY n_pos<25]" if r.get("exploratory") else ""
        print(f"  {ep}: real AUC={real.get('auc')} CI{real.get('ci95')} "
              f"n+={real.get('n_pos')}/{real.get('n_holdout_patients')} | shuffle={sh.get('auc')}"
              f" ({time.time()-t:.0f}s){flag}")
    # positive-control check
    pc=cfg["positive_control"]
    if pc and pc in results["endpoints"]:
        pca=results["endpoints"][pc].get("real",{}).get("auc")
        gate=(pca is not None and pca>=0.75)
        soft=(a.cancer=="HEADNECK_HNSC")
        results["positive_control_gate"]={"endpoint":pc,"auc":pca,"pass":gate,"soft":soft,
            "note":("morphology positive control AUROC>=0.75 means the pipeline works." if not soft else
                    "grade is a soft positive control in SCC; HPV>=0.80 is the sanity check.")}
        print(f"  [positive control {pc}] AUROC={pca} -> {'PASS' if gate else ('SOFT-FAIL' if soft else 'FAIL')}")
    # cost: no frozen_map -> per-endpoint misroute_rate (mean_cost=null)
    results["endpoint_misroute"]=misroute_per_endpoint(results)
    results["cost_note"]="No frozen_map (therapeutic distance): mean_cost=null; misroute_rate is the primary metric."
    print("  misroute(per-endpoint, cost lead):",
          json.dumps({k:{"misroute_rate":v["misroute_rate"],"auc":v["auc"]} for k,v in results["endpoint_misroute"].items()}, ensure_ascii=False))
    # output name: mil_cost_results.json for UNI, FM suffix otherwise
    fname = "mil_cost_results.json" if a.fm=="uni" else f"mil_cost_results_{a.fm}.json"
    out=HERE/a.cancer/"full"/fname
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  wrote {out}")

if __name__=="__main__":
    main()
