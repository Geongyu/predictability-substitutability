#!/usr/bin/env python3
"""Part A (CMS prediction-fidelity) + Part B (treatment-marker routing cost) for COLORECTAL.

Reuses train_eval / bootstrap_auc / patient_agg from run_mil_cost.py (CLAM-SB, site-disjoint
val+test pooled holdout, shuffle-null, prevalence). Split = existing split.csv extended by
site->fold map to all embedding patients (preserves braf comparability, site-disjoint).

Part A: cms{1..4}_vs_rest one-vs-rest. Output mil_cms_fidelity.json. Prediction-fidelity ONLY,
        not used as a treatment cost.
Part B: msi_high / anti_egfr_eligible / braf_v600 MIL + misroute_rate (measured vs predicted).
        mean_cost only for braf (frozen distance 0.868); null for MSI/anti-EGFR. Output routing_cost.json.
"""
import os, json, csv, sys, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent  # crosscancer/COLORECTAL/full
CC = HERE.parent.parent                 # crosscancer
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
EMB_DIR = DATA_ROOT / "crosscancer" / "COLORECTAL" / "uni_v1"
sys.path.insert(0, str(CC))
from run_mil_cost import train_eval, bootstrap_auc, patient_agg, default_device

COMMIT = "d3a614f88855eafd50bf764aaf1ec07e543c33b6"

def build_split():
    """site->fold from existing split.csv, extended to all embedding patients."""
    base = {r["case_id"]: r["split"] for r in csv.DictReader(open(HERE/"split.csv"))}
    site_fold = {}
    for c, f in base.items():
        site_fold[c.split("-")[1]] = f
    emb = {p.name[:12] for p in EMB_DIR.glob("*_uni_embeddings.npy")}
    split = {}
    extended = []
    for c in sorted(emb):
        s = c.split("-")[1]
        if c in base:
            split[c] = base[c]
        elif s in site_fold:
            split[c] = site_fold[s]; extended.append(c)
        else:
            split[c] = "train"; extended.append(c)  # new site (only 'BM', 1 pt) -> train, negligible
    return split, extended

def build_slides(split):
    slides = []
    for p in sorted(EMB_DIR.glob("*_uni_embeddings.npy")):
        sid = p.name.replace("_uni_embeddings.npy", "")
        cid = sid[:12]
        if cid in split:
            slides.append({"slide_id": sid, "case_id": cid, "path": p, "split": split[cid]})
    return slides

def load_cms_labels():
    lab = {}
    for r in csv.DictReader(open(HERE/"cms_labels_authoritative.csv")):
        cms = r["cms"]
        d = {}
        if cms in ("CMS1","CMS2","CMS3","CMS4"):
            for k in (1,2,3,4):
                d[f"cms{k}_vs_rest"] = 1 if cms == f"CMS{k}" else 0
        lab[r["case_id"]] = d  # NOLBL / unknown -> empty dict = "" for cms endpoints
    return lab

def load_treatment_labels():
    lab = {}
    for r in csv.DictReader(open(HERE/"labels_treatment.csv")):
        lab[r["case_id"]] = {k: int(r[k]) for k in ("msi_high","all_ras","braf_v600","anti_egfr_eligible")}
    return lab

def run_marker(slides, labels, endpoint, device):
    recs, dev_auc = train_eval(slides, labels, endpoint, device, shuffle=False)
    if recs is None:
        return None
    pa = patient_agg(recs)
    y = [v[1] for v in pa.values()]; p = [v[0] for v in pa.values()]
    auc, lo, hi = bootstrap_auc(y, p)
    srecs, _ = train_eval(slides, labels, endpoint, device, shuffle=True)
    sh = None
    if srecs:
        spa = patient_agg(srecs); sy=[v[1] for v in spa.values()]; sp=[v[0] for v in spa.values()]
        sh, _, _ = bootstrap_auc(sy, sp)
    n_pos = int(sum(y))
    return {
        "real_auroc": auc, "ci95": [lo, hi], "shuffle_null_auroc": sh,
        "prevalence_baseline_auroc": 0.5,
        "n_holdout": len(pa), "n_pos": n_pos, "dev_auc": dev_auc,
        "eval": "holdout(val+test) pooled, site-disjoint",
        "exploratory": n_pos < 25,
        "patient_proba": {c: round(v[0], 4) for c, v in pa.items()},
        "patient_true": {c: v[1] for c, v in pa.items()},
    }

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    split, extended = build_split()
    slides = build_slides(split)
    print(f"slides={len(slides)}  extended(site-map) n={len(extended)}  device={a.device}")

    # ---------- Part A: CMS prediction-fidelity ----------
    cms_lab = load_cms_labels()
    partA = {"track": "Part A: CMS prediction-fidelity (subtype recovery AUROC)",
             "note": "PREDICTION-FIDELITY ONLY. CMS is a prognostic subtype, not an NCCN treatment-routing marker (Buikhuisen JNCI 2022). "
                     "This AUROC is not a treatment substitution cost. Prior work: imCMS H&E->CMS AUC 0.84 (Sirinukunwattana Gut 2021).",
             "claim_level": "hypothesis_only", "critic_status": "pending", "commit": COMMIT,
             "eval": "site-disjoint holdout(val+test pooled)", "endpoints": {}}
    for k in (1,2,3,4):
        ep = f"cms{k}_vs_rest"; t=time.time()
        r = run_marker(slides, cms_lab, ep, a.device)
        partA["endpoints"][ep] = r
        if r: print(f"  [A] {ep}: AUROC={r['real_auroc']} CI{r['ci95']} shuffle={r['shuffle_null_auroc']} "
                    f"n+={r['n_pos']}/{r['n_holdout']} exploratory={r['exploratory']} ({time.time()-t:.0f}s)")
    (HERE/"mil_cms_fidelity.json").write_text(json.dumps(partA, indent=2, ensure_ascii=False))
    print(f"  wrote mil_cms_fidelity.json")

    # ---------- Part B: treatment-marker routing cost ----------
    tx_lab = load_treatment_labels()
    scheme = json.loads((HERE/"routing_scheme_preregistered.json").read_text())
    braf_dist = 0.868  # frozen_map antiBRAF__baseline; only frozen distance available
    partB = {"track": "Part B: treatment routing substitution cost (routing-fidelity, cost-of-substitution)",
             "note": "misroute_rate = routing disagreement between measured and H&E-predicted marker (distance-free, comparable to breast receptor routing). "
                     "mean_cost only for axes with a frozen_map distance (braf 0.868); MSI/anti-EGFR are not in the frozen_map, so null. "
                     "Not a drug response prediction.",
             "claim_level": "hypothesis_only", "critic_status": "pending", "commit": COMMIT,
             "preregistered": "routing_scheme_preregistered.json", "axes": {}}
    axis_route = {"msi_high": "anti-PD-1", "anti_egfr_eligible": "anti-EGFR", "braf_v600": "BRAF+EGFR"}
    mean_cost_map = {"braf_v600": braf_dist, "msi_high": None, "anti_egfr_eligible": None}
    for ep in ("msi_high", "anti_egfr_eligible", "braf_v600"):
        t=time.time(); r = run_marker(slides, tx_lab, ep, a.device)
        if r is None:
            partB["axes"][ep] = {"status": "skip(insufficient)"}; continue
        pp = r["patient_proba"]; pt = r["patient_true"]
        n = mis = 0
        for c, proba in pp.items():
            true = pt.get(c)
            if true is None: continue
            n += 1; mis += int(int(proba >= 0.5) != true)
        misroute = round(mis/n, 3) if n else None
        mc = None
        if mean_cost_map[ep] is not None and misroute is not None:
            mc = round(misroute * mean_cost_map[ep], 3)
        partB["axes"][ep] = {
            "route_to": axis_route[ep], "cost_hypothesis": {a2["marker"]:a2["cost_hypothesis"] for a2 in scheme["axes"]}[ep],
            "mil_auroc": r["real_auroc"], "ci95": r["ci95"], "shuffle_null_auroc": r["shuffle_null_auroc"],
            "n_holdout": r["n_holdout"], "n_pos": r["n_pos"], "exploratory": r["exploratory"],
            "misroute_rate": misroute,
            "mean_cost": mc,
            "mean_cost_note": None if mc is not None else "No frozen_map distance for this axis (ICI/antibody axes excluded); null.",
        }
        print(f"  [B] {ep}: AUROC={r['real_auroc']} CI{r['ci95']} misroute={misroute} mean_cost={mc} "
              f"n+={r['n_pos']}/{r['n_holdout']} exploratory={r['exploratory']} ({time.time()-t:.0f}s)")
    (HERE/"routing_cost.json").write_text(json.dumps(partB, indent=2, ensure_ascii=False))
    print(f"  wrote routing_cost.json")

if __name__ == "__main__":
    main()
