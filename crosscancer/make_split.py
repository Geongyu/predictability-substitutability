#!/usr/bin/env python3
"""
Cross-cancer site-disjoint patient-level split (same logic as BRCA split_policy_v0).

- Unit = patient (case_id). site = TCGA TSS code (case_id[5:7]). Each site goes to one fold only (Howard et al.).
- Greedy LPT: largest sites first, whole site to the fold furthest below target (70/15/15). No RNG (hash tie-break).
- Checks: patient overlap == 0, site-disjoint. split_hash = sha256(sorted (case_id, split))[:16].
Input: <cancer>/full/patient_labels.csv (case_id). Output: <cancer>/full/split.csv + split_meta.json.
"""
import csv, json, hashlib
from collections import defaultdict, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RATIOS = (0.70, 0.15, 0.15); SEED = 42

def tss(case_id):
    p = case_id.split("-")
    return p[1] if len(p) >= 2 else "NA"

def assign(case_ids):
    site_cases = defaultdict(set)
    for c in case_ids: site_cases[tss(c)].add(c)
    sites = sorted(site_cases.items(), key=lambda x: (-len(x[1]),
                   hashlib.sha256(f"{SEED}:{x[0]}".encode()).hexdigest()))
    total = sum(len(v) for _, v in sites) or 1
    target = {"train":RATIOS[0]*total, "val":RATIOS[1]*total, "test":RATIOS[2]*total}
    assigned = {"train":0.0,"val":0.0,"test":0.0}; site_fold = {}
    for s, cases in sites:
        fold = max(("train","val","test"), key=lambda f: target[f]-assigned[f])
        site_fold[s] = fold; assigned[fold] += len(cases)
    return {c: site_fold[tss(c)] for c in case_ids}

def split_hash(fold_map):
    items = sorted(fold_map.items())
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16]

def run(cancer):
    lab = HERE / cancer / "full" / "patient_labels.csv"
    if not lab.exists():
        print(f"[SKIP] {cancer}: patient_labels.csv not found"); return
    case_ids = sorted({r["case_id"] for r in csv.DictReader(open(lab)) if r["case_id"]})
    fold = assign(case_ids)
    # checks
    site_folds = defaultdict(set)
    for c, f in fold.items(): site_folds[tss(c)].add(f)
    bad_site = [s for s, fs in site_folds.items() if len(fs) > 1]
    assert not bad_site, f"SITE LEAK {bad_site[:5]}"
    dist = Counter(fold.values())
    out = HERE / cancer / "full" / "split.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id","tss_code","split"])
        for c in case_ids: w.writerow([c, tss(c), fold[c]])
    meta = {"cancer":cancer, "policy":"site-disjoint patient-level (split_policy_v0 logic)",
            "split_hash":split_hash(fold), "seed":SEED, "n_patients":len(case_ids),
            "n_sites":len(site_folds), "dist":dict(dist), "site_disjoint":True, "patient_overlap":0}
    (HERE / cancer / "full" / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{cancer}: {len(case_ids)} patients {len(site_folds)} sites  dist={dict(dist)}  "
          f"site-disjoint OK  hash={meta['split_hash']}")

if __name__ == "__main__":
    for c in ["LUNG_NSCLC","COLORECTAL"]:
        run(c)
