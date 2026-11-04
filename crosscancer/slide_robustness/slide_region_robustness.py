"""Slide and region robustness (HPV, MSI).

No tumour annotations are available, so tumour-only and necrosis-exclusion analyses are recorded as not feasible.
Dependence on a few tiles/regions is measured with tile subsampling and top-k concentration instead.

(A) Multi-slide concordance: agreement of predicted scores across slides of the same patient.
    Multi-slide patients have almost no positives (HNSC HPV+ 1, STAD MSI+ 0), so positive-class
    concordance cannot be measured; results are reported on scores / negatives.

(B) Tile-subsample region robustness (all patients, including positives).
    Tiles of each patient's first slide are randomly subsampled -> mean-pool -> rescored with a held-out LR.
    Small score variation (SD, call flips) across subsamples = signal spread over the slide;
    large = dependence on few tiles/regions. Out-of-fold (StratifiedKFold) models.

(C) Tile contribution concentration: per-tile linear contribution s_i = w.z_i from the held-out LR weights;
    share of the slide signal from the top 5% tiles. High = dependence on few tiles.

CPU. Uses existing UNI per-tile embeddings.
"""
import argparse, csv, json, os
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CC = Path(__file__).resolve().parents[1]           # crosscancer/
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
META = CC                                          # <cohort>/full/patient_labels.csv
EMB = str(DATA_ROOT / "crosscancer" / "{cohort}" / "uni_v1")


def load_cohort(cohort, endpoint):
    meta = Path(META) / cohort / "full"
    labels = {r["case_id"]: r for r in csv.DictReader(open(meta / "patient_labels.csv"))}
    emb = Path(EMB.format(cohort=cohort))
    bycase = defaultdict(list)
    for p in sorted(emb.glob("*_uni_embeddings.npy")):
        bycase[p.name[:12]].append(p)
    return labels, bycase


def multislide_concordance(labels, bycase, endpoint, seed=42):
    """(A) Score agreement across slides of multi-slide patients.
    Each slide is scored by an LR trained on single-slide patients."""
    single = {c: v[0] for c, v in bycase.items()
              if len(v) == 1 and c in labels and labels[c].get(endpoint) in ("0", "1")}
    multi = {c: v for c, v in bycase.items()
             if len(v) >= 2 and c in labels and labels[c].get(endpoint) in ("0", "1")}
    # train on mean-pooled single-slide patients
    Xtr = np.stack([np.load(single[c]).mean(0) for c in single]).astype(np.float32)
    ytr = np.array([int(labels[c][endpoint]) for c in single])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    rows = []
    for c, paths in multi.items():
        scores = [float(clf.predict_proba(sc.transform(np.load(p).mean(0)[None]))[0, 1]) for p in paths]
        rows.append({"case": c, "label": int(labels[c][endpoint]), "n_slides": len(paths),
                     "slide_scores": [round(s, 4) for s in scores],
                     "score_sd": round(float(np.std(scores, ddof=1)), 4),
                     "score_range": round(float(max(scores) - min(scores)), 4),
                     "calls_agree@0.5": bool(len({s >= 0.5 for s in scores}) == 1)})
    npos = sum(r["label"] for r in rows)
    agree = [r for r in rows if r["calls_agree@0.5"]]
    return {
        "n_multislide_patients": len(rows), "n_pos": npos, "n_neg": len(rows) - npos,
        "median_score_sd": round(float(np.median([r["score_sd"] for r in rows])), 4) if rows else None,
        "median_score_range": round(float(np.median([r["score_range"] for r in rows])), 4) if rows else None,
        "call_agree_rate@0.5": round(len(agree) / len(rows), 4) if rows else None,
        "CONSTRAINT": f"{npos} positive multi-slide patients: positive-class concordance cannot be measured. "
                      f"The agreement rate is dominated by negatives (confident low scores), so whether a positive "
                      f"HPV/MSI call depends on which slide is sampled cannot be answered with this data.",
        "per_patient": sorted(rows, key=lambda r: -r["label"]),
    }


def region_robustness(labels, bycase, endpoint, B=50, fracs=(0.5, 0.25), seed=42):
    """(B)+(C) Tile-subsample stability and tile contribution concentration with out-of-fold models."""
    cases = [c for c, v in bycase.items() if c in labels and labels[c].get(endpoint) in ("0", "1")]
    paths = {c: bycase[c][0] for c in cases}  # first slide per patient
    y = np.array([int(labels[c][endpoint]) for c in cases])
    Xmean = np.stack([np.load(paths[c]).mean(0) for c in cases]).astype(np.float32)

    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(cases))
    per = {c: {} for c in cases}
    idx = np.arange(len(cases))
    for tr, te in skf.split(Xmean, y):
        sc = StandardScaler().fit(Xmean[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xmean[tr]), y[tr])
        w = clf.coef_[0]; b = clf.intercept_[0]
        for i in te:
            c = cases[i]
            Z = sc.transform(np.load(paths[c]))          # (n_tiles, d) standardised tiles
            full = float(clf.predict_proba(Z.mean(0)[None])[0, 1])
            oof[i] = full
            per[c]["label"] = int(y[i]); per[c]["n_tiles"] = int(Z.shape[0]); per[c]["full_score"] = round(full, 4)
            # (B) subsample stability
            for frac in fracs:
                k = max(20, int(frac * Z.shape[0]))
                ss = []
                for _ in range(B):
                    sub = rng.choice(Z.shape[0], k, replace=False)
                    ss.append(float(clf.predict_proba(Z[sub].mean(0)[None])[0, 1]))
                ss = np.array(ss)
                per[c][f"sd@{frac}"] = round(float(ss.std(ddof=1)), 4)
                per[c][f"flip@{frac}"] = round(float((( ss >= 0.5) != (full >= 0.5)).mean()), 4)
            # (C) tile contribution concentration: s_i = w.z_i (slide logit = mean s_i + b)
            s = Z @ w
            order = np.argsort(-s)  # largest positive contribution first
            tot = s.sum()
            top5 = int(max(1, 0.05 * len(s)))
            per[c]["top5pct_tile_share"] = round(float(s[order[:top5]].sum() / tot), 4) if tot != 0 else None
    auc = roc_auc_score(y, oof)

    def agg(mask, key):
        vals = [per[c][key] for c in cases if per[c]["label"] == mask and key in per[c] and per[c][key] is not None]
        return round(float(np.median(vals)), 4) if vals else None

    summary = {"endpoint": endpoint, "n": len(cases), "n_pos": int(y.sum()),
               "oof_auroc_meanpool": round(float(auc), 4), "B_subsamples": B, "fracs": list(fracs)}
    for cls, nm in [(1, "pos"), (0, "neg")]:
        summary[nm] = {
            "median_sd@0.5": agg(cls, "sd@0.5"), "median_flip@0.5": agg(cls, "flip@0.5"),
            "median_sd@0.25": agg(cls, "sd@0.25"), "median_flip@0.25": agg(cls, "flip@0.25"),
            "median_top5pct_tile_share": agg(cls, "top5pct_tile_share"),
        }
    return summary, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=50)
    a = ap.parse_args()
    labels, bycase = load_cohort(a.cohort, a.endpoint)
    print(f"[{a.cohort}/{a.endpoint}] patients {len(bycase)}", flush=True)
    conc = multislide_concordance(labels, bycase, a.endpoint)
    print(f"  (A) multi-slide patients {conc['n_multislide_patients']} (positive {conc['n_pos']}) "
          f"agree@0.5 {conc['call_agree_rate@0.5']}, median score_sd {conc['median_score_sd']}", flush=True)
    rob, per = region_robustness(labels, bycase, a.endpoint, B=a.B)
    print(f"  (B) region-robustness oof-AUROC {rob['oof_auroc_meanpool']} "
          f"pos: sd@0.5 {rob['pos']['median_sd@0.5']} flip@0.5 {rob['pos']['median_flip@0.5']} "
          f"top5%share {rob['pos']['median_top5pct_tile_share']} | "
          f"neg: sd@0.5 {rob['neg']['median_sd@0.5']} top5%share {rob['neg']['median_top5pct_tile_share']}", flush=True)
    out = {"cohort": a.cohort, "endpoint": a.endpoint,
           "A_multislide_concordance": conc, "B_region_robustness": rob,
           "NOT_FEASIBLE": {
               "tumor_only_vs_wsi": "No tumour-region annotations; not measurable.",
               "necrosis_stroma_exclusion": "No tissue-type annotations; not measurable.",
               "attention_FM_cross": "Only uni_v1 embeddings exist for these cohorts; cross-FM attention not possible.",
               "attention_seed_cross": "No trained CLAM models saved; cross-seed attention requires CLAM training (GPU).",
           }}
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"Saved {a.out}\nDONE_{a.cohort}")


if __name__ == "__main__":
    main()
