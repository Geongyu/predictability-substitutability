"""Site/batch confounding audit.
(1) label-site imbalance: TSS site skew of endpoint positives/negatives (Cramer's V).
(2) site-predictability: H&E embeddings (mean-pool) -> TSS site, macro-AUROC + permutation null
    (high values indicate a site signature, Howard et al. 2021).
UNI embeddings. CPU. Uses existing embeddings and labels."""
import argparse, csv, json, os, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score

CC = Path(__file__).resolve().parents[1]           # crosscancer/
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
META = CC                                          # <cohort>/full/{patient_labels.csv, split.csv}
EMB = str(DATA_ROOT / "crosscancer" / "{cohort}" / "uni_v1")

def cramers_v(a, b):
    tab = defaultdict(lambda: defaultdict(int))
    for x, y in zip(a, b):
        tab[x][y] += 1
    rows = sorted({x for x in a}); cols = sorted({y for y in b})
    M = np.array([[tab[r][c] for c in cols] for r in rows], float)
    if M.shape[0] < 2 or M.shape[1] < 2:
        return None
    chi2 = chi2_contingency(M)[0]
    n = M.sum()
    return float(np.sqrt(chi2 / (n * (min(M.shape) - 1))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--anchors", required=True, help="comma-separated endpoints")
    ap.add_argument("--min_site", type=int, default=10, help="minimum slides per site for site-predictability")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = Path(META) / a.cohort / "full"
    labels = {r["case_id"]: r for r in csv.DictReader(open(d / "patient_labels.csv"))}
    site = {r["case_id"]: r["tss_code"] for r in csv.DictReader(open(d / "split.csv"))}
    emb_dir = Path(EMB.format(cohort=a.cohort))

    # slides with an embedding and a site
    slides = []
    for p in sorted(emb_dir.glob("*_uni_embeddings.npy")):
        sid = p.name.replace("_uni_embeddings.npy", ""); cid = sid[:12]
        if cid in site:
            slides.append((cid, p))
    print(f"[{a.cohort}] {len(slides)} slides (embedding+site), {len(set(c for c,_ in slides))} case, {len(set(site[c] for c,_ in slides))} site", flush=True)

    result = {"cohort": a.cohort, "n_slides": len(slides)}

    # (1) label-site imbalance
    imb = {}
    for ep in a.anchors.split(","):
        has = "has_" + ep.replace("_activating", "").replace("_v600e", "").replace("_pos", "").replace("_amp", "").replace("_h", "").replace("_diffuse", "").replace("_high", "")
        cids = [c for c, _ in slides]
        pairs = [(labels[c].get(ep), site[c]) for c in cids
                 if c in labels and labels[c].get(ep) in ("0", "1")]
        if not pairs:
            imb[ep] = {"note": "no labels"}; continue
        lab = [p[0] for p in pairs]; st = [p[1] for p in pairs]
        v = cramers_v(lab, st)
        # fraction of positives in the top site
        pos_sites = Counter(s for l, s in pairs if l == "1")
        npos = sum(pos_sites.values())
        top_frac = (pos_sites.most_common(1)[0][1] / npos) if npos else None
        imb[ep] = {"n": len(pairs), "n_pos": npos, "cramers_v_label_vs_site": round(v, 3) if v else None,
                   "top_site_pos_fraction": round(top_frac, 3) if top_frac else None}
        print(f"  [imbalance] {ep}: CramersV={imb[ep]['cramers_v_label_vs_site']} top-site pos={imb[ep]['top_site_pos_fraction']} (n_pos={npos})", flush=True)
    result["label_site_imbalance"] = imb

    # (2) site-predictability: mean-pool embedding -> TSS
    # StratifiedGroupKFold(groups=case_id) keeps all slides of a patient in one fold.
    print("  [site-pred] loading mean-pooled embeddings...", flush=True)
    X = np.stack([np.load(p).mean(0) for _, p in slides]).astype(np.float32)
    y_site = np.array([site[c] for c, _ in slides])
    groups = np.array([c for c, _ in slides])  # case_id (patient-level groups)
    # sites below min_site are merged into 'OTHER'
    cnt = Counter(y_site)
    keep = {s for s, n in cnt.items() if n >= a.min_site}
    y = np.array([s if s in keep else "OTHER" for s in y_site])
    classes = sorted(set(y))
    n_multi = len(groups) - len(set(groups))
    print(f"  [site-pred] {len(classes)} site-class(>= {a.min_site}), X={X.shape}, "
          f"{len(set(groups))} patients, {n_multi} extra slides from multi-slide patients", flush=True)

    def cv_macro_auc(Xf, yy, grp, seed=42):
        # one-vs-rest macro-AUROC. OneVsRestClassifier gives OvR (equivalent to multi_class="ovr",
        # which sklearn>=1.7 removed from LogisticRegression).
        yb = label_binarize(yy, classes=classes)
        oof = np.zeros_like(yb, float)
        # group-aware: each case_id in one fold only
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(Xf, yy, groups=grp):
            clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=1.0), n_jobs=-1)
            clf.fit(Xf[tr], yb[tr])
            proba = clf.predict_proba(Xf[te])  # (n, n_classes); columns follow clf.classes_ = 0..K-1
            for j in range(yb.shape[1]):
                oof[te, j] = proba[:, j]
        aucs = []
        for j in range(yb.shape[1]):
            if yb[:, j].sum() > 0 and yb[:, j].sum() < len(yb):
                aucs.append(roc_auc_score(yb[:, j], oof[:, j]))
        return float(np.mean(aucs))

    def permute_by_group(yy, grp, rng):
        """Permute site labels at patient level (site is constant within a patient)."""
        uniq = sorted(set(grp))
        gmap = {g: yy[grp == g][0] for g in uniq}
        perm = rng.permutation(list(gmap.values()))
        newmap = dict(zip(uniq, perm))
        return np.array([newmap[g] for g in grp])

    Xs = StandardScaler().fit_transform(X)
    real_auc = cv_macro_auc(Xs, y, groups)
    # permutation null (patient-level site label shuffle)
    rng = np.random.default_rng(0)
    null = [cv_macro_auc(Xs, permute_by_group(y, groups, rng), groups, seed=s) for s in range(5)]
    null_mean = float(np.mean(null)); null_sd = float(np.std(null, ddof=1))
    result["site_predictability"] = {
        "cv": "StratifiedGroupKFold(5) on case_id",
        "n_patients": len(set(groups)), "n_multislide_extra": n_multi,
        "n_site_classes": len(classes), "macro_auroc": round(real_auc, 4),
        "permutation_null_mean": round(null_mean, 4), "permutation_null_sd": round(null_sd, 4),
        "above_null": bool(real_auc > null_mean + 2 * null_sd),
        "interpretation": "High AUROC = H&E embeddings strongly predict TSS site, i.e. a site signature (Howard 2021). "
                          "Patient-level group-aware CV, so no multi-slide leakage.",
    }
    print(f"  [site-pred] macro-AUROC={real_auc:.4f} vs null {null_mean:.4f}±{null_sd:.4f} -> {'strong site signature' if real_auc>0.7 else 'weak site signature'}", flush=True)

    json.dump(result, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"Saved {a.out}\nDONE_{a.cohort}")

if __name__ == "__main__":
    main()
