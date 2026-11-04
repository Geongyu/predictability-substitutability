#!/usr/bin/env python3
"""Open-item resolution for COLORECTAL/full (Task1 mean_cost + Task2 5-seed shuffle-null).

- train_eval_ext = copy of run_mil_cost.train_eval that also records slide_id/n_tiles in recs
  (same training path and RNG use, so real seed 42 reproduces the saved results).
- Per endpoint: real (seed 42) + shuffle-null with seeds [42, 1, 2, 3, 4]. Slide- and patient-level probabilities.
- Output: per-endpoint partial JSON at --out (merged separately).
Usage: python run_openitems_robustness.py --device cuda --endpoints cms1_vs_rest,msi_high --out part_gpu0.json
"""
import os, json, csv, sys, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent  # crosscancer/COLORECTAL/full
CC = HERE.parent.parent                 # crosscancer
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
EMB_DIR = DATA_ROOT / "crosscancer" / "COLORECTAL" / "uni_v1"
sys.path.insert(0, str(CC))
from run_mil_cost import bootstrap_auc, patient_agg, default_device
sys.path.insert(0, str(CC.parent / "pipeline/modeling/baselines"))

SHUFFLE_SEEDS = [42, 1, 2, 3, 4]


def build_split():
    base = {r["case_id"]: r["split"] for r in csv.DictReader(open(HERE / "split.csv"))}
    site_fold = {}
    for c, f in base.items():
        site_fold[c.split("-")[1]] = f
    emb = {p.name[:12] for p in EMB_DIR.glob("*_uni_embeddings.npy")}
    split = {}
    for c in sorted(emb):
        s = c.split("-")[1]
        if c in base:
            split[c] = base[c]
        elif s in site_fold:
            split[c] = site_fold[s]
        else:
            split[c] = "train"
    return split


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
    for r in csv.DictReader(open(HERE / "cms_labels_authoritative.csv")):
        cms = r["cms"]; d = {}
        if cms in ("CMS1", "CMS2", "CMS3", "CMS4"):
            for k in (1, 2, 3, 4):
                d[f"cms{k}_vs_rest"] = 1 if cms == f"CMS{k}" else 0
        lab[r["case_id"]] = d
    return lab


def load_treatment_labels():
    lab = {}
    for r in csv.DictReader(open(HERE / "labels_treatment.csv")):
        lab[r["case_id"]] = {k: int(r[k]) for k in ("msi_high", "all_ras", "braf_v600", "anti_egfr_eligible")}
    return lab


def train_eval_ext(slides, labels, endpoint, device, shuffle=False, epochs=40, seed=42):
    """Copy of run_mil_cost.train_eval that also records slide_id/n_tiles. Same training path and RNG use."""
    import torch, torch.nn as nn
    from attention_mil import CLAMSB
    torch.manual_seed(seed); np.random.seed(seed)

    def rows(split):
        out = []
        for s in slides:
            lv = labels.get(s["case_id"], {}).get(endpoint, "")
            if s["split"] == split and lv != "":
                out.append((s, int(lv)))
        return out

    tr_all = rows("train")
    hold = rows("val") + rows("test")
    if not tr_all or not hold or len({l for _, l in tr_all}) < 2:
        return None, None
    pats = sorted({s["case_id"] for s, _ in tr_all})
    rng = np.random.default_rng(seed); rng.shuffle(pats)
    dev_pat = set(pats[:max(1, int(len(pats) * 0.15))])
    tr = [(s, y) for s, y in tr_all if s["case_id"] not in dev_pat]
    va = [(s, y) for s, y in tr_all if s["case_id"] in dev_pat]
    te = hold
    if len({l for _, l in tr}) < 2:
        return None, None
    if shuffle:
        ys = [l for _, l in tr]; np.random.default_rng(seed).shuffle(ys)
        tr = [(s, y) for (s, _), y in zip(tr, ys)]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = CLAMSB(feature_dim=1024, hidden_dim=512, att_dim=256, dropout=0.25).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    def emb(s):
        return torch.from_numpy(np.load(s["path"]).astype("float32")).to(dev)

    best_va = -1; best_state = None; patience = 7; bad = 0
    from sklearn.metrics import roc_auc_score
    for ep in range(epochs):
        model.train(); order = np.random.permutation(len(tr))
        for i in order:
            s, y = tr[i]; opt.zero_grad()
            logit, _ = model(emb(s))
            loss = lossf(logit, torch.tensor([float(y)], device=dev))
            loss.backward(); opt.step()
        model.eval(); yp = []; yt = []
        with torch.no_grad():
            for s, y in va:
                logit, _ = model(emb(s)); yp.append(torch.sigmoid(logit).item()); yt.append(y)
        va_auc = roc_auc_score(yt, yp) if len(set(yt)) > 1 else 0.5
        if va_auc > best_va:
            best_va = va_auc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval(); recs = []
    with torch.no_grad():
        for s, y in te:
            arr = np.load(s["path"]).astype("float32")
            logit, _ = model(torch.from_numpy(arr).to(dev))
            recs.append({"case_id": s["case_id"], "slide_id": s["slide_id"],
                         "proba": torch.sigmoid(logit).item(), "y": y, "n_tiles": int(arr.shape[0])})
    return recs, round(float(best_va), 4)


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None, None
    rho, pval = spearmanr(x, y)
    return round(float(rho), 4), round(float(pval), 4)


def run_endpoint(slides, labels, endpoint, device):
    t0 = time.time()
    real_recs, dev_auc = train_eval_ext(slides, labels, endpoint, device, shuffle=False, seed=42)
    if real_recs is None:
        return {"status": "skip(insufficient)"}
    pa = patient_agg(real_recs)
    y = [v[1] for v in pa.values()]; p = [v[0] for v in pa.values()]
    real_auc, lo, hi = bootstrap_auc(y, p)
    # slide-level spearman(proba, n_tiles)
    s_proba = [r["proba"] for r in real_recs]; s_n = [r["n_tiles"] for r in real_recs]
    n_at_cap = sum(1 for n in s_n if n >= 5000)
    rho_s, pval_s = spearman(s_proba, s_n)
    # patient-level spearman: patient total tiles
    ptiles = defaultdict(int)
    for r in real_recs:
        ptiles[r["case_id"]] += r["n_tiles"]
    pp_cases = list(pa.keys())
    rho_p, pval_p = spearman([pa[c][0] for c in pp_cases], [ptiles[c] for c in pp_cases])
    # misroute decomposition (patient level, pred>=0.5)
    n_fn = n_fp = n_tot = 0
    for c, (proba, true) in pa.items():
        pred = int(proba >= 0.5); n_tot += 1
        if true == 1 and pred == 0:
            n_fn += 1
        elif true == 0 and pred == 1:
            n_fp += 1
    misroute_rate = round((n_fn + n_fp) / n_tot, 3) if n_tot else None
    # 5-seed shuffle null
    null_vals = []
    for sd in SHUFFLE_SEEDS:
        srecs, _ = train_eval_ext(slides, labels, endpoint, device, shuffle=True, seed=sd)
        if srecs is None:
            null_vals.append(None); continue
        spa = patient_agg(srecs)
        sy = [v[1] for v in spa.values()]; sp = [v[0] for v in spa.values()]
        sa, _, _ = bootstrap_auc(sy, sp)
        null_vals.append(sa)
    valid = [v for v in null_vals if v is not None]
    null_mean = round(float(np.mean(valid)), 4) if valid else None
    null_sd = round(float(np.std(valid, ddof=1)), 4) if len(valid) > 1 else None
    thr = round(null_mean + 2 * null_sd, 4) if (null_mean is not None and null_sd is not None) else None
    passes = (real_auc is not None and thr is not None and real_auc > thr)
    return {
        "real_auroc_recomputed": real_auc, "ci95": [lo, hi], "dev_auc": dev_auc,
        "n_holdout": n_tot, "n_pos": int(sum(y)),
        "misroute_recompute": {"n_FN": n_fn, "n_FP": n_fp, "n_holdout": n_tot, "misroute_rate": misroute_rate},
        "shuffle_null": {
            "seeds": SHUFFLE_SEEDS, "null_seeds": null_vals,
            "null_mean": null_mean, "null_sd": null_sd,
            "null_min": (round(min(valid), 4) if valid else None),
            "null_max": (round(max(valid), 4) if valid else None),
            "null_mean_plus_2sd": thr,
        },
        "real_gt_null_mean_plus_2sd": passes,
        "proba_tilecount_spearman": {
            "slide_level": {"rho": rho_s, "pval": pval_s, "n_slides": len(s_n),
                            "n_slides_at_cap5000": n_at_cap},
            "patient_level": {"rho": rho_p, "pval": pval_p, "n_patients": len(pp_cases)},
        },
        "secs": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    split = build_split(); slides = build_slides(split)
    cms_lab = load_cms_labels(); tx_lab = load_treatment_labels()
    out = {}
    for ep in a.endpoints.split(","):
        labels = cms_lab if ep.startswith("cms") else tx_lab
        r = run_endpoint(slides, labels, ep, a.device)
        out[ep] = r
        sn = r.get("shuffle_null", {})
        print(f"[{a.device}] {ep}: real={r.get('real_auroc_recomputed')} "
              f"null_mean={sn.get('null_mean')}±{sn.get('null_sd')} seeds={sn.get('null_seeds')} "
              f"pass={r.get('real_gt_null_mean_plus_2sd')} misroute={r.get('misroute_recompute',{}).get('misroute_rate')} "
              f"spearman_slide={r.get('proba_tilecount_spearman',{}).get('slide_level',{}).get('rho')} "
              f"({r.get('secs')}s)", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[{a.device}] wrote {a.out}")


if __name__ == "__main__":
    main()
