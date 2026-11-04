"""
Run a trained CLAM-SB model on an external dataset (CPTAC), save raw predictions
(proba/pred/label) and compute a bootstrap CI. Uses the existing model.pt without retraining.

Run:
    python pipeline/modeling/scripts/eval_external.py \
        --model outputs/er_status_clam_uni_v2/model.pt \
        --config pipeline/modeling/configs/baseline_er_status_clam.yaml \
        --test_manifest $DATA_ROOT/embedding_manifest_cptac_uni.csv \
        --test_split cptac_external \
        --out_dir outputs/er_status_clam_uni_v2/

embedding_path values in the manifest are relative to DATA_ROOT.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from modeling.baselines.attention_mil import CLAMSB

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

LABEL_MAP = {"positive": 1.0, "negative": 0.0}


def load_manifest(manifest_path, label_col, split):
    rows = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split", "").lower() != split:
                continue
            label_raw = row.get(label_col, "").strip().lower()
            if label_raw not in LABEL_MAP:
                continue
            rows.append(row)
    return rows


def bootstrap_ci(y_true, y_score, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if not aucs:
        return None, None
    return round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--test_split", default="cptac_external")
    parser.add_argument("--label_col", default="", help="override label column if it differs from the training config (e.g. er_status)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    label_col = args.label_col or config["data"]["label_col"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLAMSB(
        feature_dim=config["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        att_dim=config["model"]["att_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    rows = load_manifest(args.test_manifest, label_col, args.test_split)
    print(f"External test slides: {len(rows)} (label_col={label_col})")
    if len(rows) == 0:
        print(f"[warn] 0 slides; check label_col ({label_col}) or split ({args.test_split})")
        return

    proba, pred, label = [], [], []
    with torch.no_grad():
        for row in rows:
            emb = np.load(DATA_ROOT / row["embedding_path"])
            x = torch.tensor(emb).to(device)
            logit, _ = model(x)
            p = torch.sigmoid(logit).item()
            proba.append(p)
            pred.append(int(p > 0.5))
            label.append(LABEL_MAP[row[label_col].strip().lower()])

    proba, pred, label = np.array(proba), np.array(pred), np.array(label)

    auc = round(float(roc_auc_score(label, proba)), 4)
    auprc = round(float(average_precision_score(label, proba)), 4)
    bal_acc = round(float(balanced_accuracy_score(label, pred)), 4)
    ci_lo, ci_hi = bootstrap_ci(label, proba)

    print(f"AUC={auc}  AUPRC={auprc}  BalAcc={bal_acc}  CI95=[{ci_lo}, {ci_hi}]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = np.array(list(zip(proba, pred, label)), dtype=np.float32)
    np.save(out_dir / "predictions_ext.npy", predictions)

    ext_result = {
        "test_manifest": args.test_manifest,
        "test_split": args.test_split,
        "n_test": len(rows),
        "ext_auc": auc,
        "ext_auprc": auprc,
        "ext_balanced_accuracy": bal_acc,
        "ext_auc_ci_95": [ci_lo, ci_hi] if ci_lo is not None else None,
    }
    (out_dir / "ext_eval_summary.json").write_text(json.dumps(ext_result, indent=2))
    print(f"Saved: {out_dir}/predictions_ext.npy, ext_eval_summary.json")


if __name__ == "__main__":
    main()
