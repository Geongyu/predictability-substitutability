"""
External evaluation of CLAM-MB (PAM50, multiclass); multiclass version of eval_external.py.

Run:
    python pipeline/modeling/scripts/eval_external_mb.py \
        --model outputs/pam50_clam_mb_uni_v1/model.pt \
        --config pipeline/modeling/configs/baseline_pam50_clam.yaml \
        --test_manifest $DATA_ROOT/embedding_manifest_cptac_uni.csv \
        --test_split cptac_external \
        --out_dir outputs/pam50_clam_mb_uni_v1/

The CPTAC manifest needs PAM50 labels (see pipeline/data/build_cptac_labels.py).
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
from sklearn.preprocessing import label_binarize

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from modeling.baselines.attention_mil import CLAMMB

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

PAM50_MAP = {
    "luma": 0, "lumb": 1, "basal": 2, "her2": 3,
    "normal": 4, "normal-like": 4,  # CPTAC (cBioPortal) uses "Normal-like"; TCGA uses "Normal"
}


def load_manifest(manifest_path, split, label_col="pam50"):
    rows = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split", "").lower() != split:
                continue
            label_raw = row.get(label_col, "").strip().lower()
            if label_raw not in PAM50_MAP:
                continue
            rows.append(row)
    return rows


def bootstrap_ci_multiclass(y_true, proba, num_classes, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, ys, multi_class="ovr", average="macro"))
        except ValueError:
            continue
    if not aucs:
        return None, None
    return round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--test_split", default="cptac_external")
    parser.add_argument("--label_col", default="pam50", help="PAM50 label column in the test manifest")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    num_classes = config["num_classes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLAMMB(
        feature_dim=config["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        att_dim=config["model"]["att_dim"],
        dropout=config["model"]["dropout"],
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    rows = load_manifest(args.test_manifest, args.test_split, args.label_col)
    print(f"External test slides: {len(rows)}")
    if len(rows) == 0:
        print("[warn] 0 slides; check label_col or split")
        return

    proba_list, pred_list, label_list = [], [], []
    with torch.no_grad():
        for row in rows:
            emb = np.load(DATA_ROOT / row["embedding_path"])
            x = torch.tensor(emb).to(device)
            logits, _ = model(x)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            proba_list.append(p)
            pred_list.append(int(np.argmax(p)))
            label_list.append(PAM50_MAP[row[args.label_col].strip().lower()])

    proba_arr = np.array(proba_list)
    label_arr = np.array(label_list)

    auc = auprc = bal_acc = None
    ci = (None, None)
    if len(np.unique(label_arr)) > 1:
        try:
            auc = round(float(roc_auc_score(label_arr, proba_arr, multi_class="ovr", average="macro")), 4)
            y_bin = label_binarize(label_arr, classes=list(range(num_classes)))
            auprc = round(float(average_precision_score(y_bin, proba_arr, average="macro")), 4)
            ci = bootstrap_ci_multiclass(label_arr, proba_arr, num_classes)
        except ValueError as e:
            print(f"[warn] AUC computation failed: {e}")
    bal_acc = round(float(balanced_accuracy_score(label_arr, pred_list)), 4)

    print(f"AUC(macro OvR)={auc}  CI95={list(ci) if ci[0] else None}  AUPRC={auprc}  BalAcc={bal_acc}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "predictions_ext.npz", proba=proba_arr, pred=np.array(pred_list), label=label_arr)

    result = {
        "test_manifest": args.test_manifest,
        "test_split": args.test_split,
        "n_test": len(rows),
        "ext_auc": auc,
        "ext_auc_ci_95": list(ci) if ci[0] is not None else None,
        "ext_auprc": auprc,
        "ext_balanced_accuracy": bal_acc,
    }
    (out_dir / "ext_eval_summary.json").write_text(json.dumps(result, indent=2))
    print(f"Saved: {out_dir}/predictions_ext.npz, ext_eval_summary.json")


if __name__ == "__main__":
    main()
