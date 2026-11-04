"""
CLAM-SB training script for WSI phenotype prediction.

Smoke-test (dummy data):
    python pipeline/modeling/scripts/train_mil.py \
        --config pipeline/modeling/configs/baseline_er_status_clam.yaml \
        --smoke_test

Real mode (UNI embeddings):
    python pipeline/modeling/scripts/train_mil.py \
        --config pipeline/modeling/configs/baseline_er_status_clam.yaml \
        --tag uni_v1 \
        --commit_hash $(git rev-parse HEAD)
"""

import argparse
import csv
import datetime
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from modeling.baselines.attention_mil import CLAMSB, CLAMMB

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))


LABEL_MAP = {"positive": 1.0, "negative": 0.0}

PAM50_MAP_5 = {"luma": 0, "lumb": 1, "basal": 2, "her2": 3, "normal": 4, "normal-like": 4}
PAM50_MAP_4 = {"luma": 0, "lumb": 1, "basal": 2, "her2": 3}  # excludes Normal/Normal-like
PAM50_CLASSES_5 = ["LumA", "LumB", "Basal", "HER2", "Normal"]
PAM50_CLASSES_4 = ["LumA", "LumB", "Basal", "HER2"]


def pam50_map(num_classes):
    return PAM50_MAP_4 if num_classes == 4 else PAM50_MAP_5


def pam50_classes(num_classes):
    return PAM50_CLASSES_4 if num_classes == 4 else PAM50_CLASSES_5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_dummy_dataset(n_slides: int, dim: int, seed: int, num_classes: int = 1):
    rng = np.random.default_rng(seed)
    dataset = []
    for i in range(n_slides):
        n_tiles = rng.integers(100, 500)
        emb = torch.tensor(rng.standard_normal((n_tiles, dim)).astype(np.float32))
        if num_classes > 1:
            label = torch.tensor(i % num_classes, dtype=torch.long)
        else:
            label = torch.tensor(float(i % 2))
        dataset.append((emb, label))
    return dataset


def load_manifest_dataset(manifest_path: str, label_col: str, split: str = None, num_classes: int = 1):
    is_pam50 = label_col.lower() == "pam50"
    label_lookup = pam50_map(num_classes) if is_pam50 else LABEL_MAP
    dataset = []
    skipped = 0
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if split and row.get("split", "").lower() != split:
                continue
            emb_path = row.get("embedding_path", "")
            label_raw = row.get(label_col, "").strip().lower()
            if not emb_path or label_raw not in label_lookup:
                skipped += 1
                continue
            emb = torch.tensor(np.load(DATA_ROOT / emb_path))
            label_val = label_lookup[label_raw]
            label = torch.tensor(label_val, dtype=torch.long if is_pam50 else torch.float32)
            dataset.append((emb, label))
    if skipped:
        print(f"  [skip] {skipped} rows (unknown label/missing)")
    return dataset


def train(config: dict, smoke_test: bool):
    set_seed(config["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dim = config["embedding_dim"]
    manifest = config["data"]["embedding_manifest"]

    test_manifest = config.get("_test_manifest")
    num_classes = config.get("num_classes", 1)
    is_multiclass = num_classes > 1

    if smoke_test or not Path(manifest).exists():
        print("Smoke-test mode: using dummy embeddings")
        dataset = make_dummy_dataset(config["data"]["n_dummy_slides"], dim, config["train"]["seed"], num_classes)
        n = len(dataset)
        split_idx = int(n * 0.8)
        train_set, val_set = dataset[:split_idx], dataset[split_idx:]
        ext_test_set = None
    else:
        print(f"Loading manifest: {manifest}")
        label_col = config["data"]["label_col"]
        train_set = load_manifest_dataset(manifest, label_col, split="train", num_classes=num_classes)
        val_set   = load_manifest_dataset(manifest, label_col, split="val", num_classes=num_classes)
        if config.get("_shuffle_labels"):
            # Negative control: permute train labels only (val/external keep true labels).
            shuffle_seed = config.get("_shuffle_seed", config["train"]["seed"])
            shuffle_rng = np.random.default_rng(shuffle_seed)
            embs = [e for e, _ in train_set]
            labels = [l for _, l in train_set]
            perm = shuffle_rng.permutation(len(labels))
            shuffled_labels = [labels[i] for i in perm]
            train_set = list(zip(embs, shuffled_labels))
            print(f"[label-shuffle] permuted {len(train_set)} train labels (shuffle_seed={shuffle_seed}; val/ext keep true labels)")
        # External test (CPTAC cross-dataset)
        if test_manifest and Path(test_manifest).exists():
            print(f"Loading test manifest: {test_manifest}")
            ext_test_set = load_manifest_dataset(test_manifest, label_col, split="cptac_external", num_classes=num_classes)
            print(f"External test: {len(ext_test_set)} slides")
        else:
            ext_test_set = None

    print(f"Slides: train={len(train_set)} val={len(val_set)}")

    if is_multiclass:
        model = CLAMMB(
            feature_dim=dim,
            hidden_dim=config["model"]["hidden_dim"],
            att_dim=config["model"]["att_dim"],
            dropout=config["model"]["dropout"],
            num_classes=num_classes,
        ).to(device)
    else:
        model = CLAMSB(
            feature_dim=dim,
            hidden_dim=config["model"]["hidden_dim"],
            att_dim=config["model"]["att_dim"],
            dropout=config["model"]["dropout"],
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    criterion = nn.CrossEntropyLoss() if is_multiclass else nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = config["train"].get("patience", 5)
    no_improve = 0

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for emb, label in train_set:
            emb, label = emb.to(device), label.to(device)
            optimizer.zero_grad()
            logit, _ = model(emb)
            loss = criterion(logit.unsqueeze(0) if is_multiclass else logit, label.unsqueeze(0))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for emb, label in val_set:
                emb, label = emb.to(device), label.to(device)
                logit, _ = model(emb)
                val_loss += criterion(logit.unsqueeze(0) if is_multiclass else logit, label.unsqueeze(0)).item()
                pred = logit.argmax() if is_multiclass else (torch.sigmoid(logit) > 0.5).float()
                correct += (pred == label).sum().item()

        avg_val = val_loss / max(len(val_set), 1)
        acc = correct / max(len(val_set), 1)
        print(f"Epoch {epoch:02d} | train_loss={train_loss/len(train_set):.4f} | val_loss={avg_val:.4f} | val_acc={acc:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [early stop] patience={patience} reached, stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    # Final val metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
    from sklearn.preprocessing import label_binarize

    def _predict(dataset):
        proba_list, pred_list, label_list = [], [], []
        with torch.no_grad():
            for emb, label in dataset:
                logit, _ = model(emb.to(device))
                if is_multiclass:
                    proba = torch.softmax(logit, dim=-1).cpu().numpy().tolist()
                    pred = int(logit.argmax().item())
                else:
                    proba = torch.sigmoid(logit).item()
                    pred = int(proba > 0.5)
                proba_list.append(proba)
                pred_list.append(pred)
                label_list.append(int(label.item()))
        return proba_list, pred_list, label_list

    def _compute_metrics(proba_list, pred_list, label_list):
        if len(set(label_list)) <= 1:
            return None, None, None
        bal_acc = round(float(balanced_accuracy_score(label_list, pred_list)), 4)
        try:
            if is_multiclass:
                auc = round(float(roc_auc_score(label_list, np.array(proba_list), multi_class="ovr", average="macro")), 4)
                y_bin = label_binarize(label_list, classes=list(range(num_classes)))
                auprc = round(float(average_precision_score(y_bin, np.array(proba_list), average="macro")), 4)
            else:
                auc   = round(float(roc_auc_score(label_list, proba_list)), 4)
                auprc = round(float(average_precision_score(label_list, proba_list)), 4)
        except ValueError as e:
            print(f"  [warn] AUC/AUPRC not computable (missing classes): {e}")
            auc = auprc = None
        return auc, auprc, bal_acc

    model.eval()
    all_proba, all_pred, all_label = _predict(val_set)
    auc, auprc, bal_acc = _compute_metrics(all_proba, all_pred, all_label)
    if auc is not None:
        print(f"\nVal metrics (TCGA) — AUC={auc}  AUPRC={auprc}  BalAcc={bal_acc}")

    # External test (CPTAC cross-dataset)
    ext_auc = ext_auprc = ext_bal_acc = None
    ext_proba = ext_pred = ext_label = None
    if ext_test_set:
        ext_proba, ext_pred, ext_label = _predict(ext_test_set)
        ext_auc, ext_auprc, ext_bal_acc = _compute_metrics(ext_proba, ext_pred, ext_label)
        if ext_auc is not None:
            print(f"Test metrics (CPTAC) — AUC={ext_auc}  AUPRC={ext_auprc}  BalAcc={ext_bal_acc}")

    tag    = config.get("tag", datetime.datetime.now().strftime("%Y%m%d"))
    model_tag = "clam_mb" if is_multiclass else "clam"
    suffix = f"{config['task']}_{model_tag}_{tag}" + ("_smoke" if smoke_test else "")
    out_dir = Path(config.get("output_dir", "outputs")) / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")

    if is_multiclass:
        np.savez(
            out_dir / "predictions.npz",
            proba=np.array(all_proba, dtype=np.float32),
            pred=np.array(all_pred, dtype=np.int32),
            label=np.array(all_label, dtype=np.int32),
        )
        if ext_proba is not None:
            np.savez(
                out_dir / "predictions_ext.npz",
                proba=np.array(ext_proba, dtype=np.float32),
                pred=np.array(ext_pred, dtype=np.int32),
                label=np.array(ext_label, dtype=np.int32),
            )
    else:
        predictions = np.array(list(zip(all_proba, all_pred, all_label)), dtype=np.float32)
        np.save(out_dir / "predictions.npy", predictions)

    if config.get("_config_path"):
        shutil.copy(config["_config_path"], out_dir / "config.yaml")

    commit_hash = config.get("_commit_hash") or get_git_hash()
    metrics = {
        "schema_version": "0.1",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "task": config["task"],
        "model": "CLAM-MB" if is_multiclass else "CLAM-SB",
        "num_classes": num_classes,
        "class_names": pam50_classes(num_classes) if is_multiclass else None,
        "embedding_model": config["embedding_model"],
        "smoke_test": smoke_test,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "label_shuffle_sanity_check": bool(config.get("_shuffle_labels", False)),
        "best_val_loss": round(best_val_loss, 4),
        "auc": auc,
        "auprc": auprc,
        "balanced_accuracy": bal_acc,
        "ext_test_manifest": test_manifest,
        "ext_auc": ext_auc,
        "ext_auprc": ext_auprc,
        "ext_balanced_accuracy": ext_bal_acc,
        "commit_hash": commit_hash,
        "claim_level": "hypothesis_only",
        "critic_status": "pending",
        "wandb_run_id": None,
        "mlflow_run_id": None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {out_dir}/")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--commit_hash", default="")
    parser.add_argument("--test_manifest", default="", help="external test manifest (CPTAC cross-dataset)")
    parser.add_argument("--shuffle_labels", action="store_true", help="randomly permute train labels (leakage sanity check)")
    parser.add_argument("--shuffle_seed", type=int, default=None, help="seed for label permutation (multi-seed null); defaults to train.seed")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["_config_path"] = args.config
    if args.tag:
        config["tag"] = args.tag
    if args.commit_hash:
        config["_commit_hash"] = args.commit_hash
    if args.test_manifest:
        config["_test_manifest"] = args.test_manifest
    if args.shuffle_labels:
        config["_shuffle_labels"] = True
    if args.shuffle_seed is not None:
        config["_shuffle_seed"] = args.shuffle_seed

    t0 = time.time()
    train(config, args.smoke_test)
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
