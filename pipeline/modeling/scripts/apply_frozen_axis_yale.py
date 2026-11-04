"""
Apply the frozen TCGA HER2 CLAM-SB model to Yale embeddings to get axis scores.

Frozen transfer (no fine-tuning) to the Yale cohort; outputs a per-slide anti-HER2
axis score (= predicted HER2-positive probability). Input to the pCR stratification.

Run:
    python pipeline/modeling/scripts/apply_frozen_axis_yale.py \
        --model outputs/her2_status_clam_uni_v2/model.pt \
        --config pipeline/modeling/configs/baseline_her2_status_clam.yaml \
        --manifest $DATA_ROOT/yale/embedding_manifest_yale_uni.csv \
        --out breast/models/yale_antiher2_axis_scores.csv

embedding_path values in the manifest are relative to DATA_ROOT.
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.baselines.attention_mil import CLAMSB

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLAMSB(
        feature_dim=cfg["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        att_dim=cfg["model"]["att_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()  # frozen, no fine-tuning on Yale

    rows = [r for r in csv.DictReader(open(args.manifest)) if r.get("status", "done") == "done"]
    print(f"Applying frozen HER2 axis to {len(rows)} Yale slides")

    out_rows = []
    with torch.no_grad():
        for r in rows:
            emb = np.load(DATA_ROOT / r["embedding_path"])
            x = torch.tensor(emb).to(device)
            logit, _ = model(x)
            score = float(torch.sigmoid(logit).item())
            out_rows.append({
                "slide_id": r["slide_id"],
                "case_id": r["case_id"],
                "n_tiles": r.get("n_tiles", ""),
                "antiher2_axis_score": round(score, 6),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slide_id", "case_id", "n_tiles", "antiher2_axis_score"])
        w.writeheader()
        w.writerows(out_rows)

    s = np.array([o["antiher2_axis_score"] for o in out_rows])
    print(f"Saved: {out} ({len(out_rows)} rows)")
    print(f"Axis score distribution: min={s.min():.3f} median={np.median(s):.3f} max={s.max():.3f} mean={s.mean():.3f}")
    print("Frozen transfer (no Yale fine-tuning). Source model: TCGA her2_status_clam_uni_v2.")


if __name__ == "__main__":
    main()
