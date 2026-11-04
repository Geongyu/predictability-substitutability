"""
Save CPTAC external predictions indexed by slide_id (input to cost-of-substitution routing).

Runs the ER/PR/HER2 CLAM-SB models on all CPTAC slides. Predictions do not depend on
labels, so all slides (653) are kept regardless of has_* flags. Label join is done in
breast/cost/compute_cost.py.

Output: breast/models/cptac_ext_predictions_indexed.csv
Columns: slide_id, case_id, er_pred_prob, pr_pred_prob, her2_pred_prob
embedding_path values in the manifest are relative to DATA_ROOT.
"""
import csv
import os
from pathlib import Path
import numpy as np
import torch
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.baselines.attention_mil import CLAMSB

REPO = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
BASE = Path("outputs")  # train_mil.py output dir
MANI = DATA_ROOT / "embedding_manifest_cptac_uni.csv"
SPLIT = "cptac_external"
ENDPOINTS = {
    "er":   ("er_status_clam_uni_v2",   "baseline_er_status_clam.yaml"),
    "pr":   ("pr_status_clam_uni_v2",   "baseline_pr_status_clam.yaml"),
    "her2": ("her2_status_clam_uni_v2", "baseline_her2_status_clam.yaml"),
}
CFG_DIR = REPO / "pipeline" / "modeling" / "configs"

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(exp_dir, cfg_name):
    cfg = yaml.safe_load(open(CFG_DIR / cfg_name))
    m = CLAMSB(feature_dim=cfg["embedding_dim"], hidden_dim=cfg["model"]["hidden_dim"],
               att_dim=cfg["model"]["att_dim"], dropout=cfg["model"]["dropout"]).to(device)
    m.load_state_dict(torch.load(BASE / exp_dir / "model.pt", map_location=device))
    m.eval()
    return m


rows = [r for r in csv.DictReader(open(MANI)) if r.get("split", "").strip() == SPLIT]
print(f"CPTAC external slides: {len(rows)}")

models = {ep: load_model(*v) for ep, v in ENDPOINTS.items()}

out = []
with torch.no_grad():
    for r in rows:
        emb = torch.tensor(np.load(DATA_ROOT / r["embedding_path"])).to(device)
        probs = {}
        for ep, m in models.items():
            logit, _ = m(emb)
            probs[ep] = round(float(torch.sigmoid(logit).item()), 6)
        out.append({
            "slide_id": r["slide_id"],
            "case_id": r["case_id"],
            "er_pred_prob": probs["er"],
            "pr_pred_prob": probs["pr"],
            "her2_pred_prob": probs["her2"],
        })

out_path = REPO / "breast" / "models" / "cptac_ext_predictions_indexed.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["slide_id", "case_id", "er_pred_prob", "pr_pred_prob", "her2_pred_prob"])
    w.writeheader()
    w.writerows(out)
print(f"Saved: {out_path}  ({len(out)} rows)")
print("head:")
for r in out[:3]:
    print(" ", r)
