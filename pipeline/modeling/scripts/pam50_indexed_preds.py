"""Re-run PAM50 4-class predictions with slide_id/case_id attached.
Uses the same filter and order as eval_external_mb.py and checks consistency with the saved predictions.
embedding_path values in the manifests are relative to DATA_ROOT."""
import argparse, csv, os
from pathlib import Path
import numpy as np, torch, yaml, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.baselines.attention_mil import CLAMMB

REPO = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
CFG = REPO / "pipeline" / "modeling" / "configs" / "baseline_pam50_clam_4class.yaml"
MODEL = "outputs/pam50_clam_mb_uni_v1_4class/model.pt"
EXPDIR = "outputs/pam50_clam_mb_uni_v1_4class"
PAM50_MAP4 = {"luma": 0, "lumb": 1, "basal": 2, "her2": 3}   # excludes Normal
CLASSES = ["LumA", "LumB", "Basal", "HER2"]

cfg = yaml.safe_load(open(CFG))
device = "cuda" if torch.cuda.is_available() else "cpu"
m = CLAMMB(feature_dim=cfg["embedding_dim"], hidden_dim=cfg["model"]["hidden_dim"],
           att_dim=cfg["model"]["att_dim"], dropout=cfg["model"]["dropout"], num_classes=4).to(device)
m.load_state_dict(torch.load(MODEL, map_location=device)); m.eval()


def run(manifest, split, out_csv, ref_npz):
    rows = [r for r in csv.DictReader(open(manifest))
            if r.get("split", "").strip() == split and r.get("pam50", "").strip().lower() in PAM50_MAP4]
    out = []
    with torch.no_grad():
        for r in rows:
            x = torch.tensor(np.load(DATA_ROOT / r["embedding_path"])).to(device)
            logits, _ = m(x)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append((r["slide_id"], r["case_id"], p, int(np.argmax(p)), PAM50_MAP4[r["pam50"].strip().lower()]))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide_id", "case_id"] + [f"proba_{c}" for c in CLASSES] + ["pred_class", "true_label"])
        for sid, cid, p, pr, lb in out:
            w.writerow([sid, cid] + [round(float(x), 6) for x in p] + [CLASSES[pr], CLASSES[lb]])
    # Consistency check vs saved npz: proba distribution (order-free) and label counts
    ref = np.load(ref_npz)
    new_proba = np.array([o[2] for o in out]); new_lab = np.array([o[4] for o in out])
    ok_n = len(out) == ref["label"].shape[0]
    ok_lab = np.bincount(new_lab, minlength=4).tolist() == np.bincount(ref["label"], minlength=4).tolist()
    ok_proba = np.allclose(np.sort(new_proba.ravel()), np.sort(ref["proba"].ravel()), atol=1e-4)
    ncase = len(set(o[1] for o in out))
    print(f"  {Path(out_csv).name}: {len(out)} rows = {ncase} case_id | n_match={ok_n} label_count_match={ok_lab} proba_dist_match={ok_proba}")


print("PAM50 4-class indexed predictions (Normal excluded):")
run(DATA_ROOT / "embedding_manifest_cptac_uni.csv", "cptac_external",
    f"{EXPDIR}/predictions_ext_indexed.csv", f"{EXPDIR}/predictions_ext.npz")
run(DATA_ROOT / "embedding_manifest_uni.csv", "val",
    f"{EXPDIR}/predictions_indexed.csv", f"{EXPDIR}/predictions.npz")
print("done")
