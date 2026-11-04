#!/usr/bin/env python3
"""
Validate per-slide tile embeddings against a manifest: coverage + integrity.

For every slide in the manifest, check that an embedding .npy exists and is a
2-D float array with the expected feature dim, no NaN/Inf, and >0 tiles. Prints a
summary and (optionally) writes a JSON report. Reusable across UNI/CONCH/EXAONE.

Slide stem matches run_batch_embedding.py output: Path(file_name).stem.

Example:
  python pipeline/embedding/scripts/validate_embeddings.py \
    --manifest $DATA_ROOT/tcga_brca_final_manifest.csv \
    --emb-dir  $DATA_ROOT/embeddings/tcga/uni_v1 \
    --suffix   _uni_embeddings.npy --dim 1024 \
    --report   outputs/uni_embed_validation.json
"""
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--suffix", default="_uni_embeddings.npy")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--report", default="")
    ap.add_argument("--full-nan", action="store_true", help="NaN/Inf check on full array (slower)")
    args = ap.parse_args()

    emb_dir = Path(os.path.expanduser(args.emb_dir))
    with open(os.path.expanduser(args.manifest), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    present, missing, bad = [], [], []
    for r in rows:
        fname = r.get("file_name") or r.get("slide_id", "")
        stem = Path(fname).stem
        p = emb_dir / f"{stem}{args.suffix}"
        if not p.exists():
            missing.append(r.get("slide_id", stem)); continue
        try:
            a = np.load(p, mmap_mode="r")
            arr = np.asarray(a[:64]) if not args.full_nan else np.asarray(a)
            ok = (a.ndim == 2 and a.shape[1] == args.dim and a.shape[0] > 0
                  and not np.isnan(arr).any() and not np.isinf(arr).any())
            (present if ok else bad).append(
                {"slide_id": r.get("slide_id", stem), "shape": list(a.shape)})
        except Exception as e:
            bad.append({"slide_id": r.get("slide_id", stem), "error": repr(e)})

    n = len(rows)
    summary = {
        "manifest_slides": n,
        "present_ok": len(present),
        "missing": len(missing),
        "bad": len(bad),
        "coverage_pct": round(100 * len(present) / n, 2) if n else 0.0,
        "missing_examples": missing[:10],
        "bad_examples": bad[:10],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.report:
        rp = Path(os.path.expanduser(args.report))
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({**summary, "missing_all": missing,
                                  "bad_all": bad}, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        print(f"Wrote {rp}")
    raise SystemExit(0 if (len(missing) == 0 and len(bad) == 0) else 1)


if __name__ == "__main__":
    main()
