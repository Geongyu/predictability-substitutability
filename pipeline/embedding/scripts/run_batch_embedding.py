"""
Batch runner for TCGA-BRCA WSI tiling and tile embedding extraction.

Input manifest CSV must include slide_path. Optional columns: slide_id, case_id,
file_id. The runner writes one output manifest CSV row per slide so downstream
steps can consume coords/embedding paths without scanning directories.

Example:
    python pipeline/embedding/scripts/run_batch_embedding.py \
      --manifest data/manifests/tcga_brca_subset.csv \
      --config pipeline/embedding/configs/tile_config.yaml \
      --tile_dir $DATA_ROOT/tiles \
      --embedding_dir $DATA_ROOT/embeddings/uni_v1 \
      --output_manifest $DATA_ROOT/embeddings/embedding_manifest.csv \
      --embedding_model uni
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


REQUIRED_COLUMNS = {"slide_path"}
OPTIONAL_COLUMNS = {"slide_id", "case_id", "file_id"}


def _clean_id(value: str) -> str:
    keep = []
    for ch in value.strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "unknown_slide"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        rows = []
        for i, row in enumerate(reader, start=2):
            slide_path = (row.get("slide_path") or "").strip()
            if not slide_path:
                raise ValueError(f"Empty slide_path at CSV line {i}")
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def _slide_id(row: dict[str, str]) -> str:
    explicit = row.get("slide_id", "")
    if explicit:
        return _clean_id(explicit)
    return _clean_id(Path(row["slide_path"]).stem)


def _run(cmd: list[str], dry_run: bool) -> tuple[int, float]:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0, 0.0
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    return result.returncode, time.time() - t0


def _load_tile_meta(coords_path: Path) -> dict[str, Any]:
    meta_path = coords_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _shape(path: Path) -> str:
    if not path.exists():
        return ""
    arr = np.load(path, mmap_mode="r")
    return "x".join(str(x) for x in arr.shape)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "slide_id",
        "case_id",
        "file_id",
        "slide_path",
        "coords_path",
        "coords_shape",
        "coords_json_path",
        "embedding_model",
        "embedding_path",
        "embedding_shape",
        "n_tiles",
        "n_before_cap",
        "capped",
        "tiling_seconds",
        "embedding_seconds",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def process_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    slide_id = _slide_id(row)
    slide_path = Path(row["slide_path"])
    coords_path = Path(args.tile_dir) / f"{slide_id}_coords.npy"
    coords_json_path = coords_path.with_suffix(".json")
    embedding_dir = Path(args.embedding_dir)

    result: dict[str, Any] = {
        "slide_id": slide_id,
        "case_id": row.get("case_id", ""),
        "file_id": row.get("file_id", ""),
        "slide_path": str(slide_path),
        "coords_path": str(coords_path),
        "coords_json_path": str(coords_json_path),
        "embedding_model": args.embedding_model,
        "status": "started",
        "error": "",
    }

    try:
        need_tile = args.overwrite or not coords_path.exists()
        if need_tile:
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "tile_wsi.py"),
                "--slide",
                str(slide_path),
                "--config",
                args.config,
                "--out",
                str(coords_path),
            ]
            code, elapsed = _run(cmd, args.dry_run)
            result["tiling_seconds"] = f"{elapsed:.1f}"
            if code != 0:
                result["status"] = "tile_failed"
                result["error"] = f"tile_wsi.py exited {code}"
                return result
        else:
            result["tiling_seconds"] = "0.0"

        if not args.dry_run:
            meta = _load_tile_meta(coords_path)
            result["coords_shape"] = _shape(coords_path)
            result["n_tiles"] = meta.get("n_tiles", "")
            result["n_before_cap"] = meta.get("n_before_cap", "")
            result["capped"] = meta.get("capped", "")

        if args.embedding_model == "none":
            result["status"] = "tiled"
            return result

        extractor = {
            "uni":    str(SCRIPT_DIR / "extract_uni.py"),
            "conch":  str(SCRIPT_DIR / "extract_conch.py"),
            "exaone": str(SCRIPT_DIR / "extract_exaone.py"),
            "dummy":  str(SCRIPT_DIR / "extract_dummy.py"),
        }[args.embedding_model]

        expected_suffix = {
            "uni":    "_uni_embeddings.npy",
            "conch":  "_conch_embeddings.npy",
            "exaone": "_exaone_embeddings.npy",
            "dummy":  "_dummy_embeddings.npy",
        }[args.embedding_model]
        gpu_models = {"uni", "conch", "exaone"}
        expected_stem = slide_path.stem if args.embedding_model in gpu_models else slide_id
        expected_embedding = embedding_dir / f"{expected_stem}{expected_suffix}"

        need_embed = args.overwrite or not expected_embedding.exists()
        if need_embed:
            cmd = [
                sys.executable,
                extractor,
                "--coords",
                str(coords_path),
                "--out_dir",
                str(embedding_dir),
            ]
            if args.embedding_model in gpu_models:
                cmd.extend(["--batch_size", str(args.batch_size), "--device", args.device])
            code, elapsed = _run(cmd, args.dry_run)
            result["embedding_seconds"] = f"{elapsed:.1f}"
            if code != 0:
                result["status"] = "embedding_failed"
                result["error"] = f"{Path(extractor).name} exited {code}"
                return result
        else:
            result["embedding_seconds"] = "0.0"

        result["embedding_path"] = str(expected_embedding)
        if not args.dry_run:
            result["embedding_shape"] = _shape(expected_embedding)
        result["status"] = "done"
        return result
    except Exception as exc:  # keep batch moving and record the slide-level error
        result["status"] = "error"
        result["error"] = repr(exc)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch tile WSIs and extract embeddings")
    parser.add_argument("--manifest", required=True, help="CSV with slide_path and optional slide_id/case_id/file_id")
    parser.add_argument("--config", required=True, help="Path to tile_config.yaml")
    parser.add_argument("--tile_dir", required=True, help="Directory for coords.npy/json outputs")
    parser.add_argument("--embedding_dir", required=True, help="Directory for embedding .npy outputs")
    parser.add_argument("--output_manifest", required=True, help="CSV manifest to write for downstream steps")
    parser.add_argument("--embedding_model", choices=["uni", "conch", "exaone", "dummy", "none"], default="uni")
    parser.add_argument("--batch_size", type=int, default=64, help="UNI batch size")
    parser.add_argument("--device", default="cuda", help="UNI device: cuda or cpu")
    parser.add_argument("--overwrite", action="store_true", help="Recompute outputs even if files exist")
    parser.add_argument("--dry_run", action="store_true", help="Print commands and write a plan manifest without running")
    args = parser.parse_args()

    rows = _read_manifest(Path(args.manifest))
    print(f"Loaded {len(rows)} slides from {args.manifest}")

    output_rows = []
    for idx, row in enumerate(rows, start=1):
        sid = _slide_id(row)
        print(f"\n[{idx}/{len(rows)}] {sid}", flush=True)
        result = process_row(row, args)
        output_rows.append(result)
        _write_rows(Path(args.output_manifest), output_rows)
        print(f"status={result['status']} error={result.get('error', '')}", flush=True)

    done = sum(1 for row in output_rows if row.get("status") == "done")
    failed = sum(1 for row in output_rows if row.get("status") not in {"done", "tiled"})
    print(f"\nWrote {args.output_manifest}")
    print(f"done={done} failed={failed} total={len(output_rows)}")


if __name__ == "__main__":
    main()
