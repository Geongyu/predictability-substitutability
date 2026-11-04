#!/usr/bin/env python3
"""
run_cohort_pipeline.py: one-command driver for adding a new cancer cohort to the cross-cancer analysis.

Chains the existing scripts below as subprocesses (or in-process calls); it does not reimplement any analysis.

Stages:
  0. (input) <COHORT>/full/patient_labels.csv   molecular labels (cBioPortal/GDC). Checked, not created.
  1. site-disjoint split            [CPU]  make_split.py :: run(cancer)
                                            -> <COHORT>/full/split.csv + split_meta.json (split_policy_v0 logic, seed 42)
  2. tiling -> UNI embedding        [GPU]  pipeline/embedding/scripts/run_batch_embedding.py
                                            (tile_wsi.py + extract_uni.py with tile_config.yaml)
                                            -> $DATA_ROOT/crosscancer/<COHORT>/uni_v1/<slide>_uni_embeddings.npy
  3. per-endpoint MIL + cost        [GPU]  run_mil_cost.py (LUNG_NSCLC, COLORECTAL) or
                                            sh_mil_cost.py (GASTRIC_STAD, HEADNECK_HNSC; no frozen_map)
                                            --cancer <C> --fm uni --device ...  -> <COHORT>/full/mil_cost_results.json
  4. 5-seed shuffle-null            [GPU]  sh_robustness_5seed.py (calls run_mil_cost.train_eval)
                                            --cancer <C> --fm uni  -> <COHORT>/full/shuffle_null_robustness.json

Idempotent: a stage is skipped if its outputs exist, so a rerun resumes.
--dry-run prints the plan (skip / commands) without running anything.

Conventions:
  * The driver does not compute metrics; results are read from the runners' output files.
  * The split reuses make_split.py (site-disjoint).
  * Default embedding = UNI. virchow2/uni2h are selected with embedding_model in the config.
  * Molecular labels (patient_labels.csv) and therapeutic distances (frozen_map.json) are inputs.

Inputs required for a new cohort:
  * <COHORT>/full/patient_labels.csv (label columns = endpoint names).
  * A slide manifest CSV (slide_path column) for embedding. run_embed_crosscancer.py downloads from GDC
    but has its cohorts hard-coded.
  * An entry for the cohort in the MIL runner's CANCER_CFG (endpoints, route_axis, positive_control) and in
    sh_robustness_5seed.ENDPOINTS. The driver fails if the cohort is not registered.
  * Stages 2, 3, 4 need a GPU. Stage 1 runs on CPU.

Usage:
  # read cohort settings from cohorts/<name>.yaml and run all stages
  python run_cohort_pipeline.py --cohort GASTRIC_STAD
  python run_cohort_pipeline.py --cohort NEWCANCER --config crosscancer/cohorts/newcancer.yaml
  python run_cohort_pipeline.py --cohort NEWCANCER --dry-run        # print plan only
  python run_cohort_pipeline.py --cohort NEWCANCER --only split,mil # selected stages only
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # crosscancer/
ROOT = HERE.parent                                      # repo root
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

# scripts chained by the driver
MAKE_SPLIT = HERE / "make_split.py"
BATCH_EMBED = ROOT / "pipeline/embedding/scripts/run_batch_embedding.py"
TILE_CFG = ROOT / "pipeline/embedding/configs/tile_config.yaml"
MIL_RUNNERS = {                                         # runners that hold a CANCER_CFG
    "run_mil_cost.py": HERE / "run_mil_cost.py",        # with frozen_map (cost routing): LUNG_NSCLC, COLORECTAL
    "sh_mil_cost.py":  HERE / "sh_mil_cost.py",         # no frozen_map (misrouting only): GASTRIC_STAD, HEADNECK_HNSC
}
FIVESEED = HERE / "sh_robustness_5seed.py"

# interpreter for subprocess stages
DEFAULT_PYTHON = sys.executable

# embedding file suffix per FM (matches run_mil_cost.FM_SPEC). Default UNI.
FM_SUFFIX = {"uni": "uni", "virchow2": "virchow2", "uni2h": "uni2h"}


def default_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def embedding_dir(cohort, fm):
    """Embedding directory under DATA_ROOT (same layout as run_mil_cost.FM_SPEC)."""
    if fm == "uni":
        return DATA_ROOT / "crosscancer" / cohort / "uni_v1"
    return DATA_ROOT / cohort.lower() / fm


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(cohort, config_path):
    """Load cohort settings from cohorts/<name>.yaml or --config (YAML or JSON)."""
    if config_path:
        cfg_file = Path(config_path)
        if not cfg_file.is_absolute():
            cfg_file = (ROOT / cfg_file) if (ROOT / cfg_file).exists() else (Path.cwd() / cfg_file)
    else:
        # look up cohorts/<name>.yaml (case-insensitive)
        cdir = HERE / "cohorts"
        cand = [cdir / f"{cohort}.yaml", cdir / f"{cohort}.yml",
                cdir / f"{cohort.lower()}.yaml", cdir / f"{cohort.lower()}.yml",
                cdir / f"{cohort}.json"]
        cfg_file = next((c for c in cand if c.exists()), None)
        if cfg_file is None:
            raise SystemExit(f"[ERR] no cohort config for {cohort}: create {cdir}/<name>.yaml or pass --config "
                             f"(example: {cdir}/EXAMPLE.yaml)")
    text = cfg_file.read_text(encoding="utf-8")
    if cfg_file.suffix == ".json":
        cfg = json.loads(text)
    else:
        import yaml
        cfg = yaml.safe_load(text)
    log(f"config: {cfg_file}")
    return cfg


def cohort_dir(cohort):
    return HERE / cohort / "full"


# --------------------------------------------------------------------------- #
# stage runner (idempotent + dry-run)
# --------------------------------------------------------------------------- #
def stage(name, gpu, output, cmd, dry_run, python=None, cwd=None, func=None):
    """Run or skip one stage. Skipped if output (Path or list of Paths) already exists.
    cmd: subprocess argv. func: (callable, label) to call in-process instead.
    """
    tag = "[GPU]" if gpu else "[CPU]"
    outs = output if isinstance(output, (list, tuple)) else [output]
    if outs and all(o is not None and Path(o).exists() for o in outs):
        log(f"SKIP {tag} {name}: output exists ({', '.join(str(o) for o in outs)})")
        return "skip"
    if func is not None:
        callable_, label = func
        log(f"{'PLAN' if dry_run else 'RUN '} {tag} {name}: {label}")
        if dry_run:
            return "plan"
        t0 = time.time()
        callable_()
        log(f"DONE {tag} {name} ({time.time()-t0:.0f}s)")
        return "done"
    argv = ([python] if python else []) + [str(x) for x in cmd]
    log(f"{'PLAN' if dry_run else 'RUN '} {tag} {name}: {' '.join(argv)}")
    if dry_run:
        return "plan"
    t0 = time.time()
    r = subprocess.run(argv, cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        raise SystemExit(f"[ERR] {name} failed rc={r.returncode}; stopping (rerun to resume).")
    if outs and not all(o is None or Path(o).exists() for o in outs):
        raise SystemExit(f"[ERR] {name} finished but expected output is missing: {[str(o) for o in outs]}")
    log(f"DONE {tag} {name} ({time.time()-t0:.0f}s)")
    return "done"


# --------------------------------------------------------------------------- #
# stage 1: split (make_split.run, in-process)
# --------------------------------------------------------------------------- #
def stage_split(cohort, dry_run):
    d = cohort_dir(cohort)
    labels = d / "patient_labels.csv"
    out = [d / "split.csv", d / "split_meta.json"]
    if not (out[0].exists() and out[1].exists()) and not labels.exists():
        raise SystemExit(f"[ERR] {labels} not found; molecular labels (patient_labels.csv) are required.")

    def _run():
        sys.path.insert(0, str(HERE))
        import make_split
        make_split.run(cohort)

    return stage("split(site-disjoint)", gpu=False, output=out, cmd=None,
                 dry_run=dry_run, func=(_run, f"make_split.run('{cohort}') -> split.csv/split_meta.json"))


# --------------------------------------------------------------------------- #
# stage 2: tiling + embedding (run_batch_embedding, per-slide idempotent)
# --------------------------------------------------------------------------- #
def stage_embed(cohort, cfg, python, dry_run):
    d = cohort_dir(cohort)
    fm = cfg.get("embedding_model", "uni")
    emb_dir = embedding_dir(cohort, fm)
    suffix = FM_SUFFIX.get(fm, "uni")
    manifest = cfg.get("slide_manifest")
    if not manifest:
        log(f"NOTE {cohort}: no slide_manifest; skipping embedding stage. "
            f"See run_embed_crosscancer.py for GDC download (cohorts hard-coded).")
        # existing embeddings are used as-is; otherwise the MIL stage fails
        n = len(list(emb_dir.glob(f"*_{suffix}_embeddings.npy"))) if emb_dir.exists() else 0
        log(f"  existing embeddings: {n} ({emb_dir})")
        return "skip"
    man_path = Path(manifest)
    if not man_path.is_absolute():
        man_path = ROOT / manifest
    # skip if there are already as many embeddings as manifest rows
    try:
        n_slides = sum(1 for _ in csv.DictReader(man_path.open(encoding="utf-8-sig")))
    except FileNotFoundError:
        raise SystemExit(f"[ERR] slide_manifest not found: {man_path}")
    n_have = len(list(emb_dir.glob(f"*_{suffix}_embeddings.npy"))) if emb_dir.exists() else 0
    run_manifest = d / "_embedding_run_manifest.csv"
    if n_have >= n_slides and n_slides > 0:
        log(f"SKIP [GPU] embed: {n_have}/{n_slides} embeddings already exist ({emb_dir})")
        _write_cc_manifest(cohort, emb_dir, suffix, fm, dry_run=False)
        return "skip"
    tile_cfg = cfg.get("tile_config", str(TILE_CFG))
    tile_dir = cfg.get("tile_dir", str(d / "coords"))
    device = cfg.get("device", default_device())
    dev = device.split(":")[0] if fm == "uni" else device   # extract_uni takes cuda/cpu
    cmd = [BATCH_EMBED,
           "--manifest", man_path,
           "--config", tile_cfg,
           "--tile_dir", tile_dir,
           "--embedding_dir", emb_dir,
           "--output_manifest", run_manifest,
           "--embedding_model", fm if fm in ("uni", "conch", "exaone", "dummy") else "uni",
           "--device", dev]
    st = stage("embed(tile->UNI, run_batch_embedding)", gpu=True, output=None, cmd=cmd,
               dry_run=dry_run, python=python)
    if not dry_run:
        _write_cc_manifest(cohort, emb_dir, suffix, fm, dry_run=False)
    return st


def _write_cc_manifest(cohort, emb_dir, suffix, fm, dry_run):
    """Write embedding_manifest_<cohort>_<fm>.csv (same columns as run_embed_crosscancer.build_manifest).
    Paths are relative to DATA_ROOT. load_meta globs the embedding directory and does not read this file."""
    if not emb_dir.exists():
        return
    rows = []
    for p in sorted(emb_dir.glob(f"*_{suffix}_embeddings.npy")):
        sid = p.name.replace(f"_{suffix}_embeddings.npy", "")
        rows.append({"case_id": sid[:12], "slide_id": sid, "embedding_path": os.path.relpath(p, DATA_ROOT),
                     "embedding_model": fm, "file_id": ""})
    mp = cohort_dir(cohort) / f"embedding_manifest_{cohort.lower()}_{fm}.csv"
    with mp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "slide_id", "embedding_path", "embedding_model", "file_id"])
        w.writeheader()
        w.writerows(rows)
    log(f"  manifest: {len(rows)} rows -> {mp}")


# --------------------------------------------------------------------------- #
# stage 3: MIL + cost
# --------------------------------------------------------------------------- #
def _resolve_mil_runner(cohort, cfg):
    """Pick the runner whose CANCER_CFG has the cohort. cfg.mil_runner takes priority."""
    want = cfg.get("mil_runner")
    if want:
        if want not in MIL_RUNNERS:
            raise SystemExit(f"[ERR] unknown mil_runner: {want} (choose from {list(MIL_RUNNERS)})")
        return want, MIL_RUNNERS[want]
    sys.path.insert(0, str(HERE))
    for name, path in MIL_RUNNERS.items():
        mod = __import__(name[:-3])
        if cohort in getattr(mod, "CANCER_CFG", {}):
            return name, path
    raise SystemExit(
        f"[ERR] {cohort} is not in any MIL runner's CANCER_CFG. "
        f"Add endpoints, route_axis and positive_control to CANCER_CFG in run_mil_cost.py (needs frozen_map) "
        f"or sh_mil_cost.py, or set mil_runner in cohorts/<name>.yaml.")


def stage_mil(cohort, cfg, python, dry_run):
    d = cohort_dir(cohort)
    fm = cfg.get("embedding_model", "uni")
    fm_tag = "" if fm == "uni" else f"_{fm}"
    out = d / f"mil_cost_results{fm_tag}.json"
    runner_name, runner_path = _resolve_mil_runner(cohort, cfg)
    device = cfg.get("device", default_device())
    cmd = [runner_path, "--cancer", cohort, "--fm", fm, "--device", device]
    # cwd=HERE so that the runner's `import run_mil_cost` works
    return stage(f"mil+cost ({runner_name})", gpu=True, output=out, cmd=cmd,
                 dry_run=dry_run, python=python, cwd=HERE)


# --------------------------------------------------------------------------- #
# stage 4: 5-seed shuffle-null
# --------------------------------------------------------------------------- #
def stage_fiveseed(cohort, cfg, python, dry_run):
    d = cohort_dir(cohort)
    fm = cfg.get("embedding_model", "uni")
    out = d / ("shuffle_null_robustness.json" if fm == "uni" else f"shuffle_null_robustness_{fm}.json")
    # sh_robustness_5seed accepts only cohorts listed in ENDPOINTS
    sys.path.insert(0, str(HERE))
    mod = __import__("sh_robustness_5seed")
    if cohort not in getattr(mod, "ENDPOINTS", {}):
        raise SystemExit(
            f"[ERR] {cohort} is not in sh_robustness_5seed.ENDPOINTS; add its endpoint list there.")
    device = cfg.get("device", default_device())
    cmd = [FIVESEED, "--cancer", cohort, "--fm", fm, "--device", device]
    return stage("5-seed shuffle-null", gpu=True, output=out, cmd=cmd,
                 dry_run=dry_run, python=python, cwd=HERE)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
STAGES = ["split", "embed", "mil", "fiveseed"]


def main():
    ap = argparse.ArgumentParser(description="Cross-cancer pipeline driver for a new cancer cohort")
    ap.add_argument("--cohort", required=True, help="cohort name (= <COHORT>/full directory name, e.g. GASTRIC_STAD)")
    ap.add_argument("--config", help="cohort config (YAML/JSON). Default: cohorts/<cohort>.yaml")
    ap.add_argument("--python", default=DEFAULT_PYTHON, help="Python interpreter for subprocess stages (default: current interpreter)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without running")
    ap.add_argument("--only", help="comma-separated stages to run (choices: %s)" % ",".join(STAGES))
    a = ap.parse_args()

    cfg = load_config(a.cohort, a.config)
    # warn if the config's cohort/cancer field differs from --cohort (directory name wins)
    cfg_name = cfg.get("cohort") or cfg.get("cancer")
    if cfg_name and cfg_name != a.cohort:
        log(f"NOTE config cohort='{cfg_name}' differs from --cohort '{a.cohort}'; using '{a.cohort}'")

    d = cohort_dir(a.cohort)
    d.mkdir(parents=True, exist_ok=True)
    only = set(s.strip() for s in a.only.split(",")) if a.only else set(STAGES)

    log(f"=== cohort pipeline: {a.cohort} | fm={cfg.get('embedding_model','uni')} | "
        f"dry_run={a.dry_run} | stages={sorted(only)} ===")

    results = {}
    if "split" in only:
        results["split"] = stage_split(a.cohort, a.dry_run)
    if "embed" in only:
        results["embed"] = stage_embed(a.cohort, cfg, a.python, a.dry_run)
    if "mil" in only:
        results["mil"] = stage_mil(a.cohort, cfg, a.python, a.dry_run)
    if "fiveseed" in only:
        results["fiveseed"] = stage_fiveseed(a.cohort, cfg, a.python, a.dry_run)

    log(f"=== done: {a.cohort} | {json.dumps(results, ensure_ascii=False)} ===")


if __name__ == "__main__":
    main()
