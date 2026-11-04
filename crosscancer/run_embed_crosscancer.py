#!/usr/bin/env python3
"""
Cross-cancer full-cohort embedding driver.

Per-slide streaming: GDC REST query -> download one slide -> md5 check ->
tile_wsi.py -> extract_uni.py (UNI v1) -> finite/shape check -> delete raw .svs -> checkpoint.
- UNI only.
- Multi-GPU: the master splits the slide queue into shards and spawns one worker subprocess per shard
  (CUDA_VISIBLE_DEVICES set per worker).
- Idempotent: slides with an existing valid embedding .npy are skipped.
- Cancers run sequentially. Heartbeat log, queue size log and DONE sentinel files.
- GDC access: REST /data endpoint; filters use `=` + `and` (the `in` operator returns HTTP 500);
  diagnostic slides are those with -DX in the file name.
- Embeddings are written to $DATA_ROOT/crosscancer/<cancer>/uni_v1; manifest paths are relative to DATA_ROOT.

Usage:
  master : python run_embed_crosscancer.py --cancers LUNG_NSCLC COLORECTAL
  worker : python run_embed_crosscancer.py --worker --cancer X --shard <json> --gpu i  (called by master)
  smoke  : python run_embed_crosscancer.py --cancers LUNG_NSCLC --limit 3 --smoke
"""
import argparse, json, os, sys, time, hashlib, subprocess, shutil
import urllib.request, urllib.error, urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent              # crosscancer/
ROOT = HERE.parent                                  # repo root
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
# Worker interpreter: the running Python unless WORKER_PYTHON is set.
PY = os.environ.get("WORKER_PYTHON", sys.executable)
TILE = ROOT / "pipeline/embedding/scripts/tile_wsi.py"
EXTRACT = ROOT / "pipeline/embedding/scripts/extract_uni.py"
TILE_CFG = ROOT / "pipeline/embedding/configs/tile_config.yaml"
GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data/"

CANCERS = {
    "LUNG_NSCLC": {"cohorts": ["TCGA-LUAD", "TCGA-LUSC"]},
    "COLORECTAL": {"cohorts": ["TCGA-COAD", "TCGA-READ"]},
    "UCEC": {"cohorts": ["TCGA-UCEC"]},   # exploratory extension
}
RAW_BASE = DATA_ROOT / "_crosscancer_raw"           # transient raw .svs cache


def log(msg, fp=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fp:
        with open(fp, "a") as f:
            f.write(line + "\n")


def gdc_query(cohort):
    """Open-access diagnostic WSIs (-DX in file name) for a TCGA project."""
    flt = {"op": "and", "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": cohort}},
        {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
        {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]}
    params = {"filters": json.dumps(flt), "format": "JSON", "size": "20000",
              "fields": "file_id,file_name,md5sum,file_size,cases.submitter_id"}
    q = urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{GDC_FILES}?{q}", timeout=120) as r:
                data = json.load(r)
            hits = data["data"]["hits"]
            out = []
            for h in hits:
                fn = h["file_name"]
                if "-DX" not in fn.upper():        # diagnostic slides only
                    continue
                cid = fn[:12]                        # TCGA-XX-XXXX
                out.append({"file_id": h["file_id"], "file_name": fn, "case_id": cid,
                            "md5": h.get("md5sum", ""), "size": int(h.get("file_size", 0))})
            return out
        except Exception as e:
            log(f"  GDC query {cohort} attempt {attempt+1} failed: {e}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"GDC query failed: {cohort}")


def md5_of(path, buf=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def download(rec, raw_dir):
    dst = raw_dir / rec["file_name"]
    for attempt in range(5):
        try:
            req = urllib.request.Request(GDC_DATA + rec["file_id"])
            with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if rec["md5"] and md5_of(dst) != rec["md5"]:
                raise RuntimeError("md5 mismatch")
            return dst
        except Exception as e:
            log(f"    download {rec['file_name']} attempt {attempt+1} failed: {e}")
            if dst.exists():
                dst.unlink()
            time.sleep(15 * (attempt + 1))
    return None


def process_slide(rec, dirs, wlog):
    name = rec["file_name"][:-4] if rec["file_name"].endswith(".svs") else rec["file_name"]
    emb = dirs["emb"] / f"{name}_uni_embeddings.npy"
    if emb.exists():                                  # idempotent skip (validate partial files)
        try:
            import numpy as np
            a = np.load(emb)
            if a.ndim == 2 and a.shape[1] == 1024 and np.isfinite(a).all():
                return "skip"
        except Exception:
            pass
        emb.unlink()                                  # corrupt/partial -> reprocess
    raw = download(rec, dirs["raw"])
    if raw is None:
        log(f"  FAIL download {name}", wlog)
        return "fail_dl"
    try:
        coords = dirs["coords"] / f"{name}_coords.npy"
        r1 = subprocess.run([PY, str(TILE), "--slide", str(raw), "--config", str(TILE_CFG),
                             "--out", str(coords)], capture_output=True, text=True, timeout=1800)
        if r1.returncode != 0 or not coords.exists():
            log(f"  FAIL tile {name}: {r1.stderr[-300:]}", wlog)
            return "fail_tile"
        r2 = subprocess.run([PY, str(EXTRACT), "--coords", str(coords), "--out_dir", str(dirs["emb"]),
                             "--device", "cuda"], capture_output=True, text=True, timeout=3600)
        if r2.returncode != 0 or not emb.exists():
            log(f"  FAIL embed {name}: {r2.stderr[-300:]}", wlog)
            return "fail_emb"
        import numpy as np
        arr = np.load(emb)
        if arr.ndim != 2 or arr.shape[1] != 1024 or not np.isfinite(arr).all():
            log(f"  FAIL verify {name}: shape={arr.shape}", wlog)
            return "fail_verify"
        log(f"  OK {name} shape={arr.shape}", wlog)
        return "ok"
    finally:
        if raw.exists():                              # always delete raw slide
            raw.unlink()


def run_worker(cancer, shard_path, gpu):
    # limit torch/BLAS threads per worker (inherited by tile/extract subprocesses)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "4"
    dirs = make_dirs(cancer)
    wlog = dirs["out"] / f"worker_gpu{gpu}.log"
    recs = json.loads(Path(shard_path).read_text())
    log(f"worker gpu{gpu} start: {len(recs)} slides", wlog)
    tally = {}
    for i, rec in enumerate(recs, 1):
        st = process_slide(rec, dirs, wlog)
        tally[st] = tally.get(st, 0) + 1
        if i % 10 == 0 or st.startswith("fail"):
            log(f"  gpu{gpu} progress {i}/{len(recs)} {tally}", wlog)
    log(f"worker gpu{gpu} DONE {tally}", wlog)
    (dirs["out"] / f"worker_gpu{gpu}.done").write_text(json.dumps(tally))


def make_dirs(cancer):
    out = HERE / cancer / "full"
    dirs = {"out": out, "raw": RAW_BASE / cancer, "coords": out / "coords",
            "emb": DATA_ROOT / "crosscancer" / cancer / "uni_v1"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def run_master(cancers, limit=None, smoke=False, nshards=3):
    hb = HERE / "EMBED_HEARTBEAT.log"
    log(f"=== MASTER start cancers={cancers} limit={limit} smoke={smoke} ===", hb)
    for cancer in cancers:
        dirs = make_dirs(cancer)
        mlog = dirs["out"] / "master.log"
        # step 0: build + LOG queue
        recs, seen = [], set()
        for cohort in CANCERS[cancer]["cohorts"]:
            hits = gdc_query(cohort)
            for h in hits:
                if h["file_id"] not in seen:
                    seen.add(h["file_id"]); recs.append(h)
            log(f"{cancer} {cohort}: {len(hits)} diagnostic slides", mlog)
        recs.sort(key=lambda r: r["size"])            # smallest first
        if limit:
            recs = recs[:limit]
        log(f"{cancer} QUEUE = {len(recs)} slides (step0)", mlog)
        log(f"{cancer} queued {len(recs)} slides", hb)
        (dirs["out"] / "queue.json").write_text(json.dumps(recs, indent=0))
        if not recs:
            log(f"{cancer} empty queue, skipping", mlog); continue
        # split into N shards (round-robin), shard i -> GPU (i % 3)
        nsh = nshards
        shards = [[] for _ in range(nsh)]
        for i, r in enumerate(recs):
            shards[i % nsh].append(r)
        procs = []
        for s in range(nsh):
            if not shards[s]:
                continue
            g = s % 3
            sp = dirs["out"] / f"shard_{s}_gpu{g}.json"
            sp.write_text(json.dumps(shards[s]))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            cmd = [PY, str(Path(__file__).resolve()), "--worker", "--cancer", cancer,
                   "--shard", str(sp), "--gpu", str(s)]     # --gpu is the shard index for logs; CUDA_VISIBLE_DEVICES pins the GPU
            p = subprocess.Popen(cmd, env=env)
            procs.append((s, p))
            log(f"{cancer} spawned worker shard{s}->gpu{g} pid={p.pid} n={len(shards[s])}", mlog)
        # monitor with heartbeat
        while any(p.poll() is None for _, p in procs):
            done = len(list(dirs["emb"].glob("*_uni_embeddings.npy")))
            log(f"{cancer} heartbeat: embeddings={done}/{len(recs)} "
                f"alive={[g for g,p in procs if p.poll() is None]}", hb)
            time.sleep(60)
        # collect
        n_emb = len(list(dirs["emb"].glob("*_uni_embeddings.npy")))
        log(f"{cancer} WORKERS DONE — embeddings={n_emb}/{len(recs)}", mlog)
        log(f"{cancer} COMPLETE embeddings={n_emb}/{len(recs)}", hb)
        # build embedding manifest
        build_manifest(cancer, dirs, recs)
        (dirs["out"] / "CANCER.done").write_text(json.dumps({"n_queued": len(recs), "n_emb": n_emb}))
    (HERE / "EMBED_ALL.done").write_text(
        json.dumps({"cancers": cancers, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    log(f"=== MASTER ALL DONE {cancers} ===", hb)


def build_manifest(cancer, dirs, recs):
    import csv
    rows = []
    for r in recs:
        name = r["file_name"][:-4] if r["file_name"].endswith(".svs") else r["file_name"]
        emb = dirs["emb"] / f"{name}_uni_embeddings.npy"
        if not emb.exists():
            continue
        rows.append({"case_id": r["case_id"], "slide_id": name,
                     "embedding_path": os.path.relpath(emb, DATA_ROOT), "embedding_model": "uni",
                     "file_id": r["file_id"]})
    mp = dirs["out"] / f"embedding_manifest_{cancer.lower()}_uni.csv"
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "slide_id", "embedding_path", "embedding_model", "file_id"])
        w.writeheader(); w.writerows(rows)
    log(f"{cancer} manifest: {len(rows)} rows -> {mp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancers", nargs="+", default=["LUNG_NSCLC", "COLORECTAL"])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--cancer")
    ap.add_argument("--shard")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shards", type=int, default=3, help="number of worker shards, assigned round-robin to GPUs (default 3)")
    a = ap.parse_args()
    if a.worker:
        run_worker(a.cancer, a.shard, a.gpu)
    else:
        run_master(a.cancers, limit=a.limit, smoke=a.smoke, nshards=a.shards)


if __name__ == "__main__":
    main()
