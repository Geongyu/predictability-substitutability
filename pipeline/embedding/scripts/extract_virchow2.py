"""
Virchow2 tile embedding extractor (paige-ai/Virchow2, 2560-dim).

Used for multi-FM robustness checks (model independence of the substitutability law).
The primary model is UNI v1 (1024-d).

Embedding = concat(CLS token, mean(patch tokens)) -> 1280 + 1280 = 2560-d.
Virchow2 has 4 register tokens (output[:, 1:5]); patch tokens are output[:, 5:].

Same I/O as extract_uni.py:
  coords.npy + coords.json (with read_size, from tile_wsi.py) -> <slide>_virchow2_embeddings.npy
Uses read_size from coords.json, so the MPP scaling from tile_wsi.py is applied
(for a 40x slide, reads 512x512 and resizes to 224x224).

Run:
    python pipeline/embedding/scripts/extract_virchow2.py \
        --coords $DATA_ROOT/tiles/TCGA-xxx_coords.npy \
        --out_dir $DATA_ROOT/brca/virchow2/ \
        [--batch_size 32] [--device cuda]
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # gated repo: load from local cache only
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

EMB_DIM = 2560          # CLS(1280) + mean-patch(1280)
N_REGISTER = 4          # Virchow2 register tokens; patches are [:, 1+N_REGISTER:]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_virchow2(device: str) -> tuple:
    """Load Virchow2 from local HF cache. Returns (model, transform)."""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    # SwiGLUPacked mlp + SiLU per the model card.
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


def embed_batch(model, batch: torch.Tensor) -> torch.Tensor:
    """(B,3,224,224) -> (B,2560). CLS + mean(patch), register tokens excluded."""
    out = model(batch)                              # (B, 1+N_REGISTER+P, 1280)
    cls = out[:, 0]                                 # (B, 1280)
    patch = out[:, 1 + N_REGISTER:]                 # (B, P, 1280), registers excluded
    return torch.cat([cls, patch.mean(dim=1)], dim=-1)   # (B, 2560)


# ---------------------------------------------------------------------------
# Tile reading
# ---------------------------------------------------------------------------

class TileDataset(torch.utils.data.Dataset):
    def __init__(self, slide_path, coords, read_size, transform):
        self.slide_path = str(slide_path)
        self.coords = coords
        self.read_size = read_size
        self.transform = transform
        self._h = None

    def __len__(self):
        return len(self.coords)

    def _handle(self):
        if self._h is None:                         # lazy open per worker (fork-safe)
            import openslide
            self._h = openslide.OpenSlide(self.slide_path)
        return self._h

    def __getitem__(self, idx):
        x, y = int(self.coords[idx][0]), int(self.coords[idx][1])
        img = self._handle().read_region((x, y), 0, (self.read_size, self.read_size)).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------

def extract(coords_path, out_dir, batch_size=32, device="cuda", num_workers=6):
    coords_path = Path(coords_path)
    meta = json.loads(coords_path.with_suffix(".json").read_text())
    coords = np.load(coords_path)
    slide_path = meta["slide"]
    read_size = meta["read_size"]
    name = coords_path.name.replace("_coords.npy", "")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_file = out_dir / f"{name}_virchow2_embeddings.npy"

    model, transform = load_virchow2(device)
    ds = TileDataset(slide_path, coords, read_size, transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, shuffle=False)

    t0 = time.time()
    feats = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        for batch in tqdm(dl, desc=name, unit="batch"):
            batch = batch.to(device, non_blocking=True)
            feats.append(embed_batch(model, batch).float().cpu())

    emb = torch.cat(feats).numpy().astype(np.float32)
    assert emb.shape == (len(coords), EMB_DIM), f"shape {emb.shape} != ({len(coords)}, {EMB_DIM})"
    assert np.isfinite(emb).all(), "non-finite embedding"
    np.save(emb_file, emb)
    print(f"[virchow2] {name}: {emb.shape} in {time.time()-t0:.1f}s -> {emb_file}", flush=True)
    return emb_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)   # heavier than UNI (2560-d)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=6)
    a = ap.parse_args()
    extract(a.coords, a.out_dir, a.batch_size, a.device, a.num_workers)


if __name__ == "__main__":
    main()
