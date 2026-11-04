"""
UNI2-h tile embedding extractor (MahmoodLab/UNI2-h, 1536-dim).

Used for multi-FM robustness checks; the primary model is UNI v1 (1024-d).

UNI2-h uses 8 register tokens (Virchow2 uses 4). config.json:
  architecture=vit_giant_patch14_224, num_features=1536, global_pool="token", reg_tokens=8.
Standard usage (model card) is CLS token pooling -> 1536-d. With global_pool="token",
model(x) already returns the CLS vector, so no register-token slicing is needed.

Same I/O as extract_uni.py / extract_virchow2.py:
  coords.npy + coords.json (with read_size) -> <slide>_uni2h_embeddings.npy

Run:
    python pipeline/embedding/scripts/extract_uni2h.py \
        --coords <coords.npy> --out_dir <dir> [--batch_size 32] [--device cuda]
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

EMB_DIM = 1536


def load_uni2h(device: str) -> tuple:
    """Load UNI2-h from local HF cache. Returns (model, transform). timm_kwargs follow the model card."""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


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
        if self._h is None:
            import openslide
            self._h = openslide.OpenSlide(self.slide_path)
        return self._h

    def __getitem__(self, idx):
        x, y = int(self.coords[idx][0]), int(self.coords[idx][1])
        img = self._handle().read_region((x, y), 0, (self.read_size, self.read_size)).convert("RGB")
        return self.transform(img)


def extract(coords_path, out_dir, batch_size=32, device="cuda", num_workers=6):
    coords_path = Path(coords_path)
    meta = json.loads(coords_path.with_suffix(".json").read_text())
    coords = np.load(coords_path)
    slide_path = meta["slide"]
    read_size = meta["read_size"]
    name = coords_path.name.replace("_coords.npy", "")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_file = out_dir / f"{name}_uni2h_embeddings.npy"

    model, transform = load_uni2h(device)
    ds = TileDataset(slide_path, coords, read_size, transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, shuffle=False)

    t0 = time.time()
    feats = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        for batch in tqdm(dl, desc=name, unit="batch"):
            batch = batch.to(device, non_blocking=True)
            out = model(batch)                       # global_pool="token" -> (B, 1536) CLS vector
            feats.append(out.float().cpu())

    emb = torch.cat(feats).numpy().astype(np.float32)
    assert emb.shape == (len(coords), EMB_DIM), f"shape {emb.shape} != ({len(coords)}, {EMB_DIM})"
    assert np.isfinite(emb).all(), "non-finite embedding"
    np.save(emb_file, emb)
    print(f"[uni2h] {name}: {emb.shape} in {time.time()-t0:.1f}s -> {emb_file}", flush=True)
    return emb_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=6)
    a = ap.parse_args()
    extract(a.coords, a.out_dir, a.batch_size, a.device, a.num_workers)


if __name__ == "__main__":
    main()
