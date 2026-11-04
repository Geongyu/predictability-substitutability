"""
WSI tiling — 256×256 @ 20×, Otsu tissue mask, per-patient cap 5000.
Output: coords.npy  shape (N, 2)  dtype int64  columns=(x, y) in level-0 pixels.
        coords.json  companion metadata.

Run:
    python pipeline/embedding/scripts/tile_wsi.py \
        --slide $DATA_ROOT/raw/tcga/sample.svs \
        --config pipeline/embedding/configs/tile_config.yaml \
        --out outputs/pilot/coords.npy
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Tissue masking
# ---------------------------------------------------------------------------

def _otsu_threshold(gray: np.ndarray) -> int:
    """Vectorised Otsu threshold on a uint8 grayscale array."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128

    hist /= total
    cumw = np.cumsum(hist)                          # cumulative weight
    cummu = np.cumsum(hist * np.arange(256))        # cumulative mean numerator
    mu_total = cummu[-1]

    w0 = cumw[:-1]
    w1 = 1.0 - w0
    valid = (w0 > 0) & (w1 > 0)

    mu0 = np.where(valid, cummu[:-1] / np.where(w0 > 0, w0, 1), 0)
    mu1 = np.where(valid, (mu_total - cummu[:-1]) / np.where(w1 > 0, w1, 1), 0)
    between_var = w0 * w1 * (mu0 - mu1) ** 2

    return int(np.argmax(between_var)) + 1


def _binary_close(arr: np.ndarray, radius: int = 7) -> np.ndarray:
    """Morphological closing (dilate then erode) using PIL, no cv2 needed."""
    size = 2 * radius + 1
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.filter(ImageFilter.MaxFilter(size))  # dilate
    img = img.filter(ImageFilter.MinFilter(size))  # erode
    return np.array(img)


def _thumbnail_from_level0(slide_path: str, target_ds: float) -> np.ndarray:
    """
    Build a grayscale thumbnail at ~target_ds downsample by reading level 0 in
    horizontal strips and resizing each. Fallback for slides whose downsampled
    pyramid levels have corrupt JPEG2000 codestreams (OpenJPEG "Expected a SOC
    marker") — level 0 is intact, so we derive the mask from it directly.

    Re-opens the slide from path because the handle that hit the decode error is
    poisoned (subsequent calls on it keep raising).
    """
    import openslide

    slide = openslide.OpenSlide(slide_path)
    try:
        w0, h0 = slide.level_dimensions[0]
        out_w = max(1, int(round(w0 / target_ds)))
        out_h = max(1, int(round(h0 / target_ds)))
        # ~256 output rows per strip → bounded RAM (w0 × 256·target_ds × 4 bytes)
        strip_h0 = max(int(target_ds), int(256 * target_ds))
        thumb = Image.new("L", (out_w, out_h))
        y0 = 0
        while y0 < h0:
            h = min(strip_h0, h0 - y0)
            region = slide.read_region((0, y0), 0, (w0, h)).convert("L")
            oh = max(1, int(round(h / target_ds)))
            region = region.resize((out_w, oh), Image.BILINEAR)
            thumb.paste(region, (0, int(round(y0 / target_ds))))
            y0 += h
        return np.array(thumb)
    finally:
        slide.close()


def compute_tissue_mask(slide, downsample_factor: int, slide_path: str | None = None):
    """
    Return (mask, mask_downsample) where mask is a uint8 binary image
    (255 = tissue) at the thumbnail resolution.

    Uses Otsu on grayscale: background (bright) vs tissue (darker).

    If the chosen downsampled level is unreadable (corrupt JPEG2000 frame), the
    thumbnail is rebuilt from level 0 — provided slide_path is given.
    """
    import openslide

    mask_level = slide.get_best_level_for_downsample(downsample_factor)
    target_ds = float(slide.level_downsamples[mask_level])
    try:
        dims = slide.level_dimensions[mask_level]
        thumb = slide.read_region((0, 0), mask_level, dims).convert("L")
        gray = np.array(thumb)
    except openslide.OpenSlideError as e:
        if slide_path is None:
            raise
        print(f"  [warn] mask level {mask_level} unreadable ({e}); "
              f"rebuilding mask from level 0 at downsample {target_ds:.0f}×")
        gray = _thumbnail_from_level0(slide_path, target_ds)

    thresh = _otsu_threshold(gray)
    # H&E tissue is darker than the bright glass background
    binary = ((gray < thresh).astype(np.uint8)) * 255
    binary = _binary_close(binary, radius=7)

    return binary, target_ds


# ---------------------------------------------------------------------------
# Level selection
# ---------------------------------------------------------------------------

def get_target_level(slide, target_mpp: float):
    """Return (level, actual_mpp) whose MPP is closest to target_mpp."""
    mpp_x = float(slide.properties.get("openslide.mpp-x", 0) or 0)

    if mpp_x <= 0:
        # No MPP metadata: best-effort fallback (assume L0 = 40×, L1 = 20×)
        level = 1 if slide.level_count > 1 else 0
        print("  [warn] No MPP metadata — using level", level, "as 20× approximation")
        return level, target_mpp

    best_level, best_diff = 0, float("inf")
    for lv in range(slide.level_count):
        actual = mpp_x * slide.level_downsamples[lv]
        diff = abs(actual - target_mpp)
        if diff < best_diff:
            best_diff = diff
            best_level = lv

    return best_level, mpp_x * slide.level_downsamples[best_level]


# ---------------------------------------------------------------------------
# Main tiling
# ---------------------------------------------------------------------------

def tile_slide(slide_path: str, config: dict, out_path: str) -> dict:
    import openslide
    from tqdm import tqdm

    tile_size        = config["tile_size"]
    target_mpp       = config["target_mpp"]
    tissue_threshold = config["otsu"]["tissue_threshold"]
    downsample_factor = config["otsu"]["downsample_factor"]
    per_patient_cap  = config["per_patient_cap"]
    seed             = config.get("seed", 42)

    slide = openslide.OpenSlide(slide_path)
    print(f"Slide : {slide_path}")
    print(f"  L0  : {slide.level_dimensions[0]}  levels={slide.level_count}")

    target_level, actual_mpp = get_target_level(slide, target_mpp)
    level_dims = slide.level_dimensions[target_level]
    level_ds   = slide.level_downsamples[target_level]
    print(f"  Level {target_level}: dims={level_dims}  mpp={actual_mpp:.3f} (target {target_mpp})")

    # How many native pixels to read per tile to achieve target_mpp after resize.
    # e.g. actual=0.25, target=0.5 → scale=2 → read 512×512, resize to 256×256.
    scale     = target_mpp / actual_mpp
    read_size = max(tile_size, round(tile_size * scale))
    if abs(scale - 1.0) > 0.05:
        print(f"  MPP scale: {scale:.2f}x  read_size={read_size} → resize to {tile_size}")

    print("  Computing tissue mask …")
    mask, mask_ds = compute_tissue_mask(slide, downsample_factor, slide_path=slide_path)
    mask_h, mask_w = mask.shape

    # Conversion factor: one pixel at target level → this many pixels in mask
    level_to_mask = level_ds / mask_ds

    n_x = level_dims[0] // read_size
    n_y = level_dims[1] // read_size

    valid_coords = []
    for ty in tqdm(range(n_y), desc="  Scanning rows", unit="row"):
        for tx in range(n_x):
            x_lv = tx * read_size
            y_lv = ty * read_size

            # Corresponding window in the mask image
            mx  = int(x_lv * level_to_mask)
            my  = int(y_lv * level_to_mask)
            mw  = max(1, int(read_size * level_to_mask))
            mh  = max(1, int(read_size * level_to_mask))
            mx2 = min(mx + mw, mask_w)
            my2 = min(my + mh, mask_h)

            if mx >= mx2 or my >= my2:
                continue

            tissue_frac = mask[my:my2, mx:mx2].mean() / 255.0
            if tissue_frac >= tissue_threshold:
                # Store in level-0 pixel space (openslide.read_region convention)
                valid_coords.append((int(x_lv * level_ds), int(y_lv * level_ds)))

    n_before_cap = len(valid_coords)
    coords = np.array(valid_coords, dtype=np.int64)  # shape (N, 2)

    capped = False
    if n_before_cap > per_patient_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_before_cap, per_patient_cap, replace=False)
        idx.sort()
        coords = coords[idx]
        capped = True

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, coords)

    meta = {
        "slide"          : str(slide_path),
        "slide_name"     : Path(slide_path).stem,
        "target_level"   : target_level,
        "target_mpp"     : target_mpp,
        "actual_mpp"     : round(actual_mpp, 4),
        "scale"          : round(scale, 4),
        "tile_size"      : tile_size,
        "read_size"      : read_size,
        "level_dims"     : list(level_dims),
        "n_tiles"        : int(len(coords)),
        "n_before_cap"   : n_before_cap,
        "capped"         : capped,
        "per_patient_cap": per_patient_cap,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    slide.close()

    print(f"  Tissue tiles (before cap): {n_before_cap}")
    print(f"  Saved: {len(coords)} tiles  capped={capped}")
    print(f"  Output: {out}")
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tile a WSI into 256×256 patches with Otsu tissue filtering"
    )
    parser.add_argument("--slide",  required=True, help="Path to WSI (.svs / .ndpi / .tiff)")
    parser.add_argument("--config", required=True, help="Path to tile_config.yaml")
    parser.add_argument("--out",    required=True, help="Output path for coords.npy")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    t0   = time.time()
    meta = tile_slide(args.slide, config, args.out)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s  n_tiles={meta['n_tiles']}")


if __name__ == "__main__":
    main()
