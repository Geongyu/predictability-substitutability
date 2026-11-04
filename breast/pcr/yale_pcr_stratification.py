"""Yale HER2+ trastuzumab pCR stratification test.

Pre-registered falsification test:
  H_A: the frozen TCGA-trained anti-HER2 axis score stratifies observed trastuzumab pCR
       in the Yale cohort (85 cases, not used in training).

Falsification criteria (pre-registered):
  (1) axis AUROC 95% CI includes 0.5 -> report negative
  (2) axis AUC does not exceed HER2-probability baseline AUC at DeLong p<0.05 -> report negative

Reference:
  Farahmand 2022 in-cohort CV AUC 0.80 [0.69-0.88], n=85, trastuzumab pCR.
  Success = frozen out-of-cohort transfer approaches/overlaps this range.

No fine-tuning on Yale. Only the frozen TCGA-trained model is applied.
claim_level = hypothesis_only

Usage (from repo root):
    python breast/pcr/yale_pcr_stratification.py \\
        --axis_scores breast/models/yale_antiher2_axis_scores.csv \\
        --pcr_labels  "$DATA_ROOT/yale/yale_pcr_labels.csv" \\
        --out_dir     outputs/yale_pcr \\
        --patient_col case_id \\
        --score_col   antiher2_axis_score
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Pre-registered falsification constants ──────────────────────────────────
FALSIFICATION_AUC_NULL = 0.5       # criterion (1): CI includes this value
DELONG_ALPHA = 0.05                # criterion (2): DeLong p must be below this
FARAHMAND_AUC = 0.80              # reference in-cohort CV AUC (Farahmand Mod Pathol 2022)
FARAHMAND_CI_LO = 0.69
FARAHMAND_CI_HI = 0.88
DEFAULT_N_BOOTSTRAP = 2000
RANDOM_SEED = 42


# ── Statistics ───────────────────────────────────────────────────────────────

def compute_auc_bootstrap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    """AUROC + 95% bootstrap CI (patient level, percentile method).

    Returns:
        (auc, ci_lo, ci_hi)
    """
    auc = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    boot_aucs: list[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        boot_aucs.append(roc_auc_score(yt, ys))
    arr = np.array(boot_aucs)
    ci_lo, ci_hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    log.info(f"  Bootstrap: {len(boot_aucs)}/{n_bootstrap} valid samples")
    return auc, ci_lo, ci_hi


def _structural_components(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """DeLong 1988 structural components.

    Returns:
        (auc, v10[n_pos], v01[n_neg])
    """
    pos_mask = y_true == 1
    pos_s = y_score[pos_mask]
    neg_s = y_score[~pos_mask]

    v10 = np.array([
        np.mean(ps > neg_s) + 0.5 * np.mean(ps == neg_s)
        for ps in pos_s
    ])
    v01 = np.array([
        np.mean(ns < pos_s) + 0.5 * np.mean(ns == pos_s)
        for ns in neg_s
    ])
    auc = float(v10.mean())
    return auc, v10, v01


def delong_test(
    y_true: np.ndarray,
    score1: np.ndarray,
    score2: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """DeLong (1988) test for the AUC difference of two correlated ROC curves.

    Returns:
        (auc1, auc2, delta_auc, z_stat, p_two_sided)
    """
    auc1, v10_1, v01_1 = _structural_components(y_true, score1)
    auc2, v10_2, v01_2 = _structural_components(y_true, score2)
    n_pos, n_neg = len(v10_1), len(v01_1)

    var1 = np.var(v10_1, ddof=1) / n_pos + np.var(v01_1, ddof=1) / n_neg
    var2 = np.var(v10_2, ddof=1) / n_pos + np.var(v01_2, ddof=1) / n_neg
    cov12 = (np.cov(v10_1, v10_2, ddof=1)[0, 1] / n_pos
             + np.cov(v01_1, v01_2, ddof=1)[0, 1] / n_neg)

    var_diff = var1 + var2 - 2 * cov12
    if var_diff <= 0:
        log.warning("DeLong var_diff <= 0 (numerically unstable), returning NaN")
        return auc1, auc2, auc1 - auc2, float("nan"), float("nan")

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = float(2 * stats.norm.sf(abs(z)))
    return auc1, auc2, float(auc1 - auc2), float(z), p


def falsification_check(
    auc: float,
    ci_lo: float,
    ci_hi: float,
    delong_p: Optional[float],
) -> dict:
    """Evaluate pre-registered falsification criteria (1) and (2)."""
    fail_ci = ci_lo <= FALSIFICATION_AUC_NULL <= ci_hi
    fail_delong = (delong_p is not None) and not np.isnan(delong_p) and (delong_p >= DELONG_ALPHA)

    reasons: list[str] = []
    if fail_ci:
        reasons.append(
            f"(1) AUC 95% CI [{ci_lo:.3f}, {ci_hi:.3f}] includes "
            f"{FALSIFICATION_AUC_NULL} (no signal)"
        )
    if fail_delong:
        reasons.append(
            f"(2) DeLong p={delong_p:.3f} >= {DELONG_ALPHA} (does not exceed baseline)"
        )

    status = "negative" if (fail_ci or fail_delong) else "positive"
    farahmand_overlap = (ci_lo <= FARAHMAND_CI_HI) and (ci_hi >= FARAHMAND_CI_LO)

    return {
        "falsification_status": status,
        "fail_ci_includes_null": fail_ci,
        "fail_delong_not_significant": fail_delong,
        "falsification_reasons": reasons,
        "farahmand_reference": {
            "auc": FARAHMAND_AUC,
            "ci": [FARAHMAND_CI_LO, FARAHMAND_CI_HI],
        },
        "our_ci_overlaps_farahmand": farahmand_overlap,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Yale trastuzumab pCR stratification test"
    )
    ap.add_argument(
        "--axis_scores", required=True,
        help="Axis score CSV with patient_col and score_col (optional baseline_col)",
    )
    ap.add_argument(
        "--pcr_labels", required=True,
        help="Yale pCR label CSV/XLSX with patient_col and pcr_col (1=responder, 0=non)",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--patient_col", default="case_id",
                    help="patient ID column (default: case_id)")
    ap.add_argument("--score_col", default="axis_score",
                    help="axis score column (default: axis_score)")
    ap.add_argument("--baseline_col", default=None,
                    help="baseline score column for DeLong (DeLong skipped if not set)")
    ap.add_argument("--pcr_col", default="pcr",
                    help="pCR label column (default: pcr, 1=responder)")
    ap.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    ap.add_argument("--no_plot", action="store_true", help="skip ROC plot")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("=== Yale pCR stratification test ===")
    log.info("claim_level: hypothesis_only")
    log.info(f"axis_scores : {args.axis_scores}")
    log.info(f"pcr_labels  : {args.pcr_labels}")
    log.info(f"n_bootstrap : {args.n_bootstrap}")

    # ── Load data ────────────────────────────────────────────────────────────
    score_df = pd.read_csv(args.axis_scores)
    log.info(f"axis_scores loaded: {len(score_df)} rows, columns={list(score_df.columns)}")

    label_path = Path(args.pcr_labels)
    if label_path.suffix in (".xlsx", ".xls"):
        label_df = pd.read_excel(label_path)
    else:
        label_df = pd.read_csv(label_path)
    log.info(f"pcr_labels loaded: {len(label_df)} rows, columns={list(label_df.columns)}")

    # ── Validate ─────────────────────────────────────────────────────────────
    for col, src in [(args.patient_col, "axis_scores"), (args.score_col, "axis_scores")]:
        if col not in score_df.columns:
            log.error(f"column '{col}' missing ({src}). columns: {list(score_df.columns)}")
            return 1
    for col in [args.patient_col, args.pcr_col]:
        if col not in label_df.columns:
            log.error(f"column '{col}' missing (pcr_labels). columns: {list(label_df.columns)}")
            return 1

    # ── Merge ────────────────────────────────────────────────────────────────
    merged = score_df.merge(
        label_df[[args.patient_col, args.pcr_col]].drop_duplicates(args.patient_col),
        on=args.patient_col,
        how="inner",
    )
    n_total = len(merged)
    n_pos = int(merged[args.pcr_col].sum())
    n_neg = n_total - n_pos
    log.info(f"merged: {n_total} cases (responder={n_pos}, non={n_neg})")
    log.info(f"  (before merge: axis_scores={len(score_df)}, labels={len(label_df)})")

    if n_total < 10:
        log.error(f"n={n_total} < 10 after merge; check patient_col")
        return 1
    if n_pos == 0 or n_neg == 0:
        log.error("single class; AUROC undefined")
        return 1

    y_true = merged[args.pcr_col].values.astype(int)
    y_score = merged[args.score_col].values.astype(float)

    # ── AUROC + bootstrap CI ─────────────────────────────────────────────────
    log.info("Computing AUROC + 95% bootstrap CI ...")
    auc, ci_lo, ci_hi = compute_auc_bootstrap(y_true, y_score, args.n_bootstrap)
    log.info(f"AUROC = {auc:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── DeLong test ──────────────────────────────────────────────────────────
    delong_result: Optional[dict] = None
    if args.baseline_col:
        if args.baseline_col not in merged.columns:
            log.warning(f"baseline_col '{args.baseline_col}' missing; skipping DeLong")
        else:
            baseline_score = merged[args.baseline_col].values.astype(float)
            auc1, auc2, delta, z, p = delong_test(y_true, y_score, baseline_score)
            log.info(f"DeLong: axis={auc1:.4f} vs baseline={auc2:.4f} "
                     f"Δ={delta:+.4f}  z={z:.3f}  p={p:.4f}")
            delong_result = {
                "baseline_col": args.baseline_col,
                "axis_auc": round(auc1, 4),
                "baseline_auc": round(auc2, 4),
                "delta_auc": round(delta, 4),
                "z_statistic": round(z, 4) if not np.isnan(z) else None,
                "p_value_two_sided": round(p, 4) if not np.isnan(p) else None,
            }

    # ── Falsification check ──────────────────────────────────────────────────
    delong_p = (delong_result or {}).get("p_value_two_sided")
    falsif = falsification_check(auc, ci_lo, ci_hi, delong_p)
    status = falsif["falsification_status"].upper()
    log.info(f"falsification status: {status}")
    for r in falsif["falsification_reasons"]:
        log.warning(f"  → {r}")

    # ── Result JSON ──────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "task": "Yale pCR stratification",
        "claim_level": "hypothesis_only",
        "timestamp": ts,
        "cohort": {
            "name": "Yale HER2-TUMOR-ROIS (trastuzumab pCR)",
            "n_total": n_total,
            "n_responder": n_pos,
            "n_non_responder": n_neg,
        },
        "primary": {
            "score_col": args.score_col,
            "auroc": round(auc, 4),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "n_bootstrap": args.n_bootstrap,
        },
        "delong": delong_result,
        "falsification": falsif,
        "reference": {
            "farahmand_2022": {
                "cv_auc": FARAHMAND_AUC,
                "ci": [FARAHMAND_CI_LO, FARAHMAND_CI_HI],
                "note": "in-cohort CV; this analysis is out-of-cohort frozen transfer",
            },
        },
        "inputs": {
            "axis_scores": str(args.axis_scores),
            "pcr_labels": str(args.pcr_labels),
            "patient_col": args.patient_col,
            "score_col": args.score_col,
            "baseline_col": args.baseline_col,
        },
    }

    result_path = out_dir / f"yale_pcr_a4_{ts}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.info(f"JSON saved: {result_path}")

    # ── ROC plot ─────────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fpr, tpr, _ = roc_curve(y_true, y_score)
            fig, ax = plt.subplots(figsize=(5, 5))

            ax.plot(
                fpr, tpr, lw=2, color="#1f77b4",
                label=f"Anti-HER2 axis score  AUC={auc:.3f}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
            )
            if delong_result:
                base_fpr, base_tpr, _ = roc_curve(
                    y_true, merged[args.baseline_col].values
                )
                ax.plot(
                    base_fpr, base_tpr, lw=1.5, color="#ff7f0e", ls="--",
                    label=f"HER2-prob baseline  AUC={delong_result['baseline_auc']:.3f}",
                )
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random AUC=0.50")
            ax.fill_between([0, 1],
                            [FARAHMAND_CI_LO, FARAHMAND_CI_LO],
                            [FARAHMAND_CI_HI, FARAHMAND_CI_HI],
                            alpha=0.08, color="gray",
                            label=f"Farahmand 2022 CI [{FARAHMAND_CI_LO}–{FARAHMAND_CI_HI}]")

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(
                "Yale trastuzumab pCR — anti-HER2 axis score ROC\n"
                "(hypothesis_only, frozen transfer)"
            )
            ax.legend(fontsize=8, loc="lower right")

            color = "darkred" if falsif["falsification_status"] == "negative" else "darkgreen"
            ax.text(
                0.98, 0.05,
                f"n={n_total}  (R:{n_pos} / N:{n_neg})\nfalsification: {status}",
                ha="right", va="bottom", fontsize=9,
                transform=ax.transAxes, color=color,
            )

            fig_path = out_dir / f"yale_pcr_roc_{ts}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"ROC plot saved: {fig_path}")
        except ImportError:
            log.warning("matplotlib not installed; skipping plot (use --no_plot to skip explicitly)")

    # ── Summary ──────────────────────────────────────────────────────────────
    line = "=" * 62
    print(f"\n{line}")
    print("Yale pCR stratification summary  (claim_level=hypothesis_only)")
    print(line)
    print(f"  cohort  : Yale HER2+ trastuzumab pCR  n={n_total} (R:{n_pos}/N:{n_neg})")
    print(f"  AUROC   : {auc:.4f}   95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    if delong_result:
        print(f"  DeLong  : Δ={delong_result['delta_auc']:+.4f}  "
              f"z={delong_result['z_statistic']}  p={delong_result['p_value_two_sided']}")
    print(f"  falsification: {status}")
    for r in falsif["falsification_reasons"]:
        print(f"    → {r}")
    print(f"  Farahmand ref: AUC={FARAHMAND_AUC} [{FARAHMAND_CI_LO}–{FARAHMAND_CI_HI}]")
    print(f"  CI overlap: {falsif['our_ci_overlaps_farahmand']}")
    print(f"\n  result -> {result_path}")
    print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
