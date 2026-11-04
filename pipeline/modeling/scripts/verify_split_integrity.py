"""
Verify split integrity: patient-level (case_id) and site-level (TCGA TSS code)
disjointness across the splits in a manifest's split column. Writes a JSON report.

Rules:
    - The input CSV must have split and case_id columns; otherwise the check fails.
    - Zero checked rows fails.
    - Fewer than two split groups fails.
    - The result JSON is written on failure too. Exit code 1 on failure.

Run (input = cohort split.csv):
    python pipeline/modeling/scripts/verify_split_integrity.py \
        --manifest crosscancer/LUNG_NSCLC/full/split.csv \
        --out outputs/site_disjoint_verification.json

    # embedding_manifest_*.csv has no split column; join split first.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def tss_code(case_id: str) -> str:
    parts = case_id.split("-")
    if len(parts) >= 2 and parts[0] == "TCGA":
        return parts[1]
    return f"UNKNOWN:{case_id}"


def _write(out: str, payload: dict) -> None:
    """Write the result JSON (on pass or fail)."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split_sites = defaultdict(set)
    split_cases = defaultdict(set)
    n_rows = defaultdict(int)

    REQUIRED = ("split", "case_id")
    n_skipped = 0
    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED if c not in header]
        if missing:
            print(f"[FAIL] manifest is missing required columns: {missing}\n"
                  f"       file: {args.manifest}\n"
                  f"       header: {header}\n"
                  f"       -> use split.csv, or join split into the manifest and rerun.",
                  file=sys.stderr)
            _write(args.out, {"manifest": args.manifest, "status": "FAIL",
                              "reason": "missing_required_columns", "missing": missing,
                              "header": header, "rows_checked": 0,
                              "site_disjoint": None, "patient_disjoint": None})
            return 1
        for row in reader:
            split = (row.get("split") or "").strip()
            case_id = (row.get("case_id") or "").strip()
            if not split or not case_id:
                n_skipped += 1
                continue
            split_sites[split].add(tss_code(case_id))
            split_cases[split].add(case_id)
            n_rows[split] += 1

    splits = sorted(split_sites.keys())
    site_overlaps = {}
    case_overlaps = {}
    for a, b in combinations(splits, 2):
        key = f"{a}_{b}"
        site_overlaps[key] = sorted(split_sites[a] & split_sites[b])
        case_overlaps[key] = sorted(split_cases[a] & split_cases[b])

    all_site_clear = all(len(v) == 0 for v in site_overlaps.values())
    all_case_clear = all(len(v) == 0 for v in case_overlaps.values())

    result = {
        "manifest": args.manifest,
        "n_rows_per_split": dict(n_rows),
        "n_unique_sites_per_split": {k: len(v) for k, v in split_sites.items()},
        "site_overlap": {k: len(v) for k, v in site_overlaps.items()},
        "site_overlap_detail": {k: v for k, v in site_overlaps.items() if v},
        "case_id_overlap": {k: len(v) for k, v in case_overlaps.items()},
        "case_id_overlap_detail": {k: v for k, v in case_overlaps.items() if v},
        "site_disjoint": all_site_clear,
        "patient_disjoint": all_case_clear,
    }

    n_checked = sum(n_rows.values())
    result["rows_checked"] = n_checked
    result["rows_skipped_blank"] = n_skipped

    print(json.dumps(result, indent=2))

    # zero checked rows is a failure
    if n_checked == 0:
        result["status"] = "FAIL"
        result["reason"] = "no_rows_checked"
        _write(args.out, result)
        print(f"\n[FAIL] 0 rows checked ({n_skipped} blank rows skipped).",
              file=sys.stderr)
        return 1

    # a single split group is a failure (disjointness would hold trivially)
    if len(n_rows) < 2:
        result["status"] = "FAIL"
        result["reason"] = "single_split_group"
        _write(args.out, result)
        print(f"\n[FAIL] only {len(n_rows)} split group(s) ({list(n_rows)}); nothing to compare.\n"
              f"       -> check that split.csv contains train/val/test.",
              file=sys.stderr)
        return 1

    # leakage is reported as an explicit failure (not assert), so it also works under python -O
    if not all_case_clear or not all_site_clear:
        result["status"] = "FAIL"
        result["reason"] = "leakage"
        _write(args.out, result)
        if not all_case_clear:
            print(f"[FAIL] patient-level leakage found: {result['case_id_overlap_detail']}", file=sys.stderr)
        if not all_site_clear:
            print(f"[FAIL] site-level leakage found: {result['site_overlap_detail']}", file=sys.stderr)
        return 1

    result["status"] = "PASS"
    _write(args.out, result)
    print(f"\n[PASS] site-disjoint AND patient-disjoint ({n_checked} rows checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
