#!/usr/bin/env python3
"""Check that sealed predictions were committed before their results.

For each seal, compares git committer timestamps of the prediction and the result
in a local clone of the original repository (biospin/BioProject02).

Input: seal manifest JSON
  {"seals": [
     {"claim": "...",
      "prediction_commit": "<sha>",  "result_commit": "<sha>"},        # mode A: commit SHAs
     {"claim": "...", "prediction_path": "<file>", "result_path": "<file>"}  # mode B: last commit of each file
  ]}

A seal passes if prediction commit time < result commit time; otherwise it is
retrospective and the check fails.
"""
import argparse, json, subprocess, sys
from pathlib import Path


def commit_ct(sha: str, repo: str) -> int:
    """Committer timestamp (epoch) of a commit SHA."""
    out = subprocess.run(["git", "-C", repo, "show", "-s", "--format=%ct", sha],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise ValueError(f"could not read commit time: {sha} ({out.stderr.strip()})")
    return int(out.stdout.strip().splitlines()[-1])


def path_last_ct(path: str, repo: str) -> int:
    """Committer timestamp (epoch) of the last commit touching a path."""
    out = subprocess.run(["git", "-C", repo, "log", "-1", "--format=%ct", "--", path],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise ValueError(f"could not read commit time for path: {path} ({out.stderr.strip()})")
    return int(out.stdout.strip())


def resolve(seal: dict, repo: str):
    if "prediction_commit" in seal and "result_commit" in seal:
        return commit_ct(seal["prediction_commit"], repo), commit_ct(seal["result_commit"], repo)
    if "prediction_path" in seal and "result_path" in seal:
        return path_last_ct(seal["prediction_path"], repo), path_last_ct(seal["result_path"], repo)
    raise ValueError(f"seal has no prediction/result (commit or path) pair: {seal.get('claim')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check that sealed predictions precede their results in git history")
    ap.add_argument("manifest", help="seal manifest JSON")
    ap.add_argument("--repo", default=".",
                    help="path to a local clone of the original repository biospin/BioProject02 (default: current directory)")
    ap.add_argument("--json", help="output path for the result JSON")
    a = ap.parse_args()

    data = json.loads(Path(a.manifest).read_text())
    seals = data.get("seals", [])
    if not seals:
        print("[check_seal_timeline] 0 seals, nothing to check (pass)")
        return 0

    results, failed = [], 0
    for s in seals:
        claim = s.get("claim", "(unnamed)")
        try:
            pred_ct, res_ct = resolve(s, a.repo)
        except ValueError as e:
            print(f"  ERROR {claim}: {e}")
            results.append({"claim": claim, "verdict": "ERROR", "detail": str(e)})
            failed += 1
            continue
        ok = pred_ct < res_ct
        verdict = "SEALED-FORWARD" if ok else "RETROSPECTIVE"
        if not ok:
            failed += 1
        print(f"  {'PASS' if ok else 'FAIL'} {claim}: prediction_ct={pred_ct} "
              f"{'<' if ok else '>='} result_ct={res_ct} -> {verdict}")
        results.append({"claim": claim, "prediction_ct": pred_ct,
                        "result_ct": res_ct, "verdict": verdict})

    summary = {"n_seals": len(seals), "n_failed": failed,
               "pass": failed == 0, "results": results}
    if a.json:
        Path(a.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        print(f"[check_seal_timeline] FAIL: {failed}/{len(seals)} seals are retrospective "
              f"(prediction committed after result).")
        return 1
    print(f"[check_seal_timeline] PASS: all {len(seals)} predictions were committed before their results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
