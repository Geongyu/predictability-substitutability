#!/usr/bin/env python3
"""
Cross-cancer frozen cell-line drug map and therapeutic distance.

Generalises breast/cost/build_frozen_map.py (3 fixed axes) to arbitrary treatment axes.
Per cancer: cellline_axis_table.csv (axis_* columns) + GDSC2 Z_SCORE give a per-axis drug
sensitivity vector and pairwise axis therapeutic distance. Cell lines only.

The MSI -> immunotherapy (pembrolizumab) axis is excluded from the drug map; MSI is used only as a stratifier.
"""
from pathlib import Path
import os, json, itertools, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
GDSC = pd.read_excel(DATA_ROOT / "GDSC2_fitted_dose_response_27Oct23.xlsx")
KEY = "SANGER_MODEL_ID"

# Treatment axes per cancer: label -> axis_column | ('NOT', axis_column)
CANCERS = {
 "LUNG_NSCLC": {  # ALK dropped (n=2)
   "antiEGFR": "axis_antiEGFR",
   "antiKRAS_G12C": "axis_antiKRAS_G12C",
   "chemo": "axis_chemo_noTargetDriver",
 },
 "COLORECTAL": {  # MSI (ICI) and KRAS (antibody, not in GDSC) excluded: BRAF vs baseline
   "antiBRAF": "axis_antiBRAF_V600E",
   "baseline": ("NOT", "axis_antiBRAF_V600E"),
 },
}
KNOWN = {  # positive control: drug -> expected most sensitive axis
 "LUNG_NSCLC": {"Gefitinib":"antiEGFR","Afatinib":"antiEGFR","Osimertinib":"antiEGFR"},
 "COLORECTAL": {"Dabrafenib":"antiBRAF","Trametinib":"antiBRAF","Selumetinib":"antiBRAF"},
}

def build(cancer, axes):
    tab = pd.read_csv(HERE / cancer / "cellline_counts/cellline_axis_table.csv")
    # axis -> set of SangerModelID
    members = {}
    for lab, spec in axes.items():
        if isinstance(spec, tuple) and spec[0] == "NOT":
            sel = tab[tab[spec[1]] != 1]
        else:
            sel = tab[tab[spec] == 1]
        members[lab] = set(sel["SangerModelID"].dropna())
    g = GDSC[GDSC[KEY].isin(set(tab["SangerModelID"].dropna()))].copy()
    labs = list(axes)
    rows = []
    for drug, sub in g.groupby("DRUG_NAME"):
        rec = {"drug": drug, "target": str(sub["PUTATIVE_TARGET"].iloc[0])[:30],
               "pathway": str(sub["PATHWAY_NAME"].iloc[0])[:30]}
        means = {}
        for lab in labs:
            v = sub[sub[KEY].isin(members[lab])]["Z_SCORE"].dropna().values
            rec[f"{lab}_z"] = round(float(np.mean(v)), 3) if len(v) else None
            rec[f"{lab}_n"] = int(len(v))
            if len(v): means[lab] = float(np.mean(v))
        rec["gap"] = round(max(means.values()) - min(means.values()), 3) if len(means) >= 2 else None
        rec["most_sensitive"] = min(means, key=means.get) if means else None
        rows.append(rec)
    df = pd.DataFrame(rows)
    # discriminating: value on every axis and gap >= 0.5
    common = df[df[[f"{l}_z" for l in labs]].notna().all(axis=1)].copy()
    disc = common[common["gap"] >= 0.5].sort_values("gap", ascending=False)
    zmap = {l: {r["drug"]: r[f"{l}_z"] for _, r in common.iterrows()} for l in labs}
    # therapeutic distance = 1 - Kendall tau of discriminating-drug ranks, per axis pair
    dnames = disc["drug"].tolist()
    def rank(l):
        order = sorted(dnames, key=lambda d: zmap[l][d]); return {d: i for i, d in enumerate(order)}
    rk = {l: rank(l) for l in labs}
    def tau(a, b):
        c = d = 0
        for x, y in itertools.combinations(dnames, 2):
            s = np.sign((rk[a][x]-rk[a][y])*(rk[b][x]-rk[b][y]))
            if s > 0: c += 1
            elif s < 0: d += 1
        return (c-d)/(c+d) if (c+d) else float("nan")
    dist = {}
    for a, b in itertools.combinations(labs, 2):
        tt = tau(a, b)
        dist[f"{a}__{b}"] = {"kendall_tau": round(tt, 3), "therapeutic_distance": round(1-tt, 3)}
    # positive control
    pc = {}
    for drug, exp in KNOWN.get(cancer, {}).items():
        m = df[df["drug"].str.contains(drug, case=False, na=False)]
        if len(m):
            r = m.iloc[0]; pc[drug] = {"most_sensitive": r["most_sensitive"], "ok": r["most_sensitive"] == exp,
                                       "z": {l: r.get(f"{l}_z") for l in labs}}
    out = {"cancer": cancer, "leakage_guard": "cell-line-only", "axes_n": {l: len(members[l]) for l in labs},
           "n_drugs": int(len(df)), "n_discriminating": int(len(disc)),
           "axis_pair_distance": dist,
           "positive_control": pc,
           "top_discriminating": disc.head(15)[["drug","target"]+[f"{l}_z" for l in labs]+["gap","most_sensitive"]].to_dict("records"),
           "note": "Cell-line-only frozen map. cost = misrouting frequency (WSI MIL prediction) x therapeutic distance (below). MSI (ICI) and KRAS (antibody) axes excluded."}
    (HERE / cancer / "frozen_map.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n=== {cancer} ===  axes n: {out['axes_n']}  drugs {out['n_drugs']}  disc {out['n_discriminating']}")
    print("  positive control:", {k: ('OK' if v['ok'] else v['most_sensitive']) for k, v in pc.items()})
    print("  therapeutic distance (1-tau):", {k: v['therapeutic_distance'] for k, v in dist.items()})
    print(f"  wrote {cancer}/frozen_map.json")

for c, ax in CANCERS.items():
    build(c, ax)
