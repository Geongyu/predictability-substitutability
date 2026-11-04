#!/usr/bin/env python3
"""
C1 pre-registration: freeze the discriminating drug panel.

Fixes the drugs used for cost computation before the final receptor-status routing
results are observed. Two panels are frozen:
  (1) external_anchor: targeted drugs whose axis is set by clinical pharmacology,
      independent of the map. Primary panel.
  (2) data_driven: discriminating drugs from frozen_map with gap >= 0.5. Secondary panel.
Records provenance (repo commit and timestamp). The written file is the registered panel.
"""
from pathlib import Path
import json, subprocess, datetime

HERE = Path(__file__).resolve().parent
fm = json.loads((HERE / "frozen_map.json").read_text())
zmap = fm["frozen_map_axis_to_drug_z"]
axes = list(zmap.keys())
allmap = {}  # drug -> {axis: z}
for a in axes:
    for d, z in zmap[a].items():
        allmap.setdefault(d, {})[a] = z

# (1) external anchor: established targeted drugs -> intended axis (set by pharmacology, not by the map)
ANCHOR = {
    "antiHER2":  ["Lapatinib", "Afatinib", "Sapitinib", "Tucatinib", "Neratinib"],   # HER2/EGFR TKI
    "endocrine": ["Fulvestrant", "Tamoxifen", "Palbociclib", "Ribociclib",           # ER / CDK4-6
                  "Alpelisib", "Taselisib", "Pictilisib", "AZD8186", "Capivasertib"],# PI3K/AKT (luminal-biased)
    "chemo":     ["Paclitaxel", "Docetaxel", "Cisplatin", "Gemcitabine",             # cytotoxic
                  "Doxorubicin", "Vinorelbine", "Olaparib", "Talazoparib"],          # +PARP (basal/BRCA)
}
def find(name):  # return the map drug name matching by substring
    hits = [d for d in allmap if name.lower() in d.lower()]
    return hits[0] if hits else None

anchor_resolved = {}
for axis, drugs in ANCHOR.items():
    got = []
    for nm in drugs:
        d = find(nm)
        if d:
            zs = allmap[d]
            most = min(zs, key=zs.get) if zs else None
            got.append({"drug": d, "intended_axis": axis, "in_map": True,
                        "z": {a: zs.get(a) for a in axes},
                        "map_most_sensitive_axis": most,
                        "positive_control_ok": most == axis})
        else:
            got.append({"drug": nm, "intended_axis": axis, "in_map": False})
    anchor_resolved[axis] = got

# (2) data-driven gap>=0.5
common = [d for d in allmap if len(allmap[d]) == len(axes)]
def gap(d):
    vs = list(allmap[d].values()); return round(max(vs) - min(vs), 3)
data_driven = sorted([{"drug": d, "gap": gap(d),
                       "z": {a: allmap[d][a] for a in axes},
                       "most_sensitive_axis": min(allmap[d], key=allmap[d].get)}
                      for d in common if gap(d) >= 0.5],
                     key=lambda r: r["gap"], reverse=True)

try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE).decode().strip()[:9]
except Exception:
    commit = "unknown"

out = {
    "artifact": "PRE-REGISTERED discriminating drug panel (C1 cost-of-substitution)",
    "purpose": "Fixed before observing final receptor-status routing results, to prevent cherry-picking",
    "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "frozen_map_source": "frozen_map.json",
    "repo_commit_at_freeze": commit,
    "axes": axes,
    "primary_panel_external_anchor": {
        "rationale": "Axis assignment fixed by clinical pharmacology (outside the map), avoiding subtype_only circularity",
        "n_resolved_in_map": sum(1 for a in anchor_resolved for x in anchor_resolved[a] if x.get("in_map")),
        "by_axis": anchor_resolved,
        "positive_control_summary": {
            a: f"{sum(1 for x in anchor_resolved[a] if x.get('positive_control_ok'))}/"
               f"{sum(1 for x in anchor_resolved[a] if x.get('in_map'))} map most-sensitive axis = intended axis"
            for a in axes},
    },
    "secondary_panel_data_driven": {
        "criterion": "drugs present on all 3 frozen_map axes with gap (max-min z) >= 0.5",
        "n": len(data_driven), "drugs": data_driven,
    },
}
(HERE / "preregistered_drug_panel.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("frozen_at:", out["frozen_at"], "| commit:", commit)
print("external anchor positive-control:")
for a in axes:
    print(f"  {a:10}: {out['primary_panel_external_anchor']['positive_control_summary'][a]}")
print(f"data-driven panel: {len(data_driven)} drugs (gap>=0.5)")
print("wrote preregistered_drug_panel.json")
