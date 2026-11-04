#!/usr/bin/env python3
"""Step 0 — colorectal treatment-marker labels from cBioPortal.
Markers: msi_high (MSIsensor>=3.5, fallback MANTIS>=0.4), all_ras (KRAS/NRAS codon 12/13/61
activating), braf_v600, anti_egfr_eligible (all_ras==0 AND braf_v600==0).
Joined to the 613 embedding-bearing patients. Prints denominators/prevalence vs verified targets.
"""
import os, re, csv, json
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
EMB_DIR = DATA_ROOT / "crosscancer" / "COLORECTAL" / "uni_v1"
STUDY = "coadread_tcga_pan_can_atlas_2018"
MUT_PROFILE = f"{STUDY}_mutations"
SAMPLE_LIST = f"{STUDY}_all"
API = "https://www.cbioportal.org/api"
ENTREZ = {3845: "KRAS", 4893: "NRAS", 673: "BRAF"}

def emb_patients():
    return {p.name[:12] for p in EMB_DIR.glob("*_uni_embeddings.npy")}

def fetch_mutations():
    url = f"{API}/molecular-profiles/{MUT_PROFILE}/mutations/fetch?projection=DETAILED"
    body = {"sampleListId": SAMPLE_LIST, "entrezGeneIds": list(ENTREZ.keys())}
    r = requests.post(url, json=body, headers={"accept":"application/json"}, timeout=120)
    r.raise_for_status()
    return r.json()

def fetch_msi():
    ids = requests.get(f"{API}/sample-lists/{SAMPLE_LIST}/sample-ids",
                       headers={"accept":"application/json"}, timeout=60).json()
    url = f"{API}/studies/{STUDY}/clinical-data/fetch?clinicalDataType=SAMPLE&projection=DETAILED"
    body = {"attributeIds": ["MSI_SENSOR_SCORE", "MSI_SCORE_MANTIS"], "ids": ids}
    r = requests.post(url, json=body, headers={"accept":"application/json"}, timeout=120)
    r.raise_for_status()
    return r.json()

def codon(protein):
    m = re.match(r"^[A-Z](\d+)[A-Z*]", protein or "")
    return int(m.group(1)) if m else None

def main():
    emb = emb_patients()
    print(f"embedding patients: {len(emb)}")
    muts = fetch_mutations()
    print(f"mutation records fetched: {len(muts)}")
    all_ras, braf = set(), set()
    for m in muts:
        gid = m.get("entrezGeneId"); pc = m.get("proteinChange", "")
        cid = m.get("sampleId", "")[:12]
        c = codon(pc)
        if gid in (3845, 4893) and c in (12, 13, 61):
            all_ras.add(cid)
        if gid == 673 and pc.startswith("V600"):
            braf.add(cid)
    msi_rows = fetch_msi()
    sensor, mantis = {}, {}
    for row in msi_rows:
        cid = row.get("sampleId","")[:12]; aid = row.get("clinicalAttributeId"); val = row.get("value")
        try: v = float(val)
        except (TypeError, ValueError): continue
        if aid == "MSI_SENSOR_SCORE": sensor[cid] = v
        elif aid == "MSI_SCORE_MANTIS": mantis[cid] = v
    msi_high = set()
    msi_src = {}
    for cid in set(sensor) | set(mantis):
        if cid in sensor:
            if sensor[cid] >= 3.5: msi_high.add(cid);
            msi_src[cid] = "sensor"
        elif mantis[cid] >= 0.4:
            msi_high.add(cid); msi_src[cid] = "mantis"
    # write joined to embedding patients
    out = HERE/"labels_treatment.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id","msi_high","all_ras","braf_v600","anti_egfr_eligible"])
        for cid in sorted(emb):
            mh = int(cid in msi_high); ar = int(cid in all_ras); bv = int(cid in braf)
            elig = int(ar == 0 and bv == 0)
            w.writerow([cid, mh, ar, bv, elig])
    # prevalence report (over embedding patients)
    def prev(s):
        n = sum(1 for c in emb if c in s); return n, round(100*n/len(emb),1)
    print(f"[joined to {len(emb)} embedding patients]")
    for name, s, tgt in [("msi_high", msi_high, "88/15.1%"), ("all_ras", all_ras, "219/37.5%"), ("braf_v600", braf, "48")]:
        n, p = prev(s); print(f"  {name}: {n} ({p}%)  target~{tgt}")
    elig_n = sum(1 for c in emb if c not in all_ras and c not in braf)
    print(f"  anti_egfr_eligible: {elig_n} ({round(100*elig_n/len(emb),1)}%)")
    # cross-check braf against existing patient_labels.csv braf_v600e
    pl = {r["case_id"]: r for r in csv.DictReader(open(HERE/"patient_labels.csv"))}
    both = [c for c in emb if c in pl]
    disagree = [c for c in both if int(pl[c].get("braf_v600e","0") or 0) != int(c in braf)]
    print(f"  braf cross-check vs patient_labels.csv: {len(both)} shared, {len(disagree)} disagree")
    if disagree[:10]: print("    disagree examples:", disagree[:10])
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
