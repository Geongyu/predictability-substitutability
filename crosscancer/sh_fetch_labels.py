#!/usr/bin/env python3
"""
STAD and HNSC patient label fetcher (public cBioPortal API).

Endpoints follow SUBSTITUTABILITY_LAW_PREREGISTRATION.md.
EBV/MSI/HPV are taken from the TCGA molecular SUBTYPE field.

STAD (stad_tcga_pan_can_atlas_2018):
  - erbb2_amp : GISTIC discrete (_gistic) ERBB2 (2064) alteration == 2 (counterpart of breast HER2)
  - msi_h     : MSI_SENSOR_SCORE >= 3.5 (same threshold as lung/colorectal). Cross-check: SUBTYPE STAD_MSI
  - lauren_diffuse : ICD_O_3_HISTOLOGY, diffuse (8145/3) + signet-ring (8490/3) = 1 vs intestinal (8144/3) = 0
                     (NOS 8140/3 and mucinous 8480/3 excluded). Strong H&E morphology.
  - ebv       : SUBTYPE == STAD_EBV (exploratory, lymphoepithelioma-like morphology)
HNSC (hnsc_tcga_pan_can_atlas_2018):
  - hpv_pos   : SUBTYPE == HNSC_HPV+ (viral, visible axis)
  - egfr_amp  : GISTIC discrete EGFR (1956) alteration == 2
  - grade_high: GRADE in {G3,G4} vs {G1,G2} (differentiation, soft positive control)

Denominator = patients present in each endpoint's profile (has_* flags, so negatives are true negatives).
Output: <cancer>/full/patient_labels.csv + sh_labels_prevalence_report.json
"""
import urllib.request, json, csv, sys, time
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
BASE = "https://www.cbioportal.org/api"

def post(path, body):
    for a in range(5):
        try:
            req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(body).encode(),
                headers={"Accept":"application/json","Content-Type":"application/json"}, method="POST")
            return json.load(urllib.request.urlopen(req, timeout=120))
        except Exception as e:
            print(f"  cbio POST {path} retry {a+1}: {e}"); time.sleep(8*(a+1))
    raise RuntimeError(f"cbio POST failed {path}")

def get(path):
    for a in range(5):
        try:
            return json.load(urllib.request.urlopen(f"{BASE}{path}", timeout=90))
        except Exception as e:
            print(f"  cbio GET {path} retry {a+1}: {e}"); time.sleep(8*(a+1))
    raise RuntimeError(f"cbio GET failed {path}")

def pid(x): return x[:12]  # TCGA-XX-XXXX

def subtype_map(study):
    d = post(f"/studies/{study}/clinical-data/fetch?clinicalDataType=PATIENT", {"attributeIds":["SUBTYPE"]})
    return {x["patientId"]: x["value"] for x in d}

def sample_attr(study, attr):
    """SAMPLE-level attr -> {case_id: value} (first sample per patient)."""
    d = post(f"/studies/{study}/clinical-data/fetch?clinicalDataType=SAMPLE", {"attributeIds":[attr]})
    out = {}
    for x in d:
        c = pid(x["sampleId"])
        out.setdefault(c, x["value"])
    return out

def patient_attr(study, attr):
    """PATIENT-level attr -> {case_id: value}."""
    d = post(f"/studies/{study}/clinical-data/fetch?clinicalDataType=PATIENT", {"attributeIds":[attr]})
    return {x["patientId"]: x["value"] for x in d}

def cna_amp(study, entrez):
    """Patients with GISTIC discrete amplification (== 2) and all profiled patients."""
    prof=f"{study}_gistic"; slist=f"{study}_all"
    d=post(f"/molecular-profiles/{prof}/discrete-copy-number/fetch?discreteCopyNumberEventType=ALL",
           {"sampleListId":slist,"entrezGeneIds":[entrez]})
    profiled=set(); amp=set()
    for x in d:
        c=pid(x["sampleId"]); profiled.add(c)
        if x["alteration"]==2: amp.add(c)
    return amp, profiled

def build_stad():
    study="stad_tcga_pan_can_atlas_2018"
    sub = subtype_map(study)
    msi = sample_attr(study, "MSI_SENSOR_SCORE")
    hist = patient_attr(study, "ICD_O_3_HISTOLOGY")
    erbb2_amp, cna_prof = cna_amp(study, 2064)
    # universe = patients present in any endpoint
    universe = set(sub) | set(msi) | set(hist) | cna_prof
    def msi_val(c):
        v=msi.get(c)
        try: return float(v)
        except: return None
    rows={}
    for c in sorted(universe):
        d={"case_id":c,"cohort":"STAD"}
        # ERBB2-amp
        if c in cna_prof:
            d["erbb2_amp"]=int(c in erbb2_amp); d["has_erbb2_amp"]=1
        else:
            d["erbb2_amp"]=""; d["has_erbb2_amp"]=0
        # MSI-H (MSI_SENSOR>=3.5 primary)
        mv=msi_val(c)
        if mv is not None:
            d["msi_h"]=int(mv>=3.5); d["has_msi"]=1
        else:
            d["msi_h"]=""; d["has_msi"]=0
        # Lauren diffuse vs intestinal
        h=hist.get(c,"")
        if h in ("8145/3","8490/3"):
            d["lauren_diffuse"]=1; d["has_lauren"]=1
        elif h in ("8144/3",):
            d["lauren_diffuse"]=0; d["has_lauren"]=1
        else:
            d["lauren_diffuse"]=""; d["has_lauren"]=0
        # EBV (SUBTYPE)
        s=sub.get(c,"")
        if s:
            d["ebv"]=int(s=="STAD_EBV"); d["has_ebv"]=1
        else:
            d["ebv"]=""; d["has_ebv"]=0
        d["subtype"]=s
        rows[c]=d
    # cross-check: MSI_SENSOR>=3.5 vs SUBTYPE STAD_MSI
    msi_sensor=set(c for c in rows if rows[c]["msi_h"]==1)
    msi_subtype=set(c for c in rows if sub.get(c)=="STAD_MSI")
    prev={
      "n_universe":len(rows),
      "erbb2_amp":{"n_pos":sum(1 for d in rows.values() if d["erbb2_amp"]==1),
                   "n_has":sum(d["has_erbb2_amp"] for d in rows.values())},
      "msi_h":{"n_pos":len(msi_sensor),"n_has":sum(d["has_msi"] for d in rows.values()),
               "subtype_STAD_MSI":len(msi_subtype),
               "agreement_sensor_vs_subtype":len(msi_sensor & msi_subtype)},
      "lauren_diffuse":{"n_pos":sum(1 for d in rows.values() if d["lauren_diffuse"]==1),
                        "n_neg":sum(1 for d in rows.values() if d["lauren_diffuse"]==0),
                        "n_has":sum(d["has_lauren"] for d in rows.values())},
      "ebv":{"n_pos":sum(1 for d in rows.values() if d["ebv"]==1),
             "n_has":sum(d["has_ebv"] for d in rows.values())},
    }
    for k in ("erbb2_amp","ebv"):
        prev[k]["pct"]=round(prev[k]["n_pos"]/max(prev[k]["n_has"],1)*100,1)
    prev["msi_h"]["pct"]=round(prev["msi_h"]["n_pos"]/max(prev["msi_h"]["n_has"],1)*100,1)
    prev["lauren_diffuse"]["pct_diffuse_of_classified"]=round(
        prev["lauren_diffuse"]["n_pos"]/max(prev["lauren_diffuse"]["n_pos"]+prev["lauren_diffuse"]["n_neg"],1)*100,1)
    cols=["case_id","cohort","erbb2_amp","msi_h","lauren_diffuse","ebv","subtype",
          "has_erbb2_amp","has_msi","has_lauren","has_ebv"]
    return rows, prev, cols

def build_hnsc():
    study="hnsc_tcga_pan_can_atlas_2018"
    sub = subtype_map(study)
    grade = sample_attr(study, "GRADE")
    egfr_amp, cna_prof = cna_amp(study, 1956)
    universe = set(sub) | set(grade) | cna_prof
    rows={}
    for c in sorted(universe):
        d={"case_id":c,"cohort":"HNSC"}
        s=sub.get(c,"")
        if s:
            d["hpv_pos"]=int(s=="HNSC_HPV+"); d["has_hpv"]=1
        else:
            d["hpv_pos"]=""; d["has_hpv"]=0
        if c in cna_prof:
            d["egfr_amp"]=int(c in egfr_amp); d["has_egfr_amp"]=1
        else:
            d["egfr_amp"]=""; d["has_egfr_amp"]=0
        g=grade.get(c,"")
        if g in ("G1","G2","G3","G4"):
            d["grade_high"]=int(g in ("G3","G4")); d["has_grade"]=1
        else:
            d["grade_high"]=""; d["has_grade"]=0
        d["subtype"]=s; d["grade"]=g
        rows[c]=d
    prev={
      "n_universe":len(rows),
      "hpv_pos":{"n_pos":sum(1 for d in rows.values() if d["hpv_pos"]==1),
                 "n_has":sum(d["has_hpv"] for d in rows.values())},
      "egfr_amp":{"n_pos":sum(1 for d in rows.values() if d["egfr_amp"]==1),
                  "n_has":sum(d["has_egfr_amp"] for d in rows.values())},
      "grade_high":{"n_pos":sum(1 for d in rows.values() if d["grade_high"]==1),
                    "n_neg":sum(1 for d in rows.values() if d["grade_high"]==0),
                    "n_has":sum(d["has_grade"] for d in rows.values())},
    }
    for k in ("hpv_pos","egfr_amp"):
        prev[k]["pct"]=round(prev[k]["n_pos"]/max(prev[k]["n_has"],1)*100,1)
    prev["grade_high"]["pct"]=round(prev["grade_high"]["n_pos"]/max(prev["grade_high"]["n_has"],1)*100,1)
    cols=["case_id","cohort","hpv_pos","egfr_amp","grade_high","subtype","grade",
          "has_hpv","has_egfr_amp","has_grade"]
    return rows, prev, cols

# literature prevalence ranges (%) used for the check
ANCHOR={
 "stad_erbb2_amp":(10,22), "stad_msi_h":(15,25), "stad_ebv":(5,13),
 "hnsc_hpv_pos":(10,30), "hnsc_egfr_amp":(6,20),
}
def gate(name,val,lo,hi):
    ok=lo<=val<=hi
    print(f"  [{'OK ' if ok else 'WARN'}] {name} = {val}% (expected {lo}-{hi}%)")

def write_labels(cancer, rows, cols):
    out=HERE/cancer/"full"/"patient_labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for d in rows.values(): w.writerow({k:d.get(k,"") for k in cols})
    print(f"  wrote {out} ({len(rows)} patients)")

def est_holdout_pos(prev_ep, holdout_frac=0.30):
    """Approximate holdout positives (n_pos * holdout_frac); < 25 marks an endpoint exploratory."""
    return round(prev_ep.get("n_pos",0)*holdout_frac,1)

def main():
    report={}
    print("=== STAD labels (cBioPortal) ===")
    sr,sp,sc=build_stad(); write_labels("GASTRIC_STAD",sr,sc)
    print("  prevalence:", json.dumps(sp,ensure_ascii=False))
    gate("STAD ERBB2-amp",sp["erbb2_amp"]["pct"],*ANCHOR["stad_erbb2_amp"])
    gate("STAD MSI-H",sp["msi_h"]["pct"],*ANCHOR["stad_msi_h"])
    gate("STAD EBV",sp["ebv"]["pct"],*ANCHOR["stad_ebv"])
    print(f"  MSI cross-check: sensor>=3.5={sp['msi_h']['n_pos']} vs SUBTYPE_STAD_MSI={sp['msi_h']['subtype_STAD_MSI']} "
          f"agree={sp['msi_h']['agreement_sensor_vs_subtype']}")
    print("=== HNSC labels ===")
    hr,hp,hc=build_hnsc(); write_labels("HEADNECK_HNSC",hr,hc)
    print("  prevalence:", json.dumps(hp,ensure_ascii=False))
    gate("HNSC HPV+",hp["hpv_pos"]["pct"],*ANCHOR["hnsc_hpv_pos"])
    gate("HNSC EGFR-amp",hp["egfr_amp"]["pct"],*ANCHOR["hnsc_egfr_amp"])
    # expected holdout positives (exploratory if < 25)
    est={"STAD":{k:est_holdout_pos(sp[k]) for k in ("erbb2_amp","msi_h","lauren_diffuse","ebv")},
         "HNSC":{k:est_holdout_pos(hp[k]) for k in ("hpv_pos","egfr_amp","grade_high")}}
    print("\n  expected holdout positives (~30% pooled, exploratory if < 25):", json.dumps(est,ensure_ascii=False))
    report={"stad":sp,"hnsc":hp,"anchors":ANCHOR,"est_holdout_pos_30pct":est,
            "note":"EBV/MSI/HPV from the SUBTYPE field. Primary MSI definition = MSI_SENSOR_SCORE>=3.5."}
    (HERE/"sh_labels_prevalence_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("\nwrote sh_labels_prevalence_report.json")

if __name__=="__main__":
    main()
