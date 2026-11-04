#!/usr/bin/env python3
"""
Cross-cancer MIL training + cost of substitution.

Each endpoint is evaluated under {real, shuffle-null, prevalence-baseline}; shuffle-null and baseline
separate an H&E-blind endpoint (AUROC ~0.5) from a broken MIL pipeline.
histology (LUAD/LUSC) is the positive control (morphological, expected ~0.9).
cost = routing on measured vs predicted treatment axis x frozen_map therapeutic distance.

Inputs: $DATA_ROOT/crosscancer/<cancer>/uni_v1/*_uni_embeddings.npy,
        <cancer>/full/patient_labels.csv, <cancer>/full/split.csv, <cancer>/frozen_map.json
Output: <cancer>/full/mil_cost_results.json (with per-endpoint details). Requires a GPU.
"""
import os, json, csv, sys, argparse, time
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
sys.path.insert(0, str(HERE.parent / "pipeline/modeling/baselines"))

# endpoint -> treatment axis; axis-pair distances come from frozen_map.json
CANCER_CFG = {
 "LUNG_NSCLC": {
   "endpoints": ["histology_lusc", "egfr_activating", "kras_g12c"],   # histology = positive control
   "positive_control": "histology_lusc",
   "route_axis": {"egfr_activating": "antiEGFR", "kras_g12c": "antiKRAS_G12C"},  # in priority order
   "route_default": "chemo",
   "axis_key": {"antiEGFR":"antiEGFR","antiKRAS_G12C":"antiKRAS_G12C","chemo":"chemo"},
 },
 "COLORECTAL": {
   "endpoints": ["braf_v600e"],
   "positive_control": None,
   "route_axis": {"braf_v600e": "antiBRAF"},
   "route_default": "baseline",
   "axis_key": {"antiBRAF":"antiBRAF","baseline":"baseline"},
 },
}

# Foundation model registry (multi-FM robustness). Default UNI.
# emb_dir is relative to DATA_ROOT; {cancer} = cancer name, {lc} = lower-case cancer name.
FM_SPEC = {
    "uni":      {"suffix": "uni",      "dim": 1024, "emb_dir": "crosscancer/{cancer}/uni_v1"},
    "virchow2": {"suffix": "virchow2", "dim": 2560, "emb_dir": "{lc}/virchow2"},
    "uni2h":    {"suffix": "uni2h",    "dim": 1536, "emb_dir": "{lc}/uni2h"},
}
FEATURE_DIM = 1024   # set from --fm in main


def default_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_meta(cancer, fm="uni"):
    d = HERE / cancer / "full"
    labels = {r["case_id"]: r for r in csv.DictReader(open(d/"patient_labels.csv"))}
    split = {r["case_id"]: r["split"] for r in csv.DictReader(open(d/"split.csv"))}
    spec = FM_SPEC[fm]
    suffix = spec["suffix"]
    emb_dir = DATA_ROOT / spec["emb_dir"].format(cancer=cancer, lc=cancer.lower())
    slides = []
    for p in sorted(emb_dir.glob(f"*_{suffix}_embeddings.npy")):
        sid = p.name.replace(f"_{suffix}_embeddings.npy","")
        cid = sid[:12]
        if cid in split:
            slides.append({"slide_id":sid, "case_id":cid, "path":p, "split":split[cid]})
    return labels, split, slides

def bootstrap_auc(y, p, n=1000, seed=42):
    from sklearn.metrics import roc_auc_score
    y=np.array(y); p=np.array(p)
    if len(set(y))<2: return None, None, None
    rng=np.random.default_rng(seed)
    base=roc_auc_score(y,p); boots=[]
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        if len(set(y[idx]))<2: continue
        boots.append(roc_auc_score(y[idx],p[idx]))
    lo,hi=(np.percentile(boots,[2.5,97.5]) if boots else (None,None))
    return round(float(base),4), (round(float(lo),4) if lo is not None else None), (round(float(hi),4) if hi is not None else None)

def train_eval(slides, labels, endpoint, device, shuffle=False, epochs=40, seed=42):
    """Train CLAM-SB and return held-out slide probabilities. Returns (test_slide_records, dev_auc)."""
    import torch, torch.nn as nn
    from attention_mil import CLAMSB
    torch.manual_seed(seed); np.random.seed(seed)
    def rows(split):
        out=[]
        for s in slides:
            lv=labels.get(s["case_id"],{}).get(endpoint,"")
            if s["split"]==split and lv!="":
                out.append((s, int(lv)))
        return out
    tr_all = rows("train")
    hold = rows("val") + rows("test")   # evaluation set = val + test pooled (still site-disjoint)
    if not tr_all or not hold or len({l for _,l in tr_all})<2: return None, None
    # early-stopping dev set: 15% of train patients. Reported metrics use hold (val+test) only.
    pats=sorted({s["case_id"] for s,_ in tr_all})
    rng=np.random.default_rng(seed); rng.shuffle(pats)
    dev_pat=set(pats[:max(1,int(len(pats)*0.15))])
    tr=[(s,y) for s,y in tr_all if s["case_id"] not in dev_pat]
    va=[(s,y) for s,y in tr_all if s["case_id"] in dev_pat]
    te=hold
    if len({l for _,l in tr})<2: return None, None
    if shuffle:
        ys=[l for _,l in tr]; np.random.default_rng(seed).shuffle(ys); tr=[(s,y) for (s,_),y in zip(tr,ys)]
    dev=torch.device(device if torch.cuda.is_available() else "cpu")
    model=CLAMSB(feature_dim=FEATURE_DIM, hidden_dim=512, att_dim=256, dropout=0.25).to(dev)
    opt=torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss()
    def emb(s): return torch.from_numpy(np.load(s["path"]).astype("float32")).to(dev)
    best_va=-1; best_state=None; patience=7; bad=0
    for ep in range(epochs):
        model.train(); order=np.random.permutation(len(tr))
        for i in order:
            s,y=tr[i]; opt.zero_grad()
            logit,_=model(emb(s))
            loss=lossf(logit, torch.tensor([float(y)],device=dev))
            loss.backward(); opt.step()
        # val
        model.eval(); yp=[]; yt=[]
        with torch.no_grad():
            for s,y in va:
                logit,_=model(emb(s)); yp.append(torch.sigmoid(logit).item()); yt.append(y)
        from sklearn.metrics import roc_auc_score
        va_auc=roc_auc_score(yt,yp) if len(set(yt))>1 else 0.5
        if va_auc>best_va: best_va=va_auc; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=patience: break
    if best_state: model.load_state_dict(best_state)
    model.eval(); recs=[]
    with torch.no_grad():
        for s,y in te:
            logit,_=model(emb(s)); recs.append({"case_id":s["case_id"],"proba":torch.sigmoid(logit).item(),"y":y})
    return recs, round(float(best_va),4)

def patient_agg(recs):
    """Slide probabilities -> patient (mean). Returns {case_id: (proba, y)}."""
    by=defaultdict(list)
    for r in recs: by[r["case_id"]].append((r["proba"],r["y"]))
    return {c:(float(np.mean([p for p,_ in v])), v[0][1]) for c,v in by.items()}

def run_endpoint(slides, labels, endpoint, device):
    out={}
    # real
    recs,va=train_eval(slides,labels,endpoint,device,shuffle=False)
    if recs is None: return {"status":"skip(insufficient)"}
    pa=patient_agg(recs); y=[v[1] for v in pa.values()]; p=[v[0] for v in pa.values()]
    auc,lo,hi=bootstrap_auc(y,p)
    out["real"]={"auc":auc,"ci95":[lo,hi],"eval":"holdout(val+test) pooled, site-disjoint",
                 "n_holdout_patients":len(pa),"n_pos":int(sum(y)),"dev_auc":va}
    # shuffle-null
    srecs,_=train_eval(slides,labels,endpoint,device,shuffle=True)
    if srecs:
        spa=patient_agg(srecs); sy=[v[1] for v in spa.values()]; sp=[v[0] for v in spa.values()]
        sa,_,_=bootstrap_auc(sy,sp); out["shuffle_null"]={"auc":sa}
    # prevalence baseline (constant prediction -> AUROC 0.5)
    out["prevalence_baseline"]={"auc":0.5,"note":"constant prediction, 0.5 by definition"}
    out["patient_proba"]={c:round(v[0],4) for c,v in pa.items()}
    out["patient_true"]={c:v[1] for c,v in pa.items()}
    return out

def cost_routing(cancer, ep_results, labels, split):
    cfg=CANCER_CFG[cancer]
    fm=json.loads((HERE/cancer/"frozen_map.json").read_text())
    dist=fm["axis_pair_distance"]  # keys "A__B"
    def dd(a,b):
        if a==b: return 0.0
        for k in (f"{a}__{b}",f"{b}__{a}"):
            if k in dist: return dist[k]["therapeutic_distance"]
        return None
    # endpoints used for routing
    route_eps=list(cfg["route_axis"].keys())
    # measured / predicted axis
    def axis(getter):
        # getter(ep, case) -> 0/1
        def f(case):
            for ep in route_eps:  # in priority order
                if getter(ep, case)==1: return cfg["route_axis"][ep]
            return cfg["route_default"]
        return f
    # measured
    meas_get=lambda ep,c: int(labels.get(c,{}).get(ep,"0") or 0)
    # predicted (proba >= 0.5)
    pred_tbl={ep:ep_results.get(ep,{}).get("patient_proba",{}) for ep in route_eps}
    def pred_get(ep,c):
        v=pred_tbl.get(ep,{}).get(c)
        return int(v>=0.5) if v is not None else 0
    # test patients = patients with predictions
    test_cases=set()
    for ep in route_eps: test_cases|=set(pred_tbl[ep].keys())
    m_axis=axis(meas_get); p_axis=axis(pred_get)
    per=defaultdict(lambda:{"n":0,"mis":0,"cost":0.0}); rows=[]
    for c in sorted(test_cases):
        ma=m_axis(c); pa=p_axis(c); d=dd(ma,pa)
        if d is None: continue
        st=per[ma]; st["n"]+=1; st["cost"]+=d
        if ma!=pa: st["mis"]+=1
        rows.append({"case_id":c,"measured":ma,"pred":pa,"cost":d})
    summary={a:{"n":v["n"],"mean_cost":round(v["cost"]/v["n"],3) if v["n"] else None,
                "misroute_rate":round(v["mis"]/v["n"],3) if v["n"] else None} for a,v in per.items()}
    return {"per_axis":summary, "n_routed":len(rows), "axis_pair_distance":{k:v["therapeutic_distance"] for k,v in dist.items()}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--smoke", action="store_true", help="positive-control endpoint only (quick check)")
    ap.add_argument("--fm", default="uni", choices=list(FM_SPEC),
                    help="foundation model embeddings: uni (default), virchow2, uni2h")
    a=ap.parse_args()
    global FEATURE_DIM
    FEATURE_DIM = FM_SPEC[a.fm]["dim"]      # CLAM-SB feature_dim = FM embedding dim
    cfg=CANCER_CFG[a.cancer]
    labels, split, slides=load_meta(a.cancer, a.fm)
    if a.fm != "uni":
        print(f"[multi-FM] training on {a.fm} embeddings (dim={FEATURE_DIM})")
    print(f"{a.cancer}: {len(slides)} slides with embeddings, {len(labels)} labelled patients")
    eps = [cfg["positive_control"]] if (a.smoke and cfg["positive_control"]) else cfg["endpoints"]
    results={"cancer":a.cancer,"n_slides":len(slides),"claim_level":"hypothesis_only","critic_status":"pending","endpoints":{}}
    for ep in eps:
        t=time.time(); r=run_endpoint(slides,labels,ep,a.device)
        results["endpoints"][ep]=r
        real=r.get("real",{}); sh=r.get("shuffle_null",{})
        print(f"  {ep}: real AUC={real.get('auc')} CI{real.get('ci95')} n+={real.get('n_pos')}/{real.get('n_holdout_patients')} | shuffle={sh.get('auc')} ({time.time()-t:.0f}s)")
    # positive-control check
    pc=cfg["positive_control"]
    if pc and pc in results["endpoints"]:
        pca=results["endpoints"][pc].get("real",{}).get("auc")
        gate = (pca is not None and pca>=0.75)
        results["positive_control_gate"]={"endpoint":pc,"auc":pca,"pass":gate,
            "note":"histology AUROC>=0.75 means the pipeline works; below that the MIL pipeline is suspect."}
        print(f"  [positive control {pc}] AUROC={pca} -> {'PASS' if gate else 'FAIL (check pipeline)'}")
    if not a.smoke:
        # A: targeted-only cost (measured vs predicted axis routing x frozen_map distance)
        results["cost_of_substitution_targeted"]=cost_routing(a.cancer, results["endpoints"], labels, split)
        print("  costA(targeted):", json.dumps(results["cost_of_substitution_targeted"]["per_axis"], ensure_ascii=False))
        # B: per-endpoint misrouting rate, including histology
        mr={}
        for ep, r in results["endpoints"].items():
            pp=r.get("patient_proba",{}); pt=r.get("patient_true",{})
            if not pp: continue
            n=mis=0
            for c,proba in pp.items():
                true=pt.get(c)
                if true is None: continue
                n+=1; mis+=int(int(proba>=0.5)!=true)
            mr[ep]={"n":n,"misroute_rate":round(mis/n,3) if n else None,
                    "auc":r.get("real",{}).get("auc"),
                    "type":"morphology (expected triage)" if ep=="histology_lusc" else "targeted-mutation (expected H&E-blind)"}
        results["endpoint_misroute_incl_histology"]=mr
        print("  costB(mis-route incl histology):", json.dumps(mr, ensure_ascii=False))
    # output name: mil_cost_results.json for UNI, FM suffix otherwise
    fm_tag = "" if a.fm == "uni" else f"_{a.fm}"
    base = "mil_cost_smoke" if a.smoke else "mil_cost_results"
    out=HERE/a.cancer/"full"/f"{base}{fm_tag}.json"
    results["fm"]=a.fm; results["feature_dim"]=FEATURE_DIM
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  wrote {out}")

if __name__=="__main__":
    main()
