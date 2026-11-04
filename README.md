# Predictability Is Not Substitutability

**A cost-of-substitution framework for H&E-based molecular prediction across five cancers**

Geongyu Lee<sup>1,2,\*,†</sup>, Ka-Kyung Kim<sup>1,\*</sup>, Sejin Park<sup>1,3</sup>, Jaemyun Lyu<sup>1,4</sup>, Jeong-Han Seo<sup>1,5,6</sup>, Yong Gi Ji<sup>1,7</sup>

<sup>1</sup>Pseudo Lab; <sup>2</sup>OmixAI Co. Ltd.; <sup>3</sup>Seegene Inc.; <sup>4</sup>R&D Center, Genolution Inc.; <sup>5</sup>ROKIT Genomics; <sup>6</sup>Department of Medical Science, University of Ulsan College of Medicine, Asan Medical Center; <sup>7</sup>Quantum C&S
<sup>\*</sup>Equal contribution · <sup>†</sup>Corresponding author: rjsrb365@gmail.com

Presented at **BIOINFO / GIW ISCB-Asia 2026**, Seoul, 17–20 November 2026 (poster P082). Poster and figures: [`poster/`](poster/).

## Overview

AUROC shows that a molecular marker is predictable from H&E. It does not show that the prediction can replace the molecular test. This repository contains the code and result files for a framework that evaluates H&E-based molecular prediction by the treatment assignment it would cause.

- **Cohorts:** TCGA lung (LUAD/LUSC), gastric (STAD), head and neck (HNSC), colorectal (COAD/READ) and breast (BRCA, with CPTAC-BRCA as external test); TCGA-UCEC as an exploratory extension.
- **Pipeline:** 256 × 256 tiles at 20× (Otsu tissue mask, up to 5,000 tiles per slide) → UNI v1 embeddings (1,024-d) → CLAM attention MIL, evaluated on a site-disjoint hold-out.
- **Controls:** 5-seed label-shuffled null, prevalence and pixel-mean baselines, histology-only baseline, bootstrap confidence intervals.
- **Audits:** tissue-source-site × label association (Cramér's V); repeat with UNI2-h and Virchow2.
- **Cost of substitution (breast):** H&E-predicted and measured markers are routed through the same pre-specified treatment rule; disagreements are counted as misassignments and weighted by treatment distance.

## Installation

Python 3.10 or later.

```bash
git clone <this repository>
cd <repository>
pip install -r requirements.txt
```

OpenSlide must be installed on the system for `openslide-python` (e.g. `brew install openslide` or `apt install openslide-tools`).

## Data

No slides, embeddings or model weights are included.

| Data | Source |
|---|---|
| TCGA diagnostic slides (`-DX`) | [GDC Data Portal](https://portal.gdc.cancer.gov/) |
| TCGA molecular labels | [cBioPortal](https://www.cbioportal.org/) PanCancer Atlas studies (fetched by the label scripts) |
| CPTAC-BRCA slides and clinical labels | [TCIA](https://www.cancerimagingarchive.net/) / cBioPortal `brca_cptac_2020` |
| Cell-line drug response (frozen drug map) | [GDSC2](https://www.cancerrxgene.org/) `GDSC2_fitted_dose_response_27Oct23.xlsx` |
| Foundation models | [UNI](https://huggingface.co/MahmoodLab/UNI), [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h), [Virchow2](https://huggingface.co/paige-ai/Virchow2) (gated; accept the licence and set `HF_TOKEN`) |

External data are read from `DATA_ROOT` (default `./data`):

```bash
export DATA_ROOT=/path/to/data
export HF_TOKEN=hf_...
```

Patient labels, site-disjoint splits and embedding manifests used in the study are included under `crosscancer/<COHORT>/full/`. Embedding paths in the manifests are relative to `DATA_ROOT`.

## Usage

### 1. Labels and splits

```bash
python crosscancer/fetch_labels.py        # lung, colorectal
python crosscancer/sh_fetch_labels.py     # gastric, head and neck
python pipeline/data/build_ucec_labels.py # endometrial
python crosscancer/make_split.py          # site-disjoint patient split (70/15/15 by TSS)
python pipeline/data/build_cptac_labels.py
```

### 2. Tiling and embedding

```bash
# cross-cancer cohorts: GDC download -> tiling -> UNI embedding
python crosscancer/run_embed_crosscancer.py --cancers LUNG_NSCLC COLORECTAL

# any slide list
python pipeline/embedding/scripts/run_batch_embedding.py \
  --manifest slides.csv --config pipeline/embedding/configs/tile_config.yaml \
  --tile_dir $DATA_ROOT/tiles --embedding_dir $DATA_ROOT/embeddings \
  --output_manifest $DATA_ROOT/embedding_manifest.csv --embedding_model uni
```

UNI2-h and Virchow2 embeddings: `pipeline/embedding/scripts/extract_uni2h.py`, `extract_virchow2.py`.

### 3. MIL training and evaluation

```bash
python crosscancer/run_mil_cost.py --cancer LUNG_NSCLC          # lung, colorectal
python crosscancer/sh_mil_cost.py  --cancer HEADNECK_HNSC       # gastric, head and neck
python crosscancer/sh_mil_cost.py  --cancer GASTRIC_STAD --fm virchow2
python crosscancer/run_cohort_pipeline.py --cohort UCEC --config crosscancer/cohorts/UCEC.yaml
```

Results are written to `crosscancer/<COHORT>/full/mil_cost_results*.json`.

### 4. Controls and audits

```bash
python crosscancer/sh_robustness_5seed.py --cancer HEADNECK_HNSC   # 5-seed label-shuffled null
python crosscancer/sh_baselines.py --cancer LUNG_NSCLC             # pixel-mean baseline
python crosscancer/sh_subtype_baseline.py                          # histology-only baseline (lung)
python crosscancer/site_audit/site_batch_audit.py --cohort LUNG_NSCLC \
  --anchors histology_lusc,egfr_activating,kras_g12c --out crosscancer/site_audit/site_audit_LUNG_NSCLC.json
python crosscancer/slide_robustness/slide_region_robustness.py
```

### 5. Breast cost of substitution

```bash
python pipeline/modeling/scripts/train_mil.py --config pipeline/modeling/configs/baseline_her2_status_clam.yaml
python pipeline/modeling/scripts/make_indexed_preds.py
python breast/cost/build_frozen_map.py
python breast/cost/compute_cost.py
python breast/cost/remediate_patient_ci.py
python breast/cost/make_fig3_axis_cost.py
```

## Repository structure

```
pipeline/        tiling, embedding, MIL models, breast data manifests
crosscancer/     cross-cancer labels, splits, MIL runs, controls, site audit, results per cohort
breast/          frozen drug map, treatment routing and substitution cost, pCR analysis
poster/          GIW 2026 poster (PDF) and figures
seal/            commit manifest and timeline check for the pre-specified predictions
```

`crosscancer/SUBSTITUTABILITY_LAW_PREREGISTRATION.md` is the pre-specified prediction document (Korean original; English translation in `.en.md`). Its commit history is in the original project repository, `biospin/BioProject02`, commits `77c0633` to `f4b589d`. `seal/check_seal_timeline.py --repo <clone of biospin/BioProject02>` compares those commit times with the result commits.

## Citation

```bibtex
@inproceedings{lee2026substitutability,
  title     = {Predictability Is Not Substitutability: A Cost-of-Substitution Framework for {H\&E}-Based Molecular Prediction Across Five Cancers},
  author    = {Lee, Geongyu and Kim, Ka-Kyung and Park, Sejin and Lyu, Jaemyun and Seo, Jeong-Han and Ji, Yong Gi},
  booktitle = {BIOINFO / GIW ISCB-Asia 2026},
  year      = {2026},
  address   = {Seoul, Korea}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgement

This research was supported by the "Open Source AI·SW Developer and Community Support Program" funded by the Ministry of Science and ICT (MSIT), Republic of Korea. Conducted as an independent Pseudo Lab project.
