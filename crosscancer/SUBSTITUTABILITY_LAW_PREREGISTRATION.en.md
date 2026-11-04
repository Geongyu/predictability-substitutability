English translation of the sealed document SUBSTITUTABILITY_LAW_PREREGISTRATION.md (original, Korean, unchanged).

# Preregistration — Morphological-Correlate Substitutability Law

> **Preregistered: 2026-07-12, before seeing the lung (LUNG) MIL results.** Breast and colorectal results were already observed (marked "already observed" below). For lung, only embedding was in progress (about 844/1053) and **no routing/subtype AUROC had been computed yet** → the lung predictions are the true held-out test of this law.
> Purpose: establish this law not as a post-hoc description but as **a law that predicts observations in advance**, raising it to discovery level (the single largest IF lever). The commit hash seals the timing.

> **Terminology**: the formal term is "morphological correlate"; for the definition and rationale see `../../research/paperC-positioning/PAPER_DIRECTION.md` §Terminology. The state of not being visible = "has no morphological correlate". "Silent/quiet" is not used ("silent" is confused with synonymous mutations).

## Law (falsifiable proposition)
Whether H&E can replace molecular testing in treatment routing is determined by whether the molecular alteration **has a morphological correlate visible at H&E resolution**.

- **Substitutable (low cost)**: alterations that change tissue morphology or are strongly coupled to a morphological phenotype — expression/architecture-coupled markers (ER ↔ glandular architecture, nuclear morphology), immune-coupled markers (MSI ↔ TILs, mucin, medullary type), lesions that define histologic type.
- **Molecular test required (high cost)**: alterations with no morphological correlate — point mutations without a morphological signature, amplifications not reflected in H&E architecture (HER2 amplification, RAS point mutation).
- **Graded/partial (intermediate)**: alterations with a partial morphological correlate (lung EGFR coupling to lepidic/acinar).

## Prior classification by marker (all markers, fixed before seeing results)
| Cancer | Marker | Prior class | Rationale (morphological correlate) | Status |
|---|---|---|---|---|
| Breast | ER | Substitutable | Coupled to glandular architecture and nuclear grade | Already observed (internal 0.901) ✓ consistent with law |
| Breast | HER2 (amplification) | **Required** | No morphological correlate | Already observed (internal 0.599) ✓ consistent |
| Colorectal | MSI-H | Substitutable | TILs, mucin, medullary type | Already observed (0.918) ✓ consistent |
| Colorectal | all-RAS (point mutation) | **Required** | No morphological correlate (downstream proxy only) | Already observed (0.705, misassignment 0.416) ✓ consistent |
| Colorectal | BRAF-V600 | Graded | Partially visible via co-occurring serrated/MSI features | Already observed (0.882) ✓ consistent |
| **Lung** | **Histologic type LUAD/LUSC** | **Substitutable (positive control)** | Morphology itself | **held-out → predicted AUROC ≥ 0.93** |
| **Lung** | **EGFR activating mutation** | **Graded** | lepidic/acinar high, solid/mucinous low | **held-out → predicted AUROC 0.75–0.89, not near-random** |
| **Lung** | **KRAS-G12C (point mutation)** | **Required** | No morphological correlate | **held-out → predicted AUROC ≤ 0.65, lower than EGFR** |
| **Lung** | **Transcriptomic subtype TRU** | **Substitutable (partial)** | TRU ↔ lepidic coupling | **held-out → TRU has the highest AUROC among subtypes** |

## Prior classification by marker — gastric STAD and head and neck HNSC (added 2026-07-12, sealed before results)
| Cancer | Marker | Prior class | Rationale (morphological correlate) | Status |
|---|---|---|---|---|
| Gastric STAD | **HER2/ERBB2-amp → trastuzumab (ToGA)** | **Required** | Same as breast HER2 — amplification but no morphological correlate | **held-out → AUROC ≤ 0.65** |
| Gastric STAD | MSI-H → pembrolizumab | Substitutable | TILs, medullary type, lymphoepithelioma-like | held-out → AUROC ≥ 0.82 |
| Gastric STAD | Lauren histologic type (intestinal/diffuse/signet-ring) | Substitutable (positive control) | signet-ring and diffuse = strong H&E morphology | held-out → AUROC ≥ 0.85 |
| Gastric STAD | EBV | Substitutable (partial) | lymphoepithelioma-like carcinoma morphology | held-out (exploratory) |
| Head and neck HNSC | **HPV status → treatment de-escalation/prognosis** | **Substitutable** | Non-keratinizing, basaloid morphology (a new kind of visible axis = virus) | **held-out → AUROC ≥ 0.80** |
| Head and neck HNSC | EGFR (amplification/overexpression) → cetuximab | Graded/required | Largely no morphological correlate | held-out → AUROC ≤ 0.70, lower than HPV |
| Head and neck HNSC | grade/histologic type | Substitutable | Differentiation = H&E | held-out → high |

## Held-out predictions — gastric and head and neck (sealed)
5. **Gastric HER2-amp is blind (≤0.65)** — replicates breast HER2 (observed 0.599) as *the same alteration in a different organ*. **The strongest cross-organ test of the law** (if an amplification marker is H&E-blind in both organs, the principle "amplification ≠ morphology" is established).
6. Gastric MSI-H is substitutable (≥0.82) and Lauren histologic type is a positive control (≥0.85) → within-gastric order **histologic type ≥ MSI > HER2-amp**.
7. **Head and neck HPV is visible (≥0.80)** — a *viral* axis (not a mutation) with a morphological correlate, so it tests an extension of the law's "morphological correlate" clause to a new kind of axis. Predicted to be higher than EGFR (≤0.70): order **HPV > EGFR**.

## Held-out predictions (lung, sealed)
1. Lung histologic type (LUAD/LUSC), as a positive control, has **AUROC ≥ 0.93**.
2. Lung EGFR is **graded: 0.75–0.89**, not near-random (≤0.6) — because of a partial morphological correlate.
3. Lung KRAS-G12C (point mutation, no morphological correlate) is **lower than EGFR and ≤ 0.65** — predicts the **order** of the two alterations (EGFR > KRAS).
4. Among lung transcriptomic subtypes, **TRU has the highest AUROC** (coupled to lepidic morphology).

## Falsification criteria (when the law is wrong)
- If a marker classified as "required" (KRAS-G12C) is predicted well (e.g. AUROC ≥ 0.8), or an axis classified as "substitutable" is blind (≤ 0.6), the law fails.
- If the order EGFR ≤ KRAS is reversed, the graded hypothesis fails.
- In that case the law is discarded or restated with narrower conditions (no post-hoc rescue — failure is reported as failure).

## Clarification of interpretation level (added 2026-07-12 — held-out predictions unchanged; response to recent spatial/multi-omics objections)
> This section does not change the sealed predictions (above). It clarifies at which level the law makes its claim, so that it is not falsified if high-resolution technologies such as scRNA/spatial/multiome later find correlates.

- **The law is a "substitutability" claim, not a claim of "absence of a biological correlate".** Precisely, it is an empirical and clinical boundary: "is the alteration predictable **at the resolution of deployable H&E + foundation models**". "Has no morphological correlate (silent)" does not mean "there is no correlate at any resolution" but "**not accessible with current H&E/FMs**".
- Therefore **even if high-resolution technologies later find fine spatial correlates of HER2-amp/RAS-mut, the law is not falsified.** That would either (a) reconfirm that the correlate is not accessible from H&E (boundary holds), or (b) only suggest that a better H&E model may someday access it (a future shift of the boundary). What we draw is **the boundary at which cheap H&E can replace standard testing in today's clinic**.
- **Weakest point = the colorectal RAS no-correlate ST mechanism** (point mutation, so it goes through the MAPK/EREG proxy). If multi-omics (spatial + genotype simultaneously) showed a direct morphology-coupled signature in RAS-mut spots, this *mechanism* could gain nuance → so this half is claimed only as **exploratory/proxy** and the paper does not rest on it.
- **Some high-resolution results actually strengthen our case.** CMS has large intratumoral spatial heterogeneity (Valdeolivas 2024), and bulk slide labels average it out. Spatial data may show that H&E **tracks local molecular state better** than bulk labels suggest (room for higher predictive fidelity).

## Caveats (honesty)
- Prediction itself is not novel (imCMS, Kather, Coudray). The law is **a prediction of "where substitution is possible"**, not an advance in predictive performance.
- As the CPTAC domain collapse shows, "substitutable (low cost)" is also **conditional on domain robustness** (it can collapse under stain and feature-space shift; Fernandez-Romero 2026 explains 80% of the variance). Therefore the low-cost verdict is an in-domain condition, and cross-domain application requires calibration and abstention (decision-system layer).
- claim_level: hypothesis_only.
