# MI-BCI-LI
# ERD Lateralisation Selectivity as a Cross-Population Predictor of MI-BCI Neuroplasticity

Multi-dataset validation across healthy and stroke populations.
[Author names] — [Affiliation]

## Summary
This study tests whether **ERD lateralisation selectivity** — the rate at
which sensorimotor desynchronisation becomes hemisphere-specific during
motor-imagery BCI training — predicts neuroplastic reorganisation (change
in hemispheric laterality index, ΔLI). Across five independent datasets
(272 participants, 308 sessions), selectivity predicted ΔLI at Spearman
ρ = 0.76–0.96, while non-lateralised activation predicted nothing. A model
trained on healthy participants predicted stroke responder status at
AUC = 0.92.

## Datasets (not included here — download from source)
- PhysioNet EEGMMIDB (108 healthy)
- Cho 2017, GigaDB (52 healthy)
- BCI Competition IV-2b, BNCI Horizon 2020 (9 healthy × 5 sessions)
- Lee 2019, GigaDB/OpenBMI (53 healthy)
- Liu 2024, figshare doi:10.6084/m9.figshare.21679035 (50 acute stroke)

All datasets are accessed via MOABB v1.5. Raw EEG is not redistributed here.

## Pipeline (run in order)
- `03_compute_features.py` — ERD, laterality index, selectivity per subject
- `05_compute_coherence.py` — C3–C4 connectivity
- `06_hierarchical_regression.py` — variance decomposition
- `08_dose_response_models.py` — within/cross-session trajectory models
- `09_clinical_anchoring.py` — NIHSS / hemiplegia-side analysis
- `10_ml_prediction.py` — within- and cross-population classification
- `11_surrogate_test_v2.py` — subject-label permutation (circularity control)

## Key outputs
- `MI_BCI_V8_FINAL.docx` — manuscript
- `Supplementary_Materials.docx` — supplementary tables S1–S10
- `Figure1.png`–`Figure7.png`

## Environment
Python 3.12. Install all dependencies with:
   `py -3.12 -m pip install -r requirements.txt`

## Status
Draft under internal review. Not peer-reviewed.
