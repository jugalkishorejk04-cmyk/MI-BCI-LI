# MI-BCI-LI

## Change in Hemispheric Laterality Index (ΔLI) as an EEG Marker of Motor-Imagery BCI Neuroplasticity

Analysis code and derived data for a five-dataset study of within-session
lateralisation dynamics in motor-imagery BCI training.

---

## Summary

Motor-imagery BCI protocols reward the amplitude of event-related
desynchronisation (ERD), treating a stronger cortical response as a better one.
Reorganisation, however, is not cortex working harder but work moving between
hemispheres. This study measures that redistribution directly, as the change in
task-relative hemispheric laterality index across a training session (ΔLI),
using five publicly available datasets (272 participants, 308 session-level
observations; four healthy cohorts and one acute-stroke cohort) processed
through a single harmonised pipeline.

**Main findings**

- ΔLI is unrelated to static laterality and is not explained by ERD magnitude.
- ΔLI carries no information about decodability. Two published predictors of
  BCI performance were implemented and each validated against the outcome it
  was designed for (Blankertz SMR predictor, pooled r = +0.535 against decoding
  accuracy) before being tested against ΔLI, where neither showed any
  association.
- Between-session reliability is low (ICC 0.09–0.30, n = 54), identifying ΔLI
  as a session-level state rather than a stable individual trait.
- In acute stroke, where left/right imagery decoded near chance (0.56), the
  lateralised structure of the ERD change nonetheless tracked stroke severity
  (ρ = −0.436, p = 0.002), surviving four confound controls.

**A note on ERD lateralisation selectivity.** Selectivity and ΔLI are
algebraically two summaries of the same lateralisation trajectory: selectivity
reduces to (N_late − N_early)/4 where N = |ERD_contra| − |ERD_ipsi|, and ΔLI is
the normalised endpoint difference of the same quantity. Their mutual
correlation is convergent measurement, not prediction, and is not interpreted
as an empirical finding. Analyses in this repository that regress one against
the other are superseded and retained only in `archive_superseded/`.

---

## Datasets (not redistributed here)

| Dataset | n | Population | Source |
|---|---|---|---|
| PhysioNet EEGMMIDB | 108 | Healthy | physionet.org/content/eegmmidb |
| Cho 2017 | 52 | Healthy | GigaDB, doi:10.5524/100295 |
| BCI Competition IV-2b | 9 × 5 sessions | Healthy | bnci-horizon-2020.eu/database/data-sets |
| Lee 2019 (OpenBMI) | 54 | Healthy | GigaDB, doi:10.5524/100542 |
| Liu 2024 | 50 | Acute stroke | figshare, doi:10.6084/m9.figshare.21679035 |

All accessed via MOABB v1.5. Raw EEG is not redistributed; each script
downloads from the original source on first run.

---

## Pipeline

Run in order. Scripts 00–02 verify the environment and fetch data.

**Feature extraction**
- `03_compute_features.py` — per-stage ERD, laterality index, selectivity
- `05_compute_coherence.py` — C3–C4 imaginary coherence and wPLI
- `07_compute_trial_constraints.py` — trial-level ERD CV, onset latency SD

**Core analyses**
- `08_dose_response_models.py` — within- and cross-session trajectory models
- `09_clinical_anchoring.py` — NIHSS and hemiplegia-side analysis
- `14_lee_crosssession.py` — two-session reliability (Lee 2019, n = 54)

**Published comparators**
- `17_blankertz_smr.py` — Blankertz SMR predictor
- `18_ahn_theta_alpha.py` — Ahn theta/alpha ratio
- `21_benchmark_fixes.py` — positive controls, comparator-vs-ΔLI tests,
  hierarchical regression against decoding accuracy

**Behavioural and transfer analyses**
- `15_deltali_behavioural_validation.py` — within-session learning slope
- `16_lee_acrossday_validation.py` — across-session decoding change
- `22_model_a_decontaminated.py` — leave-one-dataset-out transfer
- `24_modelB_unbiased.py` — NIHSS-band classification, direct univariate AUC

**Mechanistic tests**
- `29b_denominator_from_perstage.py` — amplitude level and drift as candidate
  explanations for the selectivity/ΔLI sensitivity difference (both null)

**Supplementary**
- `30_supplementary_tables.py` — invalidated candidates, intercorrelation matrix

---

## Derived data files

Per-subject and per-stage feature tables in CSV. Note that Lee 2019 analyses
use `lee2019_features_per_subject.csv` (n = 53); the Lee rows in
`features_per_subject.csv` are from an earlier partial extraction (n = 19) and
should not be used.

---

## Superseded analyses

`archive_superseded/` contains outputs from analyses that were withdrawn during
revision and are not reported in the manuscript:

- Hierarchical regression of ΔLI on selectivity — the two measures are
  algebraically related, so this is not a valid inferential test.
- Classification of responder status from ΔLI-derived features — the features
  are transforms of the target. Rebuilt on independent features, the model
  performs at chance across populations.
- Subject-label permutation test — shuffling subject correspondence collapses
  the correlation for an algebraically dependent pair exactly as it would for
  an independent one, so it does not discriminate between the two cases.

These are retained for transparency, not as supporting evidence.

---

## Environment

Python 3.12.

```
py -3.12 -m pip install -r requirements.txt
```

Key versions: MOABB 1.5, scikit-learn 1.9.0, numpy 2.4.6, pandas 3.0.3,
scipy 1.17.1, MNE-Python. Random seed 42 throughout.

---

## Citation

If you use this code, please cite the associated article. A `CITATION.cff`
file is provided.

## License

[MIT / CC-BY-4.0 — select one and add a LICENSE file]

## Status

Submitted for peer review. Not yet peer-reviewed.
