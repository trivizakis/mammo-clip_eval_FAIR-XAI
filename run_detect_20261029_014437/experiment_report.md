# Mammo-CLIP Default Uncalibrated Detection Report (Balanced 50–50% Cohort)

> [!NOTE]
> **Experiment Type**: Frozen Vision Backbone Detection on Balanced 50–50% Cohort at Default Threshold ($T = 0.5000$)  
> **Foundation Backbone**: Mammo-CLIP Vision Encoder (`breast_clip_b5`, `b5-model-best-epoch-7.tar`)  
> **Date**: October 29, 2026  
> **Evaluation Cohort**: Balanced RSNA Screening Mammography ($N = 2,316$ exams: 1,158 malignant vs. 1,158 benign)

---

## 1. Executive Summary

This report documents the evaluation of the **default uncalibrated Mammo-CLIP** detection model (`breast_clip_b5`) on a **strictly balanced 50–50% evaluation cohort** derived from the local RSNA Screening Mammography dataset:
- **Cohort Composition**: Balanced 50–50% dataset ($N = 2,316$ exams):
  - **All Malignant Exams**: $N = 1,158$ biopsy-proven cancer cases (100% of all malignant exams in RSNA).
  - **Sampled Benign Exams**: $N = 1,158$ benign cases, matched 1:1.
  - **True Prevalence**: $50.0\%$ to establish unbiased sensitivity and specificity baselines.
- **Default Decision Threshold**: $T = 0.5000$ (uncalibrated, frozen zero-shot detection).
- **Separated Ground-Truth XAI Organization**: Visual reports and GeoJSON contour polygons are cleanly partitioned into:
  - [`rsna_xai_outputs/malignant/`](rsna_xai_outputs/malignant/) (True cancer reports + GeoJSON files)
  - [`rsna_xai_outputs/benign/`](rsna_xai_outputs/benign/) (True benign reports + GeoJSON files)

---

## 2. Test Cohort Performance ($N = 2,316$)

Evaluating the uncalibrated model at standard threshold $T = 0.5000$ on the 50–50% balanced cohort yielded the following results:

| Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **Area Under ROC Curve (AUC)** | **0.7302** | Continuous discrimination on balanced 50–50% cohort |
| **Balanced Accuracy (BACC)** | **65.59%** | Unweighted mean of Sensitivity and Specificity |
| **Specificity** | **81.43%** | Correctly ruled out 943 of 1,158 true benign exams |
| **Sensitivity (Recall)** | **49.74%** | Detected 576 of 1,158 true malignant exams |
| **Overall Accuracy** | **65.59%** | Identical to BACC due to 50–50% balance |
| **Precision** | **72.82%** | Positive predictive value (576 TP / [576 TP + 215 FP]) |
| **Clinical F-1.5 Score** | **0.5284** | Weighted harmonic mean ($\beta = 1.5$) |
| **Standard Decision Threshold**| **0.5000** | Default uncalibrated cutoff |

### Confusion Matrix
- **True Positives (TP)**: `576`
- **True Negatives (TN)**: `943`
- **False Positives (FP)**: `215`
- **False Negatives (FN)**: `582`

---

## 3. Subgroup Fairness & Demographic Audit

| Demographic / Clinical Group | Stratum | Total Exams | True Malignant | Balanced ACC | ROC AUC | Specificity | Sensitivity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Age Group** | 20–40 | 18 | 9 | 0.6111 | **0.7407** | 0.8889 | 0.3333 |
| **Age Group** | 40–55 | 650 | 257 | 0.6558 | **0.7289** | 0.8168 | 0.4942 |
| **Age Group** | 55+ | 1,648 | 892 | 0.6548 | **0.7275** | 0.8122 | 0.4978 |
| **Acquisition Site**| Site 1 | 1,291 | 749 | 0.6860 | **0.7554** | 0.8155 | 0.5567 |
| **Acquisition Site**| Site 2 | 1,025 | 409 | 0.6206 | **0.6976** | 0.8133 | 0.4279 |
| **Breast Density** | Density A (Fatty) | 129 | 67 | 0.6508 | **0.6800** | 0.8710 | 0.4328 |
| **Breast Density** | Density B (Scattered) | 609 | 359 | 0.6993 | **0.7583** | 0.8320 | 0.5655 |
| **Breast Density** | Density C (Heterogeneous) | 519 | 297 | 0.6698 | **0.7431** | 0.8243 | 0.5152 |
| **Breast Density** | Density D (Extremely Dense) | 48 | 26 | 0.6486 | **0.7290** | 0.8182 | 0.4615 |
| **Breast Density** | Density Unknown | 1,011 | 409 | 0.6202 | **0.6975** | 0.8123 | 0.4279 |

---

## 4. Explainability Pipeline & Diagnostic Audits

Explainability maps are generated with [MammoDetectorXAI.py](../MammoDetectorXAI.py) into ground-truth separated directories:
- **Malignant Cases**: [`rsna_xai_outputs/malignant/`](rsna_xai_outputs/malignant/)
- **Benign Cases**: [`rsna_xai_outputs/benign/`](rsna_xai_outputs/benign/)

### Key Experiment Artifacts
- **Inference Predictions**: [`results/inference_predictions.csv`](results/inference_predictions.csv)
- **Inference Metrics Summary**: [`results/inference_metrics.csv`](results/inference_metrics.csv) | [`results/inference_metrics.json`](results/inference_metrics.json)
- **Subgroup Metrics Summary**: [`results/subgroup_metrics.csv`](results/subgroup_metrics.csv) | [`results/subgroup_metrics.json`](results/subgroup_metrics.json)
- **Site Parameters**: [`results/site_stats.json`](results/site_stats.json)
