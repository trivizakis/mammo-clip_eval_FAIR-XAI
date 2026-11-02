# Mammo-CLIP Detection Evaluation with Fairness & Explainable AI (FAIR-XAI)

A specialized deep learning evaluation pipeline for frozen foundation detection and classification models in digital screening mammography, integrating clinical sensitivity calibration, demographic fairness audits, and multi-modal Explainable AI (XAI).

## Core Modules & Architecture
- **`main_detection.py`**: End-to-end evaluation engine supporting balanced cohort sampling, empirical sensitivity calibration, and held-out inference.
- **`MammoDetectorXAI.py`**: Vision detection explainability suite featuring Det-CAM, D-RISE, Separated Localization vs. Classification Saliency, and Mammo-FActOR.
- **`MammoCLIPViTXAI.py`**: Vision Transformer (ViT) attention rollout and token saliency attribution engine.
- **`MammographyXAI.py`**: Traditional gradient and path attribution methods (Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion Sensitivity, Superpixel SHAP).
- **`ViTXAIEnsembler.py`**: Harmonic non-linear ensembling and dual iso-confidence contour extraction (Core Lesion Nidus vs. Peritumoral Margin).
- **`preprocessing_RSNA_MAMO.py`**: Clinical mammography preprocessing (VOI-LUT windowing, MONOCHROME standardization, unsupervised body mask extraction, anatomical MLO pectoral muscle excision, and scanner site standardization).
- **`senAnals.py`**: Clinical decision threshold calibration maximizing $F_{\beta}$ ($\beta = 1.5$) sensitivity on validation distributions.
- **`BIRADSShapeDescriptorExtractor.py`**: Extraction of BI-RADS clinical morphology descriptors from segmented contour outlines.
- **`export_contours_to_geojson.py` & `export_contours_to_dicom_rt.py`**: Clinical export utilities for DICOM-RT and standardized GeoJSON contour structures.
- **`visualize_detector_xai.py` & `visualize_clf_xai.py`**: High-resolution multi-panel diagnostic report rendering engines.

## Usage
```bash
python main_detection.py --help
```
