# Comprehensive Submission Guide: IEEE Access

**Journal**: *IEEE Access* (Multidisciplinary, Open Access, IEEE)  
**ISSN**: 2169-3536  
**Submission Portal**: [https://ieee.atyponrex.com/journal/ieee-access](https://ieee.atyponrex.com/journal/ieee-access) (IEEE Author Portal)  
**Article Type**: Regular Research Article (Original Research)  

---

## 1. IEEE Access Formatting Compliance Checklist

| IEEE Access Requirement | Manuscript Implementation Status |
| :--- | :--- |
| **Document Class & Layout** | Official Two-Column `IEEEtran` (`\documentclass[journal]{IEEEtran}`) with `IEEE Access` running headers and DOI footer banner |
| **Abstract (150–250 Words, Single Paragraph)** | **224 words**, single unstructured paragraph, no citations, all acronyms defined on first use |
| **Index Terms (Alphabetical Order)** | **10 Index Terms** in strict alphabetical order (`Deep learning, Earth observation, flood susceptibility mapping, ...`) |
| **Section Headings** | Standard IEEE Roman Numerals (`I. INTRODUCTION` through `VII. CONCLUSION`) with `\IEEEPARstart{T}{he}` drop-cap |
| **Figures & Tables** | All 9 verified data-driven figures and 12 verified tables formatted for two-column IEEE layout (`table*` / `figure*` and `\columnwidth`) |
| **Bibliography Style** | Official `ibliographystyle{IEEEtran}` numbered `[1]–[37]` (100% verified against Crossref and OpenAlex) |
| **Author Biography (Mandatory in IEEE Access)** | Included at the end of the manuscript (`egin{IEEEbiographynophoto}{Mirza Muhammad Muzzamil}`) |

---

## 2. Files in `ieee_access_submission/`

- **`paper.pdf`** / **`Research_Paper_IEEE_Access.pdf`**: Compiled 2-column IEEE Access PDF manuscript.
- **`paper.tex`** / **`Research_Paper_IEEE_Access.tex`**: Official `IEEEtran` LaTeX source file.
- **`paper.docx`** / **`Research_Paper_IEEE_Access.docx`**: Editable Microsoft Word manuscript formatted for IEEE Access.
- **`references.bib`**: Verified 37-entry BibTeX database.
- **`Cover_Letter_IEEE_Access.pdf`** & **`Cover_Letter_IEEE_Access.docx`**: Cover letter addressed to Prof. Derek Abbott, Editor-in-Chief of *IEEE Access*.
- **`graphical_abstract.png`** & **All 9 Figure Files** (`architecture.png`, `benchmark_iou_bar.png`, `calibration_curves.png`, `loss_curve_fixed.png`, `performance_metrics.png`, `pr_curves.png`, `regression_analysis_fixed.png`, `study_area.jpeg`, `visual_comparison_fixed.png`).
- **`IEEE_Access_Submission_Package.zip`**: Complete ZIP archive of all IEEE Access submission files.

---

## 3. Copy-Paste Metadata for the IEEE Author Portal

### Title
`Integrating Multimodal Earth Observations and Hydrological Regularization for Spatiotemporal Wetland Water Mapping in the Indus Delta`

### Abstract (224 Words — Single Paragraph, IEEE Access Compliant)
> Coastal deltaic wetlands are dynamic, ecologically fragile ecosystems threatened by compounding hydroclimatic anomalies and reduced upstream freshwater discharges. However, spatiotemporal surface water mapping in data-scarce deltas is constrained by severe spatial resolution disparities between multi-sensor satellite imagery and coarse hydroclimatological forcing variables, alongside the tendency of purely data-driven deep learning models to violate physical mass conservation and overfit static historical water occurrence priors. In this paper, we present the Multimodal Flood Physics-Regularized Transformer (MFPIT), a compact (6.58M-parameter) scientific machine learning framework that couples multi-sensor Earth observation sequences with regional hydroclimatological fluxes (precipitation, evapotranspiration, soil moisture, and surface runoff) via a hybrid convolutional-Transformer backbone regularized by a standardized catchment mass-balance constraint (lambda_phys = 0.1) and an inverse-rainfall weighted temporal smoothness penalty (lambda_temp = 0.1). To prevent historical prior memorization, we establish a target-leakage-prevented matched benchmark (JRC = 0) over a 23-year monthly archive (2001–2023; 279 x 502 basin-scale grid at 1 km aggregation; 1,862 annual 12-month training sequences and 196 validation and test sequences) in the Lower Indus Basin and Delta, Pakistan, supplemented by a dynamic 2022–2023 monsoon inundation sensitivity benchmark. On the leakage-free holdout, MFPIT (B = 8) achieves state-of-the-art boundary segmentation (Test IoU = 0.7102, F1 = 0.8306 at tau* = 0.330), probabilistic ranking (PR-AUC = 0.9141, ROC-AUC = 0.9296), and probability calibration (ECE = 0.0150, Brier = 0.1057), outperforming Gradient Boosted Decision Trees (IoU = 0.6872), Random Forest (IoU = 0.6839), U-Net (IoU = 0.6489), and CNN-LSTM (IoU = 0.5764), while also ranking first on the dynamic 2022–2023 flood expansion mask (IoU = 0.6456). Joint physical and temporal regularization reduces test-set mass-balance residuals by 81.7%–89.6% and temporal state jitter by 86.7%, achieving a patch water-fraction R² of 0.9564 (MAE = 0.0458) and preventing out-of-distribution collapse during the catastrophic 2022 Pakistan mega-flood (IoU = 0.6962 versus 0.4967 unconstrained).

### Index Terms (Alphabetical)
`Deep learning, Earth observation, flood susceptibility mapping, hydrological mass balance, Indus River Delta, multi-fidelity data fusion, physics-informed machine learning, remote sensing, spatiotemporal transformers, wetland conservation`
