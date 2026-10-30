# Comprehensive Submission Guide: Ecological Informatics (Elsevier)

**Journal**: *Ecological Informatics*  
**Publisher**: Elsevier  
**ISSN**: 1574-9541  
**Editor-in-Chief**: Prof. George Arhonditsis, Ph.D. (University of Toronto)  
**Submission Portal**: [https://www.editorialmanager.com/ecoinf/](https://www.editorialmanager.com/ecoinf/)  
**Target Article Type**: Research Paper (Original Research Article)  

---

## 1. Submission Package Contents

All submission files are assembled in the folder:  
`d:\all in one\Desktop\PhD-Project\ecological_informatics_submission\`

| File Name | Item Type on Editorial Manager | Description & Technical Specifications |
| :--- | :--- | :--- |
| **`Cover_Letter_Ecological_Informatics.pdf`** | Cover Letter | Addressed to Prof. George Arhonditsis; summarizes scope, novelty, and compliance. |
| **`Cover_Letter_Ecological_Informatics.docx`** | Cover Letter (Editable) | Microsoft Word editable version of the cover letter. |
| **`highlights.docx`** / **`highlights.txt`** | Highlights | 5 bullet points strictly conforming to Elsevier's $\le 85$ characters/bullet rule. |
| **`graphical_abstract.png`** | Graphical Abstract | Mandatory high-resolution landscape visual summary (4746 × 2196 px, aspect ratio 2.16:1). |
| **`paper.pdf`** | Manuscript (or Primary PDF) | Fully compiled, ready-to-read manuscript with tables, figures, and declarations. |
| **`paper.tex`** | Manuscript (LaTeX Source) | Standard Elsevier `elsarticle` document class source file. |
| **`references.bib`** | LaTeX Source / BibTeX | Clean BibTeX database of all cited literature. |
| **`Declaration_of_Competing_Interest.docx`** | Declaration of Interest | Mandatory formal statement confirming no competing financial or personal interests. |
| **`Suggested_Reviewers.txt`** | Review Preferences | Details of 5 non-conflicted international experts in remote sensing & hydrology. |
| **`Ecological_Informatics_Submission_Package.zip`** | Complete Archive | One-click ZIP containing all manuscripts, figures, and submission documents. |

---

## 2. Manuscript Highlights ($\le 85$ Characters/Bullet)

Elsevier enforces a strict maximum of 85 characters (including spaces) per highlight bullet:

1. `MFPIT fuses multi-resolution Earth observation and hydroclimate data.` *(69 chars)*
2. `Hydrological mass-balance regularization ensures physical consistency.` *(70 chars)*
3. `Multi-fidelity cross-attention resolves environmental scale disparities.` *(72 chars)*
4. `Evaluated on a 23-year spatiotemporal benchmark in the Indus Delta.` *(67 chars)*
5. `Supports wetland conservation, mangrove protection, and flood planning.` *(71 chars)*

---

## 3. Abstract (249 Words) & Keywords (6 Standard Keywords)

### Structured Abstract
> **Background:** Coastal deltaic wetlands are dynamic, ecologically fragile ecosystems threatened by compounding climatic anomalies and reduced upstream freshwater discharges.  
> **Research Gap:** Spatiotemporal water mapping in data-scarce deltas is constrained by severe resolution disparities between remote sensing imagery and hydroclimatological drivers, alongside the tendency of standard machine learning to violate physical conservation principles and overfit static historical priors.  
> **Methodology:** We present the Multimodal Flood Physics-Regularized Transformer (MFPIT), a scientific machine learning framework ($6.58\text{M}$ parameters) coupling 10 m Sentinel-2 multispectral spatial tokens with 4 km hydroclimatological fluxes via directed Multi-Fidelity Cross-Attention (MF-CA) with linear complexity, regularized by an auxiliary surrogate mass-balance loss ($\lambda_{\text{phys}} = 0.1$) and temporal persistence penalty ($\lambda_{\text{temp}} = 0.1$).  
> **Evaluation Protocol:** To prevent historical prior memorization and evaluate true generalization, we establish a target-leakage-prevented matched benchmark ($JRC=0$) over a 23-year archive (2001–2023) in the Indus River Delta, Pakistan, supplemented by disaggregated 2020–2023 monsoon evaluations and a dynamic 2022–2023 monsoon flood reference mask ($y_{\text{dyn}}$).  
> **Main Results:** On the leakage-free benchmark, MFPIT achieves state-of-the-art performance across hard boundary segmentation (Test IoU = 0.7102, F1 = 0.8306 at $\tau = 0.330$) and probabilistic ranking under severe class imbalance (PR-AUC = 0.9141, ROC-AUC = 0.9296), outperforming tree ensembles (GBDT IoU = 0.6872, Random Forest IoU = 0.6839) and deep baselines (U-Net IoU = 0.6489) with superior probability calibration (ECE = 0.0150, Brier = 0.1057). Under the dynamic 2022–2023 monsoon flood mask, static terrain baselines degrade sharply while MFPIT ranks first overall (IoU = 0.6456, F1 = 0.7847).  
> **Physical Consistency:** Mass-balance and temporal regularization reduce water-budget inconsistency by 81.7%–89.6% ($\mathcal{L}_{\text{phys}}$ from 1.3085 to 0.1355–0.2391) and temporal jitter by 86.7%, preventing catastrophic out-of-distribution degradation during the 2022 Pakistan mega-flood (maintaining IoU = 0.6962 and $R^2 = 0.9382$ versus IoU = 0.4967 and $R^2 = 0.2959$ for the unconstrained model, with $R^2 = 0.9564$ across the full test period).  
> **Applications & Impact:** This framework delivers a principled, ecologically regularized decision-support methodology for estuarine conservation, mangrove habitat protection, and coastal flood-resilience planning in vulnerable, data-scarce deltas.

### Keywords
`Ecological informatics; Multi-fidelity data fusion; Physics-informed machine learning; Water occurrence mapping; Spatiotemporal transformers; Wetland conservation`

---

## 4. Suggested Reviewers

1. **Dr. Guy J.-P. Schumann**  
   - Affiliation: RSS-Hydro / University of Bristol (Luxembourg / UK)  
   - Email: `gjpschumann@gmail.com`  
   - Expertise: Remote sensing of floods, SAR-optical integration, hydrodynamic modeling.

2. **Prof. Dr. Chaopeng Shen**  
   - Affiliation: Department of Civil and Environmental Engineering, Pennsylvania State University (USA)  
   - Email: `cshen@engr.psu.edu`  
   - Expertise: Deep learning in hydrology, physics-informed machine learning, differentiable modeling.

3. **Dr. Christian Kuenzer**  
   - Affiliation: Earth Observation Center (EOC), German Aerospace Center (DLR) (Germany)  
   - Email: `christian.kuenzer@dlr.de`  
   - Expertise: River deltas remote sensing, wetland dynamics, time-series satellite analysis.

4. **Dr. Vahid Nourani**  
   - Affiliation: Center of Excellence in Hydroinformatics, University of Tabriz / Near East University  
   - Email: `nourani@tabrizu.ac.ir`  
   - Expertise: AI in hydrology, hydroinformatics, ungauged basins, spatiotemporal modeling.

5. **Dr. Muhammad Adnan**  
   - Affiliation: Institute of Geographical Information Systems (IGIS), NUST / SUPARCO (Pakistan)  
   - Email: `adnan@igis.nust.edu.pk`  
   - Expertise: Indus Basin hydrology, remote sensing of flood disasters, Google Earth Engine.

---

## 5. Step-by-Step Editorial Manager Submission Walkthrough

### Step 1: Login & Start New Submission
1. Go to **[https://www.editorialmanager.com/ecoinf/](https://www.editorialmanager.com/ecoinf/)**.
2. Click **Login** (use your ORCID or Elsevier profile).
3. In the Author Main Menu, click **Submit New Manuscript**.

### Step 2: Select Article Type
- Choose **Research Paper** (or *Full Length Article*).
- Click **Proceed**.

### Step 3: Attach Files
Upload the following files from `ecological_informatics_submission/`:
1. **Cover Letter**: Select `Cover Letter` from dropdown -> Upload `Cover_Letter_Ecological_Informatics.pdf`.
2. **Highlights**: Select `Highlights` -> Upload `highlights.docx`.
3. **Graphical Abstract**: Select `Graphical Abstract` -> Upload `graphical_abstract.png`.
4. **Manuscript**: 
   - Option A (Direct PDF, recommended for fast initial screening under "Your Paper Your Way"): Select `Manuscript` -> Upload `paper.pdf`.
   - Option B (LaTeX source): Upload `paper.tex`, `references.bib`, and image files from the folder.
5. **Declaration of Interest**: Select `Declaration of Interest` -> Upload `Declaration_of_Competing_Interest.docx`.
- Click **Proceed**.

### Step 4: General Information & Classifications
- Select relevant classifications:
  - *Ecological Informatics*
  - *Ecosystem Modeling*
  - *Remote Sensing / GIS*
  - *Pattern Recognition / Machine Learning*
- Click **Proceed**.

### Step 5: Review Preferences (Suggested Reviewers)
- Add the 5 suggested reviewers listed in Section 4 above (providing their names, institutions, countries, and email addresses).
- Click **Proceed**.

### Step 6: Additional Information & Declarations
- **Ethics & Permissions**: Confirm the work does not involve human subjects or animal experimentation.
- **Data Availability**: Confirm public data sources (Sentinel, CHIRPS, TerraClimate, JRC) and state GitHub repository: `https://github.com/mirzamuzzamilbaig/MFPIT`.
- **Funding**: NCBC (National Center in Big Data and Cloud Computing) / NED University of Engineering and Technology.
- **Generative AI Disclosure**: Confirm generative AI was used solely for language and grammar assistance, with full scientific verification by the author.

### Step 7: Manuscript Metadata
- **Full Title**: `Integrating Multimodal Earth Observations and Hydrological Regularization for Spatiotemporal Wetland Water Mapping in the Indus Delta`
- **Abstract**: Paste the 249-word abstract from Section 3 above.
- **Keywords**: Paste the 6 keywords from Section 3 above.
- **Authors**: Confirm Mirza Muhammad Muzzamil as Corresponding Author (`mirzamuzzamil@neduet.edu.pk`).

### Step 8: Build PDF for Approval & Final Submit
1. Click **Build PDF for Approval**.
2. Wait a couple of minutes for the Editorial Manager server to generate the merged PDF.
3. Click **View PDF** to review the merged proof.
4. Click **I accept this submission** and click **Submit to Journal**.
