# DATASET — Cardiovascular Disease Risk Prediction

## 1. Source

- **Name:** Cardiovascular Disease dataset  
- **Original author:** S. Ulianova  
- **URL:** <https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>  
- **Accessed:** November 2025  

The original dataset contains **70,000** records from routine health examinations.  
Each record describes one adult patient with **11 input features plus one binary target** indicating whether the patient has been diagnosed with cardiovascular disease (`cardio`).

In this course project, we **do not upload the raw Kaggle file** to this repository.  
Instead, we:

- download the raw CSV from Kaggle locally,  
- perform cleaning and feature engineering,  
- save only the **processed and split** CSV files into the `data/` folder  
  (e.g., `cardio_exp0_train.csv`, …, `cardio_exp4_test.csv`).

---

## 2. License and Intended Use

The dataset license is specified on its Kaggle page (see the **“License”** section there).

We follow these principles:

- We use the data **only** for this university **course project** on data mining.  
- We **do not redistribute** the raw dataset; instead we refer users back to the Kaggle link above.  
- We do not attempt to re-identify individuals or link records to any external personal data.

Students or readers who want to re-run our experiments should:

1. Obtain the dataset directly from Kaggle under the corresponding license terms.  
2. Follow the preprocessing steps documented below to reconstruct our cleaned dataset and splits.

---

## 3. Raw Data Description

### 3.1 Population and size

- **Number of records:** 70,000  
- **Task:** binary classification – predict risk/diagnosis of cardiovascular disease.  
- **Target column:** `cardio`  
  - `0`: no documented cardiovascular disease  
  - `1`: presence of cardiovascular disease  

### 3.2 Original features

The original Kaggle table includes the following columns (11 input features + 1 target):

| Column       | Type  | Description (raw)                                                                 | Role     |
| ------------ | ----- | --------------------------------------------------------------------------------- | -------- |
| `age`        | int   | Age in **days**                                                                  | Feature  |
| `gender`     | int   | Biological sex; `1` = female, `2` = male                                         | Feature  |
| `height`     | int   | Height in centimeters                                                            | Feature  |
| `weight`     | float | Weight in kilograms                                                              | Feature  |
| `ap_hi`      | int   | Systolic blood pressure (mmHg)                                                   | Feature  |
| `ap_lo`      | int   | Diastolic blood pressure (mmHg)                                                  | Feature  |
| `cholesterol`| int   | Serum cholesterol category: `1` = normal, `2` = above normal, `3` = well above normal | Feature  |
| `gluc`       | int   | Fasting glucose category: `1` = normal, `2` = above normal, `3` = well above normal | Feature  |
| `smoke`      | int   | Current smoker: `0` = no, `1` = yes                                              | Feature  |
| `alco`       | int   | Regular alcohol intake: `0` = no, `1` = yes                                      | Feature  |
| `active`     | int   | Physical activity: `0` = no, `1` = yes                                           | Feature  |
| `cardio`     | int   | Cardiovascular disease label: `0` = no, `1` = yes                                | **Target** |

The dataset contains **no missing values** in these fields in its raw Kaggle version;  
most anomalies come from **out-of-range values**, not from explicit `NaN`s.

---

## 4. Derived Features Used in This Project

In addition to the original features, we construct two simple, medically interpretable derived features:

1. **Age in years**  
   - Column name: `age_years`  
   - Definition: `age_years = age / 365.25`  
   - Motivation: age in days is not intuitive; converting to years makes interpretation and reporting easier.

2. **Body Mass Index (BMI)**  
   - Column name: `bmi`  
   - Definition: `bmi = weight / ( (height / 100)^2 )`  
   - Motivation: BMI is a standard aggregated risk factor combining height and weight, widely used in cardiovascular risk models.

Both `age_years` and `bmi` are added as **additional numeric features**.  
We keep the original `age`, `height`, and `weight` columns in case other models or analyses need them.

---

## 5. Data Cleaning and Preprocessing

Starting from the raw Kaggle CSV, we perform the following cleaning steps **before** creating any train/validation/test splits:

1. **Remove duplicate rows**  
   - Exact duplicate records are dropped to avoid over-weighting identical patients.

2. **Filter out physiologically impossible values**  
   We apply simple, conservative sanity checks based on clinical plausibility (not strict clinical guidelines), for example:

   - **Age:** keep only adults in a reasonable range, e.g. 30–80 years (converted from days).  
   - **Height:** keep observations in a plausible human range (e.g. 120–220 cm).  
   - **Weight:** keep a broad but realistic range (e.g. 40–200 kg).  
   - **Blood pressure:**  
     - systolic `ap_hi` roughly within 80–250 mmHg;  
     - diastolic `ap_lo` roughly within 40–180 mmHg;  
     - enforce `ap_hi >= ap_lo`.  

   Records outside these ranges are treated as data entry errors and removed.

3. **Convert age from days to years**  
   - Add `age_years` as defined above; we keep the original `age` column for traceability.

4. **Construct BMI**  
   - Add `bmi` using the formula above.  
   - We do not apply additional clipping to BMI; extremely high or low BMI values are already largely filtered out by the earlier height/weight sanity checks.

5. **Ensure integer encoding for categorical features**  
   - We keep the original integer coding from Kaggle for `gender`, `cholesterol`, `gluc`, `smoke`, `alco`, and `active`.  
   - We do not perform one-hot encoding in the CSV files; any further encoding is handled inside the model code in `src/`.

After these steps, we obtain a **cleaned dataset** with slightly fewer than 70,000 rows  
(due to dropped duplicates and out-of-range values).  
All subsequent splits are generated from this cleaned version.

---

## 6. Train / Validation / Test Splits

To support reproducible experiments and fair model comparison, we pre-generate multiple random splits and save them into the `data/` folder.

### 6.1 Overall strategy

- We first shuffle the **cleaned** dataset with a fixed random seed (`random_state = 42`).  
- We perform **stratified splitting by the target label `cardio`** so that the positive/negative ratio is similar across all splits.  
- For each experiment index `expk` (k = 0, 1, 2, 3, 4), we create **three non-overlapping subsets**:
  - **Training set:** ~60% of the cleaned data  
  - **Validation set:** ~20% of the cleaned data  
  - **Test set:** ~20% of the cleaned data  

This yields **5 independent stratified splits** with approximately 60/20/20 proportions.

Each split is stored in three CSV files:

- `data/cardio_exp{k}_train.csv`  
- `data/cardio_exp{k}_val.csv`  
- `data/cardio_exp{k}_test.csv`  

for `k = 0, 1, 2, 3, 4`.

### 6.2 Columns in the split files

Each split CSV contains the same set of columns:

- All original features:  
  `age`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`,  
  `cholesterol`, `gluc`, `smoke`, `alco`, `active`, `cardio`  

- Two derived features:  
  `age_years`, `bmi`

There are **no missing values** in these split files;  
any rows with unrealistic values have already been removed in the cleaning stage.

---

## 7. Ethical and Practical Considerations

- The dataset is de-identified; there are **no direct personal identifiers** such as name, address, or exact dates of visits.  
- Nevertheless, this is still **health data**. We:
  - use it only for teaching and research in this course context,  
  - do not attempt any form of re-identification,  
  - do not combine it with external data sources at the individual level.  

- Our models are meant as **screening / risk stratification tools** for educational purposes,  
  not as clinically validated diagnostic systems. Any real-world deployment would require:
  - retraining on local hospital data,  
  - calibration and external validation,  
  - approval by relevant clinical and ethical committees.

---

## 8. How to Reproduce Our Data Processing

To reproduce the processed data and splits from the Kaggle raw CSV:

1. Download the raw dataset from Kaggle using the link in Section 1.  
2. Run the data preprocessing script (see `src/` in this repository) which:
   - cleans the raw table as described in Section 5,  
   - constructs `age_years` and `bmi`,  
   - generates stratified train/val/test splits with a fixed random seed,  
   - writes the five experiment splits into the `data/` folder as  
     `cardio_exp{k}_train/val/test.csv` (k = 0,…,4).

This ensures that results obtained by other students or instructors are **directly comparable**  
to the ones reported in our project.

---

## 9. Suggested Citation

If you use this dataset and processing in your own work, please cite both the original Kaggle dataset and this course project:

- **Kaggle dataset:**

  > Ulianova, S. (2021). *Cardiovascular Disease Dataset* \[Data set\]. Kaggle.  
  > Retrieved from <https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>

- **Our processed version (example):**

  > Group X (2025). *Cardiovascular risk prediction from routine health examination data*  
  > (course project, Data Mining in Frontier Application Domains, SCUT).


