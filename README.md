# Malicious URL Detection & Threat Profiling Pipeline

An end-to-end machine learning system and analytical pipeline designed to detect phishing and malicious web endpoints using a combination of lexical URL structures and pre-extracted webpage DOM telemetry.

Developed as an academic research and engineering project at **Elizabethtown College**.

---

## Overview

Traditional malicious URL detection relies heavily on static domain blacklists or purely lexical string properties, often failing to detect zero-day phishing campaigns or generating excessive false positives on complex domains. This project evaluates supervised machine learning classifiers on the **PhiUSIIL Phishing URL Dataset** (~235,000 records) to establish high-throughput offline detection.

Beyond standard binary classification (`BENIGN` vs. `MALICIOUS`), this pipeline introduces a secondary **Threat Profiler** that audits underlying DOM and structural telemetry to quantify attack vectors into three distinct severity scores:

* **Credential Harvesting Risk** (presence of password inputs, external form actions, hidden fields, submit buttons)


* **Financial Impersonation Risk** (targeting of banking, payment gateway, and cryptocurrency keywords)


* **Script & Frame Abuse Risk** (abnormal DOM manipulation via code obfuscation, popups, and iframes)



---

## Academic Context

* **Institution:** Elizabethtown College (Department of Computer Science & Department of Information Systems)


* **Course:** CS 250: Foundations of AI


* **Authors:**
* Rommel Sanchez Perez (Dept. of Computer Science)


* Scott Domboski (Dept. of Information Systems)




* **Instructor / Faculty Reviewer:** Dr. Jingwen (Jessica) Wang



---

## Dataset & Preprocessing

The pipeline trains on the **PhiUSIIL Phishing URL Dataset** (Kaggle), which captures both lexical URL features and raw HTML/DOM structural properties across a balanced distribution (57.2% legitimate, 42.8% phishing):

* **Deduplication:** Removal of duplicate URLs to prevent data leakage between train and test splits.


* **Feature Selection:** Dropping high-cardinality metadata identifiers (`FILENAME`, `URL`, `Domain`, `TLD`, `Title`) and multicollinear redundant columns (`NoOfLettersInURL`, `URLTitleMatchScore` with $r > 0.90$).


* **Label Normalization:** The raw dataset encodes $0 = \text{Phishing}$ and $1 = \text{Legitimate}$; the ingestion harness standardizes this via $1 - \text{label}$ so that malicious instances serve as the positive class ($1 = \text{Malicious}$).


* **Matrix Structure:** Standardized into a clean, 48-feature numerical feature matrix exported to `cleaned_data.csv`.



---

## Machine Learning Architecture

The training harness implements stratified 80/20 train/test splitting (`random_state=42`) across three distinct modeling paradigms:

1. **Logistic Regression (Baseline):** Wrapped in a Scikit-Learn `Pipeline` with `StandardScaler` to ensure standardized feature distribution without data leakage across linear decision boundaries.


2. **Random Forest Classifier:** 100-tree ensemble utilizing bootstrap aggregation (bagging) and randomized feature subsets to capture non-linear DOM relationships.


3. **XGBoost Classifier (`XGBClassifier`):** Optimized gradient boosted decision trees minimizing log-loss across deep interaction splits.



### Dual-Verdict Inference & Threat Profiling

For each tested URL, the inference engine produces a two-tier evaluation:

1. **Primary Confidence Rating:** Probability score ($0.0\% - 100.0\%$) compared against a configurable decision threshold (default: $0.50$) to assign a discrete verdict (`MALICIOUS` or `BENIGN`).


2. **Secondary Threat Nature:** Continuous severity percentages calculated by checking active feature activation across categorized telemetry lists:



$$\text{Severity}_{\text{category}} = \left( \frac{\sum (\text{Features}_{\text{category}} > 0)}{\text{Total Features in Category}} \right) \times 100$$

---

## Performance Benchmark & Results

Evaluated across **47,074 held-out test samples**:

| Metric | Logistic Regression (Scaled) | Random Forest | XGBoost |
| --- | --- | --- | --- |
| **Accuracy** | 99.50% | 100.00% | 99.998% |
| **ROC-AUC** | 1.0000 | 1.0000 | 1.0000 |
| **Phishing Precision** | 0.9901 | 1.0000 | 1.0000 |
| **Phishing Recall** | 1.0000 | 1.0000 | 0.99995 |
| **Macro F1-Score** | 0.9950 | 1.0000 | 1.0000 |
| **Test Misclassifications** | 1 False Positive | **0 Errors** | **1 False Negative** |

### Top Predictor Analysis (Feature Importance)

Tree split analysis reveals that classification is anchored by a mix of lexical similarity and structural DOM depth:

* **`URLSimilarityIndex` (~98.0% gain in XGBoost, 19.7% in RF):** Acts as the decisive lexical discriminator; legitimate sites consistently cluster near 100, while deceptive targets scatter widely.


* **`LineOfCode` & `NoOfExternalRef`:** Legitimate production domains host substantial codebases and link out into broad web ecosystems, whereas disposable phishing pages exhibit near-zero code volume and sparse link graphs.


* **`IsHTTPS`:** Highlights transport-layer encryption usage across deceptive hosts.



---

## Repository Structure

```text
├── CS250_finalnotebook.ipynb    # Jupyter Notebook: Data loading, cleaning, & 5 visual EDA studies
├── train.py                     # CLI training harness, model benchmarking, and artifact serialization
├── cleaned_data.csv             # Preprocessed 48-feature dataset (generated via notebook)
├── models/                      # Serialized .joblib artifacts containing model weights + feature orders
│   ├── logistical_regression.joblib
│   ├── random_forest.joblib
│   └── xgboost.joblib
├── figures/                     # Exported EDA and evaluation plots
└── README.md                    # Project documentation

```

---

## Getting Started

### 1. Prerequisites

* Python 3.10+
* Recommended: Virtual environment (`venv` or `conda`)

### 2. Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/Nietch2/URL-Prediction-Model.git
cd malicious-url-detection

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

pip install numpy pandas scikit-learn xgboost joblib matplotlib seaborn

```

### 3. Data Preparation & Exploratory Analysis

Ensure `PhiUSIIL_Phishing_URL_Dataset.csv` is placed in the project root. Launch Jupyter to execute the data cleaning pipeline and generate the five exploratory analysis figures:

```bash
jupyter notebook CS250_finalnotebook.ipynb

```

Running the notebook will audit the raw dataset, generate the distribution/correlation plots, and export `cleaned_data.csv`.

### 4. Training the Pipeline

Train and benchmark all three models, run the top-10 feature importance study, and generate console inference demonstrations:

```bash
python train.py --data cleaned_data.csv --out-dir models/ --threshold 0.5

```

---

## Future Roadmap

* **Streamlit Web Dashboard:** Interactive UI allowing security analysts to inspect arbitrary links, view real-time confidence gauges, and visualize the three threat severity distributions.


* **Transient Headless DOM Crawler:** Integration of containerized browser automation (Playwright/Selenium) to dynamically crawl live URLs, extract the 48-feature vector in an isolated sandbox, and pass live telemetry directly to the serialized `.joblib` model.
