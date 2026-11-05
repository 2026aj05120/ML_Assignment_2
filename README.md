# Salary Band Classifier Lab

## a. Problem statement
Predict which salary band (Low / Mid / High, three equally sized groups of USD mid-point salary) a job posting belongs to, using the posting's country, role family, seniority, industry, listed skills and job-title keywords. This is a **multi-class (3-class) classification** problem. Five classifiers are trained and compared, then served through an interactive Streamlit app.

## b. Dataset description
- **Source:** Kaggle — "Global Salary Data by Skill & Role" (https://www.kaggle.com/datasets/harshitkz/skill-salary-percentiles-global?resource=download). The raw file has 100,060 job postings and 18 columns.
- **Instances used:** 12,000 (stratified sample after removing duplicate postings); 9,600 train / 2,400 test.
- **Features:** 26 (after feature engineering from the raw columns; see `prepare_dataset.py`).
- **Target column:** `salary_band` — `1-Low` (< ~$37.4k), `2-Mid` (~$37.4k–$65k), `3-High` (≥ ~$65k), cut from `salary_mid_usd`. The salary columns themselves are NOT used as features, to avoid target leakage.
- **Preprocessing:** missing values filled (median / most-frequent), numeric features standardised, text features one-hot encoded, stratified 80/20 train-test split (random_state=42).
- **Feature groups:** location (country, region group); role (role family, seniority, industry); time (posted year, month); skills (number of skills listed, flags for python / sql / java / excel / javascript / aws / project management); job-title (word count, length, keyword flags such as senior, manager, engineer); has_company_domain.

## c. GitHub repository
(https://github.com/2026aj05120/ML_Assignment_2)

Live app: <https://YOUR-APP.streamlit.app>

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6161 | 0.7944 | 0.6219 | 0.6161 | 0.6174 | 0.4251 |
| Decision Tree | 0.5954 | 0.7659 | 0.6474 | 0.5954 | 0.6002 | 0.4082 |
| KNN | 0.5782 | 0.7537 | 0.5790 | 0.5782 | 0.5776 | 0.3680 |
| Naive Bayes | 0.5034 | 0.7416 | 0.6795 | 0.5034 | 0.4795 | 0.3238 |
| Support Vector Machine | 0.6058 | 0.7881 | 0.6174 | 0.6058 | 0.6092 | 0.4104 |

*Precision, Recall and F1 are weighted averages over the 3 classes; AUC is one-vs-rest weighted. **Re-run `train_models.py` yourself and paste your own numbers.***

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <your observation, e.g. strong baseline, struggles with non-linear patterns…> |
| Decision Tree | <…> |
| KNN | <…> |
| Naive Bayes | <…> |
| Support Vector Machine | <…> |
| **Overall winner for this dataset** | <model name + 2–3 sentences of justification using the metrics> |

## How to run locally
```
pip install -r requirements.txt
python prepare_dataset.py --raw salary_band_dataset.csv --out prepared_dataset.csv
python train_models.py --data salary_band_dataset.csv --target salary_band
python -m streamlit run app.py
```
