# Salary Band Classifier Lab

## a. Problem statement
Predict which salary band (Low / Mid / High, three equally sized groups of USD mid-point salary) a job posting belongs to, using the posting's country, role family, seniority, industry, listed skills and job-title keywords. This is a **multi-class (3-class) classification** problem. Five classifiers are trained and compared, then served through an interactive Streamlit app.

## b. Dataset description
- **Source:** Kaggle — "Global Salary Data by Skill & Role" (https://www.kaggle.com/datasets/harshitkz/skill-salary-percentiles-global?resource=download). The raw file has 100,060 job postings and 18 columns.
- **Instances used:** 12,000 (stratified sample after removing duplicate postings); about 9,300 train / 2,326 test.
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
| KNN | 0.5791 | 0.7529 | 0.5796 | 0.5791 | 0.5783 | 0.3693 |
| Naive Bayes | 0.5034 | 0.7416 | 0.6795 | 0.5034 | 0.4795 | 0.3238 |
| Support Vector Machine | 0.6058 | 0.7881 | 0.6174 | 0.6058 | 0.6092 | 0.4104 |

*Precision, Recall and F1 are weighted averages over the 3 classes, so weighted Recall equals Accuracy. AUC is one-vs-rest weighted. A random guess would score about 0.33 accuracy.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall model: highest Accuracy (0.6161), AUC (0.7944), F1 (0.6174) and MCC (0.4251). A linear model works well because the one-hot encoded location, role and seniority features push salary in a fairly additive way. Its confusion matrix shows most errors fall between neighbouring bands (216 Low predicted as Mid, 212 Mid predicted as High) and very few jump two bands (83 Low predicted as High, 52 High predicted as Low). Per class, Low and High are predicted best (F1 0.658 and 0.667) and Mid is hardest (F1 0.527), because it sits between the other two. |
| Decision Tree | Accuracy 0.5954 and MCC 0.4082, a little below Logistic Regression. It has the second-highest precision (0.6474) but lower recall (0.5954), so it is more selective with its predictions. It handles feature interactions like country × seniority easily, but a depth-limited tree (max_depth=6) cannot capture all of them, which probably caps its accuracy. |
| KNN | The weakest of the four balanced models (Accuracy 0.5791, MCC 0.3693). Distance-based neighbours work poorly here because one-hot encoding turns the data into many 0/1 columns, so many postings look equally "near" and the neighbours vote less reliably. Its precision, recall and F1 are all about 0.58, so it is consistent but not strong. |
| Naive Bayes | The lowest Accuracy (0.5034), F1 (0.4795) and MCC (0.3238), but the highest Precision (0.6795). It is precise when it commits to a band but misses many true cases, so its recall is poor. It assumes features are independent, but country, region and role are clearly related, which breaks that assumption. Its AUC (0.7416) is not far behind the others, so its ranking of probabilities is decent even though its final class choices are poor. |
| Support Vector Machine | The strongest alternative: Accuracy 0.6058, AUC 0.7881, F1 0.6092 and MCC 0.4104, close to Logistic Regression. The RBF kernel can model non-linear boundaries, but it gains nothing over the linear model here, which suggests the relationships in this data are mostly linear. It is slower to train and its saved model file is much larger. |
| **Overall winner for this dataset** | **Logistic Regression.** It leads on Accuracy, AUC, Recall, F1 and MCC. It is also fast to train and tiny to store (about 10 KB), whereas SVM and KNN files are about 6–7 MB. SVM is a close second. Naive Bayes has the best precision but the worst recall, so it is not the best choice overall. Performance tops out near 62% because many postings carry little information: only about 17% list skills, so the models rely mostly on location, role family and seniority. |

## How to run locally
```
pip install -r requirements.txt
python prepare_dataset.py --raw salary_band_dataset.csv --out prepared_dataset.csv
python train_models.py --data salary_band_dataset.csv --target salary_band
python -m streamlit run app.py
```
