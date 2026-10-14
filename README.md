# Deep Learning
# AI Resume Screening & ATS Hiring Prediction

A deep learning project that predicts whether a job candidate will be **Selected** or **Rejected** based on resume screening and ATS (Applicant Tracking System) data, using a neural network built with TensorFlow/Keras.

## 📁 Repository Structure
```
├── notebook0b6739e463.ipynb    # Full analysis & modeling notebook
└── README.md
```

## 📊 Dataset

**Source:** [AI Resume Screening and ATS Hiring Dataset (100K Records)](https://www.kaggle.com/datasets/mobeenfatimah/resume-screening-and-ats-hiring-dataset100k-record)

- **100,000 candidate records × 44 columns**
- **No missing values, no duplicates**

**Key features:**
| Category | Columns |
|---|---|
| **Personal** | candidate_id, full_name, email, phone, country, city, age, gender |
| **Education** | highest_education, university, field_of_study, cgpa, graduation_year |
| **Experience** | experience_years, previous_companies, current_job_title, internship_experience, leadership_experience |
| **Skills** | technical_skills, programming_languages, frameworks, databases, cloud_platform |
| **Scores** | communication_score, problem_solving_score, technical_test_score, interview_score, aptitude_score |
| **Job Preferences** | job_role, expected_salary, preferred_location, employment_type, remote_preference, availability |
| **ATS Metrics** | resume_quality_score, resume_length, keyword_match_percentage, ats_score |
| **Target** | selected (Selected / Rejected) |

## 🔍 Exploratory Data Analysis

- **Target distribution:** Perfectly balanced — 50,000 Selected / 50,000 Rejected
- **Chi-square tests** across all categorical features identified statistically significant relationships with the target for: `previous_companies`, `current_job_title`, `internship_experience`, `leadership_experience`, `programming_languages`, `databases`, and `cloud_platform`
- **Correlation heatmap** on numerical features revealed low multicollinearity
- **Age vs. experience_years** scatter plot showed no strong linear trend
- **Selection rate by education** was near-uniform across all levels (~49–51%)

## 🛠️ Modeling Approach

**Preprocessing pipeline (scikit-learn):**
- **Numerical features** → `StandardScaler`
- **Categorical features** → `OneHotEncoder(handle_unknown="ignore")`
- Combined via `ColumnTransformer`
- **Target** → `LabelEncoder`

**Data split (stratified):**
- Train: 70% | Validation: 15% | Test: 15%

**Neural Network (TensorFlow/Keras):**
```
Input (26,104 features after encoding)
├── Dense(64, ReLU)
├── Dense(32, ReLU)
├── Dense(32, ReLU)
└── Dense(3, Softmax)
```
- **Loss:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Callback:** EarlyStopping (patience=5, restore best weights)
- **Epochs:** 200 (stopped early)

## 📈 Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **100%** |
| Precision (both classes) | 1.00 |
| Recall (both classes) | 1.00 |
| F1-Score (both classes) | 1.00 |

Training and validation loss dropped to essentially zero within ~12 epochs, and the classification report showed perfect precision, recall, and F1-score on both classes.

## ⚠️ Important Note on Results

The perfect 100% accuracy strongly suggests **data leakage** — certain features (e.g., `ats_score`, `keyword_match_percentage`, or `resume_quality_score`) may be direct proxies for the `selected` label in this synthetic dataset. This is common in simulated/toy datasets. Further investigation is recommended before using this model in production:

- Check feature importances/ablations
- Test on a real-world ATS dataset
- Consider regularization / simpler baselines (Logistic Regression)

## 🚀 Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
jupyter notebook notebook0b6739e463.ipynb
```

## 👤 Author

**Stephen Omengo** — Machine Learning Portfolio
