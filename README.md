# 🏥 Hospital Readmission Analytics — Diabetic Patient Risk Prediction

End-to-end healthcare analytics project analyzing **99,493 diabetic patient 
records** from 130 US hospitals to predict 30-day readmission risk, conduct 
A/B testing on medication effectiveness, and build a machine learning model 
for patient risk stratification.

**Live Dashboard →** https://pranathi1812.github.io/healthcare-analytics/dashboard/

---

## 🎯 Project Overview

Under the US Hospital Readmissions Reduction Program (HRRP), Medicare 
penalizes hospitals for high 30-day readmission rates — costing the 
industry billions annually. This project identifies high-risk patients 
before discharge using real clinical data from 130 US hospitals (1999–2008).

---

## 📊 Key Findings

- **11.2%** of diabetic patients are readmitted within 30 days
- **Elderly patients (70+)** have the highest risk at 11.9%
- **A/B Test:** Medication changes INCREASE readmission by 1.3% (p≈0, statistically significant)
- **Top Predictor:** Number of previous inpatient visits
- **ML Model:** Catches 1,022 high-risk patients with 67% accuracy and 47% recall

---

## 🔬 A/B Testing

**Question:** Does changing medication reduce 30-day readmission?

| Group | Patients | Readmission Rate |
|---|---|---|
| Group A — Medication Changed | 45,911 | 11.9% |
| Group B — Medication Unchanged | 53,582 | 10.6% |

**Chi-square test → p-value ≈ 0 → Statistically significant**

Counterintuitive finding: medication changes correlate with higher 
readmission — suggesting they're a marker of patient complexity, 
not a cause of readmission.

---

## 🤖 Machine Learning Model

**Algorithm:** Logistic Regression (class_weight=balanced)
**Split:** 80% training / 20% testing / 14 clinical features

| Metric | Value |
|---|---|
| Accuracy | 67% |
| Readmissions Caught | 1,022 |
| Missed Readmissions | 1,153 |
| Correctly Safe | 12,317 |

Initial model achieved 89% accuracy but only caught 35 readmissions 
(predicting "safe" for almost everyone). Adding class_weight=balanced 
improved recall from 2% to 47% — making it clinically useful.

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · SciPy · Scikit-learn · 
HTML · CSS · JavaScript · Chart.js · Git · GitHub Pages

---

## 🚀 How to Run

```bash
# 1. Clone repo
git clone https://github.com/pranathi1812/healthcare-analytics.git

# 2. Install dependencies
pip3 install pandas numpy scipy scikit-learn matplotlib seaborn

# 3. Download dataset from Kaggle
# kaggle.com/datasets/brandao/diabetes
# Save as: data/diabetic_data.csv

# 4. Run pipeline
python3 healthcare_analysis.py

# 5. View dashboard
cd dashboard && python3 -m http.server 8000
# Open: http://localhost:8000
```

---

## 🎓 Skills Demonstrated

- A/B Testing with chi-square statistical significance testing
- Machine learning — logistic regression with class imbalance handling
- Healthcare domain knowledge — clinical data, readmission metrics, HRRP
- End-to-end data pipeline — Extract, Transform, Analyze, Visualize
- Feature engineering and model evaluation (recall, confusion matrix)
- Interactive dashboard with Chart.js



