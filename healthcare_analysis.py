import pandas as pd               # data cleaning and analysis
import numpy as np                # math and numbers
import matplotlib.pyplot as plt   # creating charts
import seaborn as sns             # better looking charts
from scipy import stats           # statistics and A/B testing
from sklearn.model_selection import train_test_split  # split data for ML
from sklearn.linear_model import LogisticRegression   # our ML model
from sklearn.metrics import (
    accuracy_score,               # how accurate is our model
    classification_report,        # detailed model performance
    confusion_matrix              # where model gets confused
)
from sklearn.preprocessing import LabelEncoder        # convert text to numbers
import json                       # export data for dashboard
import os                         # file and folder operations
import warnings
warnings.filterwarnings('ignore') # hide unnecessary warnings

print("✅ All libraries imported successfully!")

# ── CONFIG ───────────────────────────────────────────────
# File path to our dataset
DATA_PATH = "data/diabetic_data.csv"

# ── EXTRACT ──────────────────────────────────────────────
def extract():
    print("\n" + "="*50)
    print("STEP 1: EXTRACT")
    print("="*50)
    
    # read the CSV file into a DataFrame
    df = pd.read_csv(DATA_PATH)
    
    # print basic info about the dataset
    print(f"✅ Rows loaded: {len(df):,}")
    print(f"✅ Columns: {len(df.columns)}")
    print(f"\nColumn names:\n{list(df.columns)}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    return df

# ── TRANSFORM ────────────────────────────────────────────
def transform(df):
    print("\n" + "="*50)
    print("STEP 2: TRANSFORM")
    print("="*50)
    
    # Step 1 - Replace ? with NaN
    df = df.replace('?', np.nan)
    print(f"✅ Replaced '?' with NaN")
    
    # Step 2 - Drop columns with too many missing values
    # max_glu_serum and A1Cresult are 95%+ missing - useless
    # weight is also mostly missing
    # IDs are not useful for analysis
    cols_to_drop = [
        'max_glu_serum', 'A1Cresult', 'weight',
        'payer_code', 'medical_specialty',
        'encounter_id', 'patient_nbr',
        'diag_1', 'diag_2', 'diag_3'
    ]
    df = df.drop(columns=cols_to_drop)
    print(f"✅ Dropped {len(cols_to_drop)} unnecessary columns")
    
    # Step 3 - Drop rows with missing race or gender
    before = len(df)
    df = df.dropna(subset=['race', 'gender'])
    print(f"✅ Removed {before - len(df):,} rows with missing race/gender")
    
    # Step 4 - Convert age from text to number
    # [50-60) means age 50-59, we take the midpoint
    age_map = {
        '[0-10)':5, '[10-20)':15, '[20-30)':25,
        '[30-40)':35, '[40-50)':45, '[50-60)':55,
        '[60-70)':65, '[70-80)':75, '[80-90)':85,
        '[90-100)':95
    }
    df['age'] = df['age'].map(age_map)
    print(f"✅ Converted age to numeric")
    
    # Step 5 - Create target column
    # We simplify readmitted to binary: 1 = readmitted within 30 days, 0 = not
    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    print(f"✅ Created binary target: readmitted_30")
    print(f"   Readmitted <30 days: {df['readmitted_30'].sum():,} ({df['readmitted_30'].mean()*100:.1f}%)")
    
    # Step 6 - Create waiting days feature
    # number_inpatient = previous inpatient visits - good risk indicator
    # Already exists, just confirming
    
    # Step 7 - Create age groups for analysis
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 30, 50, 70, 100],
        labels=['Young (0-30)', 'Middle (30-50)', 
                'Senior (50-70)', 'Elderly (70+)']
    )
    print(f"✅ Created age groups")
    
    print(f"\n✅ Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df

# ── EDA ──────────────────────────────────────────────────
def eda(df):
    print("\n" + "="*50)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("="*50)

    # Overall readmission breakdown
    print("\n📊 Readmission breakdown:")
    print(df['readmitted'].value_counts())
    print(f"\n📊 30-day readmission rate: {df['readmitted_30'].mean()*100:.1f}%")

    # Readmission by age group
    print("\n📊 Readmission rate by age group:")
    age_readmit = df.groupby('age_group', observed=True)['readmitted_30'].agg(['mean','count'])
    age_readmit['mean'] = (age_readmit['mean'] * 100).round(1)
    age_readmit.columns = ['readmission_rate_%', 'patient_count']
    print(age_readmit)

    # Readmission by gender
    print("\n📊 Readmission rate by gender:")
    gender_readmit = df.groupby('gender')['readmitted_30'].mean() * 100
    print(gender_readmit.round(1))

    # Readmission by time in hospital
    print("\n📊 Avg time in hospital:")
    print(f"  Readmitted <30: {df[df['readmitted_30']==1]['time_in_hospital'].mean():.1f} days")
    print(f"  Not readmitted: {df[df['readmitted_30']==0]['time_in_hospital'].mean():.1f} days")

    # Readmission by number of medications
    print("\n📊 Avg number of medications:")
    print(f"  Readmitted <30: {df[df['readmitted_30']==1]['num_medications'].mean():.1f}")
    print(f"  Not readmitted: {df[df['readmitted_30']==0]['num_medications'].mean():.1f}")

    # Readmission by diabetes medication
    print("\n📊 Readmission rate by diabetes medication:")
    diab_readmit = df.groupby('diabetesMed')['readmitted_30'].mean() * 100
    print(diab_readmit.round(1))

    return df

# ── A/B TESTING ──────────────────────────────────────────
def ab_test(df):
    print("\n" + "="*50)
    print("STEP 4: A/B TESTING")
    print("="*50)
    print("\nQuestion: Does changing medication reduce")
    print("30-day readmission rates?")
    print("\nGroup A = Medication CHANGED (change = 'Ch')")
    print("Group B = Medication NOT changed (change = 'No')")

    # Split into two groups
    group_a = df[df['change'] == 'Ch']['readmitted_30']
    group_b = df[df['change'] == 'No']['readmitted_30']

    # Calculate readmission rates
    rate_a = group_a.mean() * 100
    rate_b = group_b.mean() * 100

    print(f"\n📊 Results:")
    print(f"  Group A (Medication Changed):")
    print(f"    Patients: {len(group_a):,}")
    print(f"    Readmission rate: {rate_a:.1f}%")
    print(f"\n  Group B (Medication NOT Changed):")
    print(f"    Patients: {len(group_b):,}")
    print(f"    Readmission rate: {rate_b:.1f}%")
    print(f"\n  Difference: {abs(rate_a - rate_b):.1f}%")

    # Chi-square test
    # Create a contingency table
    # contingency table = counts of each outcome per group
    contingency = pd.crosstab(df['change'], df['readmitted_30'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    print(f"\n📊 Statistical Test Results:")
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Degrees of freedom: {dof}")

    print(f"\n📊 Conclusion:")
    if p_value < 0.05:
        print(f"  ✅ STATISTICALLY SIGNIFICANT (p={p_value:.6f} < 0.05)")
        print(f"  The difference in readmission rates is REAL")
        print(f"  NOT due to random chance")
        if rate_a < rate_b:
            print(f"  Changing medication REDUCES readmission by {rate_b-rate_a:.1f}%")
        else:
            print(f"  Changing medication INCREASES readmission by {rate_a-rate_b:.1f}%")
    else:
        print(f"  ❌ NOT statistically significant (p={p_value:.6f} > 0.05)")
        print(f"  The difference could be due to random chance")

    # Store results for dashboard
    ab_results = {
        'group_a_rate': round(rate_a, 1),
        'group_b_rate': round(rate_b, 1),
        'group_a_size': int(len(group_a)),
        'group_b_size': int(len(group_b)),
        'chi2': round(chi2, 4),
        'p_value': round(p_value, 6),
        'significant': bool(p_value < 0.05)
    }

    return ab_results

# ── MACHINE LEARNING ─────────────────────────────────────
def machine_learning(df):
    print("\n" + "="*50)
    print("STEP 5: MACHINE LEARNING MODEL")
    print("="*50)
    print("\nBuilding Logistic Regression model to")
    print("predict 30-day hospital readmission...")

    # Step 1 - Select features for the model
    # These are the columns we'll use to make predictions
    features = [
        'age', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient',
        'number_emergency', 'number_inpatient', 'number_diagnoses',
        'gender', 'race', 'change', 'diabetesMed', 'insulin'
    ]

    # Step 2 - Prepare X (features) and y (target)
    X = df[features].copy()
    y = df['readmitted_30']

    # Step 3 - Encode text columns to numbers
    # ML models only understand numbers not text
    le = LabelEncoder()
    text_cols = ['gender', 'race', 'change', 'diabetesMed', 'insulin']
    for col in text_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    print(f"\n✅ Features selected: {len(features)}")
    print(f"✅ Target: readmitted_30")
    print(f"✅ Dataset size: {len(X):,} rows")

    # Step 4 - Split data 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n✅ Training set: {len(X_train):,} rows (80%)")
    print(f"✅ Testing set:  {len(X_test):,} rows  (20%)")

    # Step 5 - Train the model
    print("\n⏳ Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("✅ Model trained!")

    # Step 6 - Make predictions
    y_pred = model.predict(X_test)

    # Step 7 - Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Model Performance:")
    print(f"  Accuracy: {accuracy*100:.1f}%")

    # Classification report
    print(f"\n📊 Detailed Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Not Readmitted', 'Readmitted <30']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"  True Negatives  (correctly predicted NOT readmitted): {cm[0][0]:,}")
    print(f"  False Positives (wrongly predicted as readmitted):     {cm[0][1]:,}")
    print(f"  False Negatives (missed actual readmissions):          {cm[1][0]:,}")
    print(f"  True Positives  (correctly predicted readmitted):      {cm[1][1]:,}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': abs(model.coef_[0])
    }).sort_values('importance', ascending=False)

    print(f"\n📊 Top 5 Most Important Features:")
    print(feature_importance.head())

    # Store results for dashboard
    ml_results = {
        'accuracy': round(accuracy * 100, 1),
        'true_negatives':  int(cm[0][0]),
        'false_positives': int(cm[0][1]),
        'false_negatives': int(cm[1][0]),
        'true_positives':  int(cm[1][1]),
        'feature_importance': feature_importance.head(8).to_dict(orient='records')
    }

    return ml_results

# ── EXPORT JSON ──────────────────────────────────────────
def export_json(df, ab_results, ml_results):
    print("\n" + "="*50)
    print("STEP 6: EXPORT DASHBOARD DATA")
    print("="*50)

    os.makedirs("dashboard", exist_ok=True)

    # Readmission by age group
    age_data = df.groupby('age_group', observed=True)['readmitted_30']\
        .mean().mul(100).round(1).reset_index()
    age_data.columns = ['age_group', 'readmission_rate']

    # Readmission breakdown
    readmit_counts = df['readmitted'].value_counts().reset_index()
    readmit_counts.columns = ['status', 'count']

    # Medications vs readmission
    med_data = df.groupby('num_medications')['readmitted_30']\
        .mean().mul(100).round(1).reset_index()
    med_data.columns = ['num_medications', 'readmission_rate']
    med_data = med_data[med_data['num_medications'] <= 30]

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "total_patients":      int(len(df)),
        "readmission_rate":    round(df['readmitted_30'].mean()*100, 1),
        "avg_medications":     round(df['num_medications'].mean(), 1),
        "avg_hospital_days":   round(df['time_in_hospital'].mean(), 1),

        "readmission_breakdown": readmit_counts.to_dict(orient='records'),
        "readmission_by_age":    age_data.to_dict(orient='records'),
        "medications_trend":     med_data.to_dict(orient='records'),

        "ab_test": ab_results,
        "ml_model": ml_results,
    }

    with open("dashboard/data.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ dashboard/data.json created!")
    print(f"\n{'='*50}")
    print("PIPELINE COMPLETE! ✅")
    print(f"{'='*50}")
    print(f"  Total patients analyzed: {len(df):,}")
    print(f"  30-day readmission rate: {df['readmitted_30'].mean()*100:.1f}%")
    print(f"  A/B test p-value:        {ab_results['p_value']}")
    print(f"  Model accuracy:          {ml_results['accuracy']}%")
    print(f"  Readmissions caught:     {ml_results['true_positives']:,}")

# ── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw = extract()
    df_clean = transform(df_raw)
    eda(df_clean)
    ab_results = ab_test(df_clean)
    ml_results = machine_learning(df_clean)
    export_json(df_clean, ab_results, ml_results)





