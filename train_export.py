import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Dataset
df = pd.read_csv('Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')

# 2. Data Preprocessing
# Drop non-predictive columns and the descriptive target 'addiction_level'
df_clean = df.drop(['transaction_id', 'user_id', 'addiction_level'], axis=1)

# Categorical Encoding (Convert text to numbers)
# Mappings: 
# Gender: Female=0, Male=1, Other=2
# Stress: High=0, Low=1, Medium=2
# Impact: No=0, Yes=1
df_clean['gender'] = df_clean['gender'].astype('category').cat.codes
df_clean['stress_level'] = df_clean['stress_level'].astype('category').cat.codes
df_clean['academic_work_impact'] = df_clean['academic_work_impact'].astype('category').cat.codes

# 3. Split Features and Target
X = df_clean.drop('addicted_label', axis=1)
y = df_clean['addicted_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Dataset Balancing (Over-Sampling)
train_data = pd.concat([X_train, y_train], axis=1)
major_class = train_data[train_data.addicted_label == 1]
minor_class = train_data[train_data.addicted_label == 0]

minor_upsampled = resample(minor_class, 
                           replace=True,     
                           n_samples=len(major_class), 
                           random_state=42)

balanced_train = pd.concat([major_class, minor_upsampled])
X_res = balanced_train.drop('addicted_label', axis=1)
y_res = balanced_train['addicted_label']

print(f"Balanced Dataset: {len(y_res)} samples ({len(major_class)} per class)")

# 5. Model Training & Comparison
# Model A: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_res, y_res)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

# Model B: Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_res, y_res)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# 6. Export Best Model
if rf_acc > lr_acc:
    best_model = rf
    print("Selected Model: Random Forest")
else:
    best_model = lr
    print("Selected Model: Logistic Regression")

joblib.dump(best_model, 'best_model.pkl')
print("Model exported successfully as 'best_model.pkl'")