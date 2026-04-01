# Generated from: PBL PHASE 1.ipynb
# Converted at: 2026-04-01T07:31:55.881Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# IMPORTING THE LIBRARIES


# Cell 1 — Imports and settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (10,6)
RANDOM_STATE = 42


# Cell 2 — (Optional) Adjust display settings for nicer outputs
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)

# Cell 3 — Load dataset
df = pd.read_csv("D:\synthetic_heart_disease_dataset.csv")
print("Shape:", df.shape)
df.head()

# Cell 4 — Quick dataset overview
print("Columns:", df.columns.tolist())
print("\nInfo:")
display(df.info())
print("\nDescriptive statistics (numeric):")
display(df.describe().T)
print("\nTarget distribution:")
display(df['Heart_Disease'].value_counts(normalize=True).rename('proportion'))


# DATA CLEANING AND PREPROCESSING TECHNIQUES


# Cell 5 — Data cleaning: duplicates and missing values
# Duplicates
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
if dupes > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Dropped duplicates. New shape:", df.shape)

# Missing values
missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
print("\nMissing values per column (if any):")
display(missing)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("\nNumeric columns:", num_cols)
print("Categorical columns:", cat_cols)


# Cell 6 — Impute missing values
for c in df.columns:
    if df[c].isnull().any():
        if df[c].dtype in [np.float64, np.int64]:
            med = df[c].median()
            df[c] = df[c].fillna(med)
            print(f"Filled numeric {c} with median = {med}")
        else:
            mode = df[c].mode()[0]
            df[c] = df[c].fillna(mode)
            print(f"Filled categorical {c} with mode = {mode}")

print("Missing values after imputation:", df.isnull().sum().sum())

# Cell 7 — Basic EDA / Visualization 1: Distributions of some numeric columns and target
numeric_to_plot = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol_Total', 'Blood_Sugar_Fasting']
for col in numeric_to_plot:
    if col in df.columns:
        plt.figure()
        sns.histplot(df[col], kde=True, stat="density", bins=40)
        plt.title(f"Distribution of {col}")
        plt.show()

# Target countplot
plt.figure()
sns.countplot(x='Heart_Disease', data=df)
plt.title("Heart_Disease distribution")
plt.show()


# Removing outliers using IQR method
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[(df[col] >= lower) & (df[col] <= upper)]

# Reduce skewness using log transform
for col in num_cols:
    if df[col].skew() > 1:
        df[col] = np.log1p(df[col])

# Cell 8 — Correlation matrix for numeric features
num_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(14,10))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap='RdBu_r', center=0)
plt.title("Correlation matrix (numeric features)")
plt.show()

# Cell 9 — Feature engineering / create feature lists
target_col = 'Heart_Disease'

categorical_features = [c for c in df.select_dtypes(include=['object', 'category']).columns.tolist()]
if target_col in categorical_features:
    categorical_features.remove(target_col)

numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target_col]

print("Categorical features to encode:", categorical_features)
print("Numeric features to scale:", numeric_features)


# Cell 10 — Label Encoding all categorical columns
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le
    print(f"{col}: classes -> {le.classes_}")

df_encoded.head()

# Cell 11 — MinMax scaling for numeric columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_scaled = df_encoded.copy()
df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])

# Save column lists for later
X = df_scaled.drop(columns=[target_col])
y = df_scaled[target_col].astype(int)

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)


# Cell 12 — Create X and y
target_col = 'Heart_Disease'

X = df_scaled.drop(columns=[target_col])
y = df_scaled[target_col].astype(int)

print("Original Shape:", X.shape)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import pandas as pd

# Step 1 — Base model with L1 (Lasso)
log_model = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    max_iter=1000,
    random_state=RANDOM_STATE
)

# Step 2 — Pipeline
pipe = Pipeline([
    ('feature_selection', SelectFromModel(log_model)),
    ('model', LogisticRegression(
        penalty='l1',
        solver='liblinear',
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

# Step 3 — Hyperparameter grid
param_grid = {
    'feature_selection__estimator__C': [0.01, 0.1, 1, 10],
    'model__C': [0.01, 0.1, 1, 10]
}

# Step 4 — GridSearch
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)

print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# Step 5 — Extract selected features
best_selector = grid.best_estimator_.named_steps['feature_selection']

selected_features = X.columns[best_selector.get_support()]

print("\nSelected Features:", list(selected_features))
print("Number of Selected Features:", len(selected_features))

# Step 6 — Transform dataset
X_selected = best_selector.transform(X)
X = pd.DataFrame(X_selected, columns=selected_features)

print("New Shape:", X.shape)

X = X_selected
y = df_scaled['Heart_Disease']
X