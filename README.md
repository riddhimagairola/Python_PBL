# 🫀 Heart Disease Prediction (ML Classification Project)

## 📌 Overview
This project implements a complete **end-to-end machine learning pipeline** to predict the presence of heart disease using patient clinical data. It covers everything from raw data preprocessing to model evaluation and comparison.

The system outputs:
- 0 → No Heart Disease  
- 1 → Heart Disease Present  

---

## 📂 Project Structure

├── PBL-Comp.pdf        # Full code implementation (Jupyter Notebook exported)
├── README.md           # Project documentation
├── synthetic_heart_disease_dataset.csv         # Input dataset (synthetic/real)


---

## 📊 Dataset Description
- Total Records: **50,000**
- Features: **20 input features + 1 target**
- Includes:
  - Demographic: Age, Gender
  - Health metrics: BMI, BP, Cholesterol, Blood Sugar
  - Lifestyle: Smoking, Alcohol, Physical Activity, Diet
  - Medical history: Diabetes, Hypertension, Family History

From the dataset preview in the code :contentReference[oaicite:0]{index=0}:
- Slight class balance (~53% no disease, ~46% disease)
- Some missing values in Alcohol_Intake handled during preprocessing

---

## ⚙️ ML Pipeline

### 1. Data Preprocessing
- Removed duplicates (none found)
- Handled missing values:
  - Numeric → Median
  - Categorical → Mode
- Encoded categorical variables using **Label Encoding**
- Scaled numerical features using **MinMaxScaler**

---

### 2. Exploratory Data Analysis (EDA)
- Distribution plots (Age, BMI, BP, Cholesterol)
- Target class distribution
- Correlation heatmap to identify important features

Key insight:
- Features like **Hypertension, Age, Cholesterol** show higher correlation with heart disease :contentReference[oaicite:1]{index=1}

---

### 3. Feature Engineering
- Split features into:
  - Categorical → Encoding
  - Numerical → Scaling
- Final feature matrix: **(50000, 20)**

---

### 4. Model Training
Models implemented:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- (Extra) Linear Regression (converted to classifier)

Train-Test Split:
- 80% Training
- 20% Testing

---

## 📈 Results

| Model                | Accuracy |
|---------------------|---------|
| Logistic Regression | 92.38%  |
| KNN (k=5)           | 81.24%  |
| SVM                 | **92.78% (Best)** |
| Linear Regression   | 92.14%  |

From evaluation outputs :contentReference[oaicite:2]{index=2}:
- F1 Score (Logistic): **~0.917**
- ROC AUC: **~0.98 (excellent separability)**

---

## 📊 Evaluation Metrics Used
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

---

## 🔍 Sample Prediction

Patient 1: Heart Disease Patient
Patient 2: No Heart Disease



---

## 🚀 How to Run

### 1. Install dependencies
bash
pip install numpy pandas matplotlib seaborn scikit-learn


### 2. Run the notebook / code
- Open the PDF code in Jupyter Notebook or convert to .ipynb
- Ensure dataset path is correct:
  python
df = pd.read_csv("synthetic_heart_disease_dataset.csv")


### 3. Execute all cells

---

## 🧠 Key Learnings
- Proper preprocessing significantly improves model performance
- Feature scaling is critical for distance-based models (KNN, SVM)
- SVM performed best for this dataset
- Class distribution was relatively balanced → no resampling required

---

## 📌 Future Improvements
- Hyperparameter tuning (GridSearchCV)
- Use advanced models (Random Forest, XGBoost)
- Handle class imbalance (if real-world dataset is skewed)
- Deploy as a web app (Flask/Streamlit)

---
