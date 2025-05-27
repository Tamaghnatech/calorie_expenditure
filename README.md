# Predicting Calorie Expenditure - Kaggle Playground Series S5E5

Welcome to Lord Nag's end-to-end pipeline for the Kaggle Playground Series S5E5: **Predict Calorie Expenditure**. This project leverages structured tabular data to build robust machine learning models that predict calories burned during physical activities.

---

## 🔍 Problem Statement
Your objective is to predict how many calories are burned during a workout session using a dataset derived from a deep learning model trained on real-world exercise data.

---

## 📁 Dataset Overview
The dataset is provided by Kaggle and includes:
- `train.csv`: Training features + target (`Calories`)
- `test.csv`: Test features (no `Calories`)
- `sample_submission.csv`: Sample format for submissions

Key features include:
- Age, Gender (Sex_male), Height, Weight
- Duration of activity
- Heart Rate, Body Temperature

Additional features were engineered as detailed below.

---

## 🧪 Pipeline Steps

### 1. 📦 **Kaggle Setup**
Downloaded dataset via Kaggle API inside Google Colab.

### 2. 📊 **EDA (Exploratory Data Analysis)**
- Distribution plots for `Calories`
- Heatmap of feature correlations
- Inspection for skewness, outliers, and missing values

### 3. 🧠 **Feature Engineering**
New features created for deeper signal extraction:
- **BMI** = Weight / (Height/100)^2
- **HR_per_min** = Heart_Rate / Duration
- **Temp_per_min** = Body_Temp / Duration
- **HRxTemp** = Heart_Rate * Body_Temp
- **Duration_Weight** = Duration * Weight

### 4. ⚙️ **Preprocessing**
- Handled missing/infinite values
- One-hot encoded gender column
- StandardScaler applied for normalization
- Split into training/validation using 80/20 strategy

### 5. 🌲 **Random Forest Regressor**
- Baseline model
- RMSLE ≈ 0.01808

### 6. 🌳 **LightGBM Regressor** (Baseline & Tuned)
- Tuned with RandomizedSearchCV
- Best params:
```json
{
  "subsample": 1.0,
  "num_leaves": 31,
  "n_estimators": 800,
  "min_child_samples": 20,
  "max_depth": 7,
  "learning_rate": 0.05,
  "colsample_bytree": 0.6
}
```
- Final Public Leaderboard Score: **0.05892** (Top-tier performance)

---

## 📈 Model Evaluation

### 🔁 **Predicted vs Actual**
- Strong linearity around y=x
- Accurate on entire value spectrum

### 📉 **Residuals Analysis**
- Residuals centered at 0
- Low variance across prediction bands

### 🔍 **SHAP Interpretability**
- TreeExplainer used
- Key drivers:
  - `Duration`, `Heart_Rate`, `Temp_per_min`, `Age`
- Beeswarm plots and bar plots used for feature attributions

---

## 💾 Submission
- Predictions were log-transformed during training
- Inference used `np.expm1()` to return to real-world values
- Submission saved as `submission_tuned_lgb.csv`

---

## 💡 Future Enhancements
- Optuna for smarter hyperparameter search
- Blend ensemble with Random Forest
- Deploy notebook as a public kernel for reproducibility
- Implement cross-fold SHAP aggregation

---

## 📜 Author
**Tamaghna Nag (Lord Nag)**  
Machine Learning Engineer & Kaggle Warrior  
🔗 GitHub: [github.com/tamaghna-nag](https://github.com/tamaghna-nag)

---

## 🔗 Links
- [Competition Page](https://www.kaggle.com/competitions/playground-series-s5e5)
- [LightGBM Documentation](https://lightgbm.readthedocs.io)
- [SHAP Docs](https://shap.readthedocs.io/en/latest/index.html)

---

May your models be fast, your residuals be small, and your leaderboard rank ever climb. 🧠⚔️
