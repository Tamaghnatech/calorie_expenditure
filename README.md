# Predicting Calorie Expenditure - Kaggle Playground Series S5E5

Welcome to **Lord Nag's** end-to-end machine learning pipeline for the Kaggle Playground Series S5E5: **Predict Calorie Expenditure**. This repository presents a complete solution for beginners and intermediate practitioners using Python and LightGBM in Google Colab.

---

## 📌 Problem Statement
The task is to predict the number of **calories burned** during a workout based on biometric and exercise-related features. This uses a synthetic dataset modeled on real-world biometric exercise data.

---

## 📁 Dataset Overview
The dataset contains the following files:
- `train.csv`: Training dataset with features and target (`Calories`)
- `test.csv`: Test dataset with only features
- `sample_submission.csv`: Template for final submission format

Key columns:
- **Age, Height, Weight**: basic biometric info
- **Duration, Heart_Rate, Body_Temp**: workout session metrics
- **Sex_male**: encoded gender

---

## 🚀 Project Pipeline

### 1. ⚙️ Kaggle API & Data Setup
We used the Kaggle API in Google Colab to download and unzip the dataset.

### 2. 📊 Exploratory Data Analysis (EDA)
Visuals for understanding data distribution and correlations.

**Target Distribution**:
![Target Distribution](./images/download\ (82).png)

**Correlation Heatmap**:
![Correlation Heatmap](./images/download\ (83).png)

### 3. 🧠 Feature Engineering
Created several derived features to capture more meaningful relationships:
- `BMI = Weight / (Height/100)^2`
- `HR_per_min = Heart_Rate / Duration`
- `Temp_per_min = Body_Temp / Duration`
- `HRxTemp = Heart_Rate * Body_Temp`
- `Duration_Weight = Duration * Weight`

### 4. 🧼 Preprocessing
- One-hot encoded gender
- Standard scaling applied
- NaNs handled with median fill
- Log1p transformation applied to target for RMSLE stability

### 5. 🌲 Model Training
We trained and validated the following:

#### ✅ Random Forest
- RMSLE: ~0.01808

#### ✅ LightGBM (Baseline)
- RMSLE: ~0.01728

#### ✅ LightGBM (Tuned with RandomizedSearchCV)
- RMSLE Public Score: **0.05892**
- Best parameters:
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

### 6. 📈 Evaluation Metrics

**Predicted vs Actual Plot**:
![Predicted vs Actual](./images/download\ (85).png)

**Residual Histogram**:
![Residual Histogram](./images/download\ (86).png)

**Residual vs Predicted**:
![Residuals vs Predicted](./images/download\ (87).png)

### 7. 🔍 SHAP Explainability
We used SHAP values to understand model behavior.

**SHAP Beeswarm Plot**:
![SHAP Summary Plot](./images/download\ (88).png)

---

## 📤 Submission
- Inference done on log1p-transformed predictions using `np.expm1()`
- Final CSV created as: `submission_tuned_lgb.csv`

---

## 💡 Key Takeaways for Beginners
- 🔄 Always scale and encode your data properly
- 🧱 Feature engineering often improves model performance
- 🧪 Evaluate with proper plots — residuals, prediction trends
- 🔍 Use SHAP for model explainability
- ✅ Use log1p if your target is skewed

---

## 📜 Author
**Tamaghna Nag (aka Lord Nag)**  
ML Engineer | AI Strategist | Kaggle Enthusiast  
🔗 GitHub: [github.com/tamaghna-nag](https://github.com/tamaghna-nag)

---

## 📚 References
- [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s5e5)
- [LightGBM Docs](https://lightgbm.readthedocs.io)
- [SHAP Library](https://shap.readthedocs.io)

---

## 🤝 Contributions
Feel free to fork the repo, raise issues, or suggest new feature engineering ideas or model improvements!
