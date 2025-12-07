# Heart_Disease_Classification.ipynb
Heart Disease prediction using Machine Learning and Scikit-Learn. Includes EDA, classification models, tuning, evaluation metrics, ROC curve, and feature importance.

## 🧠 Project Overview

Given a patient’s clinical attributes, the goal is to predict whether they are likely to have **heart disease** (`target = 1`) or not (`target = 0`).

This notebook walks through the full machine learning workflow:

1. Problem definition  
2. Data loading & understanding  
3. Exploratory Data Analysis (EDA)  
4. Feature preparation  
5. Model training & comparison  
6. Hyperparameter tuning  
7. Model evaluation (beyond accuracy)  
8. Feature importance & interpretation  

The dataset used is the **Heart Disease** dataset derived from the **Cleveland** database (UCI Machine Learning Repository), in formatted form from **Kaggle**.  
It contains **303 samples** and **14 features**.

---

## 📊 Dataset

**Features (independent variables):**

- `age` – Age in years  
- `sex` – 1 = male, 0 = female  
- `cp` – Chest pain type (0–3)  
- `trestbps` – Resting blood pressure (mm Hg)  
- `chol` – Serum cholesterol (mg/dl)  
- `fbs` – Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
- `restecg` – Resting ECG results (0–2)  
- `thalach` – Maximum heart rate achieved  
- `exang` – Exercise-induced angina (1 = yes, 0 = no)  
- `oldpeak` – ST depression induced by exercise  
- `slope` – Slope of the peak exercise ST segment (0–2)  
- `ca` – Number of major vessels (0–3) colored by fluoroscopy  
- `thal` – Thalium stress test result  

**Target (dependent variable):**

- `target` – 1 = heart disease present, 0 = no heart disease  

---

## 🛠 Tech Stack

- **Language:** Python  
- **Libraries:**
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`

---

## 🔍 Steps in the Notebook

### 1. Exploratory Data Analysis (EDA)

- View dataset shape and head  
- Summary statistics (`df.describe()`)  
- Check data types and missing values (`df.info()`)  
- Target distribution (`value_counts` + bar plot)  
- Crosstabs and plots:
  - Heart disease vs. **sex**
  - Heart disease vs. **chest pain type (cp)**
- Scatter plots:
  - **Age** vs **Max Heart Rate (thalach)** colored by target  
- Histograms:
  - Age distribution  
- Correlation matrix and **heatmap** for all numerical features  

### 2. Feature Preparation

- (Optional) Handling missing values  
- (Optional) Encoding categorical variables  
- Defining:
  ```python

  Train/Test split:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

3. Baseline Models

Trained and compared:

K-Nearest Neighbors (KNN)

Logistic Regression

Random Forest Classifier

Accuracy on the test set (approx):

KNN: ~0.69

Logistic Regression: ~0.89

Random Forest: ~0.84

Logistic Regression performs best and is chosen as the main model.

4. Hyperparameter Tuning

KNN tuned manually over different values of n_neighbors.

Logistic Regression and Random Forest tuned using:

RandomizedSearchCV

GridSearchCV (for Logistic Regression)

The best Logistic Regression model (with tuned C and solver) is used for final evaluation.

5. Model Evaluation (Beyond Accuracy)

For the best model:

ROC Curve and AUC using RocCurveDisplay.from_estimator

Confusion matrix + heatmap via seaborn

Classification report:

Precision

Recall

F1-score

Cross-validated metrics using cross_val_score:

Accuracy

Precision

Recall

F1-score

Example average cross-validated performance (approx):

Accuracy ≈ 0.85

Precision ≈ 0.82

Recall ≈ 0.93

F1-score ≈ 0.87

6. Feature Importance

Using Logistic Regression coefficients:

Extract clf.coef_

Map coefficients to feature names

Visualize as a bar chart to see which features contribute most to predicting heart disease.

▶️ How to Run This Project

Clone the repository

git clone https://github.com/<your-username>/Heart_Disease_Classification.ipynb.git
cd Heart_Disease_Classification.ipynb


Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows


Install dependencies

pip install -r requirements.txt


(or manually install numpy, pandas, matplotlib, seaborn, scikit-learn)

Launch Jupyter Notebook

jupyter notebook


Then open: Heart_Disease_Classification.ipynb and run all cells.

📌 Results & Discussion

Logistic Regression outperformed KNN and Random Forest on this dataset.

The model achieves ~88–89% test accuracy and strong recall for the positive (disease) class.

Feature importance analysis shows certain variables (e.g., chest pain type, slope, sex, ca, thal, oldpeak) have a stronger impact on predictions.

🚀 Possible Improvements / Future Work

Try more advanced models (e.g., XGBoost, CatBoost, LightGBM).

Add more preprocessing (scaling, outlier handling, better encoding).

Use more data from other heart disease datasets to improve generalization.

Deploy the model as an API or simple web app (e.g., Streamlit or FastAPI).
  X = df.drop(columns="target")
  y = df["target"].values
