Heart Disease Prediction using Machine Learning

This project applies supervised Machine Learning techniques to predict the likelihood of heart disease based on clinical attributes such as age, sex, cholesterol, resting blood pressure, and chest pain type. The goal is to build a predictive model that helps identify patients who may be at higher cardiovascular risk.

🚀 Key Highlights

✓ End-to-end ML Pipeline
✓ Exploratory Data Analysis
✓ Baseline models comparison
✓ Hyperparameter tuning
✓ ROC curve evaluation
✓ Feature importance analysis
✓ Fully implemented in Jupyter Notebook

This project demonstrates practical clinical predictive analytics using Python and Scikit-Learn.

🧠 Project Workflow

1️⃣ Load dataset
2️⃣ Data preparation
3️⃣ Exploratory data analysis (EDA)
4️⃣ Feature engineering
5️⃣ Train/test split
6️⃣ Train multiple ML models
7️⃣ Model selection
8️⃣ Hyperparameter tuning
9️⃣ Performance evaluation
🔟 Feature importance interpretation

📦 Dataset

This dataset contains 303 samples with 14 clinical features.

Attribute	Description
age	Age in years
sex	0 = female, 1 = male
cp	Chest pain type
trestbps	Resting blood pressure
chol	Serum cholesterol
fbs	Fasting blood sugar
restecg	Resting ECG results
thalach	Max heart rate
exang	Exercise angina
oldpeak	ST depression
slope	ST segment slope
ca	Number of vessels
thal	Thalassemia
target	1 = heart disease, 0 = no disease

Source: UCI Heart Disease Dataset (Kaggle formatted)

🛠 Technologies Used

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-Learn

Jupyter Notebook

🔍 Models Trained
Model	Status
Logistic Regression	⭐ Best
Random Forest	✓
K-Nearest Neighbors	✓
🔧 Hyperparameter Tuning

RandomizedSearchCV

GridSearchCV

Manual tuning for KNN

📊 Evaluation Metrics

Evaluated using:

Accuracy

Precision

Recall

F1 score

Cross-validation

Confusion matrix

ROC curve & AUC

✔ Final Results (Approx.)
Metric	Score
Accuracy	~0.88–0.89
Precision	~0.82
Recall	~0.93
F1 Score	~0.87

Logistic Regression performed the best

📈 Visualizations Included

Target distribution

Correlation heatmap

Chest pain vs. heart disease

Age vs. maximum heart rate

ROC curve

Feature importance coefficients

▶ How to Run This Project
git clone https://github.com/yourusername/Heart-Disease-Classification.git
cd Heart-Disease-Classification
jupyter notebook


Then open:

Heart_Disease_Classification.ipynb

📌 Use Case

Early detection of cardiovascular disease can help medical professionals identify high-risk patients and provide timely treatment decisions. ML-based screening tools provide insights that assist clinical judgement (not replace it).

🚀 Future Improvements

Try XGBoost / CatBoost / LightGBM

Build a Streamlit dashboard

Deploy using Flask/FastAPI

Use larger cardiology datasets

Experiment with deep learning models

🧩 File Structure
📁 Heart-Disease-Classification
│── Heart_Disease_Classification.ipynb
│── heart disease classification dataset.csv
│── LICENSE
│── README.md
│── .gitignore

📄 License

This project is licensed under the MIT License – you are free to use and modify it.

✨ Author

Priyanka Burra
🔗 GitHub: https://github.com/burrapriyanka85-pixel

🔗 LinkedIn: https://www.linkedin.com/in/priyankaburra
