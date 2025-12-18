# Heart Disease Classification

An end-to-end machine learning project that predicts the presence of heart disease using clinical and demographic attributes. The project demonstrates a complete ML workflow including data preprocessing, exploratory analysis, model training, evaluation, and interpretation.

---

## Overview

Cardiovascular diseases are among the leading causes of mortality worldwide. Early risk prediction using patient data can support preventive care and informed decision-making.

This project applies supervised machine learning techniques to classify whether a patient is likely to have heart disease based on routinely collected medical parameters.

---

## Key Features

- End-to-end machine learning pipeline  
- Exploratory Data Analysis (EDA)  
- Data preprocessing and feature engineering  
- Training and comparison of multiple classification models  
- Model evaluation using accuracy, ROC curve, and confusion matrix  
- Feature importance analysis for interpretability  
- Implemented using Python and Scikit-Learn  

---

## Project Structure

```text
Heart_Disease_Classification/
│
├── data/
│   └── heart disease classification dataset.csv
│
├── Heart_Disease_Classification.ipynb
├── README.md
├── requirements.txt
└── .gitignore
Dataset
The dataset contains anonymized clinical attributes such as:

Age

Sex

Chest pain type

Resting blood pressure

Cholesterol levels

Fasting blood sugar

Maximum heart rate

Exercise-induced angina

The target variable indicates the presence or absence of heart disease.

Technologies Used
Programming Language
Python

Libraries
NumPy

Pandas

Matplotlib

Seaborn

Scikit-Learn

Environment
Jupyter Notebook

VS Code

Version Control
Git & GitHub

Model Workflow
Load and inspect the dataset

Perform exploratory data analysis (EDA)

Handle missing values and preprocess features

Split data into training and testing sets

Train multiple classification models

Evaluate models using performance metrics

Analyze feature importance

Interpret results

How to Run the Project
1. Clone the repository
bash
Copy code
git clone https://github.com/burrapriyanka85-pixel/Heart-Disease-Classification.git
cd Heart-Disease-Classification
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
3. Run the notebook
bash
Copy code
jupyter notebook
Open Heart_Disease_Classification.ipynb and run all cells.

Results
Models successfully classify the presence of heart disease

Performance evaluated using standard classification metrics

Feature importance highlights key clinical indicators influencing predictions

This project demonstrates the practical application of machine learning in healthcare analytics.

Future Improvements
Convert the notebook into modular Python scripts

Add model deployment using Streamlit

Include cross-validation and advanced ensemble models

Integrate a real-time patient input interface

Author
Priyanka Burra

License
This project is licensed under the MIT License.
