# 🫀 Heart Disease Prediction – Machine Learning Full Pipeline  

This project implements a **comprehensive machine learning pipeline** on the **UCI Heart Disease dataset**. The goal is to analyze, predict, and visualize heart disease risks using supervised and unsupervised learning techniques.  

## 📌 Project Overview  
- **Data Preprocessing:** Handling missing values, encoding categorical variables, and feature scaling  
- **Exploratory Data Analysis (EDA):** Correlation heatmaps, histograms, boxplots, and feature insights  
- **Dimensionality Reduction:** Principal Component Analysis (PCA)  
- **Supervised Learning Models:** Logistic Regression, Decision Trees, Random Forest, Support Vector Machine (SVM)  
- **Unsupervised Learning Models:** K-Means and Hierarchical Clustering  
- **Hyperparameter Tuning:** GridSearchCV for model optimization  
- **Deployment:** Interactive **Streamlit UI** with Ngrok for sharing  

---

## 📂 Project Structure  
import os

def print_structure(startpath, indent=""):
    for i, element in enumerate(os.listdir(startpath)):
        path = os.path.join(startpath, element)
        is_last = i == len(os.listdir(startpath)) - 1
        prefix = "└── " if is_last else "├── "
        print(indent + prefix + element)
        if os.path.isdir(path):
            new_indent = indent + ("    " if is_last else "│   ")
            print_structure(path, new_indent)

# Run this in your project root
print("Heart_Disease_Project/")
print_structure(".")

## 🚀 How to Run  
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Heart_disease_Project.git
   cd Heart_disease_Project

Install dependencies:

pip install -r requirements.txt


Launch the Streamlit app:

streamlit run ui/app.py


📊 Results & Insights

Visual insights into factors influencing heart disease

Model evaluation metrics (accuracy, precision, recall, F1-score)

Clustering analysis for hidden patterns in patient data

🛠️ Tech Stack

Python, Pandas, NumPy, Scikit-learn

Matplotlib, Seaborn (EDA & visualization)

Streamlit (UI)

GitHub + Ngrok (deployment & hosting)

📌 Future Improvements

Add deep learning models

Deploy on Heroku / Render instead of Ngrok

Enhance Streamlit UI with patient history upload
