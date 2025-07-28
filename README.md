
# 🧪 Hepatitis C Comparative Analysis Using Machine Learning

This project performs a **comparative analysis of basic machine learning models** on a Hepatitis C dataset. The goal is to evaluate and compare different classification models for predicting whether a patient is affected or not, based on clinical and diagnostic features.

---

## 📌 Objective

- Apply **7 basic ML classification algorithms**.
- Perform data preprocessing and feature analysis.
- Evaluate model performance using common evaluation metrics.
- Compare models using visualization and tabular comparison.

---

## 🧰 Tools & Technologies

- Python 3.x
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebooks (optional)

---

## 🧪 Models Used

1. Logistic Regression  
2. K-Nearest Neighbors (KNN)  
3. Decision Tree  
4. Support Vector Machine (SVM)  
5. Naive Bayes  
6. Random Forest
7. XGBoost

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 🔁 Workflow

1. **Data Collection**  
   - Load Hepatitis C dataset.

2. **Preprocessing**  
   - Handle missing values
   - Encode categorical features
   - Normalize/scale data

3. **Model Training**  
   - Train 6 models using `train_test_split`

4. **Evaluation**  
   - Evaluate models using multiple metrics
   - Save performance scores

5. **Comparison**  
   - Plot results (bar charts, confusion matrices)
   - Identify best-performing model

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/moozaheed/hepatitis-c-ml-analysis.git
cd hepatitis-ml-analysis

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

---

## 📎 Dataset Reference

- [Kaggle Dataset](https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset)

---

## 🧑‍💻 Author

**Your Name**  
Email: moozaheed@gmail.com 
GitHub: [G M Mozahed](https://github.com/moozaheed)

---

## 📄 License

This project is licensed under the MIT License.