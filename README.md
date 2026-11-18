# 👨‍💼 Employee Work Balance Analysis

A Machine Learning and Statistical Analytics project that investigates the relationship between employee work-life balance, job satisfaction, and attrition using the IBM HR Analytics Employee Attrition & Performance dataset. The project combines statistical inference, predictive modelling, and business recommendations to support data-driven Human Resource decision-making.

---

## 📌 Project Overview

Employee attrition is a major organizational challenge that increases recruitment costs, reduces productivity, and impacts employee morale. This project analyzes the key factors contributing to employee turnover and develops predictive models to identify employees at high risk of leaving.

Using the IBM HR Analytics dataset containing **1,470 employees** and **35 variables**, this project applies exploratory data analysis (EDA), statistical hypothesis testing, and machine learning to provide actionable HR insights and proposes an **Employee Wellbeing Early Warning Dashboard** for proactive retention strategies.

---

## 🎯 Objectives

- Identify the factors influencing employee attrition.
- Analyze work-life balance and employee satisfaction.
- Perform statistical hypothesis testing.
- Develop predictive machine learning models.
- Recommend evidence-based HR interventions.

---

## 📂 Dataset

**IBM HR Analytics Employee Attrition & Performance Dataset**

- **Records:** 1,470 employees
- **Features:** 35 variables
- **Target Variable:** Attrition (Yes / No)

Dataset includes employee information such as:

- Age
- Department
- Business Travel
- Job Role
- Monthly Income
- Overtime
- Work-Life Balance
- Job Satisfaction
- Environment Satisfaction
- Years at Company
- Performance Rating

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statistical Modelling
- Machine Learning
- Jupyter Notebook

---

## 📊 Exploratory Data Analysis

The project includes:

- Data cleaning
- Outlier analysis (IQR method)
- Distribution analysis
- Correlation analysis
- Feature relationship analysis
- Employee attrition visualization

Generated visualizations include:

- Attrition Overview
- Work-Life Balance Analysis
- Overtime Analysis
- Correlation Heatmaps
- Boxplots
- ROC Curves
- Confusion Matrices
- PCA Visualization
- Model Comparison

---

## 📈 Statistical Analysis

The following statistical techniques were performed:

- Independent t-test
- Two-Proportion Z-test
- One-Way ANOVA
- Chi-Square Test of Independence
- Levene's Test

These analyses were used to determine whether workplace factors had statistically significant relationships with employee attrition.

---

## 🤖 Machine Learning Models

The following classification models were developed and compared:

- Logistic Regression
- LASSO Regression
- Ridge Regression
- Gaussian Naïve Bayes

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## 🏆 Results

| Model | Accuracy | ROC-AUC |
|--------|----------|---------|
| Logistic Regression | 86.7% | 0.813 |
| Ridge Regression | - | - |
| Gaussian Naïve Bayes | - | - |
| **LASSO Regression** | **88.0%** | **0.813** |

LASSO Regression achieved the highest overall performance while maintaining model interpretability, making it the preferred model for employee attrition prediction.

---

## 🔍 Key Findings

The analysis revealed that:

- Employees working overtime had an attrition rate of **30.5%**, compared to **10.4%** for employees without overtime.
- Work-life balance has a statistically significant relationship with employee attrition.
- Business travel, overtime, time since last promotion, and environment satisfaction were among the strongest predictors of employee turnover.
- Raw accuracy alone can be misleading due to class imbalance; therefore, ROC-AUC and additional evaluation metrics were used for model comparison.

---

## 💡 Proposed Business Solution

Based on the findings, an **Employee Wellbeing Early Warning Dashboard** was proposed to help HR teams:

- Identify employees at high risk of attrition.
- Monitor employee wellbeing indicators.
- Support proactive retention strategies.
- Improve workforce planning.
- Reduce employee turnover costs.

---

## 📚 Learning Outcomes

This project strengthened my practical skills in:

- Exploratory Data Analysis (EDA)
- Statistical Hypothesis Testing
- Machine Learning Classification
- Logistic Regression
- Regularization Techniques (LASSO & Ridge)
- Model Evaluation
- Predictive Analytics
- HR Analytics
- Data-Driven Decision Making

---

## 🚀 Future Enhancements

- Deploy the predictive model as a web application.
- Develop an interactive HR dashboard using Power BI.
- Integrate Explainable AI techniques (SHAP/LIME).
- Compare advanced ensemble models such as XGBoost and Random Forest.
- Enable real-time employee attrition prediction.

---

## 👩‍💻 Author

**Tharsika Kanthavel**

BSc (Hons) Information Technology – Data Science  
Sri Lanka Institute of Information Technology (SLIIT)
