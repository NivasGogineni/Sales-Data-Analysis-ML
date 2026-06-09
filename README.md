# 📊 Sales Data Analysis & Machine Learning

This project focuses on analyzing sales data and building a machine learning model to predict future sales revenue. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, visualization, and predictive modeling using Random Forest Regression.

---

## 🚀 Features

* 📌 Data Cleaning & Preprocessing
* 🔄 Missing Value Handling
* 🔢 Categorical Data Encoding (One-Hot Encoding)
* 📊 Exploratory Data Analysis (EDA)
* 📈 Data Visualization using Matplotlib & Seaborn
* 🤖 Sales Prediction using Random Forest Regression
* 📉 Model Evaluation using MAE, RMSE, and R² Score
* 📋 Feature Importance Analysis

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Project Structure

Sales-Data-Analysis-ML/

├── sales.py

├── sales_data_sample.csv

├── README.md

---

## 📊 Workflow

1. Load dataset using Pandas
2. Handle missing values using statistical methods
3. Perform feature engineering on date columns
4. Encode categorical variables using One-Hot Encoding
5. Conduct Exploratory Data Analysis (EDA)
6. Visualize sales trends and patterns
7. Split data into training and testing sets
8. Train Random Forest Regressor
9. Evaluate model performance
10. Analyze feature importance

---

## 📈 Model Performance

### Random Forest Regressor

| Metric   | Score      |
| -------- | ---------- |
| MAE      | 302.59     |
| MSE      | 273,421.91 |
| RMSE     | 522.90     |
| R² Score | 0.9206     |

### Training vs Testing Performance

| Metric   | Score  |
| -------- | ------ |
| Train R² | 0.9908 |
| Test R²  | 0.9206 |

The model explains approximately **92% of the variance in sales data**, demonstrating strong predictive performance.

---

## 📊 Visualizations

* Sales Distribution Analysis
* Correlation Heatmap
* Monthly Sales Trends
* Actual vs Predicted Sales
* Feature Importance Analysis
* Country-wise Sales Performance

---

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

python sales.py
```

---

## 🎯 Project Outcome

Built an end-to-end machine learning pipeline for sales forecasting using Random Forest Regression. The model achieved an R² score of 92.06%, enabling accurate prediction of sales revenue based on historical order data, product information, customer location, and deal characteristics.

---

## 👨‍💻 Author

Nivas G
Data Analyst | Machine Learning Enthusiast
