# 📊 Smart Data Insight Generator

A beginner-friendly, AI-powered Streamlit app that lets you upload any CSV file and:
- 🔍 Explore data with summaries and visualizations
- 🧠 Automatically detect the ML problem type (Classification or Regression)
- 🤖 Train a machine learning model in one click
- ✅ View model accuracy or regression error (RMSE)

---

## 🚀 Features

- 📂 Upload any `.csv` dataset
- 🧪 Automatic data profiling:
  - Preview of data
  - Dataset shape, data types, and missing value counts
- 📊 Visualize any numeric column:
  - Histogram
  - Boxplot
- 🧠 ML Model Training:
  - Auto-detect classification or regression
  - One-click training with Random Forest
  - Show classification accuracy or regression RMSE

---

## 🌍 Live Demo

👉 [Try it now on Streamlit Cloud](https://your-streamlit-url-here.streamlit.app)  
*(Replace this link after deployment)*

---

## 🛠 Tech Stack

- 💻 Python
- 📊 Pandas, NumPy
- 📈 Matplotlib, Seaborn
- 🤖 Scikit-learn
- 🌐 Streamlit

---

## 💻 How to Run Locally

### Step 1: Install Requirements

```bash
pip install streamlit pandas matplotlib seaborn scikit-learn

streamlit run app.py
