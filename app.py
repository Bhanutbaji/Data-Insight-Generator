import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.multiclass import type_of_target

st.title("Smart Data Insight Generator")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write(" Preview of Data:")
    st.write(df.head())

    st.write(" Shape of Data:", df.shape)
    st.write(" Data Types:")
    st.write(df.dtypes)
    st.write(" Missing Values:")
    st.write(df.isnull().sum())

    st.subheader(" Visualizations")
    column = st.selectbox("Choose a column to visualize", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write("**Histogram:**")
        fig1, ax1 = plt.subplots()
        df[column].hist(ax=ax1)
        st.pyplot(fig1)

        st.write("**Boxplot:**")
        fig2, ax2 = plt.subplots()
        sns.boxplot(y=df[column], ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Please select a numeric column for visualization.")

    # ML section
    st.subheader(" Train a Simple ML Model")
    target = st.selectbox(" Select the target column for prediction:", df.columns)

    if target:
        df_clean = df.dropna()
        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        problem_type = type_of_target(y)

        if problem_type in ['binary', 'multiclass']:
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f" Classification Accuracy: {acc * 100:.2f}%")
        else:
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            st.success(f" Regression RMSE: {rmse:.2f}")
