# app/app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.title("❤️ ECG Anomaly Detection")
st.markdown("Upload ECG feature CSV to detect anomalies using Isolation Forest.")

# Load models
model = joblib.load("models/ecg_iso_forest_model.pkl")
scaler = joblib.load("models/ecg_scaler.pkl")

uploaded_file = st.file_uploader("Upload ECG feature file (.csv)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X = scaler.transform(df)
    df["anomaly"] = model.predict(X)

    st.success("Anomaly detection complete.")

    st.line_chart(df["mean"])

    fig, ax = plt.subplots()
    ax.plot(df["mean"], label="ECG Mean")
    ax.scatter(df.index[df["anomaly"] == -1], df["mean"][df["anomaly"] == -1], color="red", label="Anomaly")
    ax.legend()
    st.pyplot(fig)

    st.dataframe(df)
