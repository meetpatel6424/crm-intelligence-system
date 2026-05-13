import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="CRM Intelligence System",
    page_icon="📊",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("📊 CRM Intelligence System")
st.subheader("AI-Based Customer Conversion Prediction")

st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to convert or not."
)

# =========================
# LOAD MODEL
# =========================

# Load your trained model
# Make sure model.pkl exists

try:
    model = joblib.load("model.pkl")
except:
    st.warning("⚠ Model file not found. Please train model first.")
    st.stop()

# =========================
# USER INPUTS
# =========================

age = st.slider("Age", 18, 70, 25)

salary = st.number_input(
    "Annual Income",
    min_value=10000,
    max_value=1000000,
    value=50000
)

experience = st.slider(
    "Years of Experience",
    0,
    20,
    2
)

pages_visited = st.slider(
    "Website Pages Visited",
    1,
    50,
    5
)

emails_opened = st.slider(
    "Marketing Emails Opened",
    0,
    100,
    10
)

previous_purchases = st.slider(
    "Previous Purchases",
    0,
    20,
    1
)

# =========================
# CATEGORICAL INPUTS
# =========================

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

lead_source = st.selectbox(
    "Lead Source",
    ["Website", "Facebook", "LinkedIn", "Instagram"]
)

# =========================
# ENCODING
# =========================

gender = 1 if gender == "Male" else 0

married = 1 if married == "Yes" else 0

lead_source_mapping = {
    "Website": 0,
    "Facebook": 1,
    "LinkedIn": 2,
    "Instagram": 3
}

lead_source = lead_source_mapping[lead_source]

# =========================
# CREATE INPUT DATAFRAME
# =========================

input_data = pd.DataFrame({
    'age': [age],
    'salary': [salary],
    'experience': [experience],
    'pages_visited': [pages_visited],
    'emails_opened': [emails_opened],
    'previous_purchases': [previous_purchases],
    'gender': [gender],
    'married': [married],
    'lead_source': [lead_source]
})

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict Conversion"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📈 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Customer is likely to convert.")
    else:
        st.error("❌ Customer is unlikely to convert.")

    st.write(f"### Conversion Probability: {probability * 100:.2f}%")

    # =========================
    # BUSINESS INSIGHTS
    # =========================

    st.subheader("💡 Business Insights")

    if probability > 0.8:
        st.info(
            "High-value lead detected. "
            "Sales team should prioritize follow-up."
        )

    elif probability > 0.5:
        st.warning(
            "Moderate conversion chance. "
            "More engagement recommended."
        )

    else:
        st.error(
            "Low conversion probability. "
            "Consider remarketing campaigns."
        )

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption("Built using Python, Streamlit & Machine Learning")