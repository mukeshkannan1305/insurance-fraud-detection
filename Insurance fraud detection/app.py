import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer
import shap

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Explainable Insurance Fraud Detection",
    layout="wide"
)

# ----------------------------------
# Load Dataset
# ----------------------------------
data = pd.read_csv("dataset.csv")

X = data.drop("fraud", axis=1)
y = data["fraud"]
feature_names = X.columns.tolist()

# ----------------------------------
# Train-Test Split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ----------------------------------
# XGBoost Model
# ----------------------------------
model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# ----------------------------------
# LIME Explainer
# ----------------------------------
lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=["Genuine", "Fraud"],
    mode="classification"
)

# ----------------------------------
# SHAP (Future Enhancement Tool)
# ----------------------------------
shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(X_test)

# ----------------------------------
# UI Header
# ----------------------------------
st.title("🚗 Explainable & Policy-Aware Insurance Fraud Detection")
st.markdown("**XGBoost + LIME (Present) | SHAP (Future Enhancement)**")

# ----------------------------------
# Sidebar Inputs (Frontend)
# ----------------------------------
st.sidebar.header("📝 Claim Details")

age = st.sidebar.slider("Customer Age", 18, 80, 30)
claim_amount = st.sidebar.number_input("Claim Amount (₹)", 5000, 1000000, 200000)
policy_years = st.sidebar.slider("Policy Duration (Years)", 1, 20, 3)
accident_severity = st.sidebar.slider("Accident Severity (1–5)", 1, 5, 3)
previous_claims = st.sidebar.slider("Previous Claims", 0, 10, 1)
policy_score = st.sidebar.slider("Policy Compliance Score", 0, 100, 70)

input_data = np.array([[
    age,
    claim_amount,
    policy_years,
    accident_severity,
    previous_claims,
    policy_score
]])

# ----------------------------------
# Prediction
# ----------------------------------
prediction = model.predict(input_data)[0]
fraud_prob = model.predict_proba(input_data)[0][1]

# ----------------------------------
# Result Display
# ----------------------------------
st.subheader("🔍 Fraud Detection Result")

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.error(f"⚠️ FRAUD DETECTED\n\nProbability: {fraud_prob:.2f}")
    else:
        st.success(f"✅ GENUINE CLAIM\n\nFraud Probability: {fraud_prob:.2f}")

with col2:
    st.info(f"📊 Model Accuracy: {accuracy:.2f}")

# ----------------------------------
# Explainability – LIME
# ----------------------------------
st.subheader("🧠 Explainability using LIME")

lime_exp = lime_explainer.explain_instance(
    input_data[0],
    model.predict_proba,
    num_features=6
)

lime_df = pd.DataFrame(lime_exp.as_list(), columns=["Feature", "Impact"])
st.bar_chart(lime_df.set_index("Feature"))

# ----------------------------------
# Future Enhancement – SHAP
# ----------------------------------
st.subheader("🚀 Future Enhancement: Global Explainability (SHAP)")

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# ----------------------------------
# Intelligent Claim Report
# ----------------------------------
st.subheader("📄 Automated Claim Report")

decision = "REJECTED (High Fraud Risk)" if prediction == 1 else "APPROVED"

report = f"""
INSURANCE CLAIM REPORT
--------------------------------
Customer Age        : {age}
Claim Amount        : ₹{claim_amount}
Policy Duration     : {policy_years} years
Accident Severity   : {accident_severity}
Previous Claims     : {previous_claims}
Policy Score        : {policy_score}

Fraud Probability   : {fraud_prob:.2f}
Final Decision      : {decision}

Model               : XGBoost
Explainability      : LIME
Future Enhancement  : SHAP
"""

st.text(report)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.markdown("🎓 AI & Data Science Final Year Project | Explainable AI")
