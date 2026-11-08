import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

st.set_page_config(page_title="ML Classifier App", layout="wide")
st.title("🧠 Interactive ML Classification Dashboard")
st.write("Evaluate and test 5 ML models on classification data containing 15 features.")

# Sidebar Model Loading
@st.cache_resource
def load_models():
    loaded_models = {}
    model_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "k-nearest_neighbor": "K-Nearest Neighbor",
        "naive_bayes_gaussian": "Naive Bayes",
        "support_vector_machine": "Support Vector Machine"
    }
    for file_prefix, display_name in model_names.items():
        path = f"model/{file_prefix}_model.joblib"
        if os.path.exists(path):
            try:
                loaded_models[display_name] = joblib.load(path)
            except Exception as e:
                st.sidebar.error(f"Failed to load {display_name}: {e}")
    return loaded_models

models = load_models()

# Check if any model loaded successfully
if not models:
    st.error("⚠️ No models were successfully loaded! Please verify that your 'model/' directory contains the `.joblib` binary files on GitHub.")
    st.stop()

# Sidebar options
app_mode = st.sidebar.selectbox("Choose Mode", ["Batch Evaluation", "Single Instance Prediction"])

if app_mode == "Batch Evaluation":
    st.header("📊 Batch Dataset Evaluation")
    uploaded_file = st.file_uploader("Upload your evaluation CSV (must contain 15 features + 'target')", type=["csv"])

    if uploaded_file is None:
        # Load local test_data.csv as fallback
        if os.path.exists("test_data.csv"):
            df = pd.read_csv("test_data.csv")
            st.info("Using sample 'test_data.csv' loaded from the repository.")
        else: 
            st.warning("Please upload a csv file or ensure test_data.csv is in the repository directory.")
            st.stop()
    else:
        df = pd.read_csv(uploaded_file)

    # Prepare features and target
    if "target" not in df.columns:
        st.error("The dataset must contain a column named 'target' for evaluation!")
        st.stop()

    X_test_data = df.drop(columns=["target"])
    y_test_data = df["target"]

    if X_test_data.shape[1] != 15:
        st.error(f"The model expects exactly 15 feature columns. Found {X_test_data.shape[1]} columns.")
        st.stop()

    results = {}
    for name, model in models.items():
        preds = model.predict(X_test_data)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_data)[:, 1]
        else:
            probs = model.decision_function(X_test_data)

        results[name] = {
            "Accuracy": accuracy_score(y_test_data, preds),
            "AUC Score": roc_auc_score(y_test_data, probs),
            "Precision": precision_score(y_test_data, preds),
            "Recall": recall_score(y_test_data, preds),
            "F1 Score": f1_score(y_test_data, preds),
            "MCC Score": matthews_corrcoef(y_test_data, preds)
        }

    st.table(pd.DataFrame(results).T.style.highlight_max(axis=0, color="lightgreen"))

elif app_mode == "Single Instance Prediction":
    st.header("🔬 Test with Individual Feature Inputs")
    st.write("Specify the values for all 15 features to observe model outputs in real time:")

    input_features = []
    cols = st.columns(3)
    for idx in range(15):
        with cols[idx % 3]:
            val = st.number_input(f"Feature {idx+1}", value=0.0, step=0.1)
            input_features.append(val)

    input_array = np.array(input_features).reshape(1, -1)

    # Dynamically populate the selectbox based on successfully loaded models only
    selected_model_name = st.selectbox("Select Model to Predict", list(models.keys()))
    
    if selected_model_name in models:
        model_obj = models[selected_model_name]
        prediction = model_obj.predict(input_array)[0]
        prob = model_obj.predict_proba(input_array)[0][1] if hasattr(model_obj, "predict_proba") else None

        st.markdown("--- ")
        st.subheader(f"Prediction Output ({selected_model_name})")
        st.write(f"**Predicted Class:** `{prediction}`")
        if prob is not None:
            st.write(f"**Confidence Probability for Class 1:** `{prob:.4f}`")
    else:
        st.error(f"Selected model '{selected_model_name}' could not be accessed.")
