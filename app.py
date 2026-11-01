
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. Basic application settings
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Classification")
st.write(
    "This application evaluates pre-trained machine learning "
    "classifiers on uploaded bank marketing test data."
)

st.info(
    "Upload a labelled CSV file containing the model input features "
    "and the target column 'y'. The application uses saved models "
    "and does not perform any model training."
)

# --------------------------------------------------
# 2. Load the saved machine learning models
# --------------------------------------------------

MODEL_PATH = Path("model/trained_models.joblib")


@st.cache_resource
def load_saved_models():
    """Load the five trained pipelines from the model artifact."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find the model artifact: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    trained_models = load_saved_models()
except Exception as error:
    st.error(f"Unable to load the saved models: {error}")
    st.stop()


# --------------------------------------------------
# 3. Identify the required input features
# --------------------------------------------------

# The feature names are taken from the fitted pipeline.
required_features = list(
    trained_models["Logistic Regression"].feature_names_in_
)

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a classification model",
    list(trained_models.keys())
)

st.sidebar.caption(
    "All selected models are loaded from the saved artifact."
)


# --------------------------------------------------
# 4. Upload the labelled test dataset
# --------------------------------------------------

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="The CSV must contain all 19 input features and the target column y."
)

if uploaded_file is None:
    st.warning("Please upload a labelled CSV file to begin evaluation.")
    st.stop()


# --------------------------------------------------
# 5. Read and validate the uploaded data
# --------------------------------------------------

try:
    uploaded_data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read the uploaded CSV file: {error}")
    st.stop()


# Remove accidental index columns, if present.
uploaded_data = uploaded_data.loc[
    :, ~uploaded_data.columns.str.startswith("Unnamed:")
]

if "y" not in uploaded_data.columns:
    st.error("The uploaded CSV must contain a target column named 'y'.")
    st.stop()

missing_features = [
    feature for feature in required_features
    if feature not in uploaded_data.columns
]

if missing_features:
    st.error(
        "The uploaded file is missing these required features: "
        + ", ".join(missing_features)
    )
    st.stop()

if uploaded_data.empty:
    st.error("The uploaded CSV does not contain any records.")
    st.stop()

# Keep the original feature order used during model training.
X_uploaded = uploaded_data[required_features].copy()

# Convert the target labels into the numerical format used in training.
target_mapping = {
    "no": 0,
    "yes": 1,
    "0": 0,
    "1": 1
}

y_uploaded = (
    uploaded_data["y"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(target_mapping)
)

if y_uploaded.isna().any():
    st.error(
        "The target column contains unsupported values. "
        "Expected values are 'yes' and 'no' (or 1 and 0)."
    )
    st.stop()

y_uploaded = y_uploaded.astype(int)

if y_uploaded.nunique() < 2:
    st.error(
        "AUC cannot be evaluated because the uploaded data "
        "contains only one target class. Please upload data "
        "containing both 'yes' and 'no' records."
    )
    st.stop()


# --------------------------------------------------
# 6. Generate predictions using the selected model
# --------------------------------------------------

selected_model = trained_models[selected_model_name]

try:
    predictions = selected_model.predict(X_uploaded)

    # Logistic Regression, Decision Tree, KNN and Gaussian NB
    # provide probabilities. Linear SVM provides decision scores.
    if hasattr(selected_model, "predict_proba"):
        prediction_scores = selected_model.predict_proba(
            X_uploaded
        )[:, 1]
    else:
        prediction_scores = selected_model.decision_function(
            X_uploaded
        )

except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()


# --------------------------------------------------
# 7. Calculate model evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(y_uploaded, predictions)

auc = roc_auc_score(
    y_uploaded,
    prediction_scores
)

precision = precision_score(
    y_uploaded,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_uploaded,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_uploaded,
    predictions,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_uploaded,
    predictions
)


# --------------------------------------------------
# 8. Display the evaluation results
# --------------------------------------------------

st.divider()

st.subheader("Evaluation Results")

st.write(f"**Selected model:** {selected_model_name}")
st.write(f"**Number of evaluated records:** {len(X_uploaded):,}")

metric_columns = st.columns(3)

metric_values = [
    ("Accuracy", accuracy),
    ("AUC", auc),
    ("Precision", precision),
    ("Recall", recall),
    ("F1 Score", f1),
    ("MCC", mcc)
]

for index, (metric_name, metric_value) in enumerate(metric_values):
    with metric_columns[index % 3]:
        st.metric(
            label=metric_name,
            value=f"{metric_value:.4f}"
        )


# --------------------------------------------------
# 9. Display the confusion matrix
# --------------------------------------------------

st.divider()
st.subheader("Confusion Matrix")

matrix = confusion_matrix(
    y_uploaded,
    predictions,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(6, 4))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["No", "Yes"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title(f"Confusion Matrix - {selected_model_name}")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 10. Display the classification report
# --------------------------------------------------

st.divider()
st.subheader("Classification Report")

report = classification_report(
    y_uploaded,
    predictions,
    labels=[0, 1],
    target_names=["No", "Yes"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# --------------------------------------------------
# 11. Show a preview of the uploaded data
# --------------------------------------------------

with st.expander("View uploaded dataset"):
    st.dataframe(
        uploaded_data.head(10),
        use_container_width=True
    )

st.success(
    "Evaluation completed successfully using the saved model. "
    "No model retraining was performed."
)
