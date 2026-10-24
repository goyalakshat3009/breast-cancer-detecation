import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from tensorflow import keras

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("model.keras")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


@st.cache_data
def load_reference():
    ds = load_breast_cancer(as_frame=True)
    return ds.data, ds.target


model, scaler = load_artifacts()
X_ref, y_ref = load_reference()
features = list(X_ref.columns)

st.title("🩺 Breast Cancer Tumor Classifier")
st.warning(
    "Educational demo trained on the Wisconsin Diagnostic dataset. "
    "It is NOT a medical device and must not be used for diagnosis."
)

# --- Sample loader -------------------------------------------------------
if "values" not in st.session_state:
    st.session_state["values"] = X_ref.median().to_dict()

with st.sidebar:
    st.header("Quick start")
    st.write("Load a real example from the dataset, then tweak the values.")
    if st.button("Load a malignant example"):
        st.session_state["values"] = X_ref[y_ref == 0].sample(1).iloc[0].to_dict()
    if st.button("Load a benign example"):
        st.session_state["values"] = X_ref[y_ref == 1].sample(1).iloc[0].to_dict()
    if st.button("Reset to median"):
        st.session_state["values"] = X_ref.median().to_dict()

# --- Inputs, grouped by mean / error / worst -----------------------------
groups = {
    "Mean features": [f for f in features if f.startswith("mean ")],
    "Error (standard error) features": [f for f in features if f.endswith(" error")],
    "Worst features": [f for f in features if f.startswith("worst ")],
}

inputs = {}
tabs = st.tabs(list(groups))
for tab, (_, cols) in zip(tabs, groups.items()):
    with tab:
        grid = st.columns(3)
        for i, name in enumerate(cols):
            lo, hi = float(X_ref[name].min()), float(X_ref[name].max())
            inputs[name] = grid[i % 3].number_input(
                name,
                min_value=lo * 0.5,
                max_value=hi * 1.5,
                value=float(st.session_state["values"][name]),
                format="%.5f",
                key=f"{name}_{hash(tuple(st.session_state['values'].values()))}",
            )

if st.button("Predict", type="primary"):
    row = pd.DataFrame([inputs], columns=features)
    probs = model.predict(scaler.transform(row), verbose=0)[0]
    probs = probs / probs.sum()  # guard against non-normalised outputs
    label = int(np.argmax(probs))

    if label == 0:
        st.error(f"Prediction: **Malignant** ({probs[0]:.1%} confidence)")
    else:
        st.success(f"Prediction: **Benign** ({probs[1]:.1%} confidence)")

    st.bar_chart(pd.DataFrame({"probability": probs}, index=["Malignant", "Benign"]))