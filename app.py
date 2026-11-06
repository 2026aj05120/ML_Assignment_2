"""Streamlit app: upload TEST data, pick a saved model, see metrics + confusion matrix."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve

from ml_utils import compute_metrics, METRIC_COLUMNS

st.set_page_config(page_title="Global Salary Data by Skill & Role", page_icon="🧪", layout="wide")


@st.cache_resource
def load_artifacts():
    meta = joblib.load("model/meta.joblib")
    models = {n: joblib.load(f"model/{f}") for n, f in meta["model_files"].items()}
    return meta, models


meta, models = load_artifacts()
target, features = meta["target"], meta["feature_cols"]
le, class_names = meta["label_encoder"], meta["class_names"]
n_classes = len(class_names)

st.title("🧪 Global Salary Data by Skill & Role")
st.caption("Evaluate 5 pre-trained classification models on your own test data. "
           "Models are loaded from disk — nothing is retrained.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    use_sample = st.checkbox("Use bundled test_data.csv instead", value=False)
    st.header("2. Model")
    model_name = st.selectbox("Choose a model", list(models.keys()))
    st.markdown(f"**Target column:** `{target}`  \n**Classes:** {', '.join(class_names)}")

# ---------------- Load data (always into `data`) ----------------
if uploaded is not None:
    data = pd.read_csv(uploaded)
elif use_sample:
    data = pd.read_csv("test_data.csv")
else:
    st.info("👈 Upload a test CSV (or tick the bundled-sample box) to begin. "
            f"The file must contain the {len(features)} feature columns and the `{target}` column.")
    with st.expander("Required columns"):
        st.write(features + [target])
    st.stop()

# ---------------- Validate the data ----------------
missing = [c for c in features + [target] if c not in data.columns]
if missing:
    st.error(f"Missing required columns in dataset: {missing}")
    st.stop()

known = set(le.classes_.astype(str))
data = data.dropna(subset=[target])
if not set(data[target].astype(str)).issubset(known):
    st.error(f"Target column contains labels not seen in training. Expected: {class_names}")
    st.stop()

X = data[features]
y_true = le.transform(data[target].astype(le.classes_.dtype))

with st.expander("Preview uploaded data", expanded=False):
    st.write(f"{data.shape[0]} rows × {data.shape[1]} columns")
    st.dataframe(data.head(20), use_container_width=True)

# ---------------- Selected model ----------------
pipe = models[model_name]
y_pred = pipe.predict(X)
y_proba = pipe.predict_proba(X)
metrics = compute_metrics(y_true, y_pred, y_proba, n_classes)

st.subheader(f"📊 {model_name} — evaluation metrics")
cols = st.columns(6)
for col, name in zip(cols, METRIC_COLUMNS):
    val = metrics[name]
    col.metric(name, "n/a" if np.isnan(val) else f"{val:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y_true, y_pred, labels=np.arange(n_classes),
                                target_names=class_names, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

if n_classes == 2:
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.4))
    ax2.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
    ax2.legend(); ax2.set_title("ROC curve")
    st.pyplot(fig2)

# ---------------- All models side by side ----------------
st.subheader("🏁 All five models on this data")
rows = []
for name, m in models.items():
    res = compute_metrics(y_true, m.predict(X), m.predict_proba(X), n_classes)
    rows.append({"ML Model Name": name, **res})
table = pd.DataFrame(rows).set_index("ML Model Name")
st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#1f6f4a"),
             use_container_width=True)
st.bar_chart(table[["Accuracy", "F1", "MCC"]])
st.caption(f"Best by MCC on this data: **{table['MCC'].idxmax()}**")