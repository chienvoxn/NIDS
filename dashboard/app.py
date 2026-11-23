import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from sklearn.metrics import accuracy_score, f1_score
from nids.deployment_model import TokenizedCNNSEAttention


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "models" / "final"

st.set_page_config(
    page_title="NIDS Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Network Intrusion Detection System")
st.caption(
    "CNN + SE + Multi-Head Self-Attention | "
    "V10 SMOTE900 | Offline Flow Detection | CPU"
)


@st.cache_resource(show_spinner="Loading trained CNN model...")
def load_engine():
    manifest = json.loads(
        (ARTIFACTS / "deployment_manifest.json")
        .read_text(encoding="utf-8")
    )

    if manifest["architecture"] != "T4_TokenCNN_SE_MHSA":
        raise ValueError("Unsupported model architecture")

    prep = joblib.load(ARTIFACTS / "preprocessing.joblib")

    features = manifest["feature_order"]
    labels = manifest["label_order"]

    if len(features) != 60 or len(labels) != 15:
        raise ValueError("Invalid manifest dimensions")

    if list(prep["features"]) != features:
        raise ValueError("Preprocessing feature mismatch")

    if list(prep["label_encoder"].classes_) != labels:
        raise ValueError("Preprocessing label mismatch")

    ckpt = torch.load(
        ARTIFACTS / "model.pt",
        map_location="cpu",
        weights_only=True,
    )

    if ckpt["arch"] != manifest["architecture"]:
        raise ValueError("Checkpoint architecture mismatch")

    if ckpt["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Checkpoint split mismatch")

    if list(ckpt["features"]) != features:
        raise ValueError("Checkpoint feature mismatch")

    if list(ckpt["classes"]) != labels:
        raise ValueError("Checkpoint label mismatch")

    cfg = manifest["model_config"]

    model = TokenizedCNNSEAttention(
        n_features=60,
        n_classes=15,
        d_model=cfg["d_model"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
        use_cnn=True,
        use_se=True,
        use_attn=True,
        use_raw=False,
    )

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    return model, prep, manifest


def predict_flows(df, model, prep, manifest):
    features = manifest["feature_order"]
    labels = manifest["label_order"]

    if df.columns.duplicated().any():
        raise ValueError("Duplicate columns detected")

    missing = [c for c in features if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required features: " + ", ".join(missing)
        )

    if df.empty:
        raise ValueError("CSV contains no rows")

    numeric = df.loc[:, features].apply(
        pd.to_numeric, errors="coerce"
    )

    numeric = numeric.replace(
        [np.inf, -np.inf], np.nan
    )

    X = prep["imputer"].transform(numeric)
    X = prep["scaler"].transform(X)
    X = np.asarray(X, dtype=np.float32)

    if X.shape != (len(df), 60):
        raise ValueError("Unexpected feature matrix shape")

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values after preprocessing")

    benign_idx = labels.index("BENIGN")
    all_probs = []

    with torch.inference_mode():
        for start in range(0, len(X), 128):
            batch = torch.from_numpy(
                X[start:start + 128]
            )

            logits = model(batch)
            logits[:, benign_idx] += float(
                manifest["benign_logit_bias"]
            )

            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs)
    indices = probs.argmax(axis=1)

    predictions = pd.DataFrame({
        "predicted": [labels[i] for i in indices],
        "confidence": probs.max(axis=1),
    })

    predictions["is_attack"] = (
        predictions["predicted"] != "BENIGN"
    )

    if "Label" in df.columns:
        predictions.insert(
            0, "actual", df["Label"].to_numpy()
        )

    return predictions


st.sidebar.header("Model Configuration")
st.sidebar.write("Architecture: T4 TokenCNN-SE-MHSA")
st.sidebar.write("Dataset: CIC-IDS2017")
st.sidebar.write("Input: 60 flow features")
st.sidebar.write("Output: 15 classes")
st.sidebar.write("Inference: CPU")
st.sidebar.info(
    "Offline flow classification. "
    "Raw PCAP and live packet capture are not supported."
)

uploaded = st.file_uploader(
    "Upload CIC-IDS2017 flow CSV",
    type=["csv"],
)

if uploaded is None:
    st.info(
        "Upload a CSV containing all 60 required "
        "flow features to start detection."
    )
    st.stop()

if uploaded.size > 50 * 1024 * 1024:
    st.error("For this demo, CSV files must be <= 50 MB.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

    if len(df) > 20000:
        raise ValueError(
            "Demo limit: maximum 20,000 flows per upload"
        )

    model, prep, manifest = load_engine()

    with st.spinner("Running CNN inference on CPU..."):
        result = predict_flows(df, model, prep, manifest)

except Exception as exc:
    st.error(f"Detection failed: {exc}")
    st.stop()


total = len(result)
attacks = int(result["is_attack"].sum())
benign = total - attacks
attack_rate = attacks / total * 100

st.success("Inference completed successfully")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Analyzed Flows", f"{total:,}")
c2.metric("Predicted BENIGN", f"{benign:,}")
c3.metric("Predicted ATTACK", f"{attacks:,}")
c4.metric("Predicted Attack Rate", f"{attack_rate:.2f}%")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Traffic Classification")
    counts = result["predicted"].value_counts()
    st.bar_chart(counts)

with right:
    st.subheader("BENIGN vs ATTACK")
    summary = pd.DataFrame({
        "Type": ["BENIGN", "ATTACK"],
        "Flows": [benign, attacks],
    }).set_index("Type")
    st.bar_chart(summary)

if "actual" in result.columns:
    st.divider()
    st.subheader("Sample Evaluation")

    accuracy = accuracy_score(
        result["actual"], result["predicted"]
    )
    macro_f1 = f1_score(
        result["actual"],
        result["predicted"],
        labels=manifest["label_order"],
        average="macro",
        zero_division=0,
    )

    a, b = st.columns(2)
    a.metric("Sample Accuracy", f"{accuracy * 100:.2f}%")
    b.metric("Sample Macro-F1", f"{macro_f1 * 100:.2f}%")

    st.warning(
        "These metrics are computed on the uploaded CSV. "
        "They are not independent holdout results."
    )

st.divider()

st.subheader("Detection Results")

show_only_attacks = st.checkbox(
    "Show predicted attacks only",
    value=False,
)

display_df = result

if show_only_attacks:
    display_df = result[result["is_attack"]]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Predicted Attack Alerts")

alerts = result[result["is_attack"]].copy()

if not alerts.empty:
    alerts.insert(0, "flow_row", alerts.index + 1)
    alerts = alerts.sort_values(
        "confidence", ascending=False
    )
    st.dataframe(
        alerts.head(100),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No attacks predicted in this CSV.")

st.download_button(
    "Download Prediction CSV",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="nids_v10_predictions.csv",
    mime="text/csv",
)

st.caption(
    "Confidence represents the maximum softmax "
    "probability, not a calibrated certainty score. "
    "Predictions are for completed flow feature vectors."
)
