import json
import warnings
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.exceptions import InconsistentVersionWarning
from nids.deployment_model import TokenizedCNNSEAttention


ROOT = Path("models/final")
ARCHIVE = Path(
    "/mnt/c/Users/PC/Desktop/ATTT/nids_outputs_backup.zip"
)
SAMPLE = (
    "nids_outputs_backup/nids_outputs/"
    "01_CIC_EDA/cic_pilot_sample_300_per_label.csv"
)
OUTPUT = Path("outputs/v10_demo_predictions.csv")

manifest = json.loads(
    (ROOT / "deployment_manifest.json").read_text(encoding="utf-8")
)

features = manifest["feature_order"]
labels = manifest["label_order"]

# Load sample CSV
with ZipFile(ARCHIVE) as z:
    df = pd.read_csv(BytesIO(z.read(SAMPLE)))

df.columns = df.columns.str.strip()

if df.columns.duplicated().any():
    raise ValueError("Duplicate CSV column names")

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Missing features: {missing}")

print("Rows:", len(df))
print("Features:", len(features))

# Load trained preprocessing
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", InconsistentVersionWarning)
    prep = joblib.load(ROOT / "preprocessing.joblib")

if any(
    issubclass(w.category, InconsistentVersionWarning)
    for w in caught
):
    print(
        "WARNING: sklearn version mismatch. "
        "Predictions are provisional until version compatibility is verified."
    )

assert list(prep["features"]) == features
assert list(prep["label_encoder"].classes_) == labels

# Convert numeric columns and handle infinities as missing
X = df.loc[:, features].apply(
    pd.to_numeric, errors="coerce"
)
X = X.replace([np.inf, -np.inf], np.nan)

X = prep["imputer"].transform(X)
X = prep["scaler"].transform(X)
X = np.asarray(X, dtype=np.float32)

assert X.shape == (len(df), 60)
assert np.isfinite(X).all()

# Load trained CNN
ckpt = torch.load(
    ROOT / "model.pt",
    map_location="cpu",
    weights_only=True
)

assert ckpt["arch"] == manifest["architecture"]
assert ckpt["split_sha256"] == manifest["split_sha256"]
assert list(ckpt["features"]) == features
assert list(ckpt["classes"]) == labels

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

# CPU inference
all_probs = []

with torch.inference_mode():
    for start in range(0, len(X), 128):
        batch = torch.from_numpy(X[start:start + 128])
        logits = model(batch)

        logits[:, 0] += manifest["benign_logit_bias"]

        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.numpy())

probabilities = np.concatenate(all_probs, axis=0)

pred_indices = probabilities.argmax(axis=1)
predictions = [labels[i] for i in pred_indices]
confidence = probabilities.max(axis=1)

result = pd.DataFrame({
    "predicted": predictions,
    "confidence": confidence,
    "is_attack": [p != "BENIGN" for p in predictions],
})

if "Label" in df:
    result.insert(0, "actual", df["Label"].to_numpy())

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("\n=== INFERENCE RESULTS ===")
print("Total flows:", len(result))
print("BENIGN:", int((~result["is_attack"]).sum()))
print("ATTACK:", int(result["is_attack"].sum()))
print("\nPredicted classes:")
print(result["predicted"].value_counts().to_string())
print("\nFirst 10 predictions:")
print(result.head(10).to_string(index=False))
print("\nSaved:", OUTPUT)
print("V10 CSV INFERENCE PASSED")
