
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, brier_score_loss)


COHORT_NAME = "GSE..."

model_dir = r"Combat_External_dataset"
ensure_dir(model_dir)
os.chdir(model_dir)
print(f"Working directory: {os.getcwd()}")

MODEL_PATH = r"XGBoost_best_model.pkl"
SCALER_PATH = r"scaler.pkl"

CLASS_ORDER = ["Normal", "Tumor"]   # index 0 -> Normal, index 1 -> Tumor
LABEL_MAP = {name: i for i, name in enumerate(CLASS_ORDER)}

N_BOOTSTRAPS = 1000
DECISION_THRESHOLD = 0.5

external_file = r"Combat_External_dataset.csv"
external_data  = pd.read_csv(external_file, encoding="latin1", index_col=0)

print(external_data.head())
print("External dataset shape:", external_data.shape)

scaler = joblib.load(SCALER_PATH)
expected_features = list(scaler.feature_names_in_)

missing = [g for g in expected_features if g not in external_data.columns]
extra = [g for g in external_data.columns
         if g not in expected_features and g not in ("sample_type", "Patient")]

if missing:
    print(f"WARNING: {len(missing)} training genes are missing from this external "
          f"dataset and will be filled with 0 (this will distort predictions for "
          f"those genes -- ideally re-check gene symbol matching): {missing}")
if extra:
    print(f"Note: {len(extra)} extra external columns not used by the model "
          f"will be dropped: {extra[:10]}{'...' if len(extra) > 10 else ''}")

X_raw = external_data.reindex(columns=expected_features, fill_value=0)
y_raw = external_data["sample_type"]

unknown_labels = set(y_raw.unique()) - set(LABEL_MAP.keys())
if unknown_labels:
    raise ValueError(f"sample_type contains labels not in CLASS_ORDER: {unknown_labels}")

y_true = y_raw.map(LABEL_MAP).to_numpy()
n_classes_present = len(np.unique(y_true))
single_class_cohort = n_classes_present < 2
if single_class_cohort:
    present_label = CLASS_ORDER[int(np.unique(y_true)[0])]

X_scaled = pd.DataFrame(
    scaler.transform(X_raw), columns=expected_features, index=X_raw.index)

if MODEL_PATH.endswith(".pkl"):
    model = joblib.load(MODEL_PATH)  # joblib, NOT pickle -- see earlier note
    is_keras = False
elif MODEL_PATH.endswith(".keras"):
    model = load_model(MODEL_PATH)
    is_keras = True
else:
    raise ValueError("Unsupported model format (use .pkl or .keras)")

if not is_keras:
    y_prob = model.predict_proba(X_scaled.values)[:, 1]
else:
    y_prob_raw = model.predict(X_scaled.values)
    y_prob = y_prob_raw[:, 0] if y_prob_raw.ndim > 1 and y_prob_raw.shape[1] == 1 else y_prob_raw.ravel()

y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

out_dir = os.path.join(model_dir, f"external_validation_{COHORT_NAME}")
ensure_dir(out_dir)

pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}, index=X_scaled.index) \
    .to_csv(os.path.join(out_dir, f"predictions_{COHORT_NAME}.csv"))

