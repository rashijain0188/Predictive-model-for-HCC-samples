import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -----------------------
# USER MODEL PATHS
#set your paths for the model file and normalization file after downloading, 

MODEL_PKL  = r"XGBoost_best_model.pkl"
SCALER_PKL = r"scaler.pkl"

#and the directory in which you want the results to be saved
OUT_DIR = r"./results"

# -----------------------
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(OUT_DIR, "predictions.csv")

# Class mapping for numeric predictions
CLASS_MAP = {0: "Normal", 1: "Tumor"}

# -------------------------------------------------------------------------------------
# Upload function (samples as rows, features as columns)
# -------------------------------------------------------------------------------------
print("\n Upload the mRNA counts data with mRNAs as columns and samples as rows:\n")

def upload_data():
    Tk().withdraw()
    file_path = askopenfilename(title="Select data file", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    else:
        print("No file selected.")
        return None

# -----------------------
# Load data
# -----------------------
data = upload_data()

if data is None:
    raise SystemExit("No data uploaded. Exiting.")

# Use 'ID' column as index if present (optional)
if "ID" in data.columns:
    data = data.set_index("ID")

column_order = ["GPR50", "MYH1", "MT2A", "E2F8", "MYH4", "MTNR1B", "GLP2R", "GNAO1", "EGF",
    "BUB1B", "MMP3", "PLK1", "MMP1", "MT1E", "MT1F", "FOXM1", "E2F1", "MYH7",
    "CCL20", "MMP9", "CDC20", "GLP1R", "ADRA1D", "MT1G", "ADRA1A", "IGF1",
    "MT1X", "CDK1", "MYH8", "GNG4", "MYH13"]

# Reorder DataFrame columns (only those that exist in df)
data= data[[col for col in column_order if col in data.columns]]

# -----------------------
# Load Scaler
# -----------------------
try:
    scaler = joblib.load(SCALER_PKL)
except Exception:
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)

# -----------------------
# Load Model
# -----------------------
try:
    model = joblib.load(MODEL_PKL)
except Exception:
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

# -----------------------
# Basic feature check (optional but helpful)
# -----------------------
# If the scaler is a fitted transformer with feature names (e.g., a ColumnTransformer or DataFrame-aware),
# we try to check feature alignment. Otherwise, we warn if number of columns differ.
try:
    n_features_model = None
    if hasattr(scaler, "n_features_in_"):
        n_features_model = scaler.n_features_in_
    elif hasattr(model, "n_features_in_"):
        n_features_model = model.n_features_in_
except Exception:
    n_features_model = None

if n_features_model is not None and data.shape[1] != n_features_model:
    print(f"Warning: uploaded data has {data.shape[1]} features but model/scaler expects {n_features_model} features.")
    # Continue, but user should verify columns order and names match training data.

# -----------------------
# Scale data
# -----------------------
data_scaled = scaler.transform(data)  # assumes scaler implements transform

# -----------------------
# Predictions (numeric) and mapping to text labels
# -----------------------
y_pred_numeric = model.predict(data_scaled).astype(int)

# Map to "Normal"/"Tumor" using CLASS_MAP (safe even if classes differ)
predicted_text = np.vectorize(CLASS_MAP.get)(y_pred_numeric)

# -----------------------
# Predict probabilities (if available) - choose probability for numeric class 1 if present
# -----------------------
if hasattr(model, "predict_proba"):
    proba_all = model.predict_proba(data_scaled)
    # find index for numeric class == 1 if possible
    try:
        if hasattr(model, "classes_"):
            pos_idx = list(model.classes_).index(1)
        else:
            pos_idx = 1 if proba_all.shape[1] > 1 else 0
    except ValueError:
        pos_idx = 1 if proba_all.shape[1] > 1 else 0
    prob_tumor = proba_all[:, pos_idx]
else:
    # fallback: if decision_function exists, convert to probability via sigmoid; otherwise NaN
    if hasattr(model, "decision_function"):
        scores = model.decision_function(data_scaled)
        if scores.ndim == 1:
            prob_tumor = 1 / (1 + np.exp(-scores))
        else:
            # pick second column as positive-class score if available
            col = 1 if scores.shape[1] > 1 else 0
            prob_tumor = 1 / (1 + np.exp(-scores[:, col]))
    else:
        prob_tumor = np.full(data.shape[0], np.nan)

# -----------------------
# Save predictions (no loops)
# -----------------------
results_df = pd.DataFrame({
    "Predicted_class": predicted_text,
    "Prob_Tumor": prob_tumor
}, index=data.index)

results_df.to_csv(RESULTS_CSV, index=True)
print(f"\nSaved predictions: {RESULTS_CSV}")

print("\nPrediction completed. Review the CSV for predicted labels ('Normal'/'Tumor') and tumor probabilities.")
