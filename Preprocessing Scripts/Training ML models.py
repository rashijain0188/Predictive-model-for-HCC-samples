#Training ML models

import os
import numpy as np
import pandas as pd
import joblib
from collections import Counter

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# =============================================================
# SETTINGS  -- EDIT THESE PATHS
# =============================================================
train_file = r"Train.csv"
test_file  = r"Test.csv"
model_dir  = r"Model"

LABEL_COL = "sample_type"    # target column
GROUP_COL = "Patient"        # Patient for sample derivation
ID_COL    = "ID"             # sample id column, set as index

N_SPLITS   = 10   # 10-fold CV
SMOTE_K    = 6    # SMOTE k_neighbors
RANDOM_STATE = 1234

ensure_dir(model_dir)
os.chdir(model_dir)

# =============================================================
# LOAD TRAIN / TEST DATA (already split by the user)
# =============================================================
train_data = pd.read_csv(train_file)
test_data  = pd.read_csv(test_file)

train_data.set_index(ID_COL, inplace=True)
test_data.set_index(ID_COL, inplace=True)

test_data = test_data[train_data.columns]

X_train_raw = train_data.drop([LABEL_COL, GROUP_COL], axis=1)
y_train_raw = train_data[LABEL_COL]
groups_train = train_data[GROUP_COL].values

X_test_raw = test_data.drop([LABEL_COL, GROUP_COL], axis=1)
y_test_raw = test_data[LABEL_COL]
# groups_test kept only for bookkeeping -- test set is never split/CV'd
groups_test = test_data[GROUP_COL].values

feature_names = X_train_raw.columns.tolist()

# =============================================================
# LABEL ENCODING
# =============================================================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_raw)
y_test_enc = le.transform(y_test_raw)
print("Classes:", le.classes_)
print("Positive class (encoded 1):", le.classes_[1])

# =============================================================
# Z-SCORE NORMALIZATION ON GENES 
# =============================================================
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train_raw), columns=feature_names, index=X_train_raw.index)
X_test = pd.DataFrame(
    scaler.transform(X_test_raw), columns=feature_names, index=X_test_raw.index)
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

# =============================================================
# MODEL EVALUATION FUNCTION
# =============================================================

def evaluate_model(model, param_grid,
                    X_train, y_train, groups_train,
                    X_test, y_test,
                    model_name,
                    best_params_file="best_hyperparams.txt",
                    metrics_file="metrics_summary.txt",
                    cv_results_file="cv_results.txt",
                    output_dir="model_outputs",
                    cv_folds=N_SPLITS,
                    smote_k=SMOTE_K,
                    random_state=RANDOM_STATE):

    os.makedirs(output_dir, exist_ok=True)
    out_dir = os.path.join(output_dir, model_name.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else [f"f{i}" for i in range(X_train.shape[1])]

    # ---- Pipeline: SMOTE -> classifier -------------------------------
    pipeline = ImbPipeline([
        ("smote", SMOTE(k_neighbors=smote_k, sampling_strategy="not majority", random_state=random_state)),
        ("clf", model),
    ])
    prefixed_grid = {f"clf__{k}": v for k, v in param_grid.items()}

    cv_splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        pipeline, prefixed_grid, scoring="roc_auc", cv=cv_splitter,
        n_jobs=-1, verbose=2, return_train_score=True,
    )
    grid.fit(X_train, y_train, groups=groups_train)
    best_pipeline = grid.best_estimator_  # already refit on full X_train/y_train

    # ---- Save best hyperparameters ------------------------------------
    with open(best_params_file, "a") as f:
        f.write(f"{model_name} best hyperparameters:\n{grid.best_params_}\n\n")


    # ---- "Validation" = out-of-fold CV predictions ---------------------
    # Refit a fresh (unfit) clone of the tuned pipeline for each fold so
    # SMOTE is regenerated per-fold and no fold's val data ever touches
    # training/SMOTE for that fold.
    oof_pipeline = clone(best_pipeline)
    preds_val = cross_val_predict(
        oof_pipeline, X_train, y_train, cv=cv_splitter, groups=groups_train,
        method="predict", n_jobs=-1,
    )
    probs_val = cross_val_predict(
        oof_pipeline, X_train, y_train, cv=cv_splitter, groups=groups_train,
        method="predict_proba", n_jobs=-1,
    )[:, 1]

    # ---- Predictions: train (refit-on-all) and test (external) ---------
    preds_train = best_pipeline.predict(X_train)
    probs_train = best_pipeline.predict_proba(X_train)[:, 1]

    preds_test = best_pipeline.predict(X_test)
    probs_test = best_pipeline.predict_proba(X_test)[:, 1]

    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)

    return best_pipeline


# =============================================================
# MODEL TRAINING
# =============================================================
open("best_hyperparams.txt", "w").close()
ensure_dir("model_outputs")

models_grids = [
    ("SVM",
     SVC(probability=True, random_state=RANDOM_STATE),
     {"C": [0.001, 0.01, 0.1, 1, 10], "kernel": ["poly", "rbf", "sigmoid"], "gamma": ["scale", "auto"]}),

    ("Random Forest",
     RandomForestClassifier(random_state=RANDOM_STATE),
     {"n_estimators": [2, 50, 100, 200], "max_depth": [None, 2, 5, 10, 25], "min_samples_split": [2, 5, 10]}),

    ("XGBoost",
     xgb.XGBClassifier(objective="binary:logistic", random_state=RANDOM_STATE, eval_metric="logloss"),
     {"n_estimators": [20, 50, 100, 200], "max_depth": [2, 3, 5, 10, 15], "learning_rate": [0.01, 0.001, 0.05, 0.1],
      "subsample": [0.8, 1.0], "colsample_bytree": [0.3, 0.6, 0.8, 1.0]}),

    ("Lasso Logistic Regression",
     LogisticRegression(penalty="l1", solver="saga", max_iter=10000, random_state=RANDOM_STATE,
                         class_weight="balanced"),
     {"C": [0.001, 0.01, 0.1, 1, 10, 20, 50]}),
]

for name, model, grid_params in models_grids:
    best_pipeline = evaluate_model(
        model, grid_params,
        X_train, y_train_enc, groups_train,
        X_test, y_test_enc,
        model_name=name,
        best_params_file="best_hyperparams.txt",
        output_dir="model_outputs",
        cv_folds=N_SPLITS,
        smote_k=SMOTE_K,
    )

    saved_model_dir = os.path.join("model_outputs", "saved_models")
    os.makedirs(saved_model_dir, exist_ok=True)
    model_path = os.path.join(saved_model_dir, f"{name.replace(' ', '_')}_best_model.pkl")
    joblib.dump(best_pipeline, model_path)

