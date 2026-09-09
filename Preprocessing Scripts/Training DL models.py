import os
import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# =============================================================
# SETTINGS  -- EDIT THESE PATHS
# =============================================================
train_file = r"Train.csv"
test_file  = r"Test.csv"
model_dir  = r"DL"

LABEL_COL = "sample_type"
GROUP_COL = "Patient" #dataset of origin
ID_COL    = "ID"

N_SPLITS     = 10   # 10-fold CV
SMOTE_K      = 6    # SMOTE k_neighbors
RANDOM_STATE = 1234

ensure_dir(model_dir)
os.chdir(model_dir)
print(f"Working directory: {os.getcwd()}")

tf.random.set_seed(RANDOM_STATE)

# =============================================================
# LOAD TRAIN / TEST DATA
# =============================================================
train_data = pd.read_csv(train_file)
test_data  = pd.read_csv(test_file)

train_data.set_index(ID_COL, inplace=True)
test_data.set_index(ID_COL, inplace=True)

X_train_raw = train_data.drop([LABEL_COL, GROUP_COL], axis=1)
y_train_raw = train_data[LABEL_COL]
groups_train = train_data[GROUP_COL].values

X_test_raw = test_data.drop([LABEL_COL, GROUP_COL], axis=1)
y_test_raw = test_data[LABEL_COL]

feature_names = X_train_raw.columns.tolist()
missing_in_test = set(feature_names) - set(X_test_raw.columns)
if missing_in_test:
    raise ValueError(f"Test set is missing feature columns present in train: {missing_in_test}")
X_test_raw = X_test_raw[feature_names]

# =============================================================
# LABEL ENCODING 
# =============================================================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_raw)
y_test_enc = le.transform(y_test_raw)
print("Classes:", le.classes_)
print("Positive class (encoded 1):", le.classes_[1])

n_classes = len(le.classes_)
if n_classes != 2:
    raise ValueError(
        f"This script assumes binary classification (2 classes), found {n_classes}: {le.classes_}" )

# =============================================================
# Z-SCORE NORMALIZATION ON GENES 
# =============================================================
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train_raw), columns=feature_names, index=X_train_raw.index
)
X_test = pd.DataFrame(
    scaler.transform(X_test_raw), columns=feature_names, index=X_test_raw.index
)
joblib.dump(scaler, os.path.join(model_dir, "dnn_scaler.pkl"))
print("Saved dnn_scaler.pkl")


# =============================================================
# MODEL BUILDER (scikeras convention: receives `meta` for shapes)
# =============================================================

def build_dnn(meta, hidden_units=(200, 100, 50), dropout=0.4, learning_rate=0.01):
    n_features_in_ = meta["n_features_in_"]
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for units in hidden_units:
        model.add(keras.layers.Dense(units, activation="relu"))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="sigmoid"))  # binary output
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def evaluate_dnn(model_name="DNN",
                  best_params_file="best_hyperparams_dnn.txt",
                  output_dir="model_outputs",
                  cv_folds=N_SPLITS,
                  smote_k=SMOTE_K,
                  random_state=RANDOM_STATE):

    os.makedirs(output_dir, exist_ok=True)
    out_dir = os.path.join(output_dir, model_name.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    early_stop = EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)

    clf = KerasClassifier(
        model=build_dnn,
        hidden_units=(200, 100, 50),
        dropout=0.3,
        learning_rate=0.01,
        epochs=150,
        batch_size=64,
        verbose=0,
        callbacks=[early_stop],
        random_state=random_state,
    )

    pipeline = ImbPipeline([
        ("smote", SMOTE(k_neighbors=smote_k, sampling_strategy="not majority", random_state=random_state)),
        ("clf", clf),
    ])

    # ---- Hyperparameter grid ------------------------------------------
    param_grid = {
        "clf__hidden_units": [(128, 64), (200, 100, 50)], #(128, 64), (200, 100, 50)
        "clf__dropout": [0.4], #0.2, 0.4
        "clf__learning_rate": [0.001], #0.01, 0.001
        "clf__batch_size": [32], #16,32
    }

    cv_splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        pipeline, param_grid, scoring="roc_auc", cv=cv_splitter,
        n_jobs=1, verbose=2, return_train_score=True,
    )
    grid.fit(X_train_np, y_train_enc, groups=groups_train)
    best_pipeline = grid.best_estimator_

    with open(best_params_file, "a") as f:
        f.write(f"{model_name} best hyperparameters:\n{grid.best_params_}\n\n")

    # ---- "Validation" = out-of-fold CV predictions ---------------------
    oof_pipeline = clone(best_pipeline)
    preds_val = cross_val_predict(
        oof_pipeline, X_train_np, y_train_enc, cv=cv_splitter, groups=groups_train,
        method="predict", n_jobs=1,
    )
    probs_val = cross_val_predict(
        oof_pipeline, X_train_np, y_train_enc, cv=cv_splitter, groups=groups_train,
        method="predict_proba", n_jobs=1,
    )[:, 1]

    # ---- Train (refit-on-all) and Test (external) predictions ----------
    preds_train = best_pipeline.predict(X_train_np)
    probs_train = best_pipeline.predict_proba(X_train_np)[:, 1]

    preds_test = best_pipeline.predict(X_test_np)
    probs_test = best_pipeline.predict_proba(X_test_np)[:, 1]

    y_train_arr = np.asarray(y_train_enc)
    y_test_arr = np.asarray(y_test_enc)

    return best_pipeline, out_dir


# =============================================================
# RUN
# =============================================================
open("best_hyperparams_dnn.txt", "w").close()
ensure_dir("model_outputs")

best_pipeline, dnn_out_dir = evaluate_dnn(model_name="DNN")

# ---- Save the trained model -----------------------------------------
saved_model_dir = os.path.join("model_outputs", "saved_models")
os.makedirs(saved_model_dir, exist_ok=True)

keras_model = best_pipeline.named_steps["clf"].model_
keras_model.save(os.path.join(saved_model_dir, "DNN_best_model.keras"))
print(f"Saved DNN model at: {os.path.join(saved_model_dir, 'DNN_best_model.keras')}")




