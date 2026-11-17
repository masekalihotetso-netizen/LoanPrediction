import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


# ==========================================
# PROJECT SETTINGS
# ==========================================

DATA_FILE = "cleaned_loan_dataset.csv"
TARGET = "loan_status"
ID_COLUMNS = ["loan_id"]
RANDOM_STATE = 42


def calculate_metrics(y_true, predictions):
    """Calculate metrics using 'approved' as the positive class."""
    return {
        "Accuracy": accuracy_score(y_true, predictions) * 100,
        "Precision": precision_score(
            y_true, predictions, pos_label="approved", zero_division=0
        ) * 100,
        "Recall": recall_score(
            y_true, predictions, pos_label="approved", zero_division=0
        ) * 100,
        "F1_Score": f1_score(
            y_true, predictions, pos_label="approved", zero_division=0
        ) * 100
    }


def main():
    # ------------------------------------------
    # LOAD AND VALIDATE DATA
    # ------------------------------------------
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(
            f"Cannot find '{DATA_FILE}'. Put it in the same folder as this script."
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.astype(str).str.strip().str.lower()

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.drop(
        columns=[column for column in ID_COLUMNS if column in df.columns]
    )
    df = df.dropna(subset=[TARGET]).copy()

    # Normalize target labels.
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    df[TARGET] = df[TARGET].replace({
        "approve": "approved",
        "yes": "approved",
        "rejected": "rejected",
        "reject": "rejected",
        "no": "rejected"
    })

    if set(df[TARGET].unique()) != {"approved", "rejected"}:
        raise ValueError(
            "The loan_status column must contain the two classes "
            "'approved' and 'rejected'. "
            f"Found: {sorted(df[TARGET].unique().tolist())}"
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    if X.empty: