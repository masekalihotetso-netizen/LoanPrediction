
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# PROJECT SETTINGS
# --------------------------------------------------

DATA_FILE = "cleaned_loan_dataset.csv"
TARGET = "loan_status"
RANDOM_STATE = 42

print("=" * 55)
print("MOTZ FINANCIAL SERVICES")
print("MODEL TRAINING, TUNING AND EVALUATION")
print("=" * 55)

# --------------------------------------------------
# 1. LOAD THE CLEANED DATASET
# --------------------------------------------------

if not os.path.isfile(DATA_FILE):
    raise FileNotFoundError(
        f"Cannot find {DATA_FILE}. Check that it is in the project folder."
    )

df = pd.read_csv(DATA_FILE)

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
)

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )

# Remove missing target values BEFORE converting to text
df = df.dropna(subset=[TARGET]).copy()

# Standardise target labels
df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
)

df[TARGET] = df[TARGET].replace({
    "approve": "approved",
    "approved": "approved",
    "yes": "approved",
    "reject": "rejected",
    "rejected": "rejected",
    "no": "rejected"
})

if set(df[TARGET].unique()) != {"approved", "rejected"}:
    raise ValueError(
        "The loan_status column must contain 'approved' and 'rejected'. "
        f"Found: {sorted(df[TARGET].unique().tolist())}"
    )

print("\nDataset loaded:", df.shape)
print("\nLoan status distribution:")
print(df[TARGET].value_counts())

# --------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(
    columns=[TARGET, "loan_id"],
    errors="ignore"
).copy()

y = df[TARGET].copy()

if X.empty:
    raise ValueError("No input features were found in the dataset.")

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Standardise categorical values without learning from test data
for column in categorical_columns:
    X[column] = (
        X[column]
        .astype("string")
        .str.strip()
    )

# --------------------------------------------------
# 3. SPLIT DATA BEFORE IMPUTATION AND ENCODING
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nDATA SPLITTING")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 4. HANDLE MISSING VALUES USING TRAINING DATA ONLY
# --------------------------------------------------

categorical_fill_values = {}
numerical_fill_values = {}

# Learn categorical replacement values from training data
for column in categorical_columns:
    mode_values = X_train[column].mode(dropna=True)

    fill_value = (
        str(mode_values.iloc[0])
        if not mode_values.empty
        else "Unknown"
    )

    categorical_fill_values[column] = fill_value

    X_train[column] = X_train[column].fillna(fill_value)
    X_test[column] = X_test[column].fillna(fill_value)

# Learn numerical replacement values from training data
for column in numerical_columns:
    median_value = X_train[column].median()

    # Handle columns that contain no usable numerical values
    if pd.isna(median_value):
        median_value = 0

    numerical_fill_values[column] = float(median_value)

    X_train[column] = X_train[column].fillna(median_value)
    X_test[column] = X_test[column].fillna(median_value)

# Save category levels from training data only
category_levels = {
    column: sorted(
        X_train[column].astype(str).unique().tolist()
    )
    for column in categorical_columns
}

# --------------------------------------------------
# 5. ENCODE CATEGORICAL FEATURES
# --------------------------------------------------

X_train = pd.get_dummies(
    X_train,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

# Make sure training and testing have exactly the same features
feature_columns = X_train.columns.tolist()

X_test = X_test.reindex(
    columns=feature_columns,
    fill_value=0
)

X_train = X_train.astype(float)
X_test = X_test.astype(float)

print("\nPREPROCESSING")
print("Categorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)
print("Features used:", len(feature_columns))

# --------------------------------------------------
# 6. TRAIN THREE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

results = []

print("\n" + "=" * 55)
print("MODEL TRAINING")
print("=" * 55)

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results.append({
        "Model": name,

        "Accuracy": accuracy_score(
            y_test, predictions
        ) * 100,

        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ) * 100,

        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ) * 100,

        "F1_Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ) * 100
    })

# --------------------------------------------------
# 7. RANDOM FOREST HYPERPARAMETER TUNING
# --------------------------------------------------

print("\n" + "=" * 55)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 55)

parameter_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    param_grid=parameter_grid,

    cv=5,

    scoring="f1_weighted",

    n_jobs=-1,

    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1 score:")
print(f"{grid_search.best_score_ * 100:.2f}%")

# --------------------------------------------------
# 8. EVALUATE THE TUNED RANDOM FOREST
# --------------------------------------------------

tuned_predictions = best_model.predict(X_test)

tuned_results = {
    "Model": "Tuned Random Forest",

    "Accuracy": accuracy_score(
        y_test, tuned_predictions
    ) * 100,

    "Precision": precision_score(
        y_test,
        tuned_predictions,
        average="weighted",
        zero_division=0
    ) * 100,

    "Recall": recall_score(
        y_test,
        tuned_predictions,
        average="weighted",
        zero_division=0
    ) * 100,

    "F1_Score": f1_score(
        y_test,
        tuned_predictions,
        average="weighted",
        zero_division=0
    ) * 100
}

results.append(tuned_results)

# --------------------------------------------------
# 9. SAVE MODEL COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(comparison.round(2).to_string(index=False))

# --------------------------------------------------
# 10. SAVE CLASSIFICATION REPORT
# --------------------------------------------------

report = classification_report(
    y_test,
    tuned_predictions,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    "classification_report.csv"
)

# --------------------------------------------------
# 11. SAVE CONFUSION MATRIX
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    tuned_predictions,
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title("Tuned Random Forest Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# --------------------------------------------------
# 12. SAVE FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.Series(
    best_model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 6))

importance.head(10).sort_values().plot(
    kind="barh",
    ax=ax
)

ax.set_title("Top 10 Features Influencing Loan Predictions")
ax.set_xlabel("Feature Importance")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

importance.rename("Importance").to_csv(
    "feature_importance.csv"
)

# --------------------------------------------------
# 13. SAVE THE FINAL MODEL AND PREPROCESSING DETAILS
# --------------------------------------------------

model_bundle = {
    "model": best_model,

    "feature_columns": feature_columns,

    "categorical_columns": categorical_columns,

    "numerical_columns": numerical_columns,

    "category_levels": category_levels,

    "categorical_fill_values": categorical_fill_values,

    "numerical_fill_values": numerical_fill_values,

    "target_column": TARGET
}

joblib.dump(
    model_bundle,
    "best_loan_model.joblib"
)

# --------------------------------------------------
# 14. SAVE TUNING INFORMATION
# --------------------------------------------------

training_information = {
    "dataset_records": int(len(df)),

    "training_records": int(len(X_train)),

    "testing_records": int(len(X_test)),

    "number_of_features": int(len(feature_columns)),

    "categorical_columns": categorical_columns,

    "numerical_columns": numerical_columns,

    "cross_validation_folds": 5,

    "best_parameters": grid_search.best_params_,

    "best_cross_validation_f1": round(
        grid_search.best_score_ * 100, 2
    ),

    "tuned_test_accuracy": round(
        tuned_results["Accuracy"], 2
    ),

    "tuned_test_precision": round(
        tuned_results["Precision"], 2
    ),

    "tuned_test_recall": round(
        tuned_results["Recall"], 2
    ),

    "tuned_test_f1_score": round(
        tuned_results["F1_Score"], 2
    )
}

with open(
    "model_tuning_results.json",
    "w"
) as file:

    json.dump(
        training_information,
        file,
        indent=4
    )

# --------------------------------------------------
# 15. DISPLAY COMPLETION MESSAGE
# --------------------------------------------------

print("\n" + "=" * 55)
print("MODEL TRAINING AND EVALUATION COMPLETED")
print("=" * 55)

print("\nModels evaluated:")
print("1. Logistic Regression")
print("2. Decision Tree")
print("3. Random Forest")
print("4. Tuned Random Forest")

print("\nSaved files:")
print("1. best_loan_model.joblib")
print("2. model_comparison.csv")
print("3. classification_report.csv")
print("4. confusion_matrix.png")
print("5. feature_importance.png")
print("6. feature_importance.csv")
print("7. model_tuning_results.json")

print("\nAll operations completed successfully.")
