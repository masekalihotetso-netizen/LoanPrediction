
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

print("=" * 50)
print("MOTZ FINANCIAL SERVICES")
print("MODEL VISUALISATION")
print("=" * 50)

# Load dataset and saved model
data = pd.read_csv("cleaned_loan_dataset.csv")

saved = joblib.load("best_loan_model.joblib")
model = saved["model"]
feature_columns = saved["feature_columns"]

# Prepare data
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

X = pd.get_dummies(X, drop_first=True, dtype=int)
X = X.reindex(columns=feature_columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

predictions = model.predict(X_test)

# Figure 1: Model accuracy comparison
results = pd.read_csv("model_comparison.csv")

plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["Accuracy"])
plt.title("Machine Learning Model Accuracy")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.close()

# Figure 2: Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    cmap="Blues"
)
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Figure 3: Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(ascending=True)

plt.figure(figsize=(9, 6))
importance.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Figure 4: Loan status distribution
status = data["loan_status"].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(status.index, status.values)
plt.title("Loan Approval and Rejection Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig("loan_status_distribution.png")
plt.close()

print("\nVISUALISATIONS CREATED SUCCESSFULLY!")
print("\nFiles created:")
print("1. model_accuracy_comparison.png")
print("2. confusion_matrix.png")
print("3. feature_importance.png")
print("4. loan_status_distribution.png")
