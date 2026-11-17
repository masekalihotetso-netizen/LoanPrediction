
import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 50)
print("MOTZ FINANCIAL SERVICES")
print("DATA PREPROCESSING")
print("=" * 50)

# Load cleaned dataset
data = pd.read_csv("cleaned_loan_dataset.csv")

# Separate input features and target
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nPREPROCESSING COMPLETED!")

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nPrepared data is ready for machine learning!")
