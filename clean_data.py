
import pandas as pd

print("=" * 50)
print("MOTZ FINANCIAL SERVICES")
print("DATA CLEANING")
print("=" * 50)

# Load dataset
data = pd.read_csv("loan_approval_dataset.csv")

# Remove unnecessary spaces from column names
data.columns = data.columns.str.strip()

# Remove duplicate records
data = data.drop_duplicates()

# Remove applicant ID because it is not a prediction feature
data = data.drop(columns=["loan_id"])

# Save cleaned dataset
data.to_csv("cleaned_loan_dataset.csv", index=False)

print("\nDATA CLEANING COMPLETED!")

print("\nCleaned Dataset Shape:")
print(data.shape)

print("\nCleaned Column Names:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nLoan Status Distribution:")
print(data["loan_status"].value_counts())

print("\nCleaned dataset saved as cleaned_loan_dataset.csv")
