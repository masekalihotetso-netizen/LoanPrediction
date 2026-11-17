
import pandas as pd
import os

print("=" * 50)
print("MOTZ FINANCIAL SERVICES")
print("LOAN PREDICTION SYSTEM")
print("DATASET INSPECTION")
print("=" * 50)

# Dataset location
file_path = "loan_approval_dataset.csv"

# Check whether the dataset exists
if os.path.exists(file_path):

    # Load dataset
    data = pd.read_csv(file_path)

    print("\nDATASET SUCCESSFULLY LOADED!")

    # Display dataset dimensions
    print("\nDataset Shape:")
    print(data.shape)

    # Display first five records
    print("\nFirst Five Records:")
    print(data.head())

    # Display column names
    print("\nDataset Columns:")
    print(data.columns.tolist())

    # Display dataset information
    print("\nDataset Information:")
    data.info()

    # Check missing values
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Statistical summary
    print("\nDescriptive Statistics:")
    print(data.describe(include="all"))

else:
    print("\nERROR: Dataset file not found!")
    print("Please check the dataset folder and filename.")
