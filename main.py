# main.py

import pandas as pd
import os
from data_loader import load_data
from data_preprocessing import clean_data, preprocess_features
from eda import perform_eda
from feature_engineering import engineer_features
from model_training import run_all_models

def main():
    """
    Main function to run the entire microcredit loan prediction project.
    """
    print("--- Starting Microcredit Loan Prediction Project ---")

    # Define file paths
    # Ensure these CSV files are in the same directory as your Python scripts
    data_file = 'Micro-credit-Data-file.csv'
    description_file = 'Micro-credit-card-Data-Description.xlsx - Description.csv'
    output_directory = 'output' # Directory to save plots and results

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Step 1: Load Data
    print("\n### Step 1: Loading Data ###")
    df = load_data(data_file)
    if df is None:
        print("Exiting due to data loading error.")
        return

    # Load description file to understand columns
    df_desc = load_data(description_file)
    if df_desc is None:
        print("Warning: Could not load data description file. Proceeding without detailed column info.")
        # Attempt to infer target column and potential IDs if description is missing
        target_column_name = 'loan_repaid' # Default assumption based on problem
        # Infer potential ID columns (adjust based on actual data if known)
        features_to_exclude = ['ID'] # Common ID column name
    else:
        # Assuming the description file helps identify the target and ID columns
        # You might need to manually inspect the description.csv to find the exact column names
        # For this example, let's assume 'Label' is the target column in the main data
        # and 'Id' or similar is an identifier.
        # Based on problem statement, 'Label' is target.
        target_column_name = 'Label' # Target column name from problem description
        # Identify columns that are likely identifiers and should be excluded from features
        # Look for columns in description that indicate unique IDs or similar.
        # This is a guess; you'd confirm by inspecting the 'Description.csv'.
        potential_id_cols = ['Id', 'Customer_ID', 'Loan_ID', 'PROD_SUB_KEY']
        features_to_exclude = [col for col in potential_id_cols if col in df.columns]

        print("\nData Description File (first 5 rows):")
        print(df_desc.head())

        # Rename the target column for clarity if it's not already 'loan_repaid'
        if target_column_name in df.columns:
            if target_column_name != 'loan_repaid':
                df.rename(columns={target_column_name: 'loan_repaid'}, inplace=True)
                print(f"Renamed target column from '{target_column_name}' to 'loan_repaid'.")
                target_column_name = 'loan_repaid' # Update variable for consistency
        else:
            print(f"Warning: Target column '{target_column_name}' not found in the dataset. Please verify.")
            # If target column is not found, stop or assume a default.
            # For now, let's assume it *will* be renamed to 'loan_repaid' later or exists by default.
            # If not, the model training step will fail.
            print("Please ensure your data file contains a column representing loan repayment status (0 or 1).")
            print("The code expects this column to be named 'Label' initially or renamed to 'loan_repaid'.")
            # For robust execution, you might want to exit here if crucial columns are missing.
            # For this example, we'll proceed assuming 'Label' or 'loan_repaid' exists.


    # Step 2: Data Cleaning (Handle Missing Values)
    print("\n### Step 2: Data Cleaning ###")
    df_cleaned = clean_data(df)

    # Step 3: Exploratory Data Analysis (EDA)
    print("\n### Step 3: Performing EDA ###")
    # Pass 'loan_repaid' as the target column now
    perform_eda(df_cleaned.copy(), target_column='loan_repaid', output_dir=output_directory)

    # Step 4: Feature Engineering
    print("\n### Step 4: Feature Engineering ###")
    # Before engineering, check if 'LoanAmount' and 'PaybackAmount' exist or infer/rename
    # Based on the problem, these seem to be conceptual. The actual data might have different names.
    # I will modify feature_engineering.py to include checks for these columns.
    df_fe = engineer_features(df_cleaned)

    # Step 5: Feature Preprocessing (Encoding and Scaling)
    print("\n### Step 5: Feature Preprocessing ###")
    X, y, preprocessor = preprocess_features(
        df_fe,
        target_column='loan_repaid',
        features_to_exclude=features_to_exclude # Exclude ID columns from features
    )
    if y is None:
        print("Error: Target variable 'loan_repaid' not found after preprocessing. Cannot proceed with model training.")
        return

    # Step 6: Model Training, Hyperparameter Tuning, and Evaluation
    print("\n### Step 6: Model Training and Evaluation ###")
    all_model_results, best_model_name = run_all_models(X, y, output_dir=output_directory)

    print("\n--- Project Completed Successfully! ---")
    print(f"All plots and metric reports are saved in the '{output_directory}' directory.")
    print(f"The best performing model identified was: {best_model_name}")

if __name__ == "__main__":
    main()


