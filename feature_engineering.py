# feature_engineering.py

import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Creates new features based on the existing ones.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    print("\n--- Starting Feature Engineering ---")
    df_fe = df.copy()

    # Example 1: Payback Ratio
    # Assuming 'loan_amount' and 'payback_amount' columns exist.
    # We'll need to check the actual column names from the data description.
    # For this example, let's assume they are 'LoanAmount' and 'PaybackAmount'.
    # You might need to adjust these column names based on your actual data.

    # Problem statement indicates:
    # loan 5 -> payback 6 (ratio 1.2)
    # loan 10 -> payback 12 (ratio 1.2)
    # This suggests an expected payback ratio of 1.2

    # Check if required columns exist before creating features
    required_cols = ['LoanAmount', 'PaybackAmount'] # Placeholder names
    missing_cols = [col for col in required_cols if col not in df_fe.columns]

    if not missing_cols:
        # Avoid division by zero
        df_fe['PaybackRatio'] = df_fe.apply(
            lambda row: row['PaybackAmount'] / row['LoanAmount'] if row['LoanAmount'] != 0 else 0,
            axis=1
        )
        print("Created 'PaybackRatio' feature.")

        # Example 2: Deviation from Expected Payback Ratio
        # Assuming an expected ratio of 1.2 based on problem description
        expected_ratio = 1.2
        df_fe['DeviationFromExpectedPayback'] = df_fe['PaybackRatio'] - expected_ratio
        print("Created 'DeviationFromExpectedPayback' feature.")
    else:
        print(f"Warning: Missing columns for payback ratio feature engineering: {missing_cols}. Skipping related features.")


    # Example 3: Interaction features (e.g., between two important numerical features)
    # This is a generic example. You'd choose based on EDA insights.
    # Let's say 'Feature1' and 'Feature2' are numerical columns.
    # You'll need to replace these with actual column names from your data.
    # if 'Feature1' in df_fe.columns and 'Feature2' in df_fe.columns:
    #     df_fe['Feature1_x_Feature2'] = df_fe['Feature1'] * df_fe['Feature2']
    #     print("Created 'Feature1_x_Feature2' interaction feature.")

    # Example 4: Polynomial features (e.g., square of a feature)
    # if 'NumericalFeatureX' in df_fe.columns:
    #     df_fe['NumericalFeatureX_sq'] = df_fe['NumericalFeatureX']**2
    #     print("Created 'NumericalFeatureX_sq' polynomial feature.")

    # Example 5: Time-based features if date columns are present
    # Let's assume a 'LoanIssueDate' column exists and is in datetime format.
    # You would need to convert it to datetime objects first in preprocessing if it's string.
    # For now, this is commented out as a potential idea.
    # if 'LoanIssueDate' in df_fe.columns:
    #     try:
    #         df_fe['LoanIssueDate'] = pd.to_datetime(df_fe['LoanIssueDate'])
    #         df_fe['LoanDayOfWeek'] = df_fe['LoanIssueDate'].dt.dayofweek
    #         df_fe['LoanMonth'] = df_fe['LoanIssueDate'].dt.month
    #         df_fe['LoanDayOfYear'] = df_fe['LoanIssueDate'].dt.dayofyear
    #         print("Created time-based features from 'LoanIssueDate'.")
    #     except Exception as e:
    #         print(f"Could not create time-based features from 'LoanIssueDate': {e}")


    print("--- Feature Engineering Complete ---")
    return df_fe

