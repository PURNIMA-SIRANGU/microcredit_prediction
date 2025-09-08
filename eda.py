# eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import save_plot
import os

def perform_eda(df, target_column='loan_repaid', output_dir="output"):
    """
    Performs exploratory data analysis on the DataFrame and saves plots.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        output_dir (str): Directory to save the plots.
    """
    print("\n--- Starting Exploratory Data Analysis (EDA) ---")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Display basic information
    print("\nDataFrame Info:")
    df.info()

    # 2. Display descriptive statistics
    print("\nDescriptive Statistics (Numerical Columns):")
    print(df.describe())

    print("\nDescriptive Statistics (Categorical Columns):")
    print(df.describe(include='object'))

    # 3. Check for duplicates
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

    # 4. Target variable distribution
    if target_column in df.columns:
        print(f"\nDistribution of Target Variable ('{target_column}'):")
        print(df[target_column].value_counts())
        print(df[target_column].value_counts(normalize=True))

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=target_column, data=df, ax=ax)
        ax.set_title(f'Distribution of {target_column}')
        save_plot(fig, f'{target_column}_distribution.png', output_dir)
    else:
        print(f"Warning: Target column '{target_column}' not found for distribution analysis.")


    # 5. Distribution of Numerical Features
    print("\nPlotting distributions of numerical features...")
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column) # Remove target from features to plot

    for col in numerical_cols:
        if df[col].nunique() > 50: # Avoid plotting histograms for high cardinality numerical features
            # Use kde plot for smooth distribution visualization or box plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.histplot(df[col], kde=True, ax=axes[0])
            axes[0].set_title(f'Distribution of {col}')
            sns.boxplot(x=df[col], ax=axes[1])
            axes[1].set_title(f'Box Plot of {col}')
            save_plot(fig, f'{col}_distribution_boxplot.png', output_dir)
        else: # For numerical features with fewer unique values, histogram is fine
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            save_plot(fig, f'{col}_distribution.png', output_dir)

    # 6. Distribution of Categorical Features
    print("Plotting distributions of categorical features...")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax)
        ax.set_title(f'Distribution of {col}')
        save_plot(fig, f'{col}_distribution.png', output_dir)

    # 7. Correlation Matrix (for numerical features)
    print("\nGenerating correlation matrix...")
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols + [target_column]].corr() if target_column in df.columns else df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Matrix of Numerical Features')
        save_plot(fig, 'correlation_matrix.png', output_dir)
    else:
        print("Not enough numerical columns for a correlation matrix.")

    # 8. Relationship between categorical features and target
    print("Analyzing relationships between categorical features and target...")
    if target_column in df.columns:
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=col, hue=target_column, data=df, ax=ax, palette='viridis')
            ax.set_title(f'Distribution of {col} by {target_column}')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            save_plot(fig, f'{col}_vs_{target_column}_distribution.png', output_dir)
    else:
        print(f"Warning: Target column '{target_column}' not found for categorical feature relationship analysis.")

    print("\n--- EDA Complete ---")

