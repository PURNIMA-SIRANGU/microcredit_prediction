
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def clean_data(df):
    """
    Handles missing values in the DataFrame.
    For simplicity, we'll use mean imputation for numerical columns and
    most frequent imputation for categorical columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    print("Handling missing values...")

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Impute numerical columns with mean
    for col in numerical_cols:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"  Imputed numerical column '{col}' with mean: {mean_val:.2f}")

    # Impute categorical columns with most frequent value
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] # .mode() can return multiple values, take the first
            df[col] = df[col].fillna(mode_val)
            print(f"  Imputed categorical column '{col}' with mode: '{mode_val}'")

    print("Missing value handling complete.")
    return df

def preprocess_features(df, target_column='loan_repaid', features_to_exclude=None):
    """
    Preprocesses the features: one-hot encodes categorical features and scales numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        features_to_exclude (list): List of feature names to exclude from preprocessing (e.g., IDs).

    Returns:
        tuple: (pd.DataFrame, sklearn.compose.ColumnTransformer):
               Processed DataFrame (or features only), and the preprocessor pipeline.
    """
    print("Starting feature preprocessing...")

    # Make a copy to avoid modifying the original DataFrame
    df_processed = df.copy()

    # Drop features to exclude if provided
    if features_to_exclude:
        df_processed = df_processed.drop(columns=features_to_exclude, errors='ignore')
        print(f"Excluded features: {features_to_exclude}")

    # Separate target variable if present
    if target_column in df_processed.columns:
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        print(f"Separated target column: '{target_column}'")
    else:
        X = df_processed
        y = None # No target column found

    # Identify numerical and categorical columns after exclusion and target separation
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    print(f"Numerical columns identified: {list(numerical_cols)}")
    print(f"Categorical columns identified: {list(categorical_cols)}")

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Impute before scaling
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Impute before encoding
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep other columns (if any) as they are
    )

    # Apply transformations
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    new_feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            new_feature_names.extend(cols)
        elif name == 'cat':
            if hasattr(transformer['onehot'], 'get_feature_names_out'):
                # For scikit-learn 0.23+
                new_feature_names.extend(transformer['onehot'].get_feature_names_out(cols))
            else:
                # Fallback for older scikit-learn versions
                # This might not be perfect for all cases but covers the common scenario
                for i, col in enumerate(cols):
                    for category in preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i]:
                        new_feature_names.append(f"{col}_{category}")
        else: # remainder='passthrough'
            # Assuming remainder columns are not transformed and just passed through
            pass # No specific names generated for 'passthrough' in this simplified approach


    # Convert processed data back to DataFrame with new column names
    X_processed_df = pd.DataFrame(X_processed, columns=new_feature_names, index=X.index)

    print("Feature preprocessing complete.")
    return X_processed_df, y, preprocessor

