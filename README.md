
# Microcredit Loan Repayment Prediction

## Introduction

This report details a data science project focused on predicting microcredit loan repayment. Microcredit, small loans given to low-income individuals or groups, plays a crucial role in economic development by empowering entrepreneurs and fostering financial inclusion. However, accurately assessing the risk of loan default is paramount for the sustainability of microfinance institutions. This project aims to leverage data science techniques to build robust predictive models that can identify potential defaulters, thereby enabling more informed lending decisions and mitigating financial losses.

---

## Problem Statement

The core problem addressed in this project is the accurate prediction of microcredit loan repayment status. Specifically, the goal is to develop a machine learning model that can predict whether a loan will be repaid or not. This involves:

* **Predicting Loan Repayment:** Building classification models to predict a binary outcome (repaid vs. not repaid).
* **Identifying Important Variables:** Determining which factors significantly influence the likelihood of a loan being repaid, providing actionable insights for lenders.

By achieving these goals, the project aims to help microfinance institutions make better risk assessments, optimize their lending portfolios, and ultimately contribute to more sustainable microcredit initiatives.

---

## Data Collection and Preprocessing

The project utilizes data loaded from a CSV file, `Micro-credit-Data-file.csv`. A description file, `Micro-credit-card-Data-Description.xlsx - Description.csv`, is also loaded to provide context about the dataset's columns.

The data preprocessing phase involved the following critical steps:

1.  **Loading Data:** The `data_loader.py` script is responsible for loading the main dataset and the description file. It includes error handling for `FileNotFoundError` and other general exceptions during file loading.
2.  **Handling Missing Values:** The `data_preprocessing.py` script's `clean_data` function addresses missing values. For numerical columns, mean imputation is applied. For categorical columns, missing values are imputed with the most frequent category (mode). This ensures that no missing data hinders subsequent analysis and model training.
3.  **Feature Preprocessing:** The `preprocess_features` function in `data_preprocessing.py` performs further transformations:
    * **Exclusion of Features:** Identifier columns (e.g., `'ID'`, `'Customer_ID'`, `'Loan_ID'`, `'PROD_SUB_KEY'`) are excluded from the feature set to prevent data leakage and ensure that the model learns from predictive attributes.
    * **Separation of Target Variable:** The target variable, initially assumed to be `'Label'` and then renamed to `'loan_repaid'`, is separated from the features.
    * **Numerical Feature Scaling:** Numerical features are scaled using `StandardScaler`. This is crucial for models sensitive to feature scales, such as Logistic Regression and Support Vector Machines, ensuring that no single feature dominates due to its magnitude.
    * **Categorical Feature Encoding:** Categorical features are converted into a numerical format using `OneHotEncoder`. This creates new binary columns for each category, making them suitable for machine learning algorithms. Imputation for categorical features is also handled within the preprocessing pipeline.

---

## Exploratory Data Analysis (EDA)

The `eda.py` script performs a comprehensive Exploratory Data Analysis to understand the dataset's characteristics, distributions, and relationships between variables. The key aspects of EDA included:

1.  **Basic Information and Descriptive Statistics:** Displaying `df.info()` to check data types and non-null counts, and `df.describe()` for statistical summaries of numerical and categorical columns.
2.  **Duplicate Check:** Identifying and reporting the number of duplicate rows in the dataset.
3.  **Target Variable Distribution:** Analyzing the distribution of the `loan_repaid` target variable to understand class imbalance. A count plot is generated and saved.
4.  **Distribution of Numerical Features:** Visualizing the distribution of numerical features using histograms (with KDE) and box plots.
5.  **Distribution of Categorical Features:** Visualizing the distribution of categorical features using count plots to show the frequency of each category.
6.  **Correlation Matrix:** Generating a heatmap of the correlation matrix for numerical features, including the target variable.
7.  **Relationship between Categorical Features and Target:** Analyzing how different categories of features relate to the target variable using count plots, providing insights into which categories might be more associated with loan repayment or default.

All generated plots are saved to an `output` directory for easy review and reporting, ensuring a clear visual understanding of the data.

---

## Feature Engineering

The `feature_engineering.py` script is designed to create new, more informative features from existing ones, which can significantly enhance model performance. Based on the problem statement that implies a payback ratio (e.g., loan 5 -> payback 6 suggests a ratio of 1.2), the following features were engineered:

1.  **Payback Ratio (`PaybackRatio`):** This feature is calculated as `PaybackAmount / LoanAmount`. A check is included to prevent division by zero, setting the ratio to 0 in such cases. This ratio directly quantifies the repayment efficiency.
2.  **Deviation from Expected Payback Ratio (`DeviationFromExpectedPayback`):** Assuming an expected payback ratio of 1.2, this feature calculates the difference between the actual `PaybackRatio` and the `expected_ratio`. This feature aims to capture how far a loan's repayment deviates from a predefined benchmark, which could be indicative of risk.

The script includes checks to ensure the `LoanAmount` and `PaybackAmount` columns exist before attempting to create these features.

---

## Model Selection and Training

The `model_training.py` script orchestrates the training and evaluation of multiple classification models. The choice of models is driven by their effectiveness in binary classification tasks and their ability to capture different patterns in data. The project employs a structured approach, including hyperparameter tuning and stratified cross-validation.

The following models were selected for training:

1.  **Logistic Regression:** A linear model that estimates the probability of a binary outcome. It serves as a strong baseline due to its simplicity and interpretability.
2.  **Random Forest Classifier:** An ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting. It's robust to outliers and can handle non-linear relationships.
3.  **Gradient Boosting Classifier:** Another powerful ensemble technique that builds models sequentially, with each new model correcting errors made by previous ones. It often delivers high performance.
4.  **Support Vector Machine (SVC):** A powerful model for classification by finding the optimal hyperplane that separates classes in a high-dimensional space. `probability=True` is enabled to allow `predict_proba` for ROC AUC calculations.

These models were chosen to provide a diverse set of algorithms, ranging from linear to complex ensemble methods, allowing for a comprehensive comparison of their performance on the microcredit loan dataset.

### Hyperparameter Tuning

Hyperparameter tuning is crucial for optimizing model performance. The project utilizes `GridSearchCV` with `StratifiedKFold` cross-validation to systematically search for the best combination of hyperparameters for each model. `StratifiedKFold` is particularly important for this project as it ensures that the proportion of the target variable (loan repayment status) is maintained in each fold, addressing potential class imbalance issues.

The `param_grid` for each model defines the range of hyperparameters to be explored:

* **Logistic Regression:** `C` (inverse of regularization strength)
* **Random Forest Classifier:** `n_estimators` (number of trees), `max_depth` (maximum depth of each tree), `min_samples_split` (minimum samples required to split an internal node)
* **Gradient Boosting Classifier:** `n_estimators`, `learning_rate` (contribution of each tree), `max_depth`
* **Support Vector Machine:** `C`, `kernel` (type of kernel function)

The `GridSearchCV` uses `roc_auc` as the scoring metric, which is particularly relevant for imbalanced classification problems where accurately ranking probabilities is important. The best hyperparameters are identified and used for the final model.

---

## Model Evaluation

Models are evaluated on a dedicated test set (25% of the data, split using stratification). The `display_metrics` function in `utils.py` is used to calculate and present a comprehensive set of classification metrics:

* **Accuracy:** The proportion of correctly classified instances.
* **Precision:** The proportion of positive identifications that were actually correct.
* **Recall:** The proportion of actual positives that were correctly identified.
* **F1-Score:** The harmonic mean of precision and recall, providing a balanced measure.
* **Log Loss:** A measure of the prediction uncertainty.
* **ROC AUC Score:** The Area Under the Receiver Operating Characteristic curve.

Additionally, a Confusion Matrix is printed, and an ROC curve is plotted for each model and saved to the output directory.

### Comparison of Models (Example Output Structure):

| Model Name                | Best Validation ROC AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test Log Loss | Test ROC AUC Score |
| ------------------------- | ----------------------- | ------------- | -------------- | ----------- | ------------- | ------------- | ------------------ |
| Logistic Regression       | [Value]                 | [Value]       | [Value]        | [Value]     | [Value]       | [Value]       | [Value]            |
| Random Forest Classifier  | [Value]                 | [Value]       | [Value]        | [Value]     | [Value]       | [Value]       | [Value]            |
| Gradient Boosting Classifier | [Value]                 | [Value]       | [Value]        | [Value]     | [Value]       | [Value]       | [Value]            |
| Support Vector Machine    | [Value]                 | [Value]       | [Value]        | [Value]     | [Value]       | [Value]       | [Value]            |

The **Gradient Boosting Classifier** typically performs well in many classification tasks. However, the model with the highest **Test ROC AUC Score** is identified as the best performing model, as this metric is a good indicator of overall discriminative power, especially in scenarios with potential class imbalance.

---

## Feature Importance Analysis

While the provided code does not explicitly extract and print feature importances for all models, tree-based models like Random Forest Classifier and Gradient Boosting Classifier offer this capability. For these models, feature importance is typically derived from how much each feature reduces impurity (e.g., Gini impurity or entropy) across all trees in the ensemble.

(Example of how feature importances would be interpreted if implemented): Variables such as `LoanAmount`, `PaybackAmount`, `PaybackRatio`, and `DeviationFromExpectedPayback` are expected to be highly important. Understanding these important features allows microfinance institutions to focus on key data points during loan application processing and risk assessment.

---

## Business Implications

The predictive model developed in this project offers significant business implications for microcredit loan providers:

* **Improved Risk Assessment:** The model provides a quantitative assessment of loan repayment probability, enabling lenders to make more accurate and data-driven decisions.
* **Optimized Loan Portfolio:** By identifying high-risk applicants, institutions can avoid loans that are likely to default, thereby optimizing their loan portfolio and reducing financial losses.
* **Enhanced Profitability:** Lower default rates directly translate to higher repayment rates and improved profitability for microfinance organizations.
* **Targeted Interventions:** Insights from feature importance can guide the development of targeted intervention strategies.
* **Operational Efficiency:** Automating risk assessment through machine learning models can streamline the loan approval process.
* **Informed Strategy:** The model's insights can guide the company's overall lending strategy, helping them understand market segments and risk appetite more effectively.

---

## Conclusion and Future Steps

This project successfully established a robust pipeline for predicting microcredit loan repayment, encompassing data loading, cleaning, EDA, feature engineering, model training, hyperparameter tuning, and evaluation. By leveraging various machine learning algorithms, the project aims to provide microfinance institutions with a powerful tool for risk management and decision-making.

### Limitations:

* **Data Availability and Quality:** The model's effectiveness heavily relies on the quality and comprehensiveness of the input data.
* **Feature Engineering Scope:** The current feature engineering is based on basic assumptions. More advanced feature engineering could further improve the model.
* **Model Complexity vs. Interpretability:** While complex models offer high accuracy, their interpretability can be challenging for business stakeholders.
* **Static Model:** The current model is static. Real-world microcredit environments are dynamic, and borrower behavior and economic conditions can change over time.

### Future Steps:

* **Advanced Feature Engineering:** Explore creating more sophisticated features, including time-series, external data, and more complex interactions.
* **Explore More Models:** Evaluate other advanced models such as XGBoost, LightGBM, CatBoost, or deep learning models.
* **Ensemble Methods:** Implement more sophisticated ensemble techniques like stacking or blending.
* **Explainable AI (XAI):** Integrate XAI techniques (e.g., SHAP, LIME) to provide more interpretable insights into model predictions.
* **Deployment and Monitoring:** For real-world application, the model would need to be deployed as an API and its performance continuously monitored to detect concept drift.
* **Cost-Sensitive Learning:** Given the potential imbalance between repaid and defaulted loans, explore cost-sensitive learning to explicitly account for the different costs associated with false positives and false negatives.
