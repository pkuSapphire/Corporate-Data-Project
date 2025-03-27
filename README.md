### 1. **Visualization and Dataset Control Script** (`describe.py`)

**Purpose**: Handles dataset loading, filtering, visualization, and integrates financial factor computations.

- **`get_base_dataset()`: Checks for or generates `base_dataset.pkl` to load or create the dataset.**
- `clean_dataset(df)`: Filters out financial firms and future default information.
- `sample_data_info()`: Displays basic statistics and structure of the dataset.
- `statements_defaults_by_year()` / `plot_statements_and_defaults_dual_axis()`: Visualizes annual number of firms and defaults.
- `statements_defaults_by_industry()` / `plot_statements_defaults_by_industry()`: Visualizes defaults and default rates by industry sector.
- `main()`: Orchestrates plots and **AUC evaluation using factors**.

### 2. **Financial Factors Feature Engineering Script** (`financial_factors4.py`)

**Purpose**: Constructs financial ratios, handles missing values, and evaluates predictive performance via AUC.

- `get_base_dataset()`, `clean_dataset(df)`, `impute_data(df)`: Loads and pre-processes the dataset.
- `build_features(df)`: Constructs over 20 financial ratio features.
- `tobins_q_n_Altman_Z(df)`: Calculates Tobin's Q and Altman Z-score.
- `calculate_auc(df)`: Computes AUC scores for all features.
- `get_final_dataframe()`: Integrates all steps into a clean modeling dataset.

### 3. **Base Dataset Creation Script** (`base_dataset4.py`)

**Purpose**: Downloads and processes raw data from WRDS and GitHub to create `base_dataset.pkl` for modeling.

- WRDS Data Download Modules (e.g., `get_gvkey()`, `get_ratings()`): Access GVKEY identifiers and credit ratings.
- `merge_ratings_with_gvkey()`, `get_sector_info()`, `prepare_ratings()`: Combine credit ratings with industry classifications.
- `get_financials()`, `prepare_financials()`: Download and prepare annual financial statements.
- **`merge_financials_ratings()`: Merge financials and ratings data by firm and time window.**
- `compute_default_dates()`, `merge_default_dates()`: Identify default events and create binary default flags.
- `check_missing_financials_vs_ratings()`: Identifies firms with ratings but missing financials.
- `main()`: Executes full pipeline and saves the final dataset.

### 4. **Logistic Regression Modeling Script** (`model1.py`)

**Purpose**: Trains and evaluates logistic regression models (single-variable, multivariate, L1-regularized) and visualizes performance.

- `evaluate_single_var_model()`: Trains logistic regression on a single feature.
- `evaluate_multivariate_model()`: Trains logistic regression using multiple selected features.
- `evaluate_l1_model()`: Applies L1-regularization for feature selection. We can decide hyperparameter(penalty) in future using `GridSearch`.
- `plot_roc_curves()`: Compares model performance using ROC curves.
- `main()`: Runs all models and prints AUC comparison summary.

