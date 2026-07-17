# ==========================================================
# FINANCIAL INTELLIGENCE PLATFORM
# Main Pipeline
# ==========================================================

# ==========================
# 1. IMPORTS
# ==========================
import src.config
print(src.config.__file__)
import numpy as np
import joblib

# Project Configuration
from src.config import (
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    RESULTS_DIR
)

# Data Loading
from src.data_loader import load_stock_data

# Feature Engineering
from src.feature_engineering import (
    engineer_features,
    FEATURE_COLUMNS
)

# Data Preprocessing
from src.preprocessing import prepare_train_test_data

# Feature Selection
from src.feature_selection import (
    remove_features,
    select_top_features
)

# Model Training
from src.train_models import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    train_xgboost
)

# Hyperparameter Tuning
from src.hyperparameter_tuning import (
    tune_random_forest
)

# Model Evaluation
from src.evaluate import (
    evaluate_model,
    compare_models
)

# Visualization
from src.visualization import (
    plot_predictions
)

from src.feature_importance import (
    plot_feature_importance
)

from src.residual_analysis import (
    residual_analysis
)

from src.time_series_validation import (
    time_series_cv
)

# ==========================================================
# 2. LOAD DATASET
# ==========================================================

print("=" * 70)
print("FINANCIAL INTELLIGENCE PLATFORM")
print("=" * 70)

print("\nLoading stock dataset...")

stock_data = load_stock_data(DATA_DIR)

print(f"Loaded {len(stock_data)} stocks successfully.")

# Select Apple Stock
apple = stock_data["aapl"]

print("\nPreview of Dataset:")
print(apple.head())

print("\nDataset Shape:")
print(apple.shape)

print("\nColumns:")
print(apple.columns.tolist())
# ==========================================================
# 3. FEATURE ENGINEERING
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# Generate Technical Indicators & Target Variable
apple = engineer_features(apple)

print("Feature engineering completed successfully.")

print("\nTotal Features Created:", len(FEATURE_COLUMNS))

print("\nAvailable Features:")
print(FEATURE_COLUMNS)
# ==========================================================
# 4. FEATURE REMOVAL EXPERIMENTS
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE REMOVAL EXPERIMENTS")
print("=" * 70)

experiments = {

    "Baseline": [],

    "Remove_BB_Middle": [
        "BB_Middle"
    ],

    "Remove_BB_SMA20": [
        "BB_Middle",
        "SMA20"
    ],

    "Remove_BB_SMA20_RollingMax": [
        "BB_Middle",
        "SMA20",
        "Rolling_Max"
    ]
}
for experiment_name, features_to_remove in experiments.items():

    print("\n" + "=" * 60)
    print(f"Running Experiment : {experiment_name}")
    print("=" * 60)

    # ----------------------------
    # Feature Selection
    # ----------------------------

    selected_features = remove_features(
        FEATURE_COLUMNS,
        features_to_remove
    )

    print(f"Original Features : {len(FEATURE_COLUMNS)}")
    print(f"Selected Features : {len(selected_features)}")

    # ----------------------------
    # Train/Test Split
    # ----------------------------

    (
        X_train_raw,
        X_test_raw,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    ) = prepare_train_test_data(
        apple,
        selected_features
    )

    # ----------------------------
    # Train Models
    # ----------------------------

    lr_model = train_linear_regression(
        X_train_scaled,
        y_train
    )

    dt_model = train_decision_tree(
        X_train_raw,
        y_train
    )

    rf_model = train_random_forest(
        X_train_raw,
        y_train
    )

    xgb_model = train_xgboost(
        X_train_raw,
        y_train
    )

    # ----------------------------
    # Evaluate Models
    # ----------------------------

    results = {

        "Linear Regression": evaluate_model(
            lr_model,
            X_test_scaled,
            y_test
        ),

        "Decision Tree": evaluate_model(
            dt_model,
            X_test_raw,
            y_test
        ),

        "Random Forest": evaluate_model(
            rf_model,
            X_test_raw,
            y_test
        ),

        "XGBoost": evaluate_model(
            xgb_model,
            X_test_raw,
            y_test
        )

    }

    print(compare_models(results))
top_features = [
    "Return_5",
    "ROC_10",
    "Return_20",
    "Volatility_20",
    "Return_1",
    "RSI",
    "Daily_Return",
    "Volume",
    "OC_Diff",
    "MACD_Signal"
]
# ==========================================================
# 5. FINAL MODEL USING TOP FEATURES
# ==========================================================

print("\n" + "=" * 70)
print("FINAL MODEL - TOP FEATURES")
print("=" * 70)

print("\nUsing Top 10 Features:")

for feature in top_features:
    print(f"- {feature}")
(
    X_train_raw,
    X_test_raw,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler
) = prepare_train_test_data(
    apple,
    top_features
)
print("\nTraining Models...\n")

lr_model = train_linear_regression(
    X_train_scaled,
    y_train
)

dt_model = train_decision_tree(
    X_train_raw,
    y_train
)

xgb_model = train_xgboost(
    X_train_raw,
    y_train
)
print("\nOptimizing Random Forest...\n")

best_rf = tune_random_forest(
    X_train_raw,
    y_train
)
results = {

    "Linear Regression": evaluate_model(
        lr_model,
        X_test_scaled,
        y_test
    ),

    "Decision Tree": evaluate_model(
        dt_model,
        X_test_raw,
        y_test
    ),

    "Random Forest": evaluate_model(
        best_rf,
        X_test_raw,
        y_test
    ),

    "XGBoost": evaluate_model(
        xgb_model,
        X_test_raw,
        y_test
    )

}
comparison = compare_models(results)

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(comparison)
# ==========================================================
# 6. FEATURE IMPORTANCE
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

plot_feature_importance(
    best_rf,
    top_features,
    "Random Forest"
)

plot_feature_importance(
    xgb_model,
    top_features,
    "XGBoost"
)
# ==========================================================
# 7. RESIDUAL ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("RESIDUAL ANALYSIS")
print("=" * 70)

residuals = residual_analysis(
    best_rf,
    X_test_raw,
    y_test
)
# ==========================================================
# 8. PREDICTION VISUALIZATION
# ==========================================================

print("\n" + "=" * 70)
print("PREDICTION VISUALIZATION")
print("=" * 70)

plot_predictions(
    y_test,
    results["Linear Regression"]["Predictions"],
    "Linear Regression"
)

plot_predictions(
    y_test,
    results["Decision Tree"]["Predictions"],
    "Decision Tree"
)

plot_predictions(
    y_test,
    results["Random Forest"]["Predictions"],
    "Random Forest"
)

plot_predictions(
    y_test,
    results["XGBoost"]["Predictions"],
    "XGBoost"
)
# ==========================================================
# SAVE MODEL COMPARISON
# ==========================================================

comparison.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully.")
# ==========================================================
# 9. TIME SERIES CROSS VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("TIME SERIES CROSS VALIDATION")
print("=" * 70)

cv_results = time_series_cv(
    apple,
    top_features,
    n_splits=5
)

print("\nCross Validation Completed Successfully!")
# ==========================================================
# 10. SAVE BEST MODEL
# ==========================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

model_path = MODEL_DIR / "random_forest.pkl"

joblib.dump(
    best_rf,
    model_path
)

print(f"Model saved at:\n{model_path}")
scaler_path = MODEL_DIR / "scaler.pkl"

joblib.dump(
    scaler,
    scaler_path
)

print("Scaler saved successfully.")
features_path = MODEL_DIR / "top_features.pkl"

joblib.dump(
    top_features,
    features_path
)

print("Feature list saved successfully.")
print("\n" + "=" * 70)
print("VERIFY SAVED FILES")
print("=" * 70)

loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)
loaded_features = joblib.load(features_path)

print("Model Loaded Successfully!")
print("Scaler Loaded Successfully!")
print("Features Loaded Successfully!")

print("\nTop Features:")

for feature in loaded_features:
    print(feature)
print("\n" + "=" * 70)
print("PROJECT EXECUTED SUCCESSFULLY")
print("=" * 70)