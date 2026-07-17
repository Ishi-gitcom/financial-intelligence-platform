import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.train_models import train_random_forest


from src.evaluate import evaluate_model


def time_series_cv(
    df,
    feature_columns,
    n_splits=5
):

    X = df[feature_columns]
    y = df["Target"]

    tscv = TimeSeriesSplit(
        n_splits=n_splits
    )

    results = []

    for fold, (train_index, test_index) in enumerate(
        tscv.split(X),
        start=1
    ):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        rf_model = train_random_forest(
            X_train,
            y_train
        )

        rf_results = evaluate_model(
            rf_model,
            X_test,
            y_test
        )

        results.append({

            "Fold": fold,

            "MAE": rf_results["MAE"],

            "RMSE": rf_results["RMSE"],

            "R2": rf_results["R2"]

        })

    results_df = pd.DataFrame(results)

    print("\nTime Series Cross Validation Results\n")
    print(results_df)

    print("\nAverage Performance\n")
    print(results_df.mean(numeric_only=True))

    return results_df