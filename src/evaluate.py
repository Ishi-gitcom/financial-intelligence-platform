import numpy as np
import pandas as pd
from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score

)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "Predictions": predictions

    }
def compare_models(results):

    comparison = pd.DataFrame({
        "Model": results.keys(),

        "MAE": [
            results[m]["MAE"]
            for m in results
        ],

        "RMSE": [
            results[m]["RMSE"]
            for m in results
        ],

        "R2": [
            results[m]["R2"]
            for m in results
        ]
    })
    comparison = comparison.sort_values(
    by="RMSE",
    ascending=True
    ).reset_index(drop=True)
    comparison["MAE"] = comparison["MAE"].round(3)
    comparison["RMSE"] = comparison["RMSE"].round(3)
    comparison["R2"] = comparison["R2"].round(3)
    return comparison