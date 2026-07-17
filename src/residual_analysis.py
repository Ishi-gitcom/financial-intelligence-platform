import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def residual_analysis(model, X_test, y_test):

    predictions = model.predict(X_test)

    residuals = y_test - predictions

    plt.figure(figsize=(10,6))

    plt.scatter(predictions, residuals, alpha=0.6)

    plt.axhline(
        y=0,
        color='red',
        linestyle='--'
    )

    plt.xlabel("Predicted Return")
    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.tight_layout()

    # plt.show()
    plt.savefig(
    "Output/residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)
    plt.close()

    return residuals
