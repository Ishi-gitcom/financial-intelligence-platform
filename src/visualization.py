import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
PLOT_DIR = Path(r"C:\AI_ML_INTERN\Financial_Intelligence_Platform\results\plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
def plot_predictions(y_true, y_pred, model_name):

    plt.figure(figsize=(14,6))

    plt.plot(
        y_true.values,
        label="Actual"
    )

    plt.plot(
        y_pred,
        label="Predicted"
    )

    plt.title(f"{model_name} Predictions")

    plt.xlabel("Time")

    plt.ylabel("Price")

    plt.legend()

    plt.tight_layout()

    plt.show()
    filename = model_name.lower().replace(" ", "_") + ".png"
    plt.savefig(PLOT_DIR / filename)
    plt.close()

    print(f"{model_name} completed")
    