import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
def plot_feature_importance(model, feature_names, model_name):
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})
    importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)
    plt.figure(figsize=(10,6))

    plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)
    plt.title(f"{model_name} Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    # plt.show()
    plt.savefig(
    "Output/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
    plt.close()
    return importance_df