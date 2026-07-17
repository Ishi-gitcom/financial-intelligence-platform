def remove_features(feature_list, features_to_remove):
    """
    Returns a new feature list after removing selected features.
    """
    return [
        feature
        for feature in feature_list
        if feature not in features_to_remove
    ]
def select_top_features(feature_importance_df, top_n=10):
    """
    Returns the names of the top N most important features.
    """

    top_features = (
        feature_importance_df
        .sort_values(by="Importance", ascending=False)
        .head(top_n)["Feature"]
        .tolist()
    )

    return top_features
#     selected_features = [
#         feature
#         for feature in feature_list
#         if feature not in features_to_remove
#     ]
#     selected_features = [
#     feature
#     for feature in feature_list
#     if feature not in features_to_remove
# ]
#     return selected_features
