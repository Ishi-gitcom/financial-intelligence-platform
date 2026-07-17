from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


def tune_random_forest(X_train, y_train):

    param_grid = {

        "n_estimators": [100, 200, 300, 500],

        "max_depth": [5, 10, 15, 20, None],

        "min_samples_split": [2, 5, 10],

        "min_samples_leaf": [1, 2, 4],

        "max_features": ["sqrt", "log2"]
    }
    random_search = RandomizedSearchCV(

        estimator=RandomForestRegressor(
            random_state=42
        ),

        param_distributions=param_grid,

        n_iter=20,

        scoring="neg_mean_squared_error",

        cv=5,

        random_state=42,

        n_jobs=-1,

        verbose=1
    )

    random_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(random_search.best_params_)

    print("\nBest Score:")
    print(random_search.best_score_)

    return random_search.best_estimator_
    # random_search = RandomizedSearchCV(

    #     estimator=RandomForestRegressor(
    #         random_state=42
    #     ),

    #     param_distributions=param_grid,

    #     n_iter=20,

    #     scoring="neg_mean_squared_error",

    #     cv=5,

    #     random_state=42,

    #     n_jobs=-1,

    #     verbose=1
    # )