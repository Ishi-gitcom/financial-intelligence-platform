from sklearn.linear_model import LinearRegression
def train_linear_regression(X_train,y_train):
    model=LinearRegression()
    model.fit(X_train,y_train)
    return model

from sklearn.tree import DecisionTreeRegressor
def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

from sklearn.ensemble import RandomForestRegressor
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
           n_estimators=300,
           max_depth=15,
           min_samples_split=10,
           min_samples_leaf=3,
           random_state=42,
           n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
from xgboost import XGBRegressor

def train_xgboost(X_train, y_train):

    model = XGBRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42

    )

    model.fit(X_train, y_train)

    return model