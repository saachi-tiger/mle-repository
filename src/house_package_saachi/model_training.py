from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import mean_squared_error
import numpy as np


def train_linear_regression(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    return lin_reg


def train_decision_tree(X, y):
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X, y)
    return tree_reg


def train_random_forest(X, y):
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg, param_distributions=param_distribs, n_iter=10, cv=5,
        scoring="neg_mean_squared_error", random_state=42
    )
    rnd_search.fit(X, y)
    return rnd_search.best_estimator_


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    return rmse
