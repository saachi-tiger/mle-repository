from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import mean_squared_error
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def train_linear_regression(X, y):
    logging.info("Training Linear Regression model.")
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    logging.info("Linear Regression training completed.")
    return lin_reg

def train_decision_tree(X, y):
    logging.info("Training Decision Tree Regressor model.")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X, y)
    logging.info("Decision Tree Regressor training completed.")
    return tree_reg

def train_random_forest(X, y):
    logging.info("Training Random Forest Regressor model with randomized search.")
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
    logging.info("Random Forest Regressor training completed.")
    return rnd_search.best_estimator_

def evaluate_model(model, X, y):
    logging.info(f"Evaluating model: {model}")
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    logging.info(f"Model evaluation completed. RMSE: {rmse:.4f}")
    return rmse
