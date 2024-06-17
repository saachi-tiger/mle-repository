import os

import numpy as np
import pandas as pd
from house_package_saachi import housing_price
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint


def housing_data():
    housing_price.fetch_housing_data()
    data = housing_price.load_housing_data()
    return data


def test_fetch_housing_data(tmpdir):
    housing_path = tmpdir.mkdir("housing")
    housing_url = housing_price.HOUSING_URL
    housing_price.fetch_housing_data(housing_url, str(housing_path))
    assert os.path.exists(os.path.join(str(housing_path), "housing.tgz"))


def test_load_housing_data(housing_data):
    assert not housing_data.empty


def test_income_cat_proportions(housing_data):
    housing_data["income_cat"] = pd.cut(
        housing_data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    proportions = housing_price.income_cat_proportions(housing_data)
    assert proportions is not None
    assert len(proportions) == 5


def test_stratified_shuffle_split(housing_data):
    housing_data["income_cat"] = pd.cut(
        housing_data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(
        housing_data, housing_data["income_cat"]
    ):
        strat_train_set = housing_data.loc[train_index]
        strat_test_set = housing_data.loc[test_index]
    assert len(strat_train_set) > 0
    assert len(strat_test_set) > 0


def test_correlation_matrix(housing_data):
    housing_data = housing_data.drop("income_cat", axis=1)
    housing_num = housing_data.drop("ocean_proximity", axis=1)
    corr_matrix = housing_num.corr()
    assert corr_matrix is not None
    assert "median_house_value" in corr_matrix.columns


def test_imputer(housing_data):
    housing_num = housing_data.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    assert X is not None
    assert X.shape == housing_num.shape


def test_linear_regression(housing_data):
    housing_data = housing_data.drop("income_cat", axis=1)
    housing = housing_data.drop("median_house_value", axis=1)
    housing_labels = housing_data["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
        )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
        )

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    housing_predictions = lin_reg.predict(housing_prepared)

    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    assert lin_rmse > 0


def test_decision_tree(housing_data):
    housing_data = housing_data.drop("income_cat", axis=1)
    housing = housing_data.drop("median_house_value", axis=1)
    housing_labels = housing_data["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
        )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
        )

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    housing_predictions = tree_reg.predict(housing_prepared)

    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    assert tree_rmse > 0


def test_random_forest(housing_data):
    housing_data = housing_data.drop("income_cat", axis=1)
    housing = housing_data.drop("median_house_value", axis=1)
    housing_labels = housing_data["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
        )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
        )

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_

    assert rnd_search.best_estimator_ is not None
    assert len(cvres["mean_test_score"]) == 10
