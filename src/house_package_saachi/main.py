from src.house_package_saachi.data_ingestion import (
    fetch_housing_data,
    load_housing_data
)
from src.house_package_saachi.data_preprocessing import preprocess_housing_data
from src.house_package_saachi.model_training import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model,
)
from src.house_package_saachi.model_scoring import score_model
from src.house_package_saachi.utils import stratified_split
import pandas as pd
import numpy as np


def main():
    fetch_housing_data()
    housing = load_housing_data()

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    strat_train_set, strat_test_set = stratified_split(
        housing, strat_col="income_cat"
    )
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing_prepared = preprocess_housing_data(strat_train_set)
    housing_labels = strat_train_set["median_house_value"].copy()

    lin_reg = train_linear_regression(housing_prepared, housing_labels)
    tree_reg = train_decision_tree(housing_prepared, housing_labels)
    forest_reg = train_random_forest(housing_prepared, housing_labels)

    print("Linear Regression RMSE:", evaluate_model(
        lin_reg, housing_prepared, housing_labels
    ))
    print("Decision Tree RMSE:", evaluate_model(
        tree_reg, housing_prepared, housing_labels
    ))
    print("Random Forest RMSE:", evaluate_model(
        forest_reg, housing_prepared, housing_labels
    ))

    X_test_prepared = preprocess_housing_data(strat_test_set)
    y_test = strat_test_set["median_house_value"].copy()
    print("Final model RMSE:", score_model(
        forest_reg, X_test_prepared, y_test
    ))


if __name__ == "__main__":
    main()
