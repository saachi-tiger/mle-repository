from house_package_saachi.data_ingestion import (
    fetch_housing_data,
    load_housing_data
)
from house_package_saachi.data_preprocessing import preprocess_housing_data
from house_package_saachi.model_training import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model,
)
from house_package_saachi.model_scoring import score_model
from house_package_saachi.data_preprocessing import stratified_split
import pandas as pd
import numpy as np


def main():
    # Fetch and load the housing data
    fetch_housing_data()
    housing = load_housing_data()

    # Preprocess the housing data
    housing_prepared = preprocess_housing_data(housing)
    housing_prepared["income_cat"] = pd.cut(
        housing_prepared["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # Perform stratified split after preprocessing
    strat_train_set, strat_test_set = stratified_split(
        housing_prepared, strat_col="income_cat"
    )

    # Drop the 'income_cat' column after splitting
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Prepare training data
    housing_labels = strat_train_set["median_house_value"].copy()

    # Train models
    lin_reg = train_linear_regression(strat_train_set, housing_labels)
    tree_reg = train_decision_tree(strat_train_set, housing_labels)
    forest_reg = train_random_forest(strat_train_set, housing_labels)

    # Evaluate models on the training set
    print("Linear Regression RMSE:", evaluate_model(
        lin_reg, strat_train_set, housing_labels
    ))
    print("Decision Tree RMSE:", evaluate_model(
        tree_reg, strat_train_set, housing_labels
    ))
    print("Random Forest RMSE:", evaluate_model(
        forest_reg, strat_train_set, housing_labels
    ))

    # Test the final model on the test set
    y_test = strat_test_set["median_house_value"].copy()
    print("Final model RMSE:", score_model(
        forest_reg, strat_test_set, y_test
    ))


if __name__ == "__main__":
    main()
