import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_split(df, strat_col, test_size=0.2, random_state=42):
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    for train_index, test_index in split.split(df, df[strat_col]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return strat_train_set, strat_test_set


def add_extra_features(df):
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


def preprocess_housing_data(df):
    df_num = df.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    df_num_imputed = pd.DataFrame(
        imputer.fit_transform(df_num), columns=df_num.columns
    )
    df_cat = pd.get_dummies(df[["ocean_proximity"]], drop_first=True)
    df_prepared = df_num_imputed.join(df_cat)
    df_prepared = add_extra_features(df_prepared)
    return df_prepared
