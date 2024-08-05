import pandas as pd
from sklearn.impute import SimpleImputer


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
