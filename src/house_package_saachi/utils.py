<<<<<<< HEAD
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_split(df, strat_col, test_size=0.2, random_state=42):
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    for train_index, test_index in split.split(df, df[strat_col]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return strat_train_set, strat_test_set
=======
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_split(df, strat_col, test_size=0.2, random_state=42):
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    for train_index, test_index in split.split(df, df[strat_col]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return strat_train_set, strat_test_set

>>>>>>> 4720c7f (Updated refactor code)
