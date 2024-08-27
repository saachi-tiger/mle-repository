import pytest
import pandas as pd
from house_package_saachi.data_preprocessing import preprocess_housing_data
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_preprocess_housing_data():
    # Example DataFrame
    data = {
        'total_rooms': [5000, 3000],
        'households': [1000, 800],
        'total_bedrooms': [300, 200],
        'population': [1500, 1200],
        'ocean_proximity': ['NEAR BAY', 'INLAND']
    }
    df = pd.DataFrame(data)
    
    processed_df = preprocess_housing_data(df)
    assert 'rooms_per_household' in processed_df.columns, "Missing 'rooms_per_household' column."
    assert processed_df['rooms_per_household'][0] == 5, "Incorrect 'rooms_per_household' calculation."
    assert 'bedrooms_per_room' in processed_df.columns, "Missing 'bedrooms_per_room' column."
    assert 'population_per_household' in processed_df.columns, "Missing 'population_per_household' column."