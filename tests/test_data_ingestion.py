import os
import tarfile
from unittest.mock import patch
import pytest
from src.house_package_saachi.data_ingestion import fetch_housing_data

@pytest.fixture
def temp_directory(tmpdir):
    return tmpdir.mkdir("data")

@patch('urllib.request.urlretrieve')
def test_fetch_housing_data(mock_urlretrieve, temp_directory):
    # Create a dummy tarfile in the temp directory
    tgz_path = os.path.join(temp_directory.strpath, "housing.tgz")
    
    # Create a dummy .tgz file with a fake housing.csv inside
    with tarfile.open(tgz_path, "w:gz") as tar:
        csv_path = os.path.join(temp_directory.strpath, "housing.csv")
        with open(csv_path, "w") as f:
            f.write("dummy_data\n")
        tar.add(csv_path, arcname="housing.csv")

    # Mock urlretrieve to simulate downloading the tarfile to the correct location
    mock_urlretrieve.return_value = tgz_path
    
    # Call the function under test
    fetch_housing_data(housing_url="https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz", housing_path=temp_directory.strpath)
    
    # Assert that the tarfile was extracted correctly
    assert os.path.exists(os.path.join(temp_directory.strpath, "housing.csv"))

