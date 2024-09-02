import os
import tarfile
import urllib.request
import pandas as pd
import logging
from .config import HOUSING_URL, HOUSING_PATH

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    logging.info("Creating housing data directory if it doesn't exist.")
    os.makedirs(housing_path, exist_ok=True)
    
    tgz_path = os.path.join(housing_path, "housing.tgz")
    logging.info(f"Downloading housing data from {housing_url}.")
    urllib.request.urlretrieve(housing_url, tgz_path)
    
    logging.info("Extracting housing data from the tar file.")
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logging.info("Housing data extraction completed.")

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    logging.info(f"Loading housing data from {csv_path}.")
    return pd.read_csv(csv_path)
