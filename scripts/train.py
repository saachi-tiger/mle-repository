import argparse
import logging
import os
import joblib
import pandas as pd
from house_package_saachi.data_ingestion import load_housing_data
from house_package_saachi.data_preprocessing import preprocess_housing_data
from house_package_saachi.model_training import train_linear_regression, evaluate_model

def setup_logger(log_level, log_path=None, no_console_log=False):
    logger = logging.getLogger(__name__)  # Ensure the logger uses the module's name
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not no_console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

def main(input_path, output_path):
    logger = logging.getLogger(__name__)  # Get the logger for this module

    logger.info("Loading housing data.")
    housing = load_housing_data(input_path)
    
    logger.info("Separating target from features.")
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)
    
    logger.info("Preprocessing housing data.")
    housing_prepared = preprocess_housing_data(housing)
    
    logger.info("Training the linear regression model.")
    lin_reg = train_linear_regression(housing_prepared, housing_labels)
    
    logger.info("Evaluating the model.")
    rmse = evaluate_model(lin_reg, housing_prepared, housing_labels)
    logger.info(f"Training RMSE: {rmse}")
    
    logger.info("Saving the trained model.")
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(lin_reg, f"{output_path}/linear_regression.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the housing dataset.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the housing dataset.")
    parser.add_argument("--output-path", type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument('--log-level', type=str, default='INFO', help="Logging level.")
    parser.add_argument('--log-path', type=str, help="Path to log file. If not provided, logs won't be written to a file.")
    parser.add_argument('--no-console-log', action='store_true', help="Disable logging to console.")
    
    args = parser.parse_args()

    setup_logger(args.log_level, args.log_path, args.no_console_log)
    main(args.input_path, args.output_path)
