import argparse
import logging
import os
import joblib
import pandas as pd
from house_package_saachi.data_preprocessing import preprocess_housing_data
from house_package_saachi.model_scoring import score_model

def setup_logger(log_level, log_path=None, no_console_log=False):
    logger = logging.getLogger()
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

def main(model_path, dataset_path, output_path):
    logging.info("Loading the dataset.")
    housing = pd.read_csv(dataset_path)
    
    logging.info("Separating the target variable from the features.")
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)
    
    logging.info("Preprocessing the data.")
    housing_prepared = preprocess_housing_data(housing)
    
    logging.info("Loading the model.")
    model = joblib.load(model_path)
    
    logging.info("Scoring the model.")
    rmse = score_model(model, housing_prepared, housing_labels)
    logging.info(f"Model RMSE: {rmse}")
    
    logging.info("Saving the score to the specified output path.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"RMSE: {rmse}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a trained model on the housing dataset.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset for scoring.")
    parser.add_argument("--output-path", type=str, required=True, help="File to save the scoring result.")
    parser.add_argument('--log-level', type=str, default='INFO', help="Logging level.")
    parser.add_argument('--log-path', type=str, help="Path to log file. If not provided, logs won't be written to a file.")
    parser.add_argument('--no-console-log', action='store_true', help="Disable logging to console.")
    
    args = parser.parse_args()

    setup_logger(args.log_level, args.log_path, args.no_console_log)
    main(args.model_path, args.dataset_path, args.output_path)
