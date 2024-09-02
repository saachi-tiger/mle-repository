import argparse
import logging
from house_package_saachi.data_ingestion import fetch_housing_data, load_housing_data

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

def main(output_path):
    logging.info("Fetching and loading housing data.")
    fetch_housing_data()
    data = load_housing_data()
    logging.info("Saving data to the specified output path.")
    data.to_csv(output_path, index=False)
    logging.info(f"Data successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save the housing dataset.")
    parser.add_argument('--output-path', type=str, required=True, help="Output file path to save the dataset.")
    parser.add_argument('--log-level', type=str, default='INFO', help="Logging level.")
    parser.add_argument('--log-path', type=str, help="Path to log file. If not provided, logs won't be written to a file.")
    parser.add_argument('--no-console-log', action='store_true', help="Disable logging to console.")
    
    args = parser.parse_args()

    setup_logger(args.log_level, args.log_path, args.no_console_log)
    main(args.output_path)
