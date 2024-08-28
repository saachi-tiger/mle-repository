import argparse
import os
import joblib
import pandas as pd
from house_package_saachi.data_ingestion import fetch_housing_data, load_housing_data
from house_package_saachi.data_preprocessing import preprocess_housing_data
from house_package_saachi.model_training import train_linear_regression, evaluate_model

def main(input_path, output_path):
    housing = load_housing_data(input_path)
    
    # Separate the target from the features
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)
    
    housing_prepared = preprocess_housing_data(housing)
    
    # Train the model
    lin_reg = train_linear_regression(housing_prepared, housing_labels)
    
    # Evaluate the model
    rmse = evaluate_model(lin_reg, housing_prepared, housing_labels)
    print(f"Training RMSE: {rmse}")
    
    # Save the model
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(lin_reg, f"{output_path}/linear_regression.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the housing dataset.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the housing dataset.")
    parser.add_argument("--output-path", type=str, required=True, help="Directory to save the trained model.")
    args = parser.parse_args()
    main(args.input_path, args.output_path)
