import argparse
import os
import joblib
import pandas as pd
from house_package_saachi.data_preprocessing import preprocess_housing_data
from house_package_saachi.model_scoring import score_model

def main(model_path, dataset_path, output_path):
    # Load the dataset
    housing = pd.read_csv(dataset_path)
    
    # Separate the target variable from the features
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)
    
    # Preprocess the data
    housing_prepared = preprocess_housing_data(housing)
    
    # Load the model
    model = joblib.load(model_path)
    
    # Score the model
    rmse = score_model(model, housing_prepared, housing_labels)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the score
    with open(output_path, "w") as f:
        f.write(f"RMSE: {rmse}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a trained model on the housing dataset.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset for scoring.")
    parser.add_argument("--output-path", type=str, required=True, help="File to save the scoring result.")
    args = parser.parse_args()
    main(args.model_path, args.dataset_path, args.output_path)
