import argparse
from house_package_saachi.data_ingestion import fetch_housing_data, load_housing_data

def main(output_path):
    # Fetch and load the housing data
    fetch_housing_data()
    data = load_housing_data()
    # Save the loaded data to the specified output path
    data.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save the housing dataset.")
    parser.add_argument('--output-path', type=str, required=True, help="Output file path to save the dataset.")
    
    args = parser.parse_args()
    main(args.output_path)
