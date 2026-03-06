import pandas as pd
import os


def load_data(input_path: str, output_path: str):
    """
    Load raw dataset and store a copy for pipeline processing.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)

    df.to_csv(output_path, index=False)

    print("Raw data saved to:", output_path)


if __name__ == "__main__":

    input_file = "data/raw/telco_customer_churn_data.csv"
    output_file = "data/processed/raw_data.csv"

    load_data(input_file, output_file)