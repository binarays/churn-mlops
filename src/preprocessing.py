import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path: str):

    df = pd.read_csv(input_path)

    print("Starting preprocessing")

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Handle missing values
    df.fillna(0, inplace=True)

    # Convert target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop ID column
    df.drop("customerID", axis=1, inplace=True)

    # Separate features and label
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    pd.DataFrame(X_scaled).to_csv("data/processed/preprocessed.csv", index=False)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Preprocessing completed")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    preprocess_data("data/processed/raw_data.csv")