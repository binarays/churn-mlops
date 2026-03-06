import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from preprocessing import preprocess_data


# -----------------------------
# DAGsHub MLflow Configuration
# -----------------------------
USERNAME = "binarays"
REPO = "churn-mlops"   
TOKEN = "956ba71c582b8b866d8721df9cced449f381f33e"

mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO}.mlflow")

os.environ["MLFLOW_TRACKING_USERNAME"] = USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN

mlflow.set_experiment("Customer_Churn_Models")


def train_models():

    X_train, X_test, y_train, y_test = preprocess_data(
        "data/processed/raw_data.csv"
    )

    os.makedirs("models", exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(),
        "xgboost": XGBClassifier(eval_metric="logloss")
    }

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            print(f"Training {name}")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            predictions = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, predictions)

            # Log parameters
            mlflow.log_param("model_name", name)

            # Log metric
            mlflow.log_metric("accuracy", accuracy)

            # Save model locally
            model_path = f"models/{name}.pkl"
            joblib.dump(model, model_path)

            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")

            print(f"{name} completed with accuracy: {accuracy}")

    print("All models trained successfully.")


if __name__ == "__main__":
    train_models()