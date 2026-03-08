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
TOKEN = "956ba71c582b8b866d8721df9cced449f381f33e"
REPO = "churn-mlops"

if USERNAME is None or TOKEN is None:
    print("WARNING: DAGsHub credentials not found in environment variables.")

mlflow.set_tracking_uri(
    f"https://dagshub.com/{USERNAME}/{REPO}.mlflow"
)

os.environ["MLFLOW_TRACKING_USERNAME"] = USERNAME if USERNAME else ""
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN if TOKEN else ""

mlflow.set_experiment("Customer_Churn_Models")


# -----------------------------
# Training Function
# -----------------------------

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

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            print(f"Training {name}")

            # Train
            model.fit(X_train, y_train)

            # Predict
            predictions = model.predict(X_test)

            # Accuracy
            accuracy = accuracy_score(y_test, predictions)

            # Log to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", accuracy)

            # Save model
            model_path = f"models/{name}.pkl"
            joblib.dump(model, model_path)

            # Log model artifact
            mlflow.sklearn.log_model(model, "model")

            print(f"{name} completed with accuracy: {accuracy}")

            # Check if best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name

    # Save best model
    best_model_path = "models/best_model.pkl"
    joblib.dump(best_model, best_model_path)

    print("\nBest Model Selected")
    print(f"Model: {best_model_name}")
    print(f"Accuracy: {best_accuracy}")

    # Log best model to MLflow
    with mlflow.start_run(run_name="best_model"):

        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_metric("best_accuracy", best_accuracy)

        mlflow.sklearn.log_model(best_model, "best_model")

    print("Best model saved as models/best_model.pkl")


if __name__ == "__main__":
    train_models()