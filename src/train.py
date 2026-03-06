import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocessing import preprocess_data


def train_models():

    X_train, X_test, y_train, y_test = preprocess_data(
        "data/processed/raw_data.csv"
    )

    os.makedirs("models", exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            print(f"Training {name}")

            model.fit(X_train, y_train)

            # Save model
            model_path = f"models/{name}.pkl"
            joblib.dump(model, model_path)

            # Log model
            mlflow.sklearn.log_model(model, name)

            print(f"{name} training completed")


if __name__ == "__main__":

    train_models()