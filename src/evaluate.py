import joblib
import mlflow
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)

from preprocessing import preprocess_data

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("churn_evaluation")

def evaluate_models():

    X_train, X_test, y_train, y_test = preprocess_data(
        "data/processed/raw_data.csv"
    )

    models = {
        "logistic_regression": joblib.load("models/logistic_regression.pkl"),
        "random_forest": joblib.load("models/random_forest.pkl"),
        "xgboost": joblib.load("models/xgboost.pkl")
    }

    for name, model in models.items():

        with mlflow.start_run(run_name=f"{name}_evaluation"):

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            roc = roc_auc_score(y_test, probabilities)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc)

            print("Model:", name)
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("ROC AUC:", roc)
            print("-------------------")

            # Confusion Matrix
            cm = confusion_matrix(y_test, predictions)

            plt.figure()
            plt.imshow(cm)
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            cm_file = f"{name}_confusion_matrix.png"
            plt.savefig(cm_file)

            mlflow.log_artifact(cm_file)

            # ROC Curve
            RocCurveDisplay.from_predictions(y_test, probabilities)

            roc_file = f"{name}_roc_curve.png"
            plt.savefig(roc_file)

            mlflow.log_artifact(roc_file)


if __name__ == "__main__":

    evaluate_models()