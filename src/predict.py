import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("models/random_forest.pkl", "rb"))

# Load scaler if you saved one
scaler = pickle.load(open("models/scaler.pkl", "rb"))


def predict_churn(data):

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Scale features
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]

    # Probability
    probability = model.predict_proba(df_scaled)[0][1]

    return prediction, probability