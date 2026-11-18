import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


def test_model_accuracy():
    model = joblib.load("models/best_model.pkl")
    df = pd.read_csv("data/test.csv")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    ).fillna(0)

    X_test = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y_test = df["Churn"].map({"No": 0, "Yes": 1})

    assert y_test.notna().all(), "Expected Churn labels: No or Yes"

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy > 0.72, (
        f"Accuracy {accuracy:.2%} must be greater than 72%"
    )