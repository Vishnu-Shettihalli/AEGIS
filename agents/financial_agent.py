import pandas as pd
from sklearn.ensemble import IsolationForest


def analyze_financial_behavior(df):

    # Use transaction amount for anomaly detection
    amounts = df[["Amount"]]

    model = IsolationForest(contamination=0.02, random_state=42)
    preds = model.fit_predict(amounts)

    df["anomaly"] = preds

    anomalies = df[df["anomaly"] == -1]

    anomaly_ratio = len(anomalies) / len(df)

    behaviour_risk = anomaly_ratio * 100

    return round(behaviour_risk, 2)