import xgboost as xgb

def predict_credit_risk(df):

    avg_amount = df["amount"].mean()

    risk = min(avg_amount / 100, 1)

    return risk