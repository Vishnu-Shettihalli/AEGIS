import numpy as np
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def run_risk_analysis(df):

    numeric = df.select_dtypes(include="number")

    iso = IsolationForest(contamination=0.02)

    df["anomaly"] = iso.fit_predict(numeric)

    behaviour = abs(df["anomaly"].mean())

    fraud_prob = 0

    if "fraud" in df.columns:

        X = numeric
        y = df["fraud"]

        X_train,X_test,y_train,y_test = train_test_split(X,y)

        model = XGBClassifier()

        model.fit(X_train,y_train)

        fraud_prob = model.predict_proba(X_test)[:,1].mean()

    credit = numeric.mean().mean()/numeric.max().max()

    network = len(df["merchant"].unique()) / len(df)

    final = np.mean([behaviour,fraud_prob,credit,network])

    suspicious = df[df["anomaly"]==-1]

    return {
        "behaviour":round(behaviour,3),
        "fraud":round(fraud_prob,3),
        "credit":round(credit,3),
        "network":round(network,3),
        "final":round(final,3),
        "suspicious":suspicious.head(20)
    }