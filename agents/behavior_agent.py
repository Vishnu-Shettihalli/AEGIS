from sklearn.ensemble import IsolationForest

def analyze_behavior(df):

    features = df[["amount"]]

    model = IsolationForest(contamination=0.05)
    model.fit(features)

    scores = model.decision_function(features)

    risk = 1 - scores.mean()

    return max(risk, 0)