import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_fraud_model(df):

    X = df[["amount"]]
    y = df["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    return model


def fraud_probability(model, df):

    X = df[["amount"]]

    prob = model.predict_proba(X)[:,1].mean()

    return prob