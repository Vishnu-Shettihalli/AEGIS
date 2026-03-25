import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="AEGIS v4", layout="wide")

st.title("🛡️ AEGIS v4 – Financial Threat Intelligence Engine")

st.sidebar.header("Upload Transaction Dataset")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Transaction Table")
    st.dataframe(df)

    # ===============================
    # FEATURE ENGINEERING
    # ===============================

    if "amount" not in df.columns:
        st.error("CSV must contain 'amount' column")
        st.stop()

    if "type" not in df.columns:
        df["type"] = np.random.choice(["credit", "debit"], len(df))

    if "merchant" not in df.columns:
        df["merchant"] = np.random.randint(1,20,len(df))

    if "sender" not in df.columns:
        df["sender"] = np.random.randint(1,50,len(df))

    # ===============================
    # BEHAVIOR ANOMALY MODEL
    # ===============================

    iso = IsolationForest(contamination=0.05)

    df["anomaly"] = iso.fit_predict(df[["amount"]])

    anomaly_count = (df["anomaly"] == -1).sum()

    behavior_risk = anomaly_count / len(df)

    # ===============================
    # FRAUD MODEL
    # ===============================

    X = df[["amount"]]

    y = np.random.randint(0,2,len(df))

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    model.fit(X,y)

    fraud_prob = model.predict_proba(X)[:,1]

    fraud_score = fraud_prob.mean()

    # ===============================
    # NETWORK GRAPH
    # ===============================

    G = nx.from_pandas_edgelist(
        df,
        source="sender",
        target="merchant",
        edge_attr="amount"
    )

    network_risk = min(len(G.edges()) / 200, 1)

    # ===============================
    # FINANCIAL HEALTH ENGINE
    # ===============================

    revenue_volatility = df["amount"].std() / df["amount"].mean()

    merchant_concentration = df["merchant"].value_counts(normalize=True).max()

    financial_health = min(
        (revenue_volatility*0.5 + merchant_concentration*0.5),
        1
    )

    # ===============================
    # CASHFLOW ENGINE
    # ===============================

    inflow = df[df["type"]=="credit"]["amount"].mean()
    outflow = df[df["type"]=="debit"]["amount"].mean()

    if np.isnan(inflow):
        inflow = df["amount"].mean()

    if np.isnan(outflow):
        outflow = df["amount"].mean()/2

    cash_ratio = outflow / inflow

    cashflow_risk = min(cash_ratio,1)

    # ===============================
    # UNIFIED RISK ENGINE
    # ===============================

    unified_risk = (
        0.30 * financial_health +
        0.20 * cashflow_risk +
        0.20 * behavior_risk +
        0.20 * fraud_score +
        0.10 * network_risk
    )

    # ===============================
    # LOAN DECISION ENGINE
    # ===============================

    if unified_risk < 0.35:
        decision = "APPROVE"

    elif unified_risk < 0.65:
        decision = "REVIEW"

    else:
        decision = "REJECT"

    # ===============================
    # DASHBOARD METRICS
    # ===============================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Fraud Probability", round(fraud_score,2))

    col2.metric("Behavior Anomalies", anomaly_count)

    col3.metric("Cashflow Risk", round(cashflow_risk,2))

    col4.metric("Financial Health Risk", round(financial_health,2))

    # ===============================
    # RISK GAUGE
    # ===============================

    st.subheader("Unified Risk Score")

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = unified_risk,
        gauge = {
            "axis": {"range": [0,1]},
            "steps":[
                {"range":[0,0.35],"color":"green"},
                {"range":[0.35,0.65],"color":"orange"},
                {"range":[0.65,1],"color":"red"}
            ]
        }
    ))

    st.plotly_chart(fig)

    st.subheader(f"Loan Decision: {decision}")

    # ===============================
    # NETWORK GRAPH
    # ===============================

    st.subheader("Transaction Network")

    pos = nx.spring_layout(G)

    plt.figure(figsize=(6,6))

    nx.draw(
        G,
        pos,
        node_size=50,
        with_labels=False
    )

    st.pyplot(plt)

    # ===============================
    # AI INVESTIGATION REPORT
    # ===============================

    st.subheader("AI Investigation Report")

    report = f"""
    AEGIS Risk Analysis Summary

    Fraud Probability Score: {fraud_score:.2f}

    Behavior Anomalies Detected: {anomaly_count}

    Financial Health Risk: {financial_health:.2f}

    Cashflow Risk: {cashflow_risk:.2f}

    Network Risk: {network_risk:.2f}

    Unified Risk Score: {unified_risk:.2f}

    Loan Decision Recommendation: {decision}
    """

    st.text(report)

    # ===============================
    # CAM REPORT
    # ===============================

    st.subheader("Credit Assessment Memorandum")

    cam = f"""
    CREDIT ASSESSMENT MEMORANDUM
    --------------------------------------

    Borrower Transaction Analysis

    Financial Health Score: {financial_health:.2f}
    Cashflow Risk Score: {cashflow_risk:.2f}

    Behavior Anomalies: {anomaly_count}

    Fraud Probability: {fraud_score:.2f}

    Network Risk: {network_risk:.2f}

    Unified Risk Score: {unified_risk:.2f}

    Final Recommendation: {decision}

    Generated by AEGIS v4 AI Risk Engine
    """

    st.text(cam)

else:

    st.info("Upload a transaction dataset to start analysis.")