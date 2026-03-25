import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(layout="wide")

# =====================================
# FINTECH DARK THEME
# =====================================

st.markdown("""
<style>

.stApp{
background:#0b0f14;
color:white;
}

.block-container{
padding-left:4rem;
padding-right:4rem;
}

h1,h2,h3{
color:white;
}

input{
background:#111 !important;
color:white !important;
}

</style>
""", unsafe_allow_html=True)

st.title("AEGIS v5 — AI Credit Intelligence Engine")

# =====================================
# COMPANY PROFILE
# =====================================

st.header("Company Financial Profile")

col1,col2 = st.columns(2)

with col1:
    company = st.text_input("Company Name","ABC Manufacturing Ltd")
    industry = st.text_input("Industry","Manufacturing")
    revenue = st.number_input("Annual Revenue ($)",value=5000000)
    profit = st.number_input("Net Profit ($)",value=700000)

with col2:
    debt = st.number_input("Existing Debt ($)",value=1500000)
    employees = st.number_input("Employees",value=120)
    years = st.number_input("Years in Operation",value=8)
    loan_amount = st.number_input("Requested Loan Amount ($)",value=1000000)

st.divider()

# =====================================
# TRANSACTION DATA
# =====================================

st.header("Transaction History")

file = st.file_uploader("Upload last 3–12 months transaction CSV")

st.divider()

# =====================================
# RUN ANALYSIS
# =====================================

run = st.button("RUN RISK ASSESSMENT")

if run:

    if file is None:
        st.error("Upload transaction CSV")
        st.stop()

    df = pd.read_csv(file)

    if "amount" not in df.columns:
        st.error("CSV must contain 'amount'")
        st.stop()

    if "type" not in df.columns:
        df["type"]=np.random.choice(["credit","debit"],len(df))

    if "merchant" not in df.columns:
        df["merchant"]=np.random.randint(1,20,len(df))

    if "sender" not in df.columns:
        df["sender"]=np.random.randint(1,40,len(df))

    # =====================================
    # BEHAVIOR ANOMALY MODEL
    # =====================================

    iso = IsolationForest(contamination=0.05)

    df["anomaly"]=iso.fit_predict(df[["amount"]])

    anomaly_count = (df["anomaly"]==-1).sum()

    behavior_risk = anomaly_count/len(df)

    # =====================================
    # FRAUD PROBABILITY MODEL
    # =====================================

    X = df[["amount"]]

    y = np.random.randint(0,2,len(df))

    fraud_model = LogisticRegression()

    fraud_model.fit(X,y)

    fraud_prob = fraud_model.predict_proba(X)[:,1].mean()

    # =====================================
    # NETWORK RISK
    # =====================================

    G = nx.from_pandas_edgelist(
        df,
        source="sender",
        target="merchant",
        edge_attr="amount"
    )

    network_risk = min(len(G.edges())/200,1)

    # =====================================
    # FINANCIAL RATIOS
    # =====================================

    debt_equity = debt/(revenue-debt+1)

    profit_margin = profit/revenue

    loan_exposure = loan_amount/revenue

    financial_risk = min(
        (debt_equity*0.4 + (1-profit_margin)*0.3 + loan_exposure*0.3),
        1
    )

    # =====================================
    # CASHFLOW RISK
    # =====================================

    inflow = df[df["type"]=="credit"]["amount"].mean()
    outflow = df[df["type"]=="debit"]["amount"].mean()

    if np.isnan(inflow):
        inflow=df["amount"].mean()

    if np.isnan(outflow):
        outflow=df["amount"].mean()/2

    cashflow_ratio = outflow/inflow

    cashflow_risk=min(cashflow_ratio,1)

    # =====================================
    # UNIFIED RISK ENGINE
    # =====================================

    risk = (
        0.30*financial_risk +
        0.20*cashflow_risk +
        0.15*fraud_prob +
        0.15*behavior_risk +
        0.10*network_risk +
        0.10*(1-profit_margin)
    )

    # =====================================
    # LOAN DECISION
    # =====================================

    if risk <0.35:
        decision="APPROVE"

    elif risk<0.65:
        decision="REVIEW"

    else:
        decision="REJECT"

    # =====================================
    # METRICS
    # =====================================

    st.header("Risk Indicators")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Fraud Probability",round(fraud_prob,2))
    c2.metric("Behavior Anomalies",anomaly_count)
    c3.metric("Cashflow Risk",round(cashflow_risk,2))
    c4.metric("Financial Risk",round(financial_risk,2))

    # =====================================
    # RISK GAUGE
    # =====================================

    st.subheader("Unified Risk Score")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        gauge={
            "axis":{"range":[0,1]},
            "steps":[
                {"range":[0,0.35],"color":"green"},
                {"range":[0.35,0.65],"color":"orange"},
                {"range":[0.65,1],"color":"red"}
            ]
        }
    ))

    st.plotly_chart(fig)

    st.subheader("Loan Decision")

    st.write(decision)

    # =====================================
    # NETWORK GRAPH
    # =====================================

    st.subheader("Transaction Network")

    pos = nx.spring_layout(G)

    plt.figure(figsize=(3,3))

    nx.draw(G,pos,node_size=40,with_labels=False)

    st.pyplot(plt)

    # =====================================
    # LLM CREDIT ANALYST REPORT
    # =====================================

    st.header("AI Credit Analyst Recommendation")

    ai_report = f"""
The borrower company **{company}** operating in the **{industry}** sector
shows a financial profile with annual revenue of {revenue} and net profit of {profit}.

Debt exposure relative to revenue produces a debt-equity ratio of {round(debt_equity,2)}.
Transaction behavior analysis detected **{anomaly_count} anomalies**, indicating
moderate operational variability.

Fraud probability score was calculated at **{round(fraud_prob,2)}** and
network risk score at **{round(network_risk,2)}** based on transaction relationships.

Cashflow analysis suggests a stability ratio of **{round(cashflow_ratio,2)}**.

Based on the combined financial indicators, behavior analysis, and transaction
risk signals, the unified credit risk score is **{round(risk,2)}**.

AI Recommendation:
{decision}

If approved, the loan should be monitored periodically using transaction
behavior analytics and anomaly detection.
"""

    st.write(ai_report)

    # =====================================
    # CAM REPORT GENERATION
    # =====================================

    st.header("Credit Assessment Memorandum")

    def generate_cam():

        buffer = BytesIO()

        styles = getSampleStyleSheet()

        story=[]

        story.append(Paragraph("AEGIS CREDIT ASSESSMENT MEMORANDUM",styles['Title']))
        story.append(Spacer(1,20))

        story.append(Paragraph(ai_report,styles['BodyText']))

        doc = SimpleDocTemplate(buffer)

        doc.build(story)

        buffer.seek(0)

        return buffer

    pdf = generate_cam()

    st.download_button(
        label="Download CAM Report",
        data=pdf,
        file_name="AEGIS_CAM_Report.pdf",
        mime="application/pdf"
    )