import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ==========================
# DARK GRID FINTECH STYLE
# ==========================

st.markdown("""
<style>

.stApp {
    background-color:#0b0f14;
    color:white;
}

.block-container{
padding-top:2rem;
padding-left:5rem;
padding-right:5rem;
}

h1,h2,h3{
color:white;
}

input, textarea {
background-color:#111;
color:white;
}

.stTextInput>div>div>input{
background-color:#0f1419;
color:white;
border-bottom:1px solid #444;
}

.stNumberInput input{
background-color:#0f1419;
color:white;
}

.stSelectbox div{
background-color:#0f1419;
color:white;
}

.run-btn{
background:#e6e6e6;
padding:15px;
border-radius:10px;
text-align:center;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# APPLICANT PROFILE
# ==========================

st.header("Applicant Profile")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("FULL NAME", "Rahul Sharma")
    loan_amount = st.number_input("LOAN AMOUNT ($)", value=200000)
    monthly_expense = st.number_input("MONTHLY EXPENSES ($)", value=25000)

with col2:
    pan = st.text_input("PAN NUMBER", "AFGPR1234K")
    income = st.number_input("MONTHLY INCOME ($)", value=60000)
    employment = st.selectbox("EMPLOYMENT STATUS",
                              ["Full-time","Self-employed","Contract"])

st.divider()

# ==========================
# TRANSACTION UPLOAD
# ==========================

st.header("Transaction History")

st.caption("Upload the last 3–6 months of transaction data in CSV format")

file = st.file_uploader("Upload transaction CSV")

st.divider()

# ==========================
# ASSESSMENT STATUS
# ==========================

st.header("Assessment Status")

status1 = "Complete" if name else "Pending"
status2 = "Uploaded" if file else "Pending"

colA, colB = st.columns(2)

with colA:
    st.write("Profile Data:", status1)

with colB:
    st.write("Transaction CSV:", status2)

# ==========================
# RUN RISK ENGINE
# ==========================

run = st.button("RUN RISK ASSESSMENT")

if run:

    if file is None:
        st.error("Please upload transaction dataset")
        st.stop()

    df = pd.read_csv(file)

    if "amount" not in df.columns:
        st.error("CSV must contain 'amount' column")
        st.stop()

    # ==========================
    # BEHAVIOR MODEL
    # ==========================

    iso = IsolationForest(contamination=0.05)

    df["anomaly"] = iso.fit_predict(df[["amount"]])

    anomaly_count = (df["anomaly"]==-1).sum()

    behavior_risk = anomaly_count/len(df)

    # ==========================
    # CASHFLOW RISK
    # ==========================

    avg_spend = df["amount"].mean()

    cashflow_ratio = monthly_expense / income

    cashflow_risk = min(cashflow_ratio,1)

    # ==========================
    # CREDIT RISK SCORE
    # ==========================

    credit_risk = loan_amount / (income*12)

    credit_risk = min(credit_risk,1)

    # ==========================
    # UNIFIED RISK
    # ==========================

    risk = (
        0.4 * credit_risk +
        0.3 * cashflow_risk +
        0.3 * behavior_risk
    )

    # ==========================
    # DECISION
    # ==========================

    if risk < 0.35:
        decision = "APPROVE"

    elif risk < 0.65:
        decision = "REVIEW"

    else:
        decision = "REJECT"

    st.success("Risk Analysis Complete")

    # ==========================
    # RISK GAUGE
    # ==========================

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

    st.subheader("AI Risk Summary")

    report = f"""

    Applicant: {name}

    Loan Amount: {loan_amount}

    Income: {income}

    Credit Risk Score: {credit_risk:.2f}

    Cashflow Risk: {cashflow_risk:.2f}

    Behavioral Anomalies: {anomaly_count}

    Final Risk Score: {risk:.2f}

    Recommended Decision: {decision}

    """

    st.text(report)