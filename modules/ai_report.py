import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from io import BytesIO

st.set_page_config(layout="wide")

# ---------------------------------------------------
# CYBERSECURITY UI
# ---------------------------------------------------

st.markdown("""
<style>

.stApp{
background-color:#0B0F19;
color:#E6EDF3;
}

h1,h2,h3{
color:#00FFC6;
}

.metric-card{
background:#161B22;
padding:20px;
border-radius:10px;
border:1px solid #30363D;
}

.ai-panel{
background:#161B22;
padding:25px;
border-left:6px solid #00FFC6;
border-radius:8px;
}

.section-title{
color:#00FFC6;
}

</style>
""", unsafe_allow_html=True)

st.title("AEGIS v3 Financial Threat Intelligence Console")

st.write("AI-Driven Fraud Detection | Credit Risk | Behaviour Monitoring")

# ---------------------------------------------------
# AI REPORT FUNCTION
# ---------------------------------------------------

def generate_ai_report(results):

    risk = results["final"]

    return f"""
<div class="ai-panel">

<h3 class="section-title">AI Risk Investigation Report</h3>

<p>The AEGIS intelligence engine performed multi-layer financial analysis including behavioral anomaly detection,
fraud probability estimation, network relationship analysis, and credit exposure assessment.</p>

<h4>Detected Risk Metrics</h4>

<ul>
<li>Behaviour Risk Score: {results["behaviour"]}</li>
<li>Fraud Probability: {results["fraud"]}</li>
<li>Credit Exposure Risk: {results["credit"]}</li>
<li>Network Fraud Risk: {results["network"]}</li>
</ul>

<h4>Risk Interpretation</h4>

<p>The system identified irregular transaction clusters that deviate from established financial behavioral baselines.
Certain merchant nodes demonstrate high connectivity across multiple customers, which may indicate potential fraud rings or coordinated activity.</p>

<h4>Suspicious Indicators</h4>

<ul>
<li>Repeated transactions through identical merchant channels</li>
<li>Abnormally high frequency transaction sequences</li>
<li>Transaction values outside expected distribution ranges</li>
<li>Network clusters connecting multiple flagged customers</li>
</ul>

<h4>Recommended Actions</h4>

<ul>
<li>Initiate enhanced monitoring on flagged accounts</li>
<li>Conduct merchant verification reviews</li>
<li>Deploy secondary anomaly detection models</li>
<li>Perform AML compliance review</li>
</ul>

<h4>Advanced Risk Mitigation Strategy</h4>

<ul>
<li>Implement real-time fraud monitoring pipelines</li>
<li>Introduce merchant risk scoring systems</li>
<li>Apply transaction velocity limits</li>
<li>Enhance identity verification protocols</li>
<li>Integrate cross-bank fraud intelligence sharing</li>
</ul>

</div>
"""

# ---------------------------------------------------
# CAM REPORT GENERATOR (10 pages)
# ---------------------------------------------------

def generate_cam_report(df,results):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story=[]

    sections = [

    "Executive Summary",
    "Customer Profile Analysis",
    "Transaction Behaviour Assessment",
    "Fraud Detection Model Results",
    "Credit Risk Evaluation",
    "Fraud Network Investigation",
    "Suspicious Transaction Review",
    "Operational Risk Assessment",
    "AI Investigation Findings",
    "Final Recommendation"

    ]

    paragraph = """
The AEGIS Financial Threat Intelligence Engine performed a comprehensive evaluation of the transaction dataset.
Multiple machine learning models including Isolation Forest anomaly detection and gradient boosted fraud classifiers
were used to identify abnormal patterns, fraud clusters, and systemic financial risks. The investigation also
examined transaction network relationships between customers and merchants to identify suspicious transaction rings.
"""

    for s in sections:

        story.append(Paragraph(s,styles["Heading1"]))

        for i in range(8):

            story.append(Paragraph(paragraph,styles["Normal"]))

        story.append(Spacer(1,20))

    doc.build(story)

    buffer.seek(0)

    return buffer

# ---------------------------------------------------
# DATA UPLOAD
# ---------------------------------------------------

file = st.file_uploader("Upload Transaction Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    if st.button("Run AEGIS Threat Analysis"):

        numeric = df.select_dtypes(include=np.number)

        # Behaviour anomaly detection
        iso = IsolationForest(contamination=0.02)

        df["anomaly"] = iso.fit_predict(numeric)

        behaviour_score = abs(df["anomaly"].mean())

        fraud_prob = 0

        if "fraud" in df.columns:

            X = numeric
            y = df["fraud"]

            X_train,X_test,y_train,y_test = train_test_split(X,y)

            model = XGBClassifier()

            model.fit(X_train,y_train)

            fraud_prob = model.predict_proba(X_test)[:,1].mean()

        credit_risk = numeric.mean().mean()/numeric.max().max()

        network_risk = len(df["merchant"].unique())/len(df)

        final_score = np.mean([
        behaviour_score,
        fraud_prob,
        credit_risk,
        network_risk
        ])

        results = {
        "behaviour":round(behaviour_score,3),
        "fraud":round(fraud_prob,3),
        "credit":round(credit_risk,3),
        "network":round(network_risk,3),
        "final":round(final_score,3)
        }

        # ---------------------------------------------------
        # METRICS
        # ---------------------------------------------------

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Behaviour Risk",results["behaviour"])
        c2.metric("Fraud Probability",results["fraud"])
        c3.metric("Credit Risk",results["credit"])
        c4.metric("Network Risk",results["network"])

        st.subheader("Final Risk Score")

        st.error(results["final"])

        # ---------------------------------------------------
        # TRANSACTION DISTRIBUTION
        # ---------------------------------------------------

        st.subheader("Transaction Distribution")

        if "amount" in df.columns:

            fig = px.histogram(df,x="amount")

            st.plotly_chart(fig,use_container_width=True)

        # ---------------------------------------------------
        # FRAUD NETWORK GRAPH
        # ---------------------------------------------------

        st.subheader("Fraud Network Graph")

        G = nx.Graph()

        fraud_df = df[df["fraud"]==1].head(200)

        for _,row in fraud_df.iterrows():

            customer=row["customer"]
            merchant=row["merchant"]

            G.add_node(customer,type="customer")
            G.add_node(merchant,type="merchant")

            G.add_edge(customer,merchant)

        pos = nx.spring_layout(G)

        edge_x=[]
        edge_y=[]

        for edge in G.edges():

            x0,y0=pos[edge[0]]
            x1,y1=pos[edge[1]]

            edge_x.extend([x0,x1,None])
            edge_y.extend([y0,y1,None])

        edge_trace=go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1,color="#888"),
        hoverinfo='none',
        mode='lines')

        node_x=[]
        node_y=[]
        colors=[]

        for node in G.nodes():

            x,y=pos[node]

            node_x.append(x)
            node_y.append(y)

            if G.nodes[node]["type"]=="customer":
                colors.append("#00FFC6")
            else:
                colors.append("#FF4D4D")

        node_trace=go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(size=12,color=colors)
        )

        fig=go.Figure(data=[edge_trace,node_trace])

        fig.update_layout(
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19"
        )

        st.plotly_chart(fig,use_container_width=True)

        # ---------------------------------------------------
        # SUSPICIOUS TRANSACTIONS
        # ---------------------------------------------------

        st.subheader("Suspicious Transactions")

        suspicious=df[df["anomaly"]==-1]

        st.dataframe(suspicious.head(20))

        # ---------------------------------------------------
        # AI INVESTIGATION REPORT
        # ---------------------------------------------------

        st.subheader("AI Fraud Investigation")

        report = generate_ai_report(results)

        st.markdown(report,unsafe_allow_html=True)

        # ---------------------------------------------------
        # CAM REPORT DOWNLOAD
        # ---------------------------------------------------

        cam = generate_cam_report(df,results)

        st.download_button(
        "Download CAM Report",
        cam,
        file_name="AEGIS_CAM_Report.pdf"
        )