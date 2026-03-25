from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

def generate_cam_report(df,results):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story=[]

    sections = [

    "Executive Summary",
    "Customer Profile Analysis",
    "Transaction Behaviour Assessment",
    "Fraud Detection Analysis",
    "Credit Risk Evaluation",
    "Fraud Network Investigation",
    "Suspicious Transaction Review",
    "Operational Risk Assessment",
    "AI Investigation Findings",
    "Final Recommendation"

    ]

    text = """
The AEGIS Financial Threat Intelligence Engine performed a comprehensive evaluation of the provided transaction dataset.
This analysis incorporated multiple machine learning algorithms, behavioral anomaly detection models,
and network relationship mapping techniques to identify potential indicators of financial fraud.
"""

    for s in sections:

        story.append(Paragraph(s,styles["Heading1"]))

        for i in range(8):

            story.append(Paragraph(text,styles["Normal"]))

        story.append(Spacer(1,20))

    doc.build(story)

    buffer.seek(0)

    return buffer