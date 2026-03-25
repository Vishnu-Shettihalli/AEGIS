import ollama


def generate_ai_report(customer_name, behaviour_score, credit_score, network_score, final_score, recommendation):

    prompt = f"""
You are an AI financial risk analyst working for a bank.

Analyze the following risk assessment for a loan applicant.

Customer Name: {customer_name}

Behaviour Risk Score: {behaviour_score}
Credit Risk Score: {credit_score}
Network Risk Score: {network_score}

Final Risk Score: {final_score}

System Recommendation: {recommendation}

Explain the risk clearly for a bank loan officer.
Mention possible financial behaviour issues, fraud indicators,
and repayment risk.

Keep the explanation professional and concise.
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]