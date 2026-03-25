import subprocess

def generate_explanation(report):

    prompt = f"""
You are a financial fraud investigator.

Analyze this risk report and explain the threats:

{report}

Give:
1. Risk explanation
2. Suspicious behavior
3. Recommendation
"""

    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        capture_output=True,
        text=True
    )

    return result.stdout