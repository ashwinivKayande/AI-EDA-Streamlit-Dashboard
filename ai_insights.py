import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Current supported model
model = genai.GenerativeModel("gemini-2.0-flash")

def generate_ai_insights(summary_text):
    prompt = f"""
    You are a professional data analyst.

    Analyze the following dataset summary and provide:
    1. Executive Summary
    2. Data Quality Issues
    3. Missing Value Recommendations
    4. Potential Outliers
    5. Business Insights
    6. Data Cleaning Suggestions
    7. Feature Engineering Ideas

    Dataset Summary:
    {summary_text}

    Keep the response structured and concise.
    """

    response = model.generate_content(prompt)
    return response.text