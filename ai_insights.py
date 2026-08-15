import os
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )

client = genai.Client(
    api_key=api_key
)

MODEL_NAME = "gemini-3.5-flash"


# =========================================================
# CREATE SMALL DATASET SUMMARY
# =========================================================

def prepare_dataset_summary(df):

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }

    # Only first 10 rows
    sample = df.head(10).to_string(
        index=False
    )

    # Numeric statistics
    numeric_df = df.select_dtypes(
        include="number"
    )

    if not numeric_df.empty:

        statistics = (
            numeric_df
            .describe()
            .round(2)
            .to_string()
        )

    else:

        statistics = "No numerical columns."

    return summary, sample, statistics


# =========================================================
# AI INSIGHTS
# =========================================================

def generate_ai_insights(df):

    summary, sample, statistics = (
        prepare_dataset_summary(df)
    )

    prompt = f"""
You are a professional Data Analyst.

Analyze the following dataset summary.

DATASET OVERVIEW:
Rows: {summary["rows"]}
Columns: {summary["columns"]}

COLUMN NAMES:
{summary["column_names"]}

DATA TYPES:
{summary["data_types"]}

MISSING VALUES:
{summary["missing_values"]}

DUPLICATE ROWS:
{summary["duplicate_rows"]}

NUMERICAL STATISTICS:
{statistics}

SAMPLE OF DATA:
{sample}

Provide a concise professional EDA report with:

1. Dataset Overview
2. Important Data Patterns
3. Missing Value Analysis
4. Duplicate Analysis
5. Statistical Insights
6. Important Relationships
7. Potential Data Quality Issues
8. Business Insights
9. Recommendations

Do not invent information.
Use only the information provided above.

Keep the response concise.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# =========================================================
# CHAT WITH DATASET
# =========================================================

def chat_with_dataset(df, question):

    summary, sample, statistics = (
        prepare_dataset_summary(df)
    )

    prompt = f"""
You are a Data Analyst AI assistant.

Answer the user's question using only the
dataset information provided below.

DATASET:
Rows: {summary["rows"]}
Columns: {summary["columns"]}

COLUMN NAMES:
{summary["column_names"]}

DATA TYPES:
{summary["data_types"]}

MISSING VALUES:
{summary["missing_values"]}

DUPLICATE ROWS:
{summary["duplicate_rows"]}

STATISTICS:
{statistics}

SAMPLE DATA:
{sample}

USER QUESTION:
{question}

Give a clear and concise answer.
Do not invent data.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text