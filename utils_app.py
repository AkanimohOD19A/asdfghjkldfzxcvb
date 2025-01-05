import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
load_dotenv()


class TaxAnalyzer:
    def __init__(self):
        self.client = Anthropic(api_key=st.secrets("ANTHROPIC_API_KEY"))

    def analyze(self, df: pd.DataFrame, query: str) -> str:
        """
        Simply provide the data context and let Claude analyze it directly
        """
        # Create context with full relevant data
        context = f"""
        Available Records:
        {df.to_string()}

        Data Summary:
        - Total Records: {len(df)}
        - Unique Organizations: {df['business_name'].nunique()}
        - Available Columns: {', '.join(df.columns.tolist())}
        """

        # Get analysis from Claude with minimal processing
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            system="You are analyzing tax records. Look at the provided data carefully and answer questions directly "
                   "based on the records shown. If you see the information in the data, report it exactly as shown. "
                   "Do not try to calculate or estimate values - only report what you directly observe in the records.",
            messages=[
                {"role": "user", "content": f"""
                Here are the tax records:

                {context}

                Question: {query}

                Look at the records carefully and answer based on what you see in the data."""}
            ],
            max_tokens=1000
        )

        formatted_response = ""
        for item in response.content:
            formatted_response += item.text.replace("\\n", "\n").replace("(", "").replace(")", "") + "\n"

        return formatted_response.strip()
