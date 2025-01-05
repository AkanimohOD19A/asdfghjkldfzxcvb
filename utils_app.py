import os
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
load_dotenv()


class TaxAnalyzer:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def analyze(self, df: pd.DataFrame, query: str) -> str:
        """
        Analyze the tax records based on the query using direct context
        """
        # Create a more informative context string
        context = f"""
        Data Summary:
        - Total Records: {len(df)}
        - Unique Organizations: {df['business_name'].nunique()}
        - Date Range: {df['tax_period_begin'].min()} to {df['tax_period_end'].max()}
        
        Sample Record:
        {df}
        
        Available Metrics:
        Financial: total_revenue, total_expenses, total_contributions, program_service_revenue
        Efficiency: operating_margin, program_efficiency
        Personnel: total_volunteers, total_employees
        Assets: total_assets_eoy, total_liabilities_eoy
        """

        # Get analysis from Claude
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            system="You are a tax analysis assistant specializing in nonprofit tax records (Form 990). "
                   "Provide clear, accurate analysis based on the provided data. "
                   "When answering questions, reference specific numbers and fields from the data. "
                   "If certain information is not available in the provided data, clearly state that. "
                   "Provide your response in clear, natural language without any special formatting or markers.",
            messages=[
                {"role": "user", "content": f"""
                Based on these nonprofit tax records:

                {context}

                Question: {query}

                Please provide a clear, specific answer using the available data."""}
            ],
            max_tokens=1000
        )

        ## tidying the response
        formatted_response = ""
        for item in response.content:
            formatted_response += item.text.replace("\\n", "\n").replace("(", "").replace(")", "") + "\n"

        return formatted_response.strip()