import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
load_dotenv()


class TaxAnalyzer:
    def __init__(self):
        self.client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    def is_predictive_query(self, query: str) -> bool:
        """Check if query is asking for predictions/projections"""
        predictive_terms = ['predict', 'projection', 'estimate', 'forecast', 'future',
                          'next year', 'trend', 'expected', 'outlook', 'potential']
        return any(term in query.lower() for term in predictive_terms)

    def get_relevant_records(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Get relevant records with historical context for predictions if needed"""
        query_lower = query.lower()

        # If query contains an EIN, filter by it
        if 'ein:' in query_lower or 'ein' in query_lower:
            ein_val = ''.join(c for c in query_lower if c.isdigit())
            if ein_val and 'ein' in df.columns:
                return df[df['ein'].astype(str) == ein_val].sort_values('tax_period_end')

        # For predictive queries, ensure we have historical context
        if self.is_predictive_query(query):
            # Get last 3-5 years of data for the relevant organization/metric
            return df.sort_values('tax_period_end', ascending=True).tail(5)

        # For regular queries, limit to most relevant recent records
        return df.sort_values('tax_period_end', ascending=False).head(3)

    def create_context(self, df: pd.DataFrame, relevant_records: pd.DataFrame, query: str) -> str:
        """Create context with trend information for predictions if needed"""
        if self.is_predictive_query(query):
            # Calculate year-over-year changes for key metrics
            if len(relevant_records) > 1:
                records_summary = "Historical Trends:\n"
                for col in ['total_revenue', 'total_expenses', 'total_contributions']:
                    if col in relevant_records.columns:
                        records_summary += f"\n{col.replace('_', ' ').title()}:\n"
                        prev_value = None
                        for _, row in relevant_records.iterrows():
                            current_value = row[col]
                            if prev_value is not None and prev_value != 0:
                                pct_change = ((current_value - prev_value) / prev_value) * 100
                                records_summary += f"{row['tax_period_end']}: ${current_value:,.2f} ({pct_change:+.1f}%)\n"
                            else:
                                records_summary += f"{row['tax_period_end']}: ${current_value:,.2f}\n"
                            prev_value = current_value
            else:
                records_summary = relevant_records.to_string(index=False)
        else:
            records_summary = relevant_records.to_string(index=False)

        context = f"""
        {records_summary}
        
        Summary:
        - Records Shown: {len(relevant_records)}
        - Time Range: {relevant_records['tax_period_begin'].min()} to {relevant_records['tax_period_end'].max()}
        """
        return context

    def analyze(self, df: pd.DataFrame, query: str) -> str:
        """Analyze tax records with prediction support"""
        try:
            # Get relevant subset of records
            relevant_records = self.get_relevant_records(df, query)
            context = self.create_context(df, relevant_records, query)

            # Adjust system prompt based on query type
            if self.is_predictive_query(query):
                system_prompt = """You are analyzing nonprofit tax records to provide insights and projections.
                For predictive questions:
                1. Use historical trends to inform predictions
                2. Consider growth rates and patterns in the data
                3. Provide realistic ranges rather than exact numbers
                4. Include key factors that could impact future performance
                5. Note relevant assumptions and limitations
                
                Base all analysis on the provided historical data."""
            else:
                system_prompt = """You are analyzing tax records. Answer questions directly based on the records shown.
                Only report information that is explicitly present in the data."""

            # Get analysis from Claude
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Based on these tax records:\n\n{context}\n\nQuestion: {query}"
                }],
                max_tokens=1000
            )

            return " ".join(item.text for item in response.content)

        except Exception as e:
            return f"Error analyzing records: {str(e)}"
