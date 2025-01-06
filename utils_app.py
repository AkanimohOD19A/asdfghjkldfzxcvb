import os
import sqlite3
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
from fuzzywuzzy import fuzz
from datetime import datetime

load_dotenv()


def get_db_data(query=None):
    """Retrieve data from SQLite database with improved error handling"""
    try:
        conn = sqlite3.connect("tax_data.db")
        if query is None:
            query = """
            SELECT *
            FROM tax_form_basic_data
            ORDER BY tax_period_end DESC
            """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()


class TaxAnalyzer:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # self.client = Anthropic(api_key=st.secrets("ANTHROPIC_API_KEY"))
        self.conversation_history = []

    def analyze(self, df: pd.DataFrame, df_x: pd.DataFrame, query: str, ein_selected: str) -> str:
        """Analyze using complete database records and conversation history"""
        try:
            # Determine if the query requires peer comparison
            keywords = ["peer",
                        "compare"]  # , "benchmark", "contrast", "evaluate", "examine", "juxtapose", "measure", "weigh", "analyze"]
            use_df_x = any(keyword in query.lower() for keyword in keywords)

            # Select the appropriate DataFrame
            context_df = df_x if use_df_x else df

            # Create comprehensive context with ALL data
            context = "Complete Database Records:\n\n"

            # Add metadata about the dataset
            context += f"Dataset Overview:\n"
            context += f"- Total Organizations: {context_df['business_name'].nunique()}\n"
            context += f"- Total Records: {len(context_df)}\n"
            context += f"- Date Range: {context_df['tax_period_begin'].min()} to {context_df['tax_period_end'].max()}\n\n"

            # Add complete records with ALL columns
            for _, row in context_df.iterrows():
                context += f"\nOrganization: {row['business_name']}\n"
                # Include every single column and its value
                for column in context_df.columns:
                    if pd.notnull(row[column]):  # Only include non-null values
                        # Format numbers with commas and decimals
                        if isinstance(row[column], (int, float)):
                            value = f"${row[column]:,.2f}" if 'revenue' in column or 'expenses' in column or 'compensation' in column or 'assets' in column or 'liabilities' in column else f"{row[column]:,}"
                        else:
                            value = row[column]
                        context += f"- {column}: {value}\n"
                context += "-" * 50 + "\n"  # Separator between organizations

            # Add conversation history
            if self.conversation_history:
                context += "\nPrevious Conversation Context:\n"
                for q, a in self.conversation_history[-3:]:
                    context += f"\nQ: {q}\nA: {a}\n"
                context += "-" * 50 + "\n"

            # Add selected EIN context if not "General Context"
            if ein_selected != "General Context":
                selected_df = df[df['ein'] == ein_selected]
                if not selected_df.empty:
                    context += "\nSelected EIN Context:\n"
                    for _, row in selected_df.iterrows():
                        context += f"\nOrganization: {row['business_name']}\n"
                        for column in selected_df.columns:
                            if pd.notnull(row[column]):  # Only include non-null values
                                if isinstance(row[column], (int, float)):
                                    value = f"${row[column]:,.2f}" if 'revenue' in column or 'expenses' in column or 'compensation' in column or 'assets' in column or 'liabilities' in column else f"{row[column]:,}"
                                else:
                                    value = row[column]
                                context += f"- {column}: {value}\n"
                        context += "-" * 50 + "\n"

            # Get analysis from Claude
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                system="""You are analyzing nonprofit tax records with access to complete financial and operational data. For each analysis:
                1. Use ALL available metrics (financial, operational, and organizational)
                2. Consider key performance indicators like:
                   - Revenue streams (contributions, program service, other)
                   - Operational metrics (employees, volunteers, voting members)
                   - Financial health (assets, liabilities, net assets)
                   - Efficiency metrics (operating margin, program efficiency)
                3. Compare organizations when relevant
                4. Reference specific data points to support insights
                5. Consider historical context from previous conversations

                Make full use of ALL available data fields to provide comprehensive analysis.""",
                messages=[{
                    "role": "user",
                    "content": f"Using the complete tax records and our conversation history:\n\n{context}\n\nQuestion: {query}"
                }],
                max_tokens=1500
            )

            # Update conversation history
            answer = " ".join(item.text for item in response.content)

            # Clean up any markdown formatting issues
            answer = answer.replace("$,", "$")  # Fix currency formatting
            answer = answer.replace("  ", " ")  # Remove double spaces
            answer = answer.replace(" .", ".")   # Fix spacing before periods

            self.conversation_history.append((query, answer))
            if len(self.conversation_history) > 10:
                self.conversation_history.pop(0)

            return answer

        except Exception as e:
            return f"Error analyzing records: {str(e)}"
