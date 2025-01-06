import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
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
        self.client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        self.conversation_history = []

    def get_summary_stats(self, df, columns_of_interest=None):
        """Get summary statistics for specified columns"""
        if columns_of_interest is None:
            columns_of_interest = [
                'total_revenue', 'total_expenses', 'program_service_expenses',
                'admin_expenses', 'fundraising_expenses', 'total_assets',
                'total_liabilities', 'net_assets'
            ]

        stats = {}
        for col in columns_of_interest:
            if col in df.columns:
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std()
                }
        return stats

    def analyze(self, df: pd.DataFrame, df_x: pd.DataFrame, query: str, ein_selected: str) -> str:
        try:
            keywords = ["peer", "compare"]
            needs_comparison = any(keyword in query.lower() for keyword in keywords)

            if needs_comparison:
                # Use optimized comparison approach
                context = "Analysis Context:\n\n"

                # Basic metadata
                context += f"Dataset Overview:\n"
                context += f"Total Organizations: {df_x['business_name'].nunique()}\n"
                context += f"Date Range: {df_x['tax_period_begin'].min()} to {df_x['tax_period_end'].max()}\n\n"

                # Get most recent year's data for comparison
                latest_year = df_x['tax_period_end'].max()
                recent_data = df_x[df_x['tax_period_end'] == latest_year]

                # Calculate summary stats
                stats = self.get_summary_stats(recent_data)

                context += "Industry Statistics (Most Recent Year):\n"
                for metric, values in stats.items():
                    context += f"\n{metric}:\n"
                    for stat_name, value in values.items():
                        if pd.notnull(value):
                            formatted_value = f"${value:,.2f}" if any(term in metric.lower() for term in
                                                                      ['revenue', 'expenses', 'assets',
                                                                       'liabilities']) else f"{value:,.2f}"
                            context += f"- {stat_name}: {formatted_value}\n"

                # Add selected organization's metrics if available
                if ein_selected != "General Context":
                    selected_data = df[df['tax_period_end'] == latest_year]
                    if not selected_data.empty:
                        context += "\nSelected Organization Metrics:\n"
                        for col in stats.keys():
                            if col in selected_data.columns:
                                value = selected_data[col].iloc[0]
                                if pd.notnull(value):
                                    formatted_value = f"${value:,.2f}" if any(term in col.lower() for term in
                                                                              ['revenue', 'expenses', 'assets',
                                                                               'liabilities']) else f"{value:,.2f}"
                                    context += f"- {col}: {formatted_value}\n"
            else:
                # Use original approach for non-comparison queries
                context = "Complete Database Records:\n\n"

                # Add metadata about the dataset
                context += f"Dataset Overview:\n"
                context += f"- Total Organizations: {df['business_name'].nunique()}\n"
                context += f"- Total Records: {len(df)}\n"
                context += f"- Date Range: {df['tax_period_begin'].min()} to {df['tax_period_end'].max()}\n\n"

                # Add complete records with ALL columns
                for _, row in df.iterrows():
                    context += f"\nOrganization: {row['business_name']}\n"
                    for column in df.columns:
                        if pd.notnull(row[column]):
                            if isinstance(row[column], (int, float)):
                                value = f"${row[column]:,.2f}" if any(term in column.lower() for term in
                                                                      ['revenue', 'expenses', 'compensation', 'assets',
                                                                       'liabilities']) else f"{row[column]:,}"
                            else:
                                value = row[column]
                            context += f"- {column}: {value}\n"
                    context += "-" * 50 + "\n"

            # Add conversation history for both approaches
            if self.conversation_history:
                context += "\nPrevious Conversation Context:\n"
                for q, a in self.conversation_history[-3:]:
                    context += f"\nQ: {q}\nA: {a}\n"
                context += "-" * 50 + "\n"

            # Add selected EIN context if not "General Context"
            if ein_selected != "General Context" and not needs_comparison:
                selected_df = df[df['ein'] == ein_selected]
                if not selected_df.empty:
                    context += "\nSelected EIN Context:\n"
                    for _, row in selected_df.iterrows():
                        context += f"\nOrganization: {row['business_name']}\n"
                        for column in selected_df.columns:
                            if pd.notnull(row[column]):
                                if isinstance(row[column], (int, float)):
                                    value = f"${row[column]:,.2f}" if any(term in column.lower() for term in
                                                                          ['revenue', 'expenses', 'compensation',
                                                                           'assets',
                                                                           'liabilities']) else f"{row[column]:,}"
                                else:
                                    value = row[column]
                                context += f"- {column}: {value}\n"
                        context += "-" * 50 + "\n"

            # Get analysis from Claude with appropriate system message
            system_message = """You are analyzing nonprofit tax records with access to complete financial and operational data. For each analysis:
                1. Use ALL available metrics (financial, operational, and organizational)
                2. Consider key performance indicators like:
                   - Revenue streams (contributions, program service, other)
                   - Operational metrics (employees, volunteers, voting members)
                   - Financial health (assets, liabilities, net assets)
                   - Efficiency metrics (operating margin, program efficiency)
                3. Compare organizations when relevant
                4. Reference specific data points to support insights
                5. Consider historical context from previous conversations

                Make full use of ALL available data fields to provide comprehensive analysis."""

            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                system=system_message,
                messages=[{
                    "role": "user",
                    "content": f"Using the complete tax records and our conversation history:\n\n{context}\n\nQuestion: {query}"
                }],
                max_tokens=1500
            )

            # Fixed response handling
            answer = response.content[0].text if response.content else "Unable to generate analysis"

            # Clean up formatting
            answer = answer.replace("$,", "$").replace("  ", " ").replace(" .", ".")

            self.conversation_history.append((query, answer))
            if len(self.conversation_history) > 10:
                self.conversation_history.pop(0)

            return answer

        except Exception as e:
            return f"Error analyzing records: {str(e)}"
