import streamlit as st
import pandas as pd
import anthropic
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContextChatSystem:
    def __init__(self, api_key: str):
        """Initialize the chat system with tax filing data and Anthropic API client."""
        self.client = anthropic.Client(api_key=api_key)
        self.vectorizer = None
        self.document_vectors = None
        self.tax_data = None
        self.reference_files = {
            'tax_filings': 'sample_rag.csv',
            # 'organizations': './ingestion_pipeline/datasets/organizations_basic_data.csv'
            # Add more reference files as needed
        }
        self.load_data()

    def load_data(self):
        # Combine data from all reference files
        dataframes = []
        for file_path in self.reference_files.values():
            try:
                df = pd.read_csv(file_path)
                dataframes.append(df)
            except Exception as e:
                st.error(f"Error loading {file_path}: {str(e)}")

        self.tax_data = pd.concat(dataframes, ignore_index=True)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.document_vectors = self.vectorizer.fit_transform(
            self.tax_data.astype(str).apply(lambda x: ' '.join(x), axis=1)
        )

    def get_relevant_context(self, query: str, num_results: int = 3) -> List[Dict]:
        """Retrieve relevant tax filing records based on the query."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = similarities.argsort()[-num_results:][::-1]
        return self.tax_data.iloc[top_indices].to_dict('records')

    def generate_prompt(self, query: str, context: List[Dict]) -> str:
        """Generate a prompt combining the query and relevant context."""
        context_str = "\n".join([
            f"Record {i + 1}:\n" + "\n".join([f"{k}: {v}" for k, v in record.items()])
            for i, record in enumerate(context)
        ])

        return f"""You are a helpful assistant with access to tax filing information. 
        Use the following context to answer the user's question:

        Context:
        {context_str}

        User's Question: {query}

        Please provide a detailed answer based on both the provided context and your general knowledge about tax filings and organizations."""

    def get_response(self, query: str) -> str:
        """Get a response from Claude based on the query and relevant context."""
        relevant_context = self.get_relevant_context(query)
        prompt = self.generate_prompt(query, relevant_context)

        message = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            temperature=0.7,
            system="You are a helpful assistant specializing in tax filings and organizational finance.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        ## tidying the response
        formatted_response = ""
        for item in message.content:
            formatted_response += item.text.replace("\\n", "\n").replace("(", "").replace(")", "") + "\n"

        return formatted_response.strip()


def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_system' not in st.session_state:
        st.session_state.chat_system = None
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

def handle_submit():
    """Handle the submission of user input."""
    if st.session_state.user_input.strip():
        # Get the input value
        user_input = st.session_state.user_input

        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        try:
            # Get response from Claude
            response = st.session_state.chat_system.get_response(user_input)

            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

        # Clear input by resetting session state
        st.session_state.user_input = ""
