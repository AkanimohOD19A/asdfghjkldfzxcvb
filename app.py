from utils_rag import *
import streamlit as st

st.set_page_config(page_title="Tax Filing Chat Assistant", layout="wide")

initialize_session_state()

st.title("Tax Filing Chat Assistant")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Anthropic API Key:", type="password")
    # uploaded_file = st.file_uploader("Upload Tax Filings CSV", type=['csv'])

    if st.button("Initialize Chat System") and api_key: # and uploaded_file:
        with st.spinner("Initializing chat system..."):
            st.session_state.chat_system = ContextChatSystem(api_key)
            # st.session_state.chat_system.load_data(uploaded_file)
            st.success("Chat system initialized!")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.write(f"**You**: {content}")
        else:
            st.write(f"**Assistant**: {content}")

# Chat input
user_input_key = f"user_input{len(st.session_state.chat_history)}"
user_input = st.text_input("Ask a question about the tax filings:", key=user_input_key)

if st.button("Send") and user_input:
    with st.spinner("Getting response..."):
        try:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            # Get response from Claude
            response = st.session_state.chat_system.get_response(user_input)

            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

            # Rerun to update chat display
            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Add a clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
