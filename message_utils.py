import streamlit as st

def display_message(content, is_user=False):
    if is_user:
        st.text(f"User: {content}")
    else:
        st.text(f"Bot: {content}")
