import streamlit as st
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

# Function to load the language model (LLM) and conversational retrieval chain
def load_llm(uploaded_file, db):
    # Configure the ChatCSV language model (LLM) for conversation
    llm = CTransformers(
        model=" <<YOUR_MODEL_PATH>> /llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=1
    )
    # Create a conversational retrieval chain based on the LLM and provided Faiss database (retriever)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
    return llm, chain

# Function to engage in a conversational chat using the language model and chain
def conversational_chat(llm, chain, query):
    # Use the conversational retrieval chain to generate a response based on the user's query
    result = chain({"question": query, "chat_history": st.session_state['history']})
    # Append the user's query and the generated answer to the conversation history
    st.session_state['history'].append((query, result["answer"]))
    # Return the generated answer for display
    return result["answer"]

# Function to initialize the session state for the Streamlit app
def initialize_session_state(uploaded_file):
    # Initialize the conversation history if it doesn't exist in the session state
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize the generated responses list if it doesn't exist in the session state
    if 'generated' not in st.session_state:
        # Add an initial generated response mentioning the uploaded file name
        st.session_state['generated'] = ["Sure, I learned recently " + uploaded_file.name + " and I'm ready to assist."]

    # Initialize the past user inputs list if it doesn't exist in the session state
    if 'past' not in st.session_state:
        # Add an initial user input to prompt exploration of the knowledge base
        st.session_state['past'] = ["Let's explore our knowledge base."]
