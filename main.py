import streamlit as st
import tempfile
from chat_utils import load_llm, conversational_chat, initialize_session_state
from data_utils import load_csv_data, build_faiss_database, load_embeddings
from message_utils import display_message

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Apply custom CSS style to the Streamlit app
st.markdown(
    """
    <style>
    .main-title {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 30px;
        color: #1E88E5;
        font-family: 'Courier New', monospace;
    }
    .content-container {
        background-color: #FAFAFA;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .user-input {
        margin-top: 10px;
    }
    .submit-button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 20px;
        font-size: 16px;
        cursor: pointer;
        font-family: 'Courier New', monospace;
    }
    .response-container {
        margin-top: 20px;
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
    }
    .message {
        margin-top: 10px;
        padding: 8px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    .user-message {
        background-color: #4CAF50;
        color: white;
    }
    .bot-message {
        background-color: #FFC107;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the main title using the custom CSS class
st.markdown("<div class='main-title'>Craft Personalized Conversations</div>", unsafe_allow_html=True)

# File uploader centered at the top
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    knowledge_base = st.file_uploader("Upload your knowledge base", type="csv")

# Rest of the code remains the same...



# Check if a CSV file has been uploaded
if knowledge_base:
    # Create a temporary file to store the uploaded CSV content
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(knowledge_base.getvalue())
        tmp_file_path = tmp_file.name

    # Load CSV data from the temporary file
    data = load_csv_data(tmp_file_path)

    # Load pre-trained embeddings
    embeddings = load_embeddings()

    # Build a Faiss database using the loaded data and embeddings
    db = build_faiss_database(data, embeddings)

    # Save the Faiss database locally
    db.save_local(DB_FAISS_PATH)

    # Load the language model and conversation chain
    llm, chain = load_llm(knowledge_base, db)

    # Initialize session state to keep track of conversation history
    initialize_session_state(knowledge_base)

    # Create containers for displaying user input and responses
    response_container = st.container()
    container = st.container()

    # Create a form to take user input
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            # Create a text input field for the user to enter queries
            user_input = st.text_input("Query:", placeholder="Ask me anything on supplied knowledge!", key='input')
            # Create a submit button
            submit_button = st.form_submit_button(label='Send')

        # If the submit button is clicked and user input is provided
        if submit_button and user_input:
            # Generate a response using conversational_chat function
            output = conversational_chat(llm, chain, user_input)
            # Display the user's input
            display_message(user_input, is_user=True)
            # Display the generated response
            display_message(output)
            # Append the user input and generated response to session state
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # If there are generated responses in the session state
    if st.session_state['generated']:
        # Display past interactions and generated responses
        with response_container:
            for i in range(len(st.session_state['generated'])):
                # Display the user's input
                display_message(st.session_state["past"][i], is_user=True)
                # Display the generated response
                display_message(st.session_state["generated"][i])
