# Import necessary classes and modules for data loading, vector storage, and embeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Function to load data from a CSV file
def load_csv_data(file_path):
    # Create a CSVLoader instance to load data from the specified CSV file
    # Using 'utf-8' encoding and custom CSV delimiter
    loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
    
    # Load data from the CSV file using the defined loader
    data = loader.load()
    return data

# Function to build a FAISS database for efficient vector storage and retrieval
def build_faiss_database(data, embeddings):
    # Create a FAISS database from the provided data and embeddings
    db = FAISS.from_documents(data, embeddings)
    return db

# Function to load pre-trained embeddings for text data
def load_embeddings():
    # Create HuggingFaceEmbeddings using the specified model name
    # and device (CPU in this case)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    
    # Return the initialized embeddings instance
    return embeddings
