import os
import logging
from glob import glob
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Suppress low-level warnings from Google/absl libraries
# This sets the general logging level for Python, which often silences the
# non-critical warnings emitted by underlying Google client libraries.
logging.getLogger().setLevel(logging.ERROR)

load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "tauqeer_profile"
# --- End Configuration ---

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="RETRIEVAL_DOCUMENT",
    google_api_key=GEMINI_API_KEY
)

# Function to initialize the vector store (either loads existing or creates new)
def get_vectorstore():
    if os.path.exists(CHROMA_DB_PATH):
        # Load the existing persistent store
        return Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
    else:
        # Create a new persistent store
        # Find all files in the data directory (supports .txt, .md, etc.)
        all_data_files = glob("data/*")
        
        # Load documents
        documents = []
        for file_path in all_data_files:
            try:
                # Use TextLoader for generic text/markdown files
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            except Exception:
                # Silently skip files that TextLoader cannot handle (e.g., binaries)
                pass

        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Create and persist the vector store
        vectorstore = Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME
        )
        return vectorstore

# Function to perform RAG retrieval
def get_response(query):
    vectorstore = get_vectorstore()
    # Retrieve the top 3 most relevant documents (adjust k as needed)
    docs = vectorstore.similarity_search(query, k=3)
    
    # Concatenate the page content of the retrieved documents to form the context
    context = "\n---\n".join([doc.page_content for doc in docs])
    
    return context
