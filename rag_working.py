import os
import logging
import hashlib 
import base64
from glob import glob
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader # Added PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from google import genai
from langchain.docstore.document import Document # Import Document class

# Suppress low-level warnings from Google/absl libraries
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

# Initialize the Gemini Client for multimodal tasks (Image Description)
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    print("Warning: Gemini Client not initialized. Image processing will be skipped.")
    gemini_client = None


# New utility function to process images via the Gemini API
def process_image_to_document(file_path: str, client: genai.Client):
    """
    Uses the Gemini API to get a text description of an image, returning a Document object.
    """
    if not client:
        return []

    print(f"Generating text description for image: {file_path}...")
    
    try:
        with open(file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        mime_type = f"image/{os.path.splitext(file_path)[1].lstrip('.')}"
        
        prompt = (
            f"Provide a concise, detailed, and professional summary of the image content, "
            f"focusing on technical aspects, themes, or implied skills. "
            f"Do not include phrases like 'Based on the image'."
        )
        
        # API Payload for image understanding
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                {"role": "user", "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": base64_image}}
                ]}
            ]
        )
        
        doc = Document(
            page_content=f"Image description for {os.path.basename(file_path)}: {response.text}",
            metadata={"source": file_path, "type": "image_description"}
        )
        return [doc]
        
    except Exception as e:
        print(f"Error processing image {file_path} with Gemini API: {e}")
        return []


# New utility function to generate a stable ID for each chunk
def generate_chunk_id(chunk_content: str, source_path: str, chunk_index: int) -> str:
    """
    Generates a stable, unique ID for a document chunk using a hash of its content.
    """
    unique_key = f"{os.path.basename(source_path)}_{chunk_content}_c{chunk_index}"
    return hashlib.sha256(unique_key.encode('utf-8')).hexdigest()

# Function to load and prepare documents with stable IDs (REFACTORED for multimodal)
# rag_working.py

# ... (keep all imports and configuration the same)

# Function to load and prepare documents with stable IDs (REFACTORED for multimodal)
# rag_working.py (Complete updated function)

# ... (keep all imports and configuration the same)

# Function to load and prepare documents with stable IDs (REFACTORED for multimodal)
def load_and_prepare_documents():
    all_files = glob("data/*")
    documents = []

    for file_path in all_files:
        file_extension = os.path.splitext(file_path)[1].lower()
        loader = None

        if file_extension in ['.txt', '.md']:
            loader = TextLoader(file_path)
        elif file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
            print(f"PDF {file_path} loaded successfully")
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            # Handle images using the Gemini multi-modal processor
            image_docs = process_image_to_document(file_path, gemini_client)
            documents.extend(image_docs)
            print(f"Image {file_path} processed successfully")
            continue
        
        if loader:
            try:
                loaded_docs = loader.load()
                
                # --- FILTER DOCUMENTS IMMEDIATELY AFTER LOADING ---
                # Check for documents that contain non-empty, non-whitespace content
                valid_loaded_docs = [
                    doc for doc in loaded_docs if doc.page_content and doc.page_content.strip()
                ]
                if len(loaded_docs) > len(valid_loaded_docs):
                     print(f"⚠️ Warning: Skipped {len(loaded_docs) - len(valid_loaded_docs)} empty pages from {file_path}")
                     
                documents.extend(valid_loaded_docs)
                # --- END FILTER ---

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # --- CRITICAL FIX: Filter out chunks that are too short/fragmented ---
    MIN_CHUNK_LENGTH = 30  # Chunks shorter than this are likely TOC noise

    non_empty_chunks = [
        chunk for chunk in chunks 
        if chunk.page_content and len(chunk.page_content.strip()) >= MIN_CHUNK_LENGTH
    ]
    
    if len(chunks) > len(non_empty_chunks):
         print(f"Cleaned {len(chunks) - len(non_empty_chunks)} small chunks before embedding.")
    # --- END CRITICAL FIX ---
    
    # Generate stable IDs for all non-empty chunks
    chunk_ids = []
    for i, chunk in enumerate(non_empty_chunks):
        source_path = chunk.metadata.get('source', f"unknown_source_{i}")
        chunk_id = generate_chunk_id(chunk.page_content, source_path, i)
        chunk_ids.append(chunk_id)
        
    return non_empty_chunks, chunk_ids

# --- Core Database Operations (Kept Separate for Execution Control) ---

def create_initial_vectorstore():
    """Performs the slow initial indexing and creation."""
    print("ChromaDB folder not found. Starting indexing and creation...")
    chunks, chunk_ids = load_and_prepare_documents()
    
    vectorstore = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        ids=chunk_ids,
    )
    print("Chroma database created successfully.")
    return vectorstore

def load_existing_vectorstore():
    """Performs a fast load of the existing database."""
    print("Loading existing Chroma database...")
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

def perform_upsert(vectorstore: Chroma):
    """Loads all data, generates stable IDs, and performs an upsert."""
    print("\nStarting Upsert/Update process for existing Chroma database...")
    
    chunks, chunk_ids = load_and_prepare_documents()
    
    vectorstore.add_documents(
        documents=chunks,
        ids=chunk_ids
    )

    print(f"Upsert complete. Total documents/chunks processed: {len(chunks)}")
    return vectorstore

# --- Global Vector Store Assignment (FAST LOAD for HTTP Server) ---

vectorstore = None
if os.path.exists(CHROMA_DB_PATH):
    vectorstore = load_existing_vectorstore()

# --- Execution Block (ONLY runs if file is executed directly via 'python rag_working.py') ---
if __name__ == "__main__":
    print("\n--- Running RAG Data Indexer ---")
    
    # 1. Check if DB exists
    if not os.path.exists(CHROMA_DB_PATH):
        # A. Database does not exist: Create it.
        vectorstore = create_initial_vectorstore()
    else:
        # B. Database exists: Load it and run upsert.
        vectorstore_instance = load_existing_vectorstore() 
        vectorstore = perform_upsert(vectorstore_instance)
        
    print("\n--- RAG Indexing Process Complete ---")
    print(f"Database status: {CHROMA_DB_PATH} is now ready for use.")
    
# Function to perform RAG retrieval (uses the globally assigned 'vectorstore')
def get_response(query):
    global vectorstore
    
    if vectorstore is None:
        try:
            vectorstore = load_existing_vectorstore()
        except Exception:
            raise Exception("Vector store is not initialized. Please run rag_working.py directly once to index data.")
        
    # --- QUERY REWRITING LOGIC ---
    original_query = query.lower()
    
    # Check for generic contact info queries and augment them
    if "contact" in original_query:
        print(f"DEBUG: Rewriting query '{query}' to include 'Tauqeer Ali Khan'.")
        # Augment the query to include the explicit name for better retrieval
        query_for_retrieval = f"Tauqeer Ali Khan's {query}" 
    else:
        query_for_retrieval = query
    # --- END QUERY REWRITING LOGIC ---

    # Retrieve the top 5 most relevant documents
    docs = vectorstore.similarity_search(query_for_retrieval, k=10)
    
    context = "\n---\n".join([doc.page_content for doc in docs])
    
    return context
