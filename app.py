import os
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from rag_working import get_response

# Remove logging import

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define the file path for logging
IRRELEVANT_QUERY_FILE = "irrelevant_queries.txt"

def chat(query, chat_history):
    context = get_response(query)
    
    # Define the key phrase we expect the model to use when refusing to answer
    REFUSAL_PHRASE = "out of my scope"
    
    SYSTEM_PROMPT = (
    "You are an expert assistant providing information about Tauqeer Ali Khan. "
    f"Answer the user's query based on the provided context: {context}"
    "Do not use 'Based on the information provided', answer with confidence and assertiveness. "
    "Users can also address you as Tauqeer if they are asking about you it means "
    "they are asking about Tauqeer. If someone asks about your contact information, you should provide Tauqeer's contact information."
    )
    
    prompt_with_context = SYSTEM_PROMPT + context + query
    chat_history.append(prompt_with_context)
    
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=chat_history,
    )
    
    complete_response = ''
    
    for chunk in response:
        if chunk.text:
            text = chunk.text
            complete_response += text
            yield text
    
    if complete_response and not complete_response.endswith('\n'):
        yield '\n'        
            
    # --- Implement Irrelevance Check and File Writing ---
    # Check if the complete response contains the refusal phrase (case-insensitive)
    if REFUSAL_PHRASE.lower() in complete_response.lower():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Query: {query.strip()}\n"
        
        try:
            # Open the file in append mode ('a') and write the new log entry
            with open(IRRELEVANT_QUERY_FILE, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except IOError as e:
            # Fallback if file writing fails
            print(f"Error writing to file {IRRELEVANT_QUERY_FILE}: {e}")
            
    chat_history.append(complete_response)
