import os
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from rag_working import get_response

# Remove logging import

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def chat(query, chat_history):
    context = get_response(query)
    
    SYSTEM_PROMPT = (
    "You are a helpful assistant providing information about Book named Islam The Religion. "
    "Respond like a human not like a bot. And not use phrases like based on the provided context or information provided."
    f"Answer the user's query based on the provided context: {context}"
    "Crucially, **respond in the exact same language as the user's query** (English or Urdu). "
    "If the query is in Urdu, respond entirely in Urdu. If it is in English, respond in English. "
    "Do not use 'Based on the information provided', answer with confidence and assertiveness. "
    "If the user asks about the author of the book, you should say the author is Syed Anwer Ali."
    "The name of the first Surah is Al-Fatihah not Al-Fatcha."
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
            
    chat_history.append(complete_response)
