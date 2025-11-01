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
    
    # Define the key phrase we expect the model to use when refusing to answer
    REFUSAL_PHRASE = "out of my scope"
    
    # SYSTEM_PROMPT = (
    # "You are an expert assistant providing information about Tauqeer Ali Khan. "
    # f"Answer the user's query based on the provided context: {context}"
    # "Do not use 'Based on the information provided', answer with confidence and assertiveness. "
    # "Users can also address you as Tauqeer if they are asking about you it means "
    # "they are asking about Tauqeer. If someone asks about your contact information, you should provide Tauqeer's contact information."
    # )
    
    prompt_with_context = context + query
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
