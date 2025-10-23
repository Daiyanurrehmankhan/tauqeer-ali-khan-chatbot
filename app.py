from google import genai
import os
from dotenv import load_dotenv;
from rag_working import get_response
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def chat(query,chat_history):
    context = get_response(query)
    prompt_with_context = query + context
    
    chat_history.append(prompt_with_context)
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=chat_history,
    )
    
    complete_response = ''
    
    for chunk in response:
        if chunk.text:
            text=chunk.text
            complete_response +=text
            yield text
    
    if complete_response and not complete_response.endswith('\n'):
        yield '\n'        
            
    chat_history.append(complete_response)

