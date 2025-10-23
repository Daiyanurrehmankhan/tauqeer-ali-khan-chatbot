from flask import Flask, request, jsonify, Response, render_template
from app import chat

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

chat_histories={}

@app.route('/chat', methods=['POST'])
def chat_route():
    data = request.json
    query = data.get('query')
    session_id=data.get('session_id')
    
    if session_id not in chat_histories:
        chat_histories[session_id]= []
    
    def stream_with_context(query):
        for chunk in chat(query,chat_histories[session_id]):
            yield chunk
    
    return Response(stream_with_context(query), mimetype='text/event-stream')       

if __name__ == '__main__':
    app.run(debug=True, port=8000)