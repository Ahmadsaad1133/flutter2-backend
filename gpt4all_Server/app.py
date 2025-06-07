import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_GREETING = "Hey, welcome to Sleep Moon AI assistant! How can I help you today?"

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set in environment variables!")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt.strip():
        return jsonify({"response": DEFAULT_GREETING})
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        res = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Request failed", "details": str(e)}), 500

    if res.status_code != 200:
        return jsonify({"error": "Groq API error", "details": res.text}), 500
    
    result = res.json()
    message = result["choices"][0]["message"]["content"]
    
    return jsonify({"response": message})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port)

