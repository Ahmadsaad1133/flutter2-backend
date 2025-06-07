import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set in environment variables!")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are a friendly and supportive sleep coach named Silent Veil. "
        "Your job is to help users improve their sleep habits, reduce stress, manage anxiety, "
        "and build relaxing bedtime routines. Respond in a calm, reassuring, and conversational tone. "
        "Avoid repeating greetings like 'hello there' unless the user greets you first. "
        "Offer helpful suggestions without using technical or medical jargon unless asked."
    )

    # Compose messages to always include the system prompt
    messages = [{"role": "system", "content": system_prompt}]

    if prompt.strip():
        # Normal user input
        messages.append({"role": "user", "content": prompt})
    else:
        # If user sends empty input, treat as a greeting to start the chat
        messages.append({"role": "user", "content": "Hi, what can you help me with?"})

    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
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

