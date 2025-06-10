import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")
DEEPAI_URL = "https://api.deepai.org/api/text2img"

if not GROQ_API_KEY:
    app.logger.warning("GROQ_API_KEY is not set!")
if not DEEPAI_API_KEY:
    app.logger.warning("DEEPAI_API_KEY is not set!")

@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    system_prompt = (
        "You are a friendly sleep coach named Silent Veil: calm, reassuring, "
        "help users improve sleep habits and reduce stress. Avoid jargon."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt or "Hi, what can you help me with?"}
    ]
    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.7
    }
    res = requests.post(GROQ_API_URL, headers={
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }, json=payload, timeout=10)
    if res.status_code != 200:
        return jsonify({"error": res.text}), 500
    content = res.json()["choices"][0]["message"]["content"]
    return jsonify({"response": content})

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    res = requests.post(DEEPAI_URL, headers={
        "Api-Key": DEEPAI_API_KEY
    }, data={"text": prompt}, timeout=20)

    if res.status_code != 200:
        return jsonify({"error": res.text}), 500

    output = res.json().get("output_url")
    if not output:
        return jsonify({"error": "No image returned"}), 500

    return jsonify({"imageUrl": output})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

