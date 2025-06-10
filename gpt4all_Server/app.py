# app.py (Flask on Render)

import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")
GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
DEEPAI_API_URL = "https://api.deepai.org/api/text2img"

@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.get_json()
    prompt = data.get("prompt", "")
    messages = [
        {"role": "system", "content": "You are Silent Veil, a calm sleep coach."},
        {"role": "user",   "content": prompt}
    ]
    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.7
    }
    res = requests.post(GROQ_API_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=15
    )
    if res.status_code != 200:
        return jsonify(error=res.text), 500
    return jsonify(response=res.json()["choices"][0]["message"]["content"])

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify(error="Missing prompt"), 400

    res = requests.post(
        DEEPAI_API_URL,
        headers={"Api-Key": DEEPAI_API_KEY},
        data={"text": prompt},
        timeout=30
    )
    if res.status_code != 200:
        return jsonify(error=res.text), 500

    url = res.json().get("output_url")
    if not url:
        return jsonify(error="No image returned"), 500

    return jsonify(imageUrl=url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))

