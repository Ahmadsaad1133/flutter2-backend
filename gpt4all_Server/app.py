import json
import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

# API endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
STABILITY_API_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"

@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify(error="Missing prompt"), 400

    messages = [
        {"role": "system", "content": "You are Silent Veil, a calm sleep coach."},
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.7
    }

    res = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=15
    )

    if res.status_code != 200:
        return jsonify(error=res.text), 500

    content = res.json().get("choices", [])[0].get("message", {}).get("content")
    return jsonify(response=content)

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify(error="Missing prompt"), 400

    payload = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "samples": 1,
        "width": 512,
        "height": 512,
        "steps": 50
    }

    res = requests.post(
        STABILITY_API_URL,
        headers={
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "application/json"  # <-- Important fix here
        },
        files={
            'init_image': (None, ''),  # required field placeholder
            'options': (None, json.dumps(payload), 'application/json')
        },
        timeout=30
    )

    print("[Stability] status:", res.status_code)
    print("[Stability] body snippet:", res.text[:200])

    if res.status_code != 200:
        return jsonify(error=res.text), 500

    result = res.json()
    artifacts = result.get("artifacts", [])
    if not artifacts:
        return jsonify(error="No artifacts returned"), 500

    raw_b64 = artifacts[0].get("base64") or artifacts[0].get("b64_encoded_image")
    if not raw_b64:
        return jsonify(error="No base64 image in artifacts"), 500
    clean_b64 = "".join(raw_b64.split())
    data_uri = f"data:image/png;base64,{clean_b64}"

    print("[Stability] returning data URI snippet:", data_uri[:50], "...")
    return jsonify(imageUrl=data_uri)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


