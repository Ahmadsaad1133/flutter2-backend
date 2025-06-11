import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

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
    data = request.get_json(silent=True)
    # Allow prompt via JSON or form-data
    prompt = None
    if data and data.get("prompt"):
        prompt = data.get("prompt").strip()
    else:
        prompt = request.form.get("prompt", "").strip()

    if not prompt:
        return jsonify(error="Missing prompt"), 400

    # Send multipart/form-data with required stability parameters
    files = {
        'prompt': (None, prompt),
        'cfg_scale': (None, '7'),
        'samples': (None, '1'),
        'width': (None, '512'),
        'height': (None, '512'),
        'steps': (None, '50'),
        'output_format': (None, 'png')
    }
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "application/json"
        # Do not set Content-Type manually; requests sets multipart boundary
    }

    res = requests.post(
        STABILITY_API_URL,
        headers=headers,
        files=files,
        timeout=30
    )

    if res.status_code != 200:
        return jsonify(error=res.text), 500

    result = res.json()
    artifacts = result.get("artifacts", [])
    if not artifacts or not artifacts[0].get("url"):
        return jsonify(error="No image returned"), 500

    return jsonify(imageUrl=artifacts[0]["url"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))


