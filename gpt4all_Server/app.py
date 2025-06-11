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


def generate_bedtime_story(mood: str, sleep_quality: str):
    """Generate a bedtime story based on mood and sleep quality using Groq API."""
    prompt = (
        f"You are Silent Veil, a calm sleep coach.\n"
        f"Based on the user's mood: '{mood}' and how they slept last night: '{sleep_quality}', "
        f"compose a gentle bedtime story to help them relax."
    )
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
        return None, f"Groq API error: {res.text}"

    content = res.json().get("choices", [])[0].get("message", {}).get("content")
    return content, None


def generate_image_from_story(story: str):
    """Generate an image from the bedtime story text using Stability API."""
    prompt = story.strip()
    if not prompt:
        return None, "Empty prompt for image generation."

    options_payload = {
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
            "Accept": "application/json"
        },
        files={
            "init_image": (None, ""),
            "options": (None, json.dumps(options_payload), "application/json")
        },
        timeout=30
    )

    if res.status_code != 200:
        return None, f"Stability API error: {res.text}"

    result = res.json()
    artifacts = result.get("artifacts", [])
    if not artifacts:
        return None, "No artifacts returned from image generation."

    raw_b64 = artifacts[0].get("base64") or artifacts[0].get("b64_encoded_image")
    if not raw_b64:
        return None, "No base64 image found in artifacts."

    clean_b64 = "".join(raw_b64.split())
    data_uri = f"data:image/png;base64,{clean_b64}"
    return data_uri, None


@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()

    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    story, err = generate_bedtime_story(mood, sleep_quality)
    if err:
        return jsonify(error=err), 500

    return jsonify(story=story)


@app.route("/generate-story-and-image", methods=["POST"])
def generate_story_and_image():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()

    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    # Generate story
    story, err = generate_bedtime_story(mood, sleep_quality)
    if err:
        return jsonify(error=err), 500

    # Generate image based on story
    image_url, err = generate_image_from_story(story)
    if err:
        return jsonify(error=err), 500

    return jsonify(story=story, imageUrl=image_url)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



