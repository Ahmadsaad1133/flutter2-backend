import json
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

def generate_bedtime_story(mood, sleep_quality):
    prompt = (
        f"Based on the user's mood: '{mood}' and how they slept tonight: '{sleep_quality}', "
        "create a calm, soothing bedtime story to help them relax."
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
        return None, res.text

    story = res.json().get("choices", [])[0].get("message", {}).get("content")
    return story, None

def generate_image_from_prompt(prompt):
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
            "Accept": "application/json"
        },
        files={
            'init_image': (None, ''),
            'options': (None, json.dumps(payload), 'application/json')
        },
        timeout=30
    )

    if res.status_code != 200:
        return None, res.text

    result = res.json()
    artifacts = result.get("artifacts", [])
    if not artifacts:
        return None, "No artifacts returned"

    raw_b64 = artifacts[0].get("base64") or artifacts[0].get("b64_encoded_image")
    if not raw_b64:
        return None, "No base64 image in artifacts"
    clean_b64 = "".join(raw_b64.split())
    data_uri = f"data:image/png;base64,{clean_b64}"
    return data_uri, None

def create_image_prompt_from_story(story_text):
    # You can make this smarter with NLP or prompt engineering,
    # but for now, a simple heuristic:
    # Extract key elements or just say "illustration of: {story snippet}"
    snippet = story_text[:150].replace('\n', ' ')  # first 150 chars
    return f"Calm, soothing bedtime scene, inspired by: {snippet}"

@app.route("/generate-story-and-image", methods=["POST"])
def generate_story_and_image():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()

    if not mood or not sleep_quality:
        return jsonify(error="Missing mood or sleep_quality"), 400

    story, err = generate_bedtime_story(mood, sleep_quality)
    if err:
        return jsonify(error=err), 500

    image_prompt = create_image_prompt_from_story(story)
    image_data_url, err = generate_image_from_prompt(image_prompt)
    if err:
        return jsonify(error=err), 500

    return jsonify(
        story=story,
        imageUrl=image_data_url,
        imagePrompt=image_prompt
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

