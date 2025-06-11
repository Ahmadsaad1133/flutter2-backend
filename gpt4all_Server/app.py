# backend/app.py

import json
import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from environment
groq_api_key = os.getenv("GROQ_API_KEY")
stability_api_key = os.getenv("STABILITY_API_KEY")

# API endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
STABILITY_API_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"


def generate_bedtime_story(mood: str, sleep_quality: str) -> (str, str):
    prompt = (
        f"You are Silent Veil, a calm sleep coach.\n"
        f"Create a short (3-5 sentence) bedtime story based on:\n"
        f"- Current mood: {mood}\n"
        f"- Last night's sleep quality: {sleep_quality}\n"
        f"Make it calming and suitable for sleep."
    )
    return _call_groq(prompt)


def generate_sleep_analysis(mood: str, sleep_quality: str) -> (str, str):
    prompt = (
        f"You are Silent Veil, an expert sleep coach.\n"
        f"Analyze the user's sleep data and mood:\n"
        f"- Mood: {mood}\n"
        f"- Sleep quality: {sleep_quality}\n"
        f"Provide a concise analysis and personalized tips to improve sleep."
    )
    return _call_groq(prompt)


def _call_groq(user_prompt: str) -> (str, str):
    try:
        messages = [
            {"role": "system", "content": "You are a sleep coach assistant."},
            {"role": "user", "content": user_prompt}
        ]
        payload = {"model": "llama3-70b-8192", "messages": messages, "temperature": 0.7, "max_tokens": 300}
        res = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {groq_api_key}","Content-Type": "application/json"}, json=payload, timeout=20)
        if res.status_code != 200:
            logger.error(f"Groq API error: {res.status_code} - {res.text}")
            return None, f"Groq API error: {res.status_code}"
        data = res.json()
        content = data.get("choices", [])[0].get("message", {}).get("content")
        if not content:
            return None, "Empty response from Groq"
        return content.strip(), None
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        return None, str(e)


def generate_image_from_story(story: str, tone: str = None, theme: str = None) -> (str, str):
    sentences = [s.strip() for s in story.split('.') if s.strip()]
    short_prompt = " ".join(sentences[:3]) + ". Calm, peaceful, bedtime story style."
    if theme:
        short_prompt += f" Theme: {theme}."
    if tone:
        short_prompt += f" Tone: {tone}."

    form_data = {
        "prompt": (None, short_prompt[:1000]),
        "model": (None, "core"),
        "output_format": (None, "png"),
        "cfg_scale": (None, "6"),
        "samples": (None, "1"),
        "width": (None, "512"),
        "height": (None, "512"),
        "steps": (None, "30"),
        "style_preset": (None, "fantasy-art")
    }
    try:
        res = requests.post(STABILITY_API_URL, headers={"Authorization": f"Bearer {stability_api_key}","Accept": "application/json"}, files=form_data, timeout=45)
        if res.status_code != 200:
            err = res.json().get('message', res.text)
            logger.error(f"Stability API error: {err}")
            return None, err
        artifacts = res.json().get("artifacts", [])
        raw_b64 = (artifacts[0].get("base64") or artifacts[0].get("image")) if artifacts else None
        if not raw_b64:
            return None, "No image data"
        data_uri = f"data:image/png;base64,{''.join(raw_b64.split())}"
        return data_uri, None
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return None, str(e)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify(error="Missing 'prompt'"), 400
    response, err = _call_groq(prompt)
    if err:
        return jsonify(error=err), 500
    return jsonify(response=response)


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
    tone = data.get("tone")
    theme = data.get("theme")
    # allow optional parameters but require mood and sleep_quality
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400
    story, err = generate_bedtime_story(mood, sleep_quality)
    if err:
        return jsonify(error=err), 500
    image_url, img_err = generate_image_from_story(story, tone=tone, theme=theme)
    if img_err:
        return jsonify(story=story, error=img_err), 207
    return jsonify(story=story, imageUrl=image_url)


@app.route("/analyze-sleep", methods=["POST"])
def analyze_sleep():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400
    analysis, err = generate_sleep_analysis(mood, sleep_quality)
    if err:
        return jsonify(error=err), 500
    return jsonify(analysis=analysis)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
