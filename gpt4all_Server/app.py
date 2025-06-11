import json
import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import logging
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_api_key = os.getenv("GROQ_API_KEY")
pixabay_api_key = os.getenv("PIXABAY_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PIXABAY_SEARCH_URL = "https://pixabay.com/api/"

def extract_text(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        for key in ['text', 'content', 'en', 'value']:
            if key in value and isinstance(value[key], str):
                return value[key]
        return json.dumps(value)
    else:
        return str(value) if value is not None else ''

def _call_groq(user_prompt: str) -> (str, str):
    try:
        messages = [
            {"role": "system", "content": "You are a sleep coach assistant."},
            {"role": "user", "content": user_prompt}
        ]
        payload = {"model": "llama3-70b-8192", "messages": messages, "temperature": 0.7, "max_tokens": 300}
        res = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=20
        )
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

def search_cartoon_image(query: str) -> str | None:
    if not pixabay_api_key:
        logger.error("Missing PIXABAY_API_KEY environment variable.")
        return None
    params = {
        "key": pixabay_api_key,
        "q": query,
        "image_type": "illustration",
        "per_page": 10,
        "safesearch": "true"
    }
    try:
        resp = requests.get(PIXABAY_SEARCH_URL, params=params, timeout=10)
        if resp.status_code != 200:
            logger.error(f"Pixabay API error {resp.status_code}: {resp.text}")
            return None
        hits = resp.json().get("hits", [])
        if not hits:
            return None
        return hits[0].get("webformatURL")
    except Exception as e:
        logger.error(f"Pixabay search failed: {e}")
        return None

@app.route("/generate-stories", methods=["POST"])
def generate_stories():
    """
    Generate multiple bedtime stories, each with title, description, imageUrl, durationMinutes, content.
    """
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    count = int(data.get("count", 3))  # number of stories to generate, default 3

    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    stories = []

    for i in range(count):
        prompt = (
            f"You are Silent Veil, a calm sleep coach.\n"
            f"Create a short bedtime story with:\n"
            f"- A calming title\n"
            f"- A 1-2 sentence description\n"
            f"- A 3-5 sentence story content\n"
            f"Based on mood: {mood} and sleep quality: {sleep_quality}.\n"
            f"Format your response as JSON with fields: title, description, content."
        )
        story_json_str, err = _call_groq(prompt)
        if err:
            logger.error(f"Error generating story {i+1}: {err}")
            continue
        
        try:
            story_data = json.loads(story_json_str)
        except Exception:
            story_data = {
                "title": f"Dream Story {i+1}",
                "description": f"A calming story based on your mood: {mood}",
                "content": story_json_str
            }

        keywords = extract_text(story_data.get("title")) or extract_text(story_data.get("description")) or mood
        image_url = search_cartoon_image(keywords)
        duration_minutes = random.choice([4, 5, 6])

        story = {
            "title": extract_text(story_data.get("title", f"Dream Story {i+1}")),
            "description": extract_text(story_data.get("description", "")),
            "imageUrl": image_url or "",
            "durationMinutes": duration_minutes,
            "content": extract_text(story_data.get("content", ""))
        }
        stories.append(story)

    if not stories:
        return jsonify(error="Failed to generate any stories"), 500

    return jsonify(stories=stories)


@app.route("/generate-story-and-image", methods=["POST"])
def generate_story_and_image():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    prompt = (
        f"You are Silent Veil, a calm sleep coach.\n"
        f"Create a short bedtime story with:\n"
        f"- A calming title\n"
        f"- A 1-2 sentence description\n"
        f"- A 3-5 sentence story content\n"
        f"Based on mood: {mood} and sleep quality: {sleep_quality}.\n"
        f"Format your response as JSON with fields: title, description, content."
    )
    story_json_str, err = _call_groq(prompt)
    if err:
        return jsonify(error=err), 500

    try:
        story_data = json.loads(story_json_str)
    except Exception:
        story_data = {
            "title": "Dream Story",
            "description": f"A calming story based on your mood: {mood}",
            "content": story_json_str
        }

    keywords = extract_text(story_data.get("title")) or extract_text(story_data.get("description")) or mood
    image_url = search_cartoon_image(keywords)
    duration_minutes = random.choice([4, 5, 6])

    story = {
        "title": extract_text(story_data.get("title", "Dream Story")),
        "description": extract_text(story_data.get("description", "")),
        "imageUrl": image_url or "",
        "durationMinutes": duration_minutes,
        "content": extract_text(story_data.get("content", ""))
    }

    return jsonify(story=story)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


