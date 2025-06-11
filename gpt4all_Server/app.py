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
            {"role": "system", "content": (
                "You are Silent Veil, a compassionate and skilled sleep coach who acts like a dream therapist. "
                "You help users fall asleep by creating calming, deeply relaxing bedtime stories tailored to their emotional and physical state. "
                "Every story should feel therapeutic, imaginative, and lull the user into rest."
            )},
            {"role": "user", "content": user_prompt}
        ]
        payload = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.95,
            "max_tokens": 600
        }
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
        logger.debug(f"Groq raw content: {content}")
        if not content:
            return None, "Empty response from Groq"
        return content.strip(), None
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        return None, str(e)

def clean_json_output(json_text: str) -> dict:
    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            content = parsed.get("content", "")
            if isinstance(content, dict):
                parsed["content"] = json.dumps(content, indent=2)
        return parsed
    except Exception:
        return {
            "title": None,
            "description": "",
            "content": json_text
        }

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
        return random.choice(hits).get("webformatURL")
    except Exception as e:
        logger.error(f"Pixabay search failed: {e}")
        return None

@app.route("/generate-stories", methods=["POST"])
def generate_stories():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    count = int(data.get("count", 3))

    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    stories = []
    seen_titles = set()

    prompt_base = (
        f"A user is feeling '{mood}' and reports their sleep quality as '{sleep_quality}'. "
        "You are a dream therapist helping them fall asleep. "
        "Write a magical, calming, 5-minute bedtime story designed to ease them into sleep. "
        "Every story should be totally original, using a unique setting, calming tone, and gentle pacing. "
        "Avoid using repeated titles like 'Oneiric Dream'. "
        "Respond strictly in JSON with the following fields (strings only): title, description, content."
    )

    for i in range(count):
        prompt = f"{prompt_base} This is story #{i+1}. Make it unique and healing."
        story_json_str, err = _call_groq(prompt)
        story_data = clean_json_output(story_json_str or "")

        raw_title = extract_text(story_data.get("title")).strip()
        if not raw_title or raw_title.lower() == "oneiric dream":
            raw_title = f"Dream #{i+1} - {mood.title()}"

        unique_title = raw_title
        suffix = 1
        while unique_title in seen_titles:
            suffix += 1
            unique_title = f"{raw_title} ({suffix})"
        seen_titles.add(unique_title)

        description = extract_text(story_data.get("description", "")).strip()
        content = extract_text(story_data.get("content", "")).strip()
        if not content or "{" in content:  # probably malformed JSON
            content = "No content available. Please try again."

        image_url = search_cartoon_image(unique_title or mood) or ""
        duration = random.choice([4, 5, 6])

        stories.append({
            "title": unique_title,
            "description": description,
            "content": content,
            "imageUrl": image_url,
            "durationMinutes": duration
        })

    if not stories:
        return jsonify(error="Failed to generate stories"), 500

    return jsonify(stories=stories)

@app.route("/generate-story-and-image", methods=["POST"])
def generate_story_and_image():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    prompt = (
        f"A user is experiencing mood '{mood}' and reports sleep quality '{sleep_quality}'. "
        "You are a sleep therapist crafting a personal bedtime story to help them rest. "
        "Write a story that is gentle, comforting, and imaginative. "
        "Respond in JSON with fields: title, description, content (all as strings)."
    )
    story_json_str, err = _call_groq(prompt)
    story_data = clean_json_output(story_json_str or "")

    title = extract_text(story_data.get("title", f"{mood.title()} Dream")).strip()
    if not title or title.lower() == "oneiric dream":
        title = f"{mood.title()} Dream"

    description = extract_text(story_data.get("description", "")).strip()
    content = extract_text(story_data.get("content", "")).strip()
    if not content or "{" in content:
        content = "No content available. Please try again."

    image_url = search_cartoon_image(title or mood) or ""
    duration = random.choice([4, 5, 6])

    return jsonify({
        "title": title,
        "description": description,
        "content": content,
        "imageUrl": image_url,
        "durationMinutes": duration
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


