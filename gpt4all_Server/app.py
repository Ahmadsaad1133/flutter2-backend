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
pixabay_api_key = os.getenv("PIXABAY_API_KEY")

# API endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PIXABAY_SEARCH_URL = "https://pixabay.com/api/"


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


def extract_keywords_from_story(story: str) -> (str, str):
    """
    Use the Groq API to extract 2-3 keywords that best describe
    the story's setting or theme, separated by commas.
    """
    prompt = (
        "Extract 2-3 keywords that best describe the setting or theme of the following story, "
        "separated by commas:\n\n"
        f"{story}"
    )
    return _call_groq(prompt)


def _call_groq(user_prompt: str) -> (str, str):
    try:
        messages = [
            {"role": "system", "content": "You are a sleep coach assistant."},
            {"role": "user", "content": user_prompt}
        ]
        payload = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300
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
        if not content:
            return None, "Empty response from Groq"
        return content.strip(), None
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        return None, str(e)


def search_cartoon_image(query: str) -> str | None:
    """
    Search Pixabay for a cartoon/illustration matching the query.
    Returns the URL of the first hit or None.
    """
    if not pixabay_api_key:
        logger.error("Missing PIXABAY_API_KEY environment variable.")
        return None

    params = {
        "key": pixabay_api_key,
        "q": query,
        "image_type": "illustration",
        "per_page": 1,
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
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    # 1) Generate bedtime story
    story, err = generate_bedtime_story(mood, sleep_quality)
    if err:
        return jsonify(error=err), 500

    # 2) Extract keywords using AI
    keywords, err = extract_keywords_from_story(story)
    if err or not keywords:
        # fallback to old method if keyword extraction fails
        keywords = story.split('.')[0]

    # Clean keywords (remove newlines, extra spaces)
    keywords = keywords.replace('\n', ' ').strip()

    # 3) Search cartoon image via Pixabay using keywords
    image_url = search_cartoon_image(keywords)
    if not image_url:
        # return story only, code 207 indicates partial success
        return jsonify(story=story), 207

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

