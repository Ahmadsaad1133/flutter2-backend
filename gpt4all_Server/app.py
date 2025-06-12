import json
import os
import logging
import random
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
pixabay_api_key = os.getenv("PIXABAY_API_KEY")

# API endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PIXABAY_SEARCH_URL = "https://pixabay.com/api/"

def extract_text(value):
    """Extract text from various response formats."""
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        for key in ['text', 'content', 'en', 'value']:
            if key in value and isinstance(value[key], str):
                return value[key]
        return json.dumps(value)
    else:
        return str(value) if value is not None else ''

def call_groq(user_prompt: str) -> (str, str):
    """Call Groq API with the given prompt."""
    try:
        messages = [
            {"role": "system", "content": "You are Silent Veil, a calm sleep coach assistant."},
            {"role": "user", "content": user_prompt}
        ]
        payload = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.95,
            "max_tokens": 1200
        }
        res = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
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

def clean_json_output(json_text: str) -> dict:
    """Handle different JSON response formats."""
    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            content = parsed.get("content", "")
            if isinstance(content, dict):
                parsed["content"] = json.dumps(content, indent=2)
        return parsed
    except Exception:
        return {
            "title": "Oneiric Dream",
            "description": "A calm bedtime story.",
            "content": json_text
        }

def search_cartoon_image(query: str) -> str | None:
    """Search for cartoon images on Pixabay."""
    if not pixabay_api_key:
        logger.error("Missing PIXABAY_API_KEY environment variable.")
        return None
    
    # Clean up query for Pixabay
    clean_query = query.replace(":", "").replace("'", "")[:50]
    
    params = {
        "key": pixabay_api_key,
        "q": clean_query,
        "image_type": "illustration",
        "per_page": 10,
        "safesearch": "true"
    }
    try:
        resp = requests.get(PIXABAY_SEARCH_URL, params=params, timeout=15)
        if resp.status_code != 200:
            logger.error(f"Pixabay API error {resp.status_code}: {resp.text}")
            return None
        hits = resp.json().get("hits", [])
        if not hits:
            logger.warning(f"No Pixabay results for query: {clean_query}")
            return None
        return random.choice(hits).get("webformatURL")
    except Exception as e:
        logger.error(f"Pixabay search failed: {e}")
        return None

@app.route("/chat", methods=["POST"])
def chat():
    """Handle general chat requests."""
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    
    if not prompt:
        return jsonify(error="Missing 'prompt'"), 400
    
    response, error = call_groq(prompt)
    if error:
        return jsonify(error=error), 500
        
    return jsonify(response=response)

@app.route("/generate", methods=["POST"])
def generate_story():
    """Generate plain text story without images."""
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    prompt = (
        f"You are Silent Veil, a calm sleep coach. Based on mood '{mood}' "
        f"and sleep quality '{sleep_quality}', create a calming bedtime story. "
        "Return only the story text, no JSON or formatting."
    )
    
    story, error = call_groq(prompt)
    if error:
        return jsonify(error=error), 500
        
    return jsonify(story=story)

@app.route("/generate-stories", methods=["POST"])
def generate_stories():
    """Generate multiple stories with images."""
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    count = int(data.get("count", 5))

    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    stories = []
    seen_titles = set()

    for i in range(count):
        prompt = (
            f"You are Silent Veil. Based on mood '{mood}' and sleep quality '{sleep_quality}', "
            f"create unique bedtime story #{i+1}. "
            "Respond in JSON with: title, description, content. "
            "All values must be plain strings. No markdown or nested data."
        )

        story_json_str, err = call_groq(prompt)
        story_data = clean_json_output(story_json_str or "")

        raw_title = extract_text(story_data.get("title", f"Oneiric Journey #{i+1}")).strip()
        unique_title = raw_title
        suffix = 2
        while unique_title in seen_titles:
            unique_title = f"{raw_title} ({suffix})"
            suffix += 1
        seen_titles.add(unique_title)

        description = extract_text(story_data.get("description", "")).strip()
        content = extract_text(story_data.get("content", "")).strip()
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
    """Generate single story with image."""
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    prompt = (
        f"You are Silent Veil. Based on mood '{mood}' and sleep quality '{sleep_quality}', "
        "create a calming bedtime story. Respond in JSON with: title, description, content. "
        "All values must be plain strings. No markdown or nested data."
    )
    
    story_json_str, err = call_groq(prompt)
    story_data = clean_json_output(story_json_str or "")

    title = extract_text(story_data.get("title", "Oneiric Dream")).strip()
    description = extract_text(story_data.get("description", "")).strip()
    content = extract_text(story_data.get("content", "")).strip()
    image_url = search_cartoon_image(title or mood) or ""
    duration = random.choice([4, 5, 6])

    return jsonify({
        "title": title,
        "description": description,
        "content": content,
        "imageUrl": image_url,
        "durationMinutes": duration
    })

@app.route("/sleep-analysis", methods=["POST"])
def sleep_analysis():
    """Analyze sleep logs and provide recommendations."""
    data = request.get_json() or {}
    sleep_data = data.get("sleep_data", {})
    
    if not sleep_data:
        return jsonify(error="Missing 'sleep_data'"), 400
    
    try:
        # Build comprehensive prompt
        prompt = (
            "You are a professional sleep coach AI, speaking with empathy and warmth. "
            "Analyze this sleep data and provide detailed recommendations using these exact headers:\n"
            "### Summary\n"
            "### Weekly Goals\n"
            "### Bedtime Routine Suggestions\n"
            "### Relaxation Techniques\n"
            "### Environmental Adjustments\n"
            "### Warnings\n"
            "### Analyze for Disorders\n\n"
            f"Sleep Data:\n{json.dumps(sleep_data, indent=2)}\n\n"
            "Provide conversational but professional analysis with bullet points under each header."
        )
        
        # Get analysis from Groq
        analysis, error = call_groq(prompt)
        if error:
            return jsonify(error=error), 500
            
        return jsonify(analysis=analysis)
        
    except Exception as e:
        logger.error(f"Sleep analysis failed: {e}")
        return jsonify(error="Internal server error"), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify(status="ok", groq_ready=bool(groq_api_key))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("DEBUG", "false").lower() == "true")
