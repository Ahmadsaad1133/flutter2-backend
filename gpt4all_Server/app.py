import json
import os
import logging
import random
import time
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# Load API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
pixabay_api_key = os.getenv("PIXABAY_API_KEY")

# API endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PIXABAY_SEARCH_URL = "https://pixabay.com/api/"

# Track request count for overload protection
app.request_count = 0
app.last_reset = time.time()

@app.before_request
def check_overload():
    """Protect against request overload"""
    current_time = time.time()
    
    # Reset counter every minute
    if current_time - app.last_reset > 60:
        app.request_count = 0
        app.last_reset = current_time
    
    app.request_count += 1
    
    # Return 503 if overloaded
    if app.request_count > 100:
        logger.warning("Server overload detected. Request count: %d", app.request_count)
        return jsonify(
            error="Server is currently busy. Please try again later."
        ), 503

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
        if not groq_api_key:
            logger.error("GROQ_API_KEY environment variable is missing")
            return None, "AI service is not configured"
            
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
        
        # Handle Groq API errors
        if res.status_code == 429:
            logger.error("Groq API rate limit exceeded")
            return None, "AI service is busy. Please try again later."
        elif res.status_code >= 500:
            logger.error(f"Groq API server error: {res.status_code}")
            return None, "AI service is temporarily unavailable"
        elif res.status_code != 200:
            logger.error(f"Groq API error: {res.status_code} - {res.text}")
            return None, f"AI service error: {res.status_code}"
        
        data = res.json()
        content = data.get("choices", [])[0].get("message", {}).get("content")
        if not content:
            return None, "Empty response from AI service"
        return content.strip(), None
    except requests.exceptions.Timeout:
        logger.error("Groq API timeout")
        return None, "AI service timeout. Please try again."
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        return None, "AI service error"

def clean_json_output(json_text: str) -> dict:
    """Handle different JSON response formats."""
    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            content = parsed.get("content", "")
            if isinstance(content, dict):
                parsed["content"] = json.dumps(content, indent=2)
        return parsed
    except Exception as e:
        logger.warning(f"JSON cleaning failed: {e}")
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
            logger.info(f"No Pixabay results for query: {clean_query}")
            return None
        return random.choice(hits).get("webformatURL")
    except Exception as e:
        logger.error(f"Pixabay search failed: {e}")
        return None

@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    """Handle general chat requests."""
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "").strip()
        
        if not prompt:
            return jsonify(error="Missing 'prompt'"), 400
        
        response, error = call_groq(prompt)
        if error:
            return jsonify(error=error), 500
            
        return jsonify(response=response)
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify(error="Internal server error"), 500

@app.route("/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate_story():
    """Generate plain text story without images."""
    try:
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
    except Exception as e:
        logger.error(f"Story generation error: {e}")
        return jsonify(error="Internal server error"), 500

@app.route("/generate-stories", methods=["POST"])
@limiter.limit("3 per minute")
def generate_stories():
    """Generate multiple stories with images."""
    try:
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
            if err:
                logger.error(f"Story generation failed: {err}")
                continue

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
            return jsonify(error="Failed to generate any stories"), 500

        return jsonify(stories=stories)
    except Exception as e:
        logger.error(f"Multi-story generation error: {e}")
        return jsonify(error="Internal server error"), 500

@app.route("/generate-story-and-image", methods=["POST"])
@limiter.limit("5 per minute")
def generate_story_and_image():
    """Generate single story with image."""
    try:
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
        if err:
            return jsonify(error=err), 500

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
    except Exception as e:
        logger.error(f"Story+image generation error: {e}")
        return jsonify(error="Internal server error"), 500

@app.route("/sleep-analysis", methods=["POST"])
@limiter.limit("3 per minute")  # Strict limit due to complexity
def sleep_analysis():
    """Analyze sleep logs with clinical precision."""
    try:
        data = request.get_json() or {}
        sleep_data = data.get("sleep_data", {})
        
        if not sleep_data:
            return jsonify(error="Missing 'sleep_data'"), 400
        
        # Build medical-grade analysis prompt
        prompt = (
            "You are Dr. Somnus, a board-certified sleep medicine specialist with 20 years of experience. "
            "Conduct a clinical analysis of this patient's sleep data using ICSD-3 diagnostic criteria and AASM guidelines.\n\n"
            
            "**ANALYSIS PROTOCOL**:\n"
            "1. Calculate sleep efficiency: (TST / TIB) × 100\n"
            "2. Assess sleep continuity: WASO, SOL, sleep fragmentation index\n"
            "3. Evaluate circadian rhythm consistency\n"
            "4. Analyze lifestyle factors against clinical thresholds\n"
            "5. Formulate diagnosis based on quantitative metrics\n\n"
            
            "**REQUIRED SECTIONS**:\n"
            "### Quantitative Analysis\n"
            "### Diagnostic Impression (ICSD-3 codes)\n"
            "### Severity Assessment (mild/moderate/severe)\n"
            "### Evidence-Based Treatment Plan\n"
            "### Prognosis\n"
            "### Referral Recommendations\n\n"
            
            "**DATA RULES**:\n"
            "- Use ONLY provided data\n"
            "- Include calculations for all metrics\n"
            "- Reference clinical thresholds (AASM)\n"
            "- Never speculate beyond data\n"
            "- Quantify all observations\n\n"
            
            "**KEY MEDICAL TERMS**:\n"
            "- TST: Total Sleep Time\n"
            "- TIB: Time In Bed\n"
            "- SE: Sleep Efficiency\n"
            "- SOL: Sleep Onset Latency\n"
            "- WASO: Wake After Sleep Onset\n"
            "- AHI: Apnea-Hypopnea Index\n\n"
            
            "Patient's Sleep Data:\n"
            f"{json.dumps(sleep_data, indent=2)}\n\n"
            
            "**Begin clinical analysis**:"
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
    status = {
        "status": "ok",
        "groq_ready": bool(groq_api_key),
        "pixabay_ready": bool(pixabay_api_key),
        "request_count": app.request_count,
        "last_reset": time.ctime(app.last_reset)
    }
    return jsonify(status)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(
        error="Too many requests. Please wait a minute and try again."
    ), 429

@app.errorhandler(503)
def service_unavailable(e):
    return jsonify(
        error="Service temporarily overloaded. Please try again later."
    ), 503

if __name__ == "__main__":
    # Production configuration
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    # Don't run in debug mode for production
    if os.getenv("FLASK_ENV") == "production":
        app.config["DEBUG"] = False
        
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
