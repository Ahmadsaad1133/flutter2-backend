import json
import os
import logging
import random
import traceback
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
            error_msg = f"Groq API error: {res.status_code} - {res.text[:200]}"
            logger.error(error_msg)
            return None, error_msg
        
        data = res.json()
        content = data.get("choices", [])[0].get("message", {}).get("content")
        if not content:
            return None, "Empty response from Groq"
        return content.strip(), None
    except Exception as e:
        error_msg = f"Groq call failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg

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
            logger.error(f"Pixabay API error {resp.status_code}: {resp.text[:200]}")
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
        logger.warning("Chat request with empty prompt")
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
        logger.warning("Generate story request missing parameters")
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
        logger.warning("Generate stories request missing parameters")
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

@app.route("/generate-story-and-image", methods=["POST"])
def generate_story_and_image():
    """Generate single story with image."""
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    
    if not mood or not sleep_quality:
        logger.warning("Generate story+image request missing parameters")
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    prompt = (
        f"You are Silent Veil. Based on mood '{mood}' and sleep quality '{sleep_quality}', "
        "create a calming bedtime story. Respond in JSON with: title, description, content. "
        "All values must be plain strings. No markdown or nested data."
    )
    
    story_json_str, err = call_groq(prompt)
    if err:
        logger.error(f"Story+image generation failed: {err}")
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

@app.route("/sleep-analysis", methods=["POST"])
def sleep_analysis():
    """Analyze sleep logs with clinical precision."""
    data = request.get_json() or {}
    sleep_data = data.get("sleep_data", {})
    
    logger.info(f"Received sleep analysis request: {sleep_data}")
    
    if not sleep_data:
        logger.warning("Sleep analysis called with empty sleep_data")
        return jsonify(error="Missing 'sleep_data'"), 400
    
    try:
        # Enhanced input validation
        if not isinstance(sleep_data, dict):
            logger.error(f"Invalid sleep_data type: {type(sleep_data)}")
            return jsonify(error="'sleep_data' must be a JSON object"), 400
        
        # Detect input type with more robust checking
        quantitative_keys = ['TST', 'TIB', 'SE', 'SOL', 'WASO', 'AHI', 'sleep_efficiency']
        is_quantitative = any(key in sleep_data for key in quantitative_keys)
        
        # Build dynamic prompt based on input type
        if is_quantitative:
            logger.info("Processing quantitative sleep data")
            prompt = (
                "You are Dr. Somnus, a board-certified sleep specialist. Analyze this quantitative sleep data:\n\n"
                "1. Calculate sleep efficiency: (TST / TIB) × 100 (if not provided)\n"
                "2. Assess sleep continuity metrics\n"
                "3. Compare against AASM clinical thresholds\n"
                "4. Identify potential sleep disorders\n"
                "5. Provide evidence-based recommendations\n\n"
                "Required sections:\n"
                "### Quantitative Analysis\n"
                "### Clinical Assessment\n"
                "### Treatment Recommendations\n\n"
                f"Data: {json.dumps(sleep_data)}"
            )
        else:
            # Handle symptom-based input
            logger.info("Processing symptom-based sleep data")
            symptoms = sleep_data.get("symptoms", [])
            if not symptoms:
                # Extract all string values as symptoms
                symptoms = [v for k, v in sleep_data.items() if isinstance(v, str)]
            
            if not symptoms:
                logger.error("No symptoms found in sleep_data")
                return jsonify(error="No symptoms provided for analysis"), 400
            
            logger.info(f"Analyzing symptoms: {', '.join(symptoms)}")
            prompt = (
                "You are Dr. Somnus, a board-certified sleep specialist. "
                "Analyze these patient-reported symptoms:\n\n"
                "1. Identify potential sleep disorders (use ICSD-3 terminology)\n"
                "2. Relate symptoms to possible physiological causes\n"
                "3. Provide clinical recommendations\n\n"
                "Required sections:\n"
                "### Symptom Analysis\n"
                "### Clinical Assessment\n"
                "### Personalized Recommendations\n\n"
                f"Symptoms: {', '.join(symptoms)}"
            )
        
        # Get analysis from Groq
        logger.debug(f"Sending prompt to Groq API: {prompt[:200]}...")
        analysis, error = call_groq(prompt)
        if error:
            logger.error(f"Groq API failure: {error}")
            return jsonify(
                error="Analysis service error",
                details=error,
                code="SleepAnalysisException"
            ), 500
            
        if not analysis:
            logger.error("Empty analysis response from Groq")
            return jsonify(
                error="Analysis service returned empty response",
                code="SleepAnalysisException"
            ), 500
        
        logger.info("Successfully generated sleep analysis")
        return jsonify(analysis=analysis)
        
    except Exception as e:
        logger.error(f"Sleep analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify(
            error="Sleep analysis failed",
            details=str(e),
            code="SleepAnalysisException"
        ), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Enhanced health check endpoint."""
    status = {
        "status": "ok",
        "services": {
            "groq": "available" if groq_api_key else "missing_api_key",
            "pixabay": "available" if pixabay_api_key else "missing_api_key"
        },
        "endpoints": [
            "/chat (POST)",
            "/generate (POST)",
            "/generate-stories (POST)",
            "/generate-story-and-image (POST)",
            "/sleep-analysis (POST)"
        ]
    }
    return jsonify(status)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    logger.info(f"Starting server on port {port} in {'debug' if debug_mode else 'production'} mode")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
