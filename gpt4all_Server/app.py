import json
import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

# API endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
STABILITY_API_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"

def generate_bedtime_story(mood: str, sleep_quality: str):
    """Generate a bedtime story based on mood and sleep quality using Groq API."""
    try:
        prompt = (
            f"You are Silent Veil, a calm sleep coach.\n"
            f"Create a short (3-5 sentence) bedtime story based on:\n"
            f"- Current mood: {mood}\n"
            f"- Last night's sleep quality: {sleep_quality}\n"
            f"Make it calming and suitable for sleep."
        )
        
        messages = [
            {"role": "system", "content": "You are a sleep coach who creates calming bedtime stories."},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300  # Limit story length
        }

        res = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=20
        )

        if res.status_code != 200:
            logger.error(f"Groq API error: {res.status_code} - {res.text}")
            return None, f"Groq API error: {res.status_code}"

        data = res.json()
        if not data.get("choices"):
            return None, "No story generated"
            
        content = data["choices"][0].get("message", {}).get("content")
        if not content:
            return None, "Empty story content"
            
        return content, None
        
    except Exception as e:
        logger.error(f"Story generation error: {str(e)}")
        return None, f"Story generation failed: {str(e)}"

def generate_image_from_story(story: str):
    """Generate an image from the bedtime story text using Stability API."""
    try:
        # Extract key elements for image prompt
        sentences = [s.strip() for s in story.split('.') if s.strip()]
        short_prompt = " ".join(sentences[:3]) + ". Calm, peaceful, bedtime story style."
        
        if not short_prompt:
            return None, "Empty prompt for image generation."

        # Prepare form data with optimized parameters
        form_data = {
            "prompt": (None, short_prompt[:1000]),  # Truncate if too long
            "model": (None, "core"),
            "output_format": (None, "png"),
            "cfg_scale": (None, "6"),  # Better for creative results
            "samples": (None, "1"),
            "width": (None, "512"),
            "height": (None, "512"),
            "steps": (None, "30"),  # Reduced for faster generation
            "style_preset": (None, "fantasy-art")  # Better style for stories
        }

        res = requests.post(
            STABILITY_API_URL,
            headers={
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Accept": "application/json",
            },
            files=form_data,
            timeout=45
        )

        if res.status_code != 200:
            error_msg = res.text
            try:
                error_data = res.json()
                if 'errors' in error_data:
                    error_msg = ", ".join(error_data['errors'])
                elif 'message' in error_data:
                    error_msg = error_data['message']
            except:
                pass
            logger.error(f"Stability API error: {error_msg}")
            return None, f"Image generation failed: {error_msg}"

        result = res.json()
        artifacts = result.get("artifacts", [])
        
        if not artifacts:
            # Check for alternative response format
            if 'image' in result:
                raw_b64 = result['image']
            else:
                logger.error(f"No artifacts in response: {result}")
                return None, "Image generation produced no results"

        raw_b64 = artifacts[0].get("base64") or artifacts[0].get("image")
        if not raw_b64:
            return None, "No image data found"

        clean_b64 = "".join(raw_b64.split())
        data_uri = f"data:image/png;base64,{clean_b64}"
        return data_uri, None

    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        return None, f"Image generation failed: {str(e)}"

@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        data = request.get_json() or {}
        mood = data.get("mood", "").strip()
        sleep_quality = data.get("sleep_quality", "").strip()

        if not mood or not sleep_quality:
            return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

        story, err = generate_bedtime_story(mood, sleep_quality)
        if err:
            return jsonify(error=err), 500

        return jsonify(story=story)
        
    except Exception as e:
        logger.error(f"Generate text endpoint error: {str(e)}")
        return jsonify(error="Internal server error"), 500

@app.route("/generate-story-and-image", methods=["POST"])
def generate_story_and_image():
    try:
        data = request.get_json() or {}
        mood = data.get("mood", "").strip()
        sleep_quality = data.get("sleep_quality", "").strip()

        if not mood or not sleep_quality:
            return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

        story, err = generate_bedtime_story(mood, sleep_quality)
        if err:
            return jsonify(error=err), 500

        image_url, err = generate_image_from_story(story)
        if err:
            # Return story even if image fails
            return jsonify(story=story, error=err), 207  # Multi-status
            
        return jsonify(story=story, imageUrl=image_url)
        
    except Exception as e:
        logger.error(f"Generate story+image endpoint error: {str(e)}")
        return jsonify(error="Internal server error"), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
