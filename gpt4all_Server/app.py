import json
import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import logging
import random

from mood_utils import MoodAnalyzer  # Your utility class

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_api_key = os.getenv("GROQ_API_KEY")
pixabay_api_key = os.getenv("PIXABAY_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PIXABAY_SEARCH_URL = "https://pixabay.com/api/"

# Therapeutic instructions by category
THERAPY_INSTRUCTIONS = {
    "anger": {
        "english": "The user feels angry or frustrated—guide them through gentle breathing and calming imagery.",
        "arabic": "المستخدم يشعر بالغضب أو الإحباط – قدّم له تمارين تنفس هادئة وصوراً مهدئة."
    },
    "sadness": {
        "english": "The user feels sad—infuse empathy, healing metaphors, and hope into the narrative.",
        "arabic": "المستخدم يشعر بالحزن – أضف تعاطفاً واستعارات شفائية وأملاً في القصة."
    },
    "stress": {
        "english": "The user is stressed or anxious—include relaxation techniques like progressive muscle relaxation.",
        "arabic": "المستخدم متوتر أو قلق – اشمل تقنيات الاسترخاء مثل الاسترخاء التدريجي للعضلات."
    },
    "lonely": {
        "english": "The user feels lonely—create warm companionship characters and reassuring dialogue.",
        "arabic": "المستخدم يشعر بالوحدة – ابتكر شخصيات رفيقة ودية وحواراً مطمئناً."
    },
    "sexual": {
        "english": "The user experiences sexual frustration—focus on self-care, gentle body awareness, and comfort.",
        "arabic": "المستخدم يعاني من إحباط جنسي – ركز على العناية الذاتية والوعي الجسدي اللطيف والراحة."
    },
    "general": {
        "english": "The user seeks a calm, restorative bedtime story.",
        "arabic": "المستخدم يبحث عن قصة هادئة ومُنعِشة قبل النوم."
    }
}

# System personas to vary tone
SYSTEM_PERSONAS = {
    "english": [
        "You are Silent Veil, a calm sleep coach assistant.",
        "You are Dr. Somnus, an empathetic AI doctor specializing in sleep therapy.",
        "You are Nightingale, a soothing storyteller who weaves healing into every word."
    ],
    "arabic": [
        "أنت الصمت الستار، مساعد مهدئ في تحسين النوم.",
        "أنت الدكتور سومنوس، طبيب ذكاء اصطناعي متعاطف متخصص في علاج الأرق.",
        "أنت طائر الليل، راوي قصص هادئة ينسج الشفاء في كل كلمة."
    ]
}


def extract_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "en", "value"):
            if key in value and isinstance(value[key], str):
                return value[key]
        return json.dumps(value)
    return str(value) if value is not None else ""


def _call_groq(messages: list) -> (str, str):
    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 700
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


def clean_json_output(json_text: str) -> dict:
    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            c = parsed.get("content", "")
            if isinstance(c, dict):
                parsed["content"] = json.dumps(c, indent=2)
        return parsed
    except Exception:
        return {"title": "Oneiric Dream", "description": "A calm bedtime story.", "content": json_text}


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
        return random.choice(hits)["webformatURL"]
    except Exception as e:
        logger.error(f"Pixabay search failed: {e}")
        return None


def build_prompt(i: int, mood: str, sleep_quality: str, category: str, language: str) -> list:
    persona = random.choice(SYSTEM_PERSONAS[language])
    therapy = THERAPY_INSTRUCTIONS[category][language]
    system_content = (
        f"System: {persona}\n"
        f"Instruction: {therapy}\n\n"
        f"User Mood: '{mood}', Sleep Quality: '{sleep_quality}'.\n"
        f"Task: Create bedtime story #{i+1} with a unique title, setting, characters, "
        "and emotional arc tailored to the user's state. "
        "Output strictly in JSON (title, description, content) as flat strings."
    )
    if language == "arabic":
        system_content += "\nPlease respond in Arabic only."
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": ""}
    ]


@app.route("/generate-stories", methods=["POST"])
def generate_stories():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    count = int(data.get("count", 5))
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    # Determine category & language
    category, language = MoodAnalyzer.categorize(mood)

    stories = []
    seen_titles = set()

    for i in range(count):
        messages = build_prompt(i, mood, sleep_quality, category, language)
        raw_json, err = _call_groq(messages)
        story_data = clean_json_output(raw_json or "")

        raw_title = extract_text(story_data.get("title", f"Oneiric Journey #{i+1}")).strip()
        unique_title = raw_title
        suffix = 2
        while unique_title in seen_titles:
            unique_title = f"{raw_title} ({suffix})"
            suffix += 1
        seen_titles.add(unique_title)

        stories.append({
            "title": unique_title,
            "description": extract_text(story_data.get("description", "")).strip(),
            "content":     extract_text(story_data.get("content", "")).strip(),
            "imageUrl":    search_cartoon_image(unique_title or mood) or "",
            "durationMinutes": random.choice([4, 5, 6])
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

    category, language = MoodAnalyzer.categorize(mood)
    messages = build_prompt(1, mood, sleep_quality, category, language)
    raw_json, err = _call_groq(messages)
    story_data = clean_json_output(raw_json or "")

    return jsonify({
        "title": extract_text(story_data.get("title", "Oneiric Dream")).strip(),
        "description": extract_text(story_data.get("description", "")).strip(),
        "content": extract_text(story_data.get("content", "")).strip(),
        "imageUrl": search_cartoon_image(mood) or "",
        "durationMinutes": random.choice([4, 5, 6])
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


