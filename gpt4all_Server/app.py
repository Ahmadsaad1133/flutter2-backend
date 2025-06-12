import json
import os
import random
import logging
import re
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_api_key = os.getenv("GROQ_API_KEY")
pixabay_api_key = os.getenv("PIXABAY_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PIXABAY_SEARCH_URL = "https://pixabay.com/api/"

# Mood Analyzer
class MoodAnalyzer:
    MOOD_KEYWORDS = {
        "anger": ["angry", "mad", "furious", "rage", "غاضب", "عصبي", "منفعل"],
        "sadness": ["sad", "depressed", "crying", "broken", "حزين", "مكسور", "كئيب"],
        "stress": ["stressed", "anxious", "nervous", "توتر", "قلق", "مضغوط"],
        "lonely": ["lonely", "alone", "isolated", "وحيد", "عزلة", "منعزل"],
        "sexual": ["aroused", "sensual", "horny", "جنسي", "مثير", "رغبة"]
    }
    GENERAL_CATEGORY = "general"
    ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")

    @classmethod
    def detect_language(cls, text: str) -> str:
        return 'arabic' if cls.ARABIC_CHAR_PATTERN.search(text) else 'english'

    @classmethod
    def categorize(cls, raw_mood: str) -> tuple[str, str]:
        lang = cls.detect_language(raw_mood)
        rm = raw_mood.lower()
        for category, keywords in cls.MOOD_KEYWORDS.items():
            for kw in keywords:
                if kw in rm:
                    return category, lang
        return cls.GENERAL_CATEGORY, lang

# Instructions and Persona
THERAPY_INSTRUCTIONS = {
    "anger": {"english": "I see you're feeling angry...", "arabic": "أرى أنك غاضب..."},
    "sadness": {"english": "I see you're feeling sad...", "arabic": "أشعر أنك حزين..."},
    "stress": {"english": "I see you're stressed...", "arabic": "أشعر أنك متوتر..."},
    "lonely": {"english": "Feeling lonely is hard...", "arabic": "الوحدة صعبة..."},
    "sexual": {"english": "Let's redirect those thoughts...", "arabic": "دعنا نحول هذه المشاعر..."},
    "general": {"english": "You're looking for a calming bedtime story...", "arabic": "تبحث عن قصة هادئة تساعدك على النوم..."},
}

SYSTEM_PERSONAS = {
    "english": ["You are Nightingale, a wise and gentle storyteller for children and adults."],
    "arabic": ["أنت Nightingale، راوي حكايات هادئ وحنون يساعد الناس على النوم."]
}

# Utilities
def extract_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "en", "value"):
            if key in value and isinstance(value[key], str):
                return value[key]
        return json.dumps(value)
    return str(value) if value else ""

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
        return "", f"Groq API error: {res.status_code}"
    data = res.json()
    return data["choices"][0]["message"]["content"].strip(), None

def clean_json_output(raw_text: str, language: str) -> dict:
    if language == "arabic":
        return {"story": raw_text.strip() if raw_text else "قصة قبل النوم"}
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return {
                "title": extract_text(parsed.get("title", "Untitled")),
                "description": extract_text(parsed.get("description", "")),
                "content": extract_text(parsed.get("content", raw_text))
            }
    except Exception:
        pass
    return {
        "title": "Untitled",
        "description": "",
        "content": raw_text.strip()
    }

def search_cartoon_image(query: str) -> str | None:
    if not pixabay_api_key:
        return None
    if MoodAnalyzer.detect_language(query) == 'arabic':
        query = 'calm night'
    params = {"key": pixabay_api_key, "q": query, "image_type": "illustration", "per_page": 10, "safesearch": "true"}
    try:
        resp = requests.get(PIXABAY_SEARCH_URL, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        hits = resp.json().get("hits", [])
        return random.choice(hits).get("webformatURL") if hits else None
    except Exception as e:
        logger.error(f"Pixabay search failed: {e}")
        return None

# Build LLM Prompt
def build_prompt(i: int, mood: str, sleep_quality: str, category: str, language: str) -> list:
    persona = random.choice(SYSTEM_PERSONAS[language])
    therapy = THERAPY_INSTRUCTIONS[category][language]
    if language == 'english':
        prompt = f"{persona}\n{therapy}\nUser Mood: {mood}, Sleep Quality: {sleep_quality}.\nCreate a bedtime story in JSON format with title, description, and content."
    else:
        prompt = f"{persona}\n{therapy}\nمزاج المستخدم: {mood}، جودة النوم: {sleep_quality}.\nاكتب قصة نوم جميلة باللغة العربية بدون JSON."
    return [{"role": "system", "content": prompt}, {"role": "user", "content": ""}]

# Routes

@app.route("/generate-stories", methods=["POST"])
def generate_stories():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    count = int(data.get("count", 5))
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400
    category, language = MoodAnalyzer.categorize(mood)
    stories, seen = [], set()
    for i in range(count):
        msgs = build_prompt(i, mood, sleep_quality, category, language)
        raw, err = _call_groq(msgs)
        parsed = clean_json_output(raw or "", language)
        title = parsed.get("title", f"Dream #{i+1}")
        base, suffix = title, 2
        while title in seen:
            title = f"{base} ({suffix})"; suffix += 1
        seen.add(title)
        stories.append({
            "title": title if language == "english" else "",
            "description": parsed.get("description", "") if language == "english" else "",
            "content": parsed.get("content", "") if language == "english" else "",
            "story": parsed.get("story", "") if language == "arabic" else "",
            "imageUrl": search_cartoon_image(title),
            "durationMinutes": random.choice([4, 5, 6])
        })
    return jsonify(stories=stories)

@app.route("/generate-story-and-image", methods=["POST"])
def generate_story_and_image():
    data = request.get_json() or {}
    mood, sq = data.get("mood", "").strip(), data.get("sleep_quality", "").strip()
    if not mood or not sq:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"),400
    cat, lang = MoodAnalyzer.categorize(mood)
    msgs = build_prompt(1, mood, sq, cat, lang)
    raw, err = _call_groq(msgs)
    parsed = clean_json_output(raw or "", lang)
    return jsonify({
        "title": parsed.get("title", "") if lang == "english" else "",
        "description": parsed.get("description", "") if lang == "english" else "",
        "content": parsed.get("content", "") if lang == "english" else "",
        "story": parsed.get("story", "") if lang == "arabic" else "",
        "imageUrl": search_cartoon_image(mood),
        "durationMinutes": random.choice([4, 5, 6])
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    if not mood or not sleep_quality:
        return jsonify(error="Missing mood or sleep_quality"), 400
    msgs = [
        {"role": "system", "content": "You are a sleep therapist who gives clear advice based on user mood and sleep quality."},
        {"role": "user", "content": f"My mood is '{mood}' and my sleep quality is '{sleep_quality}'. Can you give me insights and tips?"}
    ]
    response, _ = _call_groq(msgs)
    return jsonify({"analysis": response or "No response available."})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    history = data.get("history", [])
    if not history:
        return jsonify(error="Missing 'history'"), 400
    response, _ = _call_groq(history)
    return jsonify({"response": response or "No response."})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


