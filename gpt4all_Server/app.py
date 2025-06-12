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
typing_pattern = re.compile(r"[؀-ۿ]")
class MoodAnalyzer:
    MOOD_KEYWORDS = {
        "anger": ["angry", "mad", "furious", "rage", "غاضب", "عصبي", "منفعل"],
        "sadness": ["sad", "depressed", "crying", "broken", "حزين", "مكسور", "كئيب"],
        "stress": ["stressed", "anxious", "nervous", "توتر", "قلق", "مضغوط"],
        "lonely": ["lonely", "alone", "isolated", "وحيد", "عزلة", "منعزل"],
        "sexual": ["aroused", "sensual", "horny", "جنسي", "مثير", "رغبة"]
    }
    GENERAL_CATEGORY = "general"

    @classmethod
    def detect_language(cls, text: str) -> str:
        return 'arabic' if typing_pattern.search(text) else 'english'

    @classmethod
    def categorize(cls, raw_mood: str) -> tuple[str, str]:
        lang = cls.detect_language(raw_mood)
        rm = raw_mood.lower()
        for category, keywords in cls.MOOD_KEYWORDS.items():
            if any(kw in rm for kw in keywords):
                return category, lang
        return cls.GENERAL_CATEGORY, lang

# Instructions and Persona
THERAPY_INSTRUCTIONS = {
    "anger": {"english": "I see you're feeling angry...", "arabic": "أرى أنك غاضب..."},
    "sadness": {"english": "I see you're feeling sad...", "arabic": "أشعر أنك حزين..."},
    "stress": {"english": "I see you're stressed...", "arabic": "الوحدة صعبة..."},
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
    if isinstance(value, str): return value
    if isinstance(value, dict):
        for key in ("text", "content", "en", "value"):
            if key in value and isinstance(value[key], str):
                return value[key]
        return json.dumps(value)
    return str(value) if value else ""


def _call_groq(messages: list) -> tuple[str, str]:
    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 700
    }
    try:
        res = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=20
        )
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"].strip(), None
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return "", str(e)


def clean_json_output(raw_text: str, language: str) -> dict:
    if language == "arabic":
        return {"story": raw_text.strip() or "قصة قبل النوم"}
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return {
                "title": extract_text(parsed.get("title", "Untitled")),
                "description": extract_text(parsed.get("description", "")),
                "content": extract_text(parsed.get("content", raw_text))
            }
    except json.JSONDecodeError:
        pass
    return {"title": "Untitled", "description": "", "content": raw_text.strip()}


def search_cartoon_image(query: str) -> str | None:
    if not pixabay_api_key:
        return None
    lang = MoodAnalyzer.detect_language(query)
    q = query if lang == 'english' else query  # use mood or title directly
    params = {"key": pixabay_api_key, "q": q, "image_type": "illustration", "per_page": 10, "safesearch": "true"}
    try:
        resp = requests.get(PIXABAY_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        return random.choice(hits).get("webformatURL") if hits else None
    except Exception as e:
        logger.error(f"Pixabay search failed: {e}")
        return None

# Prompt builder

def build_prompt(i: int, mood: str, sleep_quality: str, category: str, language: str) -> list:
    persona = random.choice(SYSTEM_PERSONAS[language])
    therapy = THERAPY_INSTRUCTIONS[category][language]
    if language == 'english':
        user_instr = (
            f"User Mood: {mood}, Sleep Quality: {sleep_quality}."
            " Please produce a bedtime story in strict JSON with fields `title`, `description`, and `content`."
        )
    else:
        user_instr = (
            f"مزاج المستخدم: {mood}، جودة النوم: {sleep_quality}."
            " اكتب قصة نوم جميلة باللغة العربية بدون تنسيق JSON، فقط السرد."
        )
    return [
        {"role": "system", "content": persona},
        {"role": "system", "content": therapy},
        {"role": "user", "content": user_instr}
    ]

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
        if err:
            logger.error("Error generating story %d: %s", i, err)
            continue
        parsed = clean_json_output(raw, language)
        title = parsed.get("title", f"Dream #{i+1}")
        base, suffix = title, 2
        while title in seen:
            title = f"{base} ({suffix})"
            suffix += 1
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
    mood = data.get("mood", "").strip()
    sq = data.get("sleep_quality", "").strip()
    if not mood or not sq:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400
    category, language = MoodAnalyzer.categorize(mood)
    msgs = build_prompt(1, mood, sq, category, language)
    raw, err = _call_groq(msgs)
    if err:
        return jsonify(error=err), 500
    parsed = clean_json_output(raw, language)
    return jsonify({
        "title": parsed.get("title", "") if language == "english" else "",
        "description": parsed.get("description", "") if language == "english" else "",
        "content": parsed.get("content", "") if language == "english" else "",
        "story": parsed.get("story", "") if language == "arabic" else "",
        "imageUrl": search_cartoon_image(mood),
        "durationMinutes": random.choice([4, 5, 6])
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    sleep_quality = data.get("sleep_quality", "").strip()
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400
    messages = [
        {"role": "system", "content": "You are a sleep therapist who gives clear advice based on user mood and sleep quality."},
        {"role": "user",   "content": f"My mood is '{mood}' and my sleep quality is '{sleep_quality}'. Can you give me insights and tips?"}
    ]
    response, err = _call_groq(messages)
    if err:
        logger.error("Groq analyze error: %s", err)
        return jsonify(error="Failed to get analysis"), 500
    return jsonify(analysis=response)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    history = data.get("history", [])
    if not isinstance(history, list) or not history:
        return jsonify(error="Missing or invalid 'history'"), 400
    system_prompt = {"role": "system", "content": "You are Nightingale, a wise and gentle sleep coach who responds based on the conversation so far."}
    messages = [system_prompt] + history
    response, err = _call_groq(messages)
    if err:
        logger.error("Groq chat error: %s", err)
        return jsonify(error=err), 500
    return jsonify(response=response)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


