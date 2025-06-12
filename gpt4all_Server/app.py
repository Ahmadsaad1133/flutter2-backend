import json
import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import logging
import random
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_api_key = os.getenv("GROQ_API_KEY")
pixabay_api_key = os.getenv("PIXABAY_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PIXABAY_SEARCH_URL = "https://pixabay.com/api/"

# -----------------------------------------------------------------------------
# MoodAnalyzer class inlined
# -----------------------------------------------------------------------------
class MoodAnalyzer:
    """
    Analyze user mood and detect language (English/Arabic) based on keyword matching.
    Categories: anger, sadness, stress, lonely, sexual, general.
    """
    MOOD_KEYWORDS = {
        "anger": [
            "angry", "mad", "furious", "irritated", "annoyed", "resentful", "outraged", "enraged", "cross", "indignant",
            "غاضب", "عصبي", "مستاء", "غضبان", "مغتاظ", "ساخط", "مهيج", "منزعج", "مستاء جداً", "غاضب بشدة"
        ],
        "sadness": [
            "sad", "down", "unhappy", "depressed", "melancholy", "gloomy", "tearful", "despondent", "mournful", "blue",
            "حزين", "كسول", "كئيب", "مكتئب", "متضايق", "بكاء", "محزون", "مثبط", "وحيد القلب", "مكتئب جداً"
        ],
        "stress": [
            "stressed", "anxious", "nervous", "tense", "overwhelmed", "worried", "panicked", "frazzled", "restless", "uptight",
            "متوتر", "قلق", "منزعج", "مرهق", "منهك", "مذعور", "مرتبك", "قلقان", "مضطرب", "متحفز"
        ],
        "lonely": [
            "lonely", "alone", "isolated", "abandoned", "forsaken", "solitary", "lonesome", "secluded", "detached", "alienated",
            "وحيد", "منعزل", "مهجور", "متروك", "منفرد", "مقصي", "معزول", "منفصل", "مغترب", "منفرد جداً"
        ],
        "sexual": [
            "sexually frustrated", "sexual frustration", "libido", "erotic", "desire", "arousal", "sensual", "intimate", "lustful", "yearning",
            "إحباط جنسي", "إثارة", "رغبة جنسية", "شهوة", "حسّي", "حميمي", "شهواني", "تهيّج", "لذة", "حيوية"
        ]
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

# -----------------------------------------------------------------------------
# Therapeutic instructions by category (refined teacher Arabic)
# -----------------------------------------------------------------------------
THERAPY_INSTRUCTIONS = {
    "anger": {
        "english": "The user feels angry or frustrated—guide them through gentle breathing and calming imagery.",
        "arabic": "أرى أنك غاضب أو متأزم؛ دعنا نمارس معاً تنفساً عميقاً مع تخيل سماء هادئة وأمواج ناعمة."  
    },
    "sadness": {
        "english": "The user feels sad—infuse empathy, healing metaphors, and hope into the narrative.",
        "arabic": "أشعر بحزنك؛ سأروي لك قصة دافئة مفعمة بالتعاطف والاستعارات الشفائية لتعزيز الأمل في قلبك."  
    },
    "stress": {
        "english": "The user is stressed or anxious—include relaxation techniques like progressive muscle relaxation.",
        "arabic": "يبدو أنك متوتر أو قلق؛ سنتبع خطوات الاسترخاء التدريجي للعضلات مع توجيه ذهنك نحو الراحة."  
    },
    "lonely": {
        "english": "The user feels lonely—create warm companionship characters and reassuring dialogue.",
        "arabic": "أرى شعور الوحدة يرافقك؛ سأحكي لك حكاية يملؤها دفء الصداقة وحوار يلمس القلب."  
    },
    "sexual": {
        "english": "The user experiences sexual frustration—focus on self-care, gentle body awareness, and comfort.",
        "arabic": "أشعر بإحباطك الجنسي؛ سنركز على العناية بالجسد بلطف والوعي الحسي لتهدئة الحواس وإيجاد الراحة."  
    },
    "general": {
        "english": "The user seeks a calm, restorative bedtime story.",
        "arabic": "تبحث عن قصة هادئة تصحبك إلى النوم بسلام؛ استمع لهذه الحكاية الهادئة."  
    }
}

# -----------------------------------------------------------------------------
# System personas to vary tone (teacher style Arabic)
# -----------------------------------------------------------------------------
SYSTEM_PERSONAS = {
    "english": [
        "You are Silent Veil, a calm sleep coach assistant.",
        "You are Dr. Somnus, an empathetic AI doctor specializing in sleep therapy.",
        "You are Nightingale, a soothing storyteller who weaves healing into every word."
    ],
    "arabic": [
        "أنت Silent Veil، مدرّس فذ في فن الاسترخاء قبل النوم.",
        "أنت دكتور Somnus، أستاذ في سحر القصص المعالجة للذهن والجسد.",
        "أنت Nightingale، معلم حكيم ينسج الحكايات لتبعث فيك السكينة والطمأنينة."
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
    # fallback English query if Arabic
    if MoodAnalyzer.detect_language(query) == 'arabic':
        query = 'calm night'
    params = {"key": pixabay_api_key, "q": query, "image_type": "illustration", "per_page": 10, "safesearch": "true"}
    try:
        resp = requests.get(PIXABAY_SEARCH_URL, params=params, timeout=10)
        if resp.status_code != 200:
            logger.error(f"Pixabay API error {resp.status_code}: {resp.text}")
            return None
        hits = resp.json().get("hits", [])
        return random.choice(hits).get("webformatURL") if hits else None
    except Exception as e:
        logger.error(f"Pixabay search failed: {e}")
        return None


def build_prompt(i: int, mood: str, sleep_quality: str, category: str, language: str) -> list:
    persona = random.choice(SYSTEM_PERSONAS[language])
    therapy = THERAPY_INSTRUCTIONS[category][language]
    prompt = (
        f"System: {persona}\n"
        f"Instruction: {therapy}\n\n"
        f"User Mood: '{mood}', Sleep Quality: '{sleep_quality}'.\n"
        f"Task: Create bedtime story #{i+1} with a unique title, setting, characters, and emotional arc tailored to the user's state. Output strictly in JSON (title, description, content) as flat strings."
    )
    if language == 'arabic':
        prompt += "\nPlease respond in Arabic only."
    return [{"role": "system", "content": prompt}, {"role": "user", "content": ""}]

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
        data = clean_json_output(raw or "")
        title = extract_text(data.get("title", f"Dream #{i+1}")).strip()
        base, suffix = title, 2
        while title in seen:
            title = f"{base} ({suffix})"; suffix+=1
        seen.add(title)
        stories.append({
            "title": title,
            "description": extract_text(data.get("description", "")).strip(),
            "content": extract_text(data.get("content", "")).strip(),
            "imageUrl": search_cartoon_image(title),
            "durationMinutes": random.choice([4,5,6])
        })
    return jsonify(stories=stories) if stories else jsonify(error="Failed"), 500

@app.route("/generate-story-and-image", methods=["POST"])
def generate_story_and_image():
    data = request.get_json() or {}
    mood, sq = data.get("mood", "").strip(), data.get("sleep_quality", "").strip()
    if not mood or not sq:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"),400
    cat, lang = MoodAnalyzer.categorize(mood)
    msgs = build_prompt(1, mood, sq, cat, lang)
    raw, err = _call_groq(msgs)
    d = clean_json_output(raw or "")
    return jsonify({
        "title": extract_text(d.get("title","")).strip(),
        "description": extract_text(d.get("description","")).strip(),
        "content": extract_text(d.get("content","")).strip(),
        "imageUrl": search_cartoon_image(mood),
        "durationMinutes": random.choice([4,5,6])
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT",5000))
    app.run(host="0.0.0.0",port=port)


