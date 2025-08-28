import json
import os
import logging
import random
import traceback
import re
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- Config / Keys ----------------
groq_api_key = os.getenv("GROQ_API_KEY")
pixabay_api_key = os.getenv("PIXABAY_API_KEY")

# API endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PIXABAY_SEARCH_URL = "https://pixabay.com/api/"

# ===================== Helpers (shared) =====================

def extract_text(value):
    """Extract text from various response formats."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ['text', 'content', 'en', 'value']:
            if key in value and isinstance(value[key], str):
                return value[key]
        return json.dumps(value)
    return str(value) if value is not None else ''

def call_groq(user_prompt):
    """Call Groq API with the given prompt. Returns (content, error)."""
    try:
        messages = [
            {"role": "system", "content": "You are Silent Veil, a calm sleep coach assistant."},
            {"role": "user", "content": user_prompt}
        ]
        payload = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.6,
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

def clean_json_output(json_text):
    """Try to parse JSON; if fails, wrap raw text."""
    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            content = parsed.get("content", "")
            if isinstance(content, dict):
                parsed["content"] = json.dumps(content, indent=2)
        return parsed
    except Exception:
        return {"raw": json_text}

def _extract_json_block(text):
    """Extract first JSON object or array from freeform text."""
    if not text:
        return None
    brace = re.search(r"\{[\s\S]*\}", text)
    brack = re.search(r"\[[\s\S]*\]", text)
    m = None
    if brace and brack:
        m = brace if brace.start() < brack.start() else brack
    else:
        m = brace or brack
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def search_cartoon_image(query):
    """Search for cartoon images on Pixabay."""
    if not pixabay_api_key:
        logger.warning("Missing PIXABAY_API_KEY; returning no image.")
        return None

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
            logger.info(f"No Pixabay results for query: {clean_query}")
            return None
        return random.choice(hits).get("webformatURL")
    except Exception as e:
        logger.error(f"Pixabay search failed: {e}")
        return None

# ---- Numeric helpers ----

def _num_or_0(v):
    try:
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return float(str(v))
    except Exception:
        return 0.0

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _minutes_between(start_iso, end_iso):
    """Compute minutes between two ISO strings; handles cross-midnight."""
    try:
        if not start_iso or not end_iso:
            return 0
        s = datetime.fromisoformat(str(start_iso).replace('Z', '+00:00'))
        e = datetime.fromisoformat(str(end_iso).replace('Z', '+00:00'))
        if e < s:
            e = e + timedelta(days=1)
        return int((e - s).total_seconds() // 60)
    except Exception:
        return 0

def _hm_to_minutes(txt):
    try:
        hh, mm = str(txt).split(":")
        return int(hh) * 60 + int(mm)
    except Exception:
        return 0

def _parse_duration_flexible(raw):
    """Understand '7h 30m', '450 min', '07:30', 'PT7H30M', and plain numbers."""
    if raw is None:
        return 0
    s = str(raw).strip()
    if not s:
        return 0

    # ISO-8601 duration PT#H#M#S
    if s.upper().startswith("P"):
        try:
            m = re.search(r"P(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)", s, re.I)
            if m:
                h = int(m.group(1) or 0)
                mi = int(m.group(2) or 0)
                sec = int(m.group(3) or 0)
                return h * 60 + mi + round(sec / 60)
        except Exception:
            pass

    # HH:mm
    if re.match(r"^\d{1,2}:\d{2}$", s):
        return _hm_to_minutes(s)

    # "7h 30m", "7h", "30m"
    h = re.search(r"(\d+)\s*h", s, re.I)
    m = re.search(r"(\d+)\s*m", s, re.I)
    if h or m:
        hh = int(h.group(1)) if h else 0
        mm = int(m.group(1)) if m else 0
        return hh * 60 + mm

    # "450 min" / "450m"
    mins = re.search(r"(\d+)\s*(?:min|mins|m)$", s, re.I)
    if mins:
        return int(mins.group(1))

    # plain number
    try:
        return int(float(s))
    except Exception:
        return 0

# ===================== Routes =====================

@app.route("/", methods=["GET"])
def root():
    """Root endpoint with basic info."""
    return jsonify({
        "message": "Silent Veil backend is online 💤",
        "available_routes": [
            "/chat",
            "/generate",
            "/generate-stories",
            "/generate-story-and-image",
            "/sleep-analysis",
            "/compare-sleep-logs",
            "/ai-highlights",
            "/readiness",
            "/lifestyle-correlations",
            "/health"
        ]
    }), 200

# ---------- Chat ----------

@app.route("/chat", methods=["POST"])
def chat():
    """Handle general chat requests."""
    data = request.get_json() or {}
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
        logger.warning("Chat request with empty prompt")
        return jsonify(error="Missing 'prompt'"), 400

    response, error = call_groq(prompt)
    if error:
        return jsonify(error=error), 500

    return jsonify(response=response)

# ---------- Generate (plain) ----------

@app.route("/generate", methods=["POST"])
def generate_story():
    """Generate plain text story without images."""
    data = request.get_json() or {}
    mood = (data.get("mood") or "").strip()
    sleep_quality = (data.get("sleep_quality") or "").strip()

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

# ---------- Generate (multiple stories with images) ----------

@app.route("/generate-stories", methods=["POST"])
def generate_stories():
    """Generate multiple stories with images."""
    data = request.get_json() or {}
    mood = (data.get("mood") or "").strip()
    sleep_quality = (data.get("sleep_quality") or "").strip()
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

        # Try to parse as JSON block
        story_data = _extract_json_block(story_json_str or "") or clean_json_output(story_json_str or "")

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

# ---------- Generate (single story + image) ----------

@app.route("/generate-story-and-image", methods=["POST"])
def generate_story_and_image():
    """Generate single story with image."""
    data = request.get_json() or {}
    mood = (data.get("mood") or "").strip()
    sleep_quality = (data.get("sleep_quality") or "").strip()

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

    story_data = _extract_json_block(story_json_str or "") or clean_json_output(story_json_str or "")

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

# ---------- Sleep Analysis ----------

@app.route("/sleep-analysis", methods=["POST"])
def sleep_analysis():
    """Analyze sleep logs with clinical precision."""
    data = request.get_json() or {}
    logger.info(f"Received sleep analysis request: {str(data)[:500]}")

    # Flexible input handling
    sleep_data = data.get("sleep_data")
    if sleep_data is None:
        # If sleep_data key is missing, try using the root object
        sleep_data = data
        logger.info("Using root object as sleep_data")

    if not sleep_data:
        logger.warning("Sleep analysis called with empty input")
        return jsonify(
            error="Missing sleep data",
            details="Please provide sleep_data or parameters in the request body",
            code="SleepAnalysisException"
        ), 400

    try:
        # Handle different input formats
        if isinstance(sleep_data, str):
            logger.info("Converting string input to symptom list")
            symptoms = [s.strip() for s in sleep_data.split(",") if s.strip()]
            sleep_data = {"symptoms": symptoms}
        elif isinstance(sleep_data, list):
            logger.info("Converting array input to symptom list")
            sleep_data = {"symptoms": [str(item) for item in sleep_data]}
        elif not isinstance(sleep_data, dict):
            logger.error(f"Invalid sleep_data type: {type(sleep_data)}")
            return jsonify(
                error="Invalid input format",
                details="sleep_data must be a JSON object, string, or array",
                code="SleepAnalysisException"
            ), 400

        # Detect input type
        quantitative_keys = ['TST', 'TIB', 'SE', 'SOL', 'WASO', 'AHI', 'sleep_efficiency']
        is_quantitative = any(key in sleep_data for key in quantitative_keys)

        # Build prompt
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
            logger.info("Processing symptom-based sleep data")
            symptoms = sleep_data.get("symptoms", [])
            if not symptoms:
                symptoms = [str(v) for v in sleep_data.values() if isinstance(v, (str, int, float))]

            if not symptoms:
                logger.error("No symptoms found in sleep_data")
                return jsonify(
                    error="No symptoms provided for analysis",
                    code="SleepAnalysisException"
                ), 400

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

        # Call Groq
        logger.debug(f"Sending prompt to Groq: {prompt[:200]}...")
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

# ---------- Compare Sleep Logs ----------

def _pick_metric_for_compare(log):
    """
    Choose a single numeric metric for comparison.
    Priority:
      1) sleep_score
      2) duration (duration_minutes/totalSleepMinutes or computed from times)
      3) sum of stage minutes
      4) quality * duration (last resort)
    Supports snake_case and camelCase.
    """
    # 1) explicit score
    score = _num_or_0(log.get("sleep_score") or log.get("sleepScore") or log.get("score"))
    if score > 0:
        return score

    # 2) duration
    duration = _num_or_0(
        log.get("duration_minutes") or log.get("durationMinutes") or log.get("totalSleepMinutes") or log.get("duration")
    )
    if duration <= 0:
        # from ISO times (preferred if present)
        bed_iso = log.get("bedTime") or log.get("bed_time") or log.get("sleepStart") or log.get("sleep_start")
        wake_iso = log.get("wakeTime") or log.get("wake_time") or log.get("sleepEnd") or log.get("sleep_end")
        if bed_iso and wake_iso:
            duration = float(_minutes_between(str(bed_iso), str(wake_iso)))
        else:
            # from HH:mm strings
            bed = log.get("bedtime")
            wake = log.get("wake_time")
            if isinstance(bed, str) and isinstance(wake, str) and ":" in bed and ":" in wake:
                b = _hm_to_minutes(bed)
                w = _hm_to_minutes(wake)
                duration = float(w - b) if w > b else float((24*60 - b) + w)
    if duration > 0:
        return duration

    # 3) stage minutes
    deep  = _num_or_0(log.get("deep_sleep_minutes")  or log.get("deepSleepMinutes"))
    rem   = _num_or_0(log.get("rem_sleep_minutes")   or log.get("remSleepMinutes"))
    light = _num_or_0(log.get("light_sleep_minutes") or log.get("lightSleepMinutes"))
    stages_total = deep + rem + light
    if stages_total > 0:
        return stages_total

    # 4) quality * duration
    quality = _num_or_0(log.get("quality") or log.get("sleepQuality"))
    if quality > 0 and duration > 0:
        return quality * duration

    return 0.0

@app.route("/compare-sleep-logs", methods=["POST"])
def compare_sleep_logs():
    """
    Body:
      { "current_log": {...}, "previous_log": {...} }
    Returns:
      { today, yesterday, delta, better, worse }
    """
    try:
        data = request.get_json(silent=True) or {}
        cur = data.get("current_log") or {}
        prev = data.get("previous_log") or {}

        if not isinstance(cur, dict) or not isinstance(prev, dict):
            return jsonify({"error": "Invalid payload"}), 400

        today_val = _pick_metric_for_compare(cur)
        yest_val  = _pick_metric_for_compare(prev)
        delta = round(today_val - yest_val, 1)

        return jsonify({
            "today": today_val,
            "yesterday": yest_val,
            "delta": delta,
            "better": f"{today_val:.1f}" if delta >= 0 else f"{yest_val:.1f}",
            "worse":  f"{yest_val:.1f}" if delta >= 0 else f"{today_val:.1f}",
        }), 200

    except Exception as e:
        logger.error("/compare-sleep-logs failed", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------- AI Highlights ----------

def _summarize_logs_for_highlights(logs):
    """Produce compact numeric summary and deltas from logs array (latest first)."""
    if not logs:
        return {"count": 0}

    latest = logs[0] or {}
    prev = logs[1] if len(logs) > 1 else {}

    def getn(m, *keys):
        if not isinstance(m, dict):
            return 0.0
        for k in keys:
            if isinstance(k, (list, tuple)):
                # nested path
                cur = m
                ok = True
                for seg in k:
                    if isinstance(cur, dict) and seg in cur:
                        cur = cur[seg]
                    else:
                        ok = False
                        break
                if ok:
                    n = _num_or_0(cur)
                    if n:
                        return n
            else:
                n = _num_or_0(m.get(k))
                if n:
                    return n
        return 0.0

    summary = {
        "count": len(logs),
        "latest": {
            "sleep_score": getn(latest, "sleep_score", "sleepScore", ["metrics", "sleepScore"]),
            "duration_minutes": getn(latest, "duration_minutes", "durationMinutes", "totalSleepMinutes", ["metrics", "durationMinutes"]),
            "deep": getn(latest, "deep_sleep_minutes", "deepSleepMinutes", ["stages", "deepMinutes"]),
            "rem": getn(latest, "rem_sleep_minutes", "remSleepMinutes", ["stages", "remMinutes"]),
            "light": getn(latest, "light_sleep_minutes", "lightSleepMinutes", ["stages", "lightMinutes"]),
            "quality": getn(latest, "quality", "sleepQuality", ["metrics", "sleepQuality"]),
            "stress": getn(latest, "stress_level", "stressLevel"),
            "caffeine": getn(latest, "caffeine_intake", "caffeineIntake"),
            "exercise": getn(latest, "exercise_minutes", "exerciseMinutes"),
            "screen": getn(latest, "screen_time_before_bed", "screenTimeBeforeBed"),
        },
        "delta_vs_prev": {}
    }

    if prev:
        def pv(key):
            mapping1 = {
                "sleep_score":"sleep_score","duration_minutes":"duration_minutes","deep":"deep_sleep_minutes",
                "rem":"rem_sleep_minutes","light":"light_sleep_minutes","quality":"quality",
                "stress":"stress_level","caffeine":"caffeine_intake","exercise":"exercise_minutes","screen":"screen_time_before_bed"
            }
            mapping2 = {
                "sleep_score":"sleepScore","duration_minutes":"durationMinutes","deep":"deepSleepMinutes",
                "rem":"remSleepMinutes","light":"lightSleepMinutes","quality":"sleepQuality",
                "stress":"stressLevel","caffeine":"caffeineIntake","exercise":"exerciseMinutes","screen":"screenTimeBeforeBed"
            }
            return _num_or_0(prev.get(mapping1[key]) or prev.get(mapping2[key]))
        summary["delta_vs_prev"] = {k: round(summary["latest"][k] - pv(k), 1) for k in summary["latest"].keys()}

    return summary

@app.route("/ai-highlights", methods=["POST"])
def ai_highlights():
    """
    Body:
      { "logs": [ {sleep_score, duration_minutes, ...}, ... ] } // latest first (recommended)
    Response:
      { "highlights": [ {title, value, change, insight}, ... ] }
    """
    try:
        data = request.get_json(silent=True) or {}
        logs = data.get("logs") or []
        if not isinstance(logs, list):
            return jsonify({"error": "logs must be a list"}), 400

        summary = _summarize_logs_for_highlights(logs)

        # LLM attempt (if key present)
        prompt = (
            "You are Silent Veil, an expert sleep coach. Given the following numeric summary of a user's recent sleep logs, "
            "produce 4–6 concise highlights strictly in JSON array format. Each item must be an object with keys:\n"
            "title (<= 40 chars), value (short stat string), change (one of: up, down, flat), insight (<= 140 chars).\n\n"
            "Focus on actionable, positive coaching. No markdown, no extra text.\n\n"
            f"SUMMARY_JSON = {json.dumps(summary)}"
        )

        text, err = call_groq(prompt) if groq_api_key else (None, "No GROQ_API_KEY")
        parsed = _extract_json_block(text or "") if not err else None

        highlights = []
        if isinstance(parsed, list):
            for item in parsed[:6]:
                if not isinstance(item, dict):
                    continue
                title = extract_text(item.get("title", "")).strip()[:60]
                value = extract_text(item.get("value", "")).strip()[:40]
                change = extract_text(item.get("change", "flat")).lower()
                if change not in ("up", "down", "flat"):
                    change = "flat"
                insight = extract_text(item.get("insight", "")).strip()[:160]
                if title:
                    highlights.append({
                        "title": title,
                        "value": value,
                        "change": change,
                        "insight": insight
                    })

        # Local fallback if LLM fails
        if not highlights:
            lat = summary.get("latest", {})
            dlt = summary.get("delta_vs_prev", {})
            def chg(key):
                v = dlt.get(key, 0.0)
                return "up" if v > 0.5 else ("down" if v < -0.5 else "flat")

            highlights = [
                {
                    "title": "Sleep Duration",
                    "value": f"{int(lat.get('duration_minutes', 0))} min",
                    "change": chg("duration_minutes"),
                    "insight": "Aim for 420–480 mins most nights for optimal recovery."
                },
                {
                    "title": "Sleep Score",
                    "value": f"{int(lat.get('sleep_score', 0))}/100",
                    "change": chg("sleep_score"),
                    "insight": "Consistent schedule & wind-down can lift your score."
                },
                {
                    "title": "Deep + REM",
                    "value": f"{int(lat.get('deep', 0)+lat.get('rem', 0))} min",
                    "change": "flat",
                    "insight": "Protect last cycle: reduce late screens & bright light."
                },
                {
                    "title": "Caffeine & Stress",
                    "value": f"{int(lat.get('caffeine', 0))}mg / {int(lat.get('stress', 0))}/10",
                    "change": "flat",
                    "insight": "Cut caffeine after 14:00 and try 5-min breathing pre-bed."
                },
            ]

        return jsonify({"highlights": highlights})

    except Exception as e:
        logger.error("/ai-highlights failed", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------- Readiness ----------

@app.route("/readiness", methods=["POST"])
def readiness():
    """
    Body: { "log": { ... latest log ... } }
    Returns: { score: 0-100, components: {...}, advice: "..." }
    """
    try:
        data = request.get_json(silent=True) or {}
        log = data.get("log") or {}
        if not isinstance(log, dict):
            return jsonify({"error": "log must be an object"}), 400

        duration = _num_or_0(log.get("duration_minutes") or log.get("durationMinutes"))
        if duration <= 0:
            # compute from times if needed
            bed_iso = log.get("bedTime") or log.get("bed_time") or log.get("sleepStart") or log.get("sleep_start")
            wake_iso = log.get("wakeTime") or log.get("wake_time") or log.get("sleepEnd") or log.get("sleep_end")
            if bed_iso and wake_iso:
                duration = float(_minutes_between(str(bed_iso), str(wake_iso)))
            else:
                bed = log.get("bedtime")
                wake = log.get("wake_time")
                if isinstance(bed, str) and isinstance(wake, str) and ":" in bed and ":" in wake:
                    b = _hm_to_minutes(bed)
                    w = _hm_to_minutes(wake)
                    duration = float(w - b) if w > b else float((24*60 - b) + w)

        quality  = _num_or_0(log.get("quality") or log.get("sleepQuality"))
        stress   = _clamp(_num_or_0(log.get("stress_level") or log.get("stressLevel")), 0, 10)
        caffeine = _clamp(_num_or_0(log.get("caffeine_intake") or log.get("caffeineIntake")), 0, 600)
        exercise = _clamp(_num_or_0(log.get("exercise_minutes") or log.get("exerciseMinutes")), 0, 180)
        latency  = _num_or_0(log.get("latency_minutes") or log.get("latencyMinutes"))
        in_bed   = _num_or_0(log.get("time_in_bed_minutes") or log.get("timeInBedMinutes"))

        eff = 0.0
        if in_bed > 0:
            denom = in_bed + latency if (in_bed + latency) > 0 else 0
            eff = _clamp((duration / denom) * 100.0, 0, 100) if denom > 0 else 0.0

        comp = {
            "duration":  _clamp((duration / 480.0) * 100.0, 0, 100),
            "quality":   _clamp((quality / 10.0) * 100.0, 0, 100),
            "efficiency": eff,
            "stress":    _clamp(100.0 - (stress * 10.0), 0, 100),
            "caffeine":  _clamp(100.0 - (caffeine / 6.0), 0, 100),  # 60mg steps
            "exercise":  _clamp(min(exercise, 60) / 60.0 * 100.0, 0, 100),
        }

        score = (
            0.32 * comp["duration"] +
            0.20 * comp["quality"] +
            0.18 * comp["efficiency"] +
            0.12 * comp["stress"] +
            0.08 * comp["caffeine"] +
            0.10 * comp["exercise"]
        )
        score = round(_clamp(score, 0, 100), 1)

        advice = (
            "Solid recovery ahead. Keep caffeine <200mg after 14:00 and aim for 7–8h sleep."
            if score >= 75 else
            "Moderate recovery. Prioritize 7.5h sleep, light cardio, and earlier wind-down."
            if score >= 55 else
            "Take it easy today. Short naps, hydration, and gentle movement recommended."
        )

        return jsonify({
            "score": score,
            "components": comp,
            "advice": advice,
        })
    except Exception as e:
        logger.error("/readiness failed", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------- Lifestyle Correlations ----------

def _series_from_logs(logs, key_aliases):
    """Return numeric series from logs by trying aliases (snake/camel)."""
    series = []
    for log in logs:
        if not isinstance(log, dict):
            continue
        val = None
        for k in key_aliases:
            if isinstance(k, (list, tuple)):
                # nested path
                cur = log
                ok = True
                for seg in k:
                    if isinstance(cur, dict) and seg in cur:
                        cur = cur[seg]
                    else:
                        ok = False
                        break
                if ok:
                    val = cur
                    break
            else:
                if k in log:
                    val = log.get(k)
                    break
        n = _num_or_0(val)
        # accept zero as value; skip only None/non-numeric that became 0 while genuinely missing?
        # We'll include zeros, but require variance later.
        series.append(n)
    return series

def _pearson_corr(x, y):
    """Compute Pearson r with basic safeguards. Returns (r, n)."""
    if not x or not y or len(x) != len(y):
        return 0.0, 0
    n = len(x)
    # Remove pairs where both are exactly 0 AND likely missing everywhere
    # But better: keep all, variance check will handle flat lines.
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    vx = sum((xi - mean_x) ** 2 for xi in x)
    vy = sum((yi - mean_y) ** 2 for yi in y)
    if vx == 0 or vy == 0:
        return 0.0, n
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    r = cov / (vx ** 0.5) / (vy ** 0.5)
    # clamp to [-1, 1] for numerical safety
    if r != r:  # NaN
        r = 0.0
    r = max(-1.0, min(1.0, r))
    return r, n

@app.route("/lifestyle-correlations", methods=["POST"])
def lifestyle_correlations():
    """
    Body: { "logs": [ {...}, {...}, ... ] }  // order doesn't matter
    Returns:
    {
      "n": <num_samples>,
      "method": "pearson",
      "correlations": {
        "caffeine_intake": {"sleep_score":  ..., "duration_minutes": ..., "deep_sleep_minutes": ..., "rem_sleep_minutes": ..., "efficiency": ...},
        "exercise_minutes": {...},
        "screen_time_before_bed": {...},
        "water_intake": {...},
        "stress_level": {...},
        "medications": {...}   // if numeric present
      },
      "top_positive": [ {"pair":"exercise_minutes vs sleep_score","r":0.42}, ...],
      "top_negative": [ {"pair":"caffeine_intake vs duration_minutes","r":-0.37}, ...],
      "caveats": [
        "Correlations are associative, not causal.",
        "Require at least ~7–10 nights for stable signals.",
        "Flat/constant series are reported as 0."
      ]
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        logs = data.get("logs") or []
        if not isinstance(logs, list) or len(logs) < 2:
            return jsonify({"error": "Provide logs as an array with at least 2 entries"}), 400

        # Define lifestyle inputs and sleep outcomes (aliases included)
        lifestyle_keys = {
            "caffeine_intake": ["caffeine_intake", "caffeineIntake"],
            "exercise_minutes": ["exercise_minutes", "exerciseMinutes"],
            "screen_time_before_bed": ["screen_time_before_bed", "screenTimeBeforeBed", "screen_time"],
            "water_intake": ["water_intake", "waterIntake"],
            "stress_level": ["stress_level", "stressLevel"],
            "medications": ["medications", "meds_count", "meds"]  # if numeric count present
        }
        outcome_keys = {
            "sleep_score": ["sleep_score", "sleepScore", "score", ["metrics","sleepScore"]],
            "duration_minutes": ["duration_minutes", "durationMinutes", "totalSleepMinutes", "duration"],
            "deep_sleep_minutes": ["deep_sleep_minutes", "deepSleepMinutes", ["stages","deepMinutes"]],
            "rem_sleep_minutes": ["rem_sleep_minutes", "remSleepMinutes", ["stages","remMinutes"]],
            "light_sleep_minutes": ["light_sleep_minutes", "lightSleepMinutes", ["stages","lightMinutes"]],
            "efficiency": ["sleep_efficiency", "efficiency", "efficiencyScore", "sleepEfficiency"]
        }

        # Pre-extract series for all keys
        series_lifestyle = {lk: _series_from_logs(logs, aliases) for lk, aliases in lifestyle_keys.items()}
        series_outcomes  = {ok: _series_from_logs(logs, aliases) for ok, aliases in outcome_keys.items()}

        # Compute correlations
        correlations = {}
        pairs = []
        for lk, x in series_lifestyle.items():
            correlations[lk] = {}
            for ok, y in series_outcomes.items():
                # Align lengths (they should match; _series_from_logs always creates same len)
                r, n = _pearson_corr(x, y)
                # If the lifestyle column is clearly non-numeric (all zeros with no variance and no non-zero),
                # the correlation isn't meaningful; keep r=0 but note n.
                correlations[lk][ok] = round(r, 3)
                if abs(r) > 0:  # collect for ranking (ignore exact zeros)
                    pairs.append((lk, ok, r))

        # Rank top positive/negative (take top 5 each)
        positives = sorted([p for p in pairs if p[2] > 0], key=lambda p: p[2], reverse=True)[:5]
        negatives = sorted([p for p in pairs if p[2] < 0], key=lambda p: p[2])[:5]

        top_positive = [{"pair": f"{lk} vs {ok}", "r": round(r, 3)} for (lk, ok, r) in positives]
        top_negative = [{"pair": f"{lk} vs {ok}", "r": round(r, 3)} for (lk, ok, r) in negatives]

        response = {
            "n": len(logs),
            "method": "pearson",
            "correlations": correlations,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "caveats": [
                "Correlations are associative, not causal.",
                "At least 7–10 nights usually needed for stable signals.",
                "Flat/constant series (no variance) produce r=0."
            ]
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error("/lifestyle-correlations failed", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------- Health ----------

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
            "/sleep-analysis (POST)",
            "/compare-sleep-logs (POST)",
            "/ai-highlights (POST)",
            "/readiness (POST)",
            "/lifestyle-correlations (POST)"
        ]
    }
    return jsonify(status)

# ===================== Main =====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    logger.info(f"Starting server on port {port} in {'debug' if debug_mode else 'production'} mode")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

