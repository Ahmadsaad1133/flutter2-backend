import json
import os
import logging
import random
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
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ['text', 'content', 'en', 'value']:
            if key in value and isinstance(value[key], str):
                return value[key]
        return json.dumps(value)
    return str(value) if value is not None else ''

def call_groq(user_prompt):
    try:
        messages = [
            {"role": "system", "content": "You are Silent Veil, a calm sleep coach assistant."},
            {"role": "user", "content": user_prompt}
        ]
        payload = {
            "model": "gemma2-9b-it",
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
            return None, f"Groq API error: {res.status_code} - {res.text[:200]}"
        data = res.json()
        content = data.get("choices", [])[0].get("message", {}).get("content")
        if not content:
            return None, "Empty response from Groq"
        return content.strip(), None
    except Exception as e:
        return None, f"Groq call failed: {str(e)}"

def clean_json_output(json_text):
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
            return None
        hits = resp.json().get("hits", [])
        if not hits:
            return None
        return random.choice(hits).get("webformatURL")
    except Exception:
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
    if raw is None:
        return 0
    s = str(raw).strip()
    if not s:
        return 0
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
    if re.match(r"^\d{1,2}:\d{2}$", s):
        return _hm_to_minutes(s)
    h = re.search(r"(\d+)\s*h", s, re.I)
    m = re.search(r"(\d+)\s*m", s, re.I)
    if h or m:
        hh = int(h.group(1)) if h else 0
        mm = int(m.group(1)) if m else 0
        return hh * 60 + mm
    mins = re.search(r"(\d+)\s*(?:min|mins|m)$", s, re.I)
    if mins:
        return int(mins.group(1))
    try:
        return int(float(s))
    except Exception:
        return 0

# ===================== Routes =====================

@app.route("/", methods=["GET"])
def root():
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
            "/insights",
            "/health",
            "/report"
        ]
    }), 200

# ---------- Chat ----------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="Missing 'prompt'"), 400
    content, err = call_groq(prompt)
    if err:
        return jsonify(error=err), 500
    return jsonify(response=content)

# ---------- Generate (plain) ----------

@app.route("/generate", methods=["POST"])
def generate_story():
    data = request.get_json() or {}
    mood = (data.get("mood") or "").strip()
    sleep_quality = (data.get("sleep_quality") or "").strip()
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400
    prompt = (
        f"You are Silent Veil, a calm sleep coach. Based on mood '{mood}' "
        f"and sleep quality '{sleep_quality}', create a calming bedtime story. "
        "Return only the story text, no JSON or formatting."
    )
    story, err = call_groq(prompt)
    if err:
        return jsonify(error=err), 500
    return jsonify(story=story)

# ---------- Generate (multiple stories with images) ----------

@app.route("/generate-stories", methods=["POST"])
def generate_stories():
    data = request.get_json() or {}
    mood = (data.get("mood") or "").strip()
    sleep_quality = (data.get("sleep_quality") or "").strip()
    count = int(data.get("count", 5))
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400

    stories = []
    seen = set()
    for i in range(count):
        prompt = (
            f"You are Silent Veil. Based on mood '{mood}' and sleep quality '{sleep_quality}', "
            f"create unique bedtime story #{i+1}. "
            "Respond in JSON with: title, description, content. "
            "All values must be plain strings. No markdown or nested data."
        )
        text, err = call_groq(prompt)
        if err:
            continue
        parsed = _extract_json_block(text) or {}
        title = extract_text((parsed or {}).get("title") or f"Oneiric Journey #{i+1}").strip()
        base = title; n = 2
        while title in seen:
            title = f"{base} ({n})"; n += 1
        seen.add(title)
        description = extract_text((parsed or {}).get("description") or "").strip()
        content = extract_text((parsed or {}).get("content") or "").strip()
        image_url = search_cartoon_image(title or mood) or ""
        duration = random.choice([4, 5, 6])
        stories.append({
            "title": title,
            "description": description,
            "content": content,
            "imageUrl": image_url,
            "durationMinutes": duration,
        })
    if not stories:
        return jsonify(error="Failed to generate any stories"), 500
    return jsonify(stories=stories)

# ---------- Generate (single story + image) ----------

@app.route("/generate-story-and-image", methods=["POST"])
def generate_story_and_image():
    data = request.get_json() or {}
    mood = (data.get("mood") or "").strip()
    sleep_quality = (data.get("sleep_quality") or "").strip()
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400
    prompt = (
        f"You are Silent Veil. Based on mood '{mood}' and sleep quality '{sleep_quality}', "
        "create a calming bedtime story. Respond in JSON with: title, description, content. "
        "All values must be plain strings. No markdown or nested data."
    )
    text, err = call_groq(prompt)
    if err:
        return jsonify(error=err), 500
    parsed = _extract_json_block(text) or {}
    title = extract_text(parsed.get("title") or "Oneiric Dream").strip()
    description = extract_text(parsed.get("description") or "").strip()
    content = extract_text(parsed.get("content") or "").strip()
    image_url = search_cartoon_image(title or mood) or ""
    duration = random.choice([4, 5, 6])
    return jsonify({
        "title": title,
        "description": description,
        "content": content,
        "imageUrl": image_url,
        "durationMinutes": duration,
    })

# ---------- Sleep Analysis ----------

@app.route("/sleep-analysis", methods=["POST"])
def sleep_analysis():
    data = request.get_json() or {}
    logger.info(f"Received sleep analysis request: {str(data)[:500]}")

    sleep_data = data.get("sleep_data")
    if sleep_data is None:
        sleep_data = data

    if not sleep_data:
        return jsonify(
            error="Missing sleep data",
            details="Provide sleep_data or parameters in the request body",
            code="SleepAnalysisException"
        ), 400

    try:
        if isinstance(sleep_data, str):
            symptoms = [s.strip() for s in sleep_data.split(",") if s.strip()]
            sleep_data = {"symptoms": symptoms}
        elif isinstance(sleep_data, list):
            sleep_data = {"symptoms": [str(item) for item in sleep_data]}
        elif not isinstance(sleep_data, dict):
            return jsonify(
                error="Invalid input format",
                details="sleep_data must be an object, string, or array",
                code="SleepAnalysisException"
            ), 400

        quantitative_keys = ['TST', 'TIB', 'SE', 'SOL', 'WASO', 'AHI', 'sleep_efficiency']
        is_quantitative = any(key in sleep_data for key in quantitative_keys)

        if is_quantitative:
            prompt = (
                "You are Dr. Somnus, a board-certified sleep specialist. Analyze this quantitative sleep data:\n\n"
                "1. Calculate sleep efficiency: (TST / TIB) × 100 (if not provided)\n"
                "2. Assess sleep continuity metrics\n"
                "3. Compare against AASM clinical thresholds\n"
                "4. Identify potential sleep disorders\n"
                "5. Provide evidence-based recommendations\n\n"
                "Sections:\n"
                "### Quantitative Analysis\n"
                "### Clinical Assessment\n"
                "### Treatment Recommendations\n\n"
                f"Data: {json.dumps(sleep_data)}"
            )
        else:
            symptoms = sleep_data.get("symptoms", [])
            if not symptoms:
                symptoms = [str(v) for v in sleep_data.values() if isinstance(v, (str, int, float))]
            if not symptoms:
                return jsonify(error="No symptoms provided", code="SleepAnalysisException"), 400
            prompt = (
                "You are Dr. Somnus, a board-certified sleep specialist. "
                "Analyze these patient-reported symptoms:\n\n"
                "1. Identify potential sleep disorders (ICSD-3)\n"
                "2. Relate symptoms to physiological causes\n"
                "3. Provide clinical recommendations\n\n"
                "Sections:\n"
                "### Symptom Analysis\n"
                "### Clinical Assessment\n"
                "### Personalized Recommendations\n\n"
                f"Symptoms: {', '.join(symptoms)}"
            )

        text, err = call_groq(prompt)
        if err or not text:
            return jsonify(error="Analysis service error", details=err or "empty"), 500
        return jsonify(analysis=text)
    except Exception as e:
        logger.error("Sleep analysis failed", exc_info=True)
        return jsonify(error="Sleep analysis failed", details=str(e), code="SleepAnalysisException"), 500

# ---------------- Daily Compare ----------------

def _pick_metric_for_compare(log):
    score = _num_or_0(log.get("sleep_score") or log.get("sleepScore") or log.get("score"))
    if score > 0:
        return score
    duration = _num_or_0(log.get("duration_minutes") or log.get("durationMinutes") or log.get("totalSleepMinutes") or log.get("duration"))
    if duration <= 0:
        bed_iso = log.get("bedTime") or log.get("bed_time") or log.get("sleepStart") or log.get("sleep_start")
        wake_iso = log.get("wakeTime") or log.get("wake_time") or log.get("sleepEnd") or log.get("sleep_end")
        if bed_iso and wake_iso:
            duration = float(_minutes_between(str(bed_iso), str(wake_iso)))
        else:
            bed = log.get("bedtime"); wake = log.get("wake_time")
            if isinstance(bed, str) and isinstance(wake, str) and ":" in bed and ":" in wake:
                b = _hm_to_minutes(bed); w = _hm_to_minutes(wake)
                duration = float(w - b) if w > b else float((24*60 - b) + w)
    if duration > 0:
        return duration
    deep  = _num_or_0(log.get("deep_sleep_minutes")  or log.get("deepSleepMinutes"))
    rem   = _num_or_0(log.get("rem_sleep_minutes")   or log.get("remSleepMinutes"))
    light = _num_or_0(log.get("light_sleep_minutes") or log.get("lightSleepMinutes"))
    total = deep + rem + light
    if total > 0:
        return total
    quality = _num_or_0(log.get("quality") or log.get("sleepQuality"))
    if quality > 0 and duration > 0:
        return quality * duration
    return 0.0

@app.route("/compare-sleep-logs", methods=["POST"])
def compare_sleep_logs():
    try:
        data = request.get_json(silent=True) or {}
        cur = data.get("current_log") or {}
        prev = data.get("previous_log") or {}
        if not isinstance(cur, dict) or not isinstance(prev, dict):
            return jsonify({"error": "Invalid payload"}), 400
        today_val = _pick_metric_for_compare(cur)
        yest_val = _pick_metric_for_compare(prev)
        delta = round(today_val - yest_val, 1)
        return jsonify({
            "today": today_val,
            "yesterday": yest_val,
            "delta": delta,
            "better": f"{today_val:.1f}" if delta >= 0 else f"{yest_val:.1f}",
            "worse": f"{yest_val:.1f}" if delta >= 0 else f"{today_val:.1f}",
        }), 200
    except Exception as e:
        logger.error("/compare-sleep-logs failed", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------- AI Highlights ----------

def _summarize_logs_for_highlights(logs):
    if not logs:
        return {"count": 0}
    latest = logs[0] or {}
    prev = logs[1] if len(logs) > 1 else {}
    def getn(m, *keys):
        if not isinstance(m, dict):
            return 0.0
        for k in keys:
            if isinstance(k, (list, tuple)):
                cur = m; ok = True
                for seg in k:
                    if isinstance(cur, dict) and seg in cur:
                        cur = cur[seg]
                    else:
                        ok = False; break
                if ok:
                    n = _num_or_0(cur)
                    if n: return n
            else:
                n = _num_or_0(m.get(k))
                if n: return n
        return 0.0

    summary = {
        "count": len(logs),
        "latest": {
            "sleep_score": getn(latest, "sleep_score", "sleepScore", ["metrics","sleepScore"]),
            "duration_minutes": getn(latest, "duration_minutes", "durationMinutes", "totalSleepMinutes", ["metrics","durationMinutes"]),
            "deep": getn(latest, "deep_sleep_minutes", "deepSleepMinutes", ["stages","deepMinutes"]),
            "rem": getn(latest, "rem_sleep_minutes", "remSleepMinutes", ["stages","remMinutes"]),
            "light": getn(latest, "light_sleep_minutes", "lightSleepMinutes", ["stages","lightMinutes"]),
            "quality": getn(latest, "quality", "sleepQuality", ["metrics","sleepQuality"]),
            "stress": getn(latest, "stress_level", "stressLevel"),
            "caffeine": getn(latest, "caffeine_intake", "caffeineIntake"),
            "exercise": getn(latest, "exercise_minutes", "exerciseMinutes"),
            "screen": getn(latest, "screen_time_before_bed", "screenTimeBeforeBed"),
        },
        "delta_vs_prev": {}
    }
    if prev:
        def pv(key):
            m1 = {"sleep_score":"sleep_score","duration_minutes":"duration_minutes","deep":"deep_sleep_minutes",
                  "rem":"rem_sleep_minutes","light":"light_sleep_minutes","quality":"quality",
                  "stress":"stress_level","caffeine":"caffeine_intake","exercise":"exercise_minutes","screen":"screen_time_before_bed"}
        # camelCase too
            m2 = {"sleep_score":"sleepScore","duration_minutes":"durationMinutes","deep":"deepSleepMinutes",
                  "rem":"remSleepMinutes","light":"lightSleepMinutes","quality":"sleepQuality",
                  "stress":"stressLevel","caffeine":"caffeineIntake","exercise":"exerciseMinutes","screen":"screenTimeBeforeBed"}
            return _num_or_0(prev.get(m1[key]) or prev.get(m2[key]))
        summary["delta_vs_prev"] = {k: round(summary["latest"][k] - pv(k), 1) for k in summary["latest"].keys()}
    return summary

@app.route("/ai-highlights", methods=["POST"])
def ai_highlights():
    try:
        data = request.get_json(silent=True) or {}
        logs = data.get("logs") or []
        if not isinstance(logs, list):
            return jsonify({"error": "logs must be a list"}), 400
        summary = _summarize_logs_for_highlights(logs)
        prompt = (
            "You are Silent Veil, an expert sleep coach. Given the following numeric summary of a user's recent sleep logs, "
            "produce 4–6 concise highlights strictly in JSON array format. Each item must be an object with keys:\n"
            "title (<= 40 chars), value (short stat string), change (one of: up, down, flat), insight (<= 140 chars).\n\n"
            "No markdown, no extra text.\n\n"
            f"SUMMARY_JSON = {json.dumps(summary)}"
        )
        text, err = call_groq(prompt) if groq_api_key else (None, "No GROQ_API_KEY")
        parsed = _extract_json_block(text or "") if not err else None
        highlights = []
        if isinstance(parsed, list):
            for item in parsed[:6]:
                if not isinstance(item, dict): continue
                title = extract_text(item.get("title","")).strip()[:60]
                value = extract_text(item.get("value","")).strip()[:40]
                change = extract_text(item.get("change","flat")).lower()
                if change not in ("up","down","flat"): change = "flat"
                insight = extract_text(item.get("insight","")).strip()[:160]
                if title:
                    highlights.append({"title":title,"value":value,"change":change,"insight":insight})
        if not highlights:
            lat = summary.get("latest", {}); dlt = summary.get("delta_vs_prev", {})
            def chg(k): 
                v = dlt.get(k,0.0); return "up" if v > 0.5 else ("down" if v < -0.5 else "flat")
            highlights = [
                {"title":"Sleep Duration","value":f"{int(lat.get('duration_minutes',0))} min","change":chg('duration_minutes'),
                 "insight":"Aim for 420–480 mins most nights for optimal recovery."},
                {"title":"Sleep Score","value":f"{int(lat.get('sleep_score',0))}/100","change":chg('sleep_score'),
                 "insight":"Consistent schedule & wind-down can lift your score."},
                {"title":"Deep + REM","value":f"{int(lat.get('deep',0)+lat.get('rem',0))} min","change":"flat",
                 "insight":"Protect last cycle by reducing late screens & bright light."},
                {"title":"Caffeine & Stress","value":f"{int(lat.get('caffeine',0))}mg / {int(lat.get('stress',0))}/10","change":"flat",
                 "insight":"Cut caffeine after 14:00 and try 5-min breathing pre-bed."},
            ]
        return jsonify({"highlights": highlights})
    except Exception as e:
        logger.error("/ai-highlights failed", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------- Readiness ----------

@app.route("/readiness", methods=["POST"])
def readiness():
    try:
        data = request.get_json(silent=True) or {}
        log = data.get("log") or {}
        if not isinstance(log, dict):
            return jsonify({"error": "log must be an object"}), 400

        duration = _num_or_0(log.get("duration_minutes") or log.get("durationMinutes"))
        if duration <= 0:
            bed_iso = log.get("bedTime") or log.get("bed_time") or log.get("sleepStart") or log.get("sleep_start")
            wake_iso = log.get("wakeTime") or log.get("wake_time") or log.get("sleepEnd") or log.get("sleep_end")
            if bed_iso and wake_iso:
                duration = float(_minutes_between(str(bed_iso), str(wake_iso)))
            else:
                bed = log.get("bedtime"); wake = log.get("wake_time")
                if isinstance(bed, str) and isinstance(wake, str) and ":" in bed and ":" in wake:
                    b = _hm_to_minutes(bed); w = _hm_to_minutes(wake)
                    duration = float(w - b) if w > b else float((24*60 - b) + w)

        quality = _num_or_0(log.get("quality") or log.get("sleepQuality"))
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
            "caffeine":  _clamp(100.0 - (caffeine / 6.0), 0, 100),
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
        advice = "Solid recovery ahead. Keep caffeine <200mg after 14:00 and aim for 7–8h sleep." if score >= 75 else \
                 "Moderate recovery. Prioritize 7.5h sleep, light cardio, and earlier wind-down." if score >= 55 else \
                 "Take it easy today. Short naps, hydration, and gentle movement recommended."
        return jsonify({"score": score, "components": comp, "advice": advice})
    except Exception as e:
        logger.error("/readiness failed", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------- Lifestyle Correlations (standalone) ----------

def _series_from_logs(logs, key_aliases):
    series = []
    for log in logs:
        if not isinstance(log, dict): 
            series.append(0.0); continue
        val = None
        for k in key_aliases:
            if isinstance(k, (list, tuple)):
                cur = log; ok = True
                for seg in k:
                    if isinstance(cur, dict) and seg in cur: cur = cur[seg]
                    else: ok = False; break
                if ok: val = cur; break
            else:
                if k in log: val = log.get(k); break
        series.append(_num_or_0(val))
    return series

def _pearson_corr(x, y):
    if not x or not y or len(x) != len(y):
        return 0.0, 0
    n = len(x)
    mean_x = sum(x) / n; mean_y = sum(y) / n
    vx = sum((xi - mean_x) ** 2 for xi in x)
    vy = sum((yi - mean_y) ** 2 for yi in y)
    if vx == 0 or vy == 0:
        return 0.0, n
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    r = cov / (vx ** 0.5) / (vy ** 0.5)
    if r != r: r = 0.0
    r = max(-1.0, min(1.0, r))
    return r, n

@app.route("/lifestyle-correlations", methods=["POST"])
def lifestyle_correlations():
    try:
        data = request.get_json(silent=True) or {}
        logs = data.get("logs") or []
        if not isinstance(logs, list) or len(logs) < 2:
            return jsonify({"error": "Provide logs as an array with at least 2 entries"}), 400

        lifestyle_keys = {
            "caffeine_intake": ["caffeine_intake", "caffeineIntake"],
            "exercise_minutes": ["exercise_minutes", "exerciseMinutes"],
            "screen_time_before_bed": ["screen_time_before_bed", "screenTimeBeforeBed", "screen_time"],
            "water_intake": ["water_intake", "waterIntake"],
            "stress_level": ["stress_level", "stressLevel"],
            "medications": ["medications", "meds_count", "meds"]
        }
        outcome_keys = {
            "sleep_score": ["sleep_score", "sleepScore", "score", ["metrics","sleepScore"]],
            "duration_minutes": ["duration_minutes", "durationMinutes", "totalSleepMinutes", "duration"],
            "deep_sleep_minutes": ["deep_sleep_minutes", "deepSleepMinutes", ["stages","deepMinutes"]],
            "rem_sleep_minutes": ["rem_sleep_minutes", "remSleepMinutes", ["stages","remMinutes"]],
            "light_sleep_minutes": ["light_sleep_minutes", "lightSleepMinutes", ["stages","lightMinutes"]],
            "efficiency": ["sleep_efficiency", "efficiency", "efficiencyScore", "sleepEfficiency"]
        }
        series_l = {k: _series_from_logs(logs, v) for k, v in lifestyle_keys.items()}
        series_o = {k: _series_from_logs(logs, v) for k, v in outcome_keys.items()}

        correlations = {}
        pairs = []
        for lk, x in series_l.items():
            correlations[lk] = {}
            for ok, y in series_o.items():
                r, n = _pearson_corr(x, y)
                correlations[lk][ok] = round(r, 3)
                if abs(r) > 0:
                    pairs.append((lk, ok, r))
        positives = sorted([p for p in pairs if p[2] > 0], key=lambda p: p[2], reverse=True)[:5]
        negatives = sorted([p for p in pairs if p[2] < 0], key=lambda p: p[2])[:5]
        return jsonify({
            "n": len(logs),
            "method": "pearson",
            "correlations": correlations,
            "top_positive": [{"pair": f"{a} vs {b}", "r": round(r,3)} for a,b,r in positives],
            "top_negative": [{"pair": f"{a} vs {b}", "r": round(r,3)} for a,b,r in negatives],
            "caveats": [
                "Correlations are associative, not causal.",
                "At least 7–10 nights usually needed for stable signals.",
                "Flat/constant series (no variance) produce r=0."
            ]
        }), 200
    except Exception as e:
        logger.error("/lifestyle-correlations failed", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------- Insights (used by client and /report) ----------

def _labels_from_key(k):
    pretty = {
        "caffeine_intake": "Caffeine",
        "exercise_minutes": "Exercise",
        "screen_time_before_bed": "Screen Time",
        "water_intake": "Water Intake",
        "stress_level": "Stress",
        "medications": "Medications",
        "sleep_score": "Sleep Score",
        "duration_minutes": "Duration",
        "deep_sleep_minutes": "Deep Sleep",
        "rem_sleep_minutes": "REM Sleep",
        "light_sleep_minutes": "Light Sleep",
        "efficiency": "Efficiency",
    }
    return pretty.get(k, k.replace("_", " ").title())

@app.route("/insights", methods=["POST"])
def insights():
    """
    Accepts either:
      - { ...single sleep log fields... }
      - { "current": {...}, "logs": [ {...}, {...}, ... ] }
      - { "sleep_data": {...}, "logs": [...] }
    Returns a JSON object that includes 'lifestyleCorrelations'.
    """
    try:
        payload = request.get_json(silent=True) or {}
        # Flexible extraction
        current = payload.get("current") or payload.get("sleep_data") or payload
        logs = payload.get("logs") or payload.get("recent_logs") or []

        lifestyleCorrelations = []  # what your UI expects

        # If we have history, compute real correlations against sleep outcomes
        if isinstance(logs, list) and len(logs) >= 2:
            lifestyle_keys = {
                "caffeine_intake": ["caffeine_intake", "caffeineIntake"],
                "exercise_minutes": ["exercise_minutes", "exerciseMinutes"],
                "screen_time_before_bed": ["screen_time_before_bed", "screenTimeBeforeBed", "screen_time"],
                "water_intake": ["water_intake", "waterIntake"],
                "stress_level": ["stress_level", "stressLevel"],
                "medications": ["medications", "meds_count", "meds"]
            }
            outcome_keys = {
                "sleep_score": ["sleep_score", "sleepScore", "score", ["metrics","sleepScore"]],
                "duration_minutes": ["duration_minutes", "durationMinutes", "totalSleepMinutes", "duration"],
                "deep_sleep_minutes": ["deep_sleep_minutes", "deepSleepMinutes", ["stages","deepMinutes"]],
                "rem_sleep_minutes": ["rem_sleep_minutes", "remSleepMinutes", ["stages","remMinutes"]],
                "light_sleep_minutes": ["light_sleep_minutes", "lightSleepMinutes", ["stages","lightMinutes"]],
                "efficiency": ["sleep_efficiency", "efficiency", "efficiencyScore", "sleepEfficiency"]
            }
            series_l = {k: _series_from_logs(logs, v) for k, v in lifestyle_keys.items()}
            series_o = {k: _series_from_logs(logs, v) for k, v in outcome_keys.items()}

            # Choose a single outcome to rank by (sleep_score > duration > efficiency)
            outcome_order = ["sleep_score", "duration_minutes", "efficiency"]
            chosen_outcome = None
            for o in outcome_order:
                if any(val != 0 for val in series_o[o]): 
                    chosen_outcome = o; break
            chosen_outcome = chosen_outcome or "sleep_score"

            scored = []
            for lk, x in series_l.items():
                r, n = _pearson_corr(x, series_o[chosen_outcome])
                scored.append({"label": _labels_from_key(lk), "value": round(r, 3)})
            # sort by magnitude desc, keep 6 items
            scored.sort(key=lambda d: abs(d["value"]), reverse=True)
            lifestyleCorrelations = scored[:6]

        else:
            # No history provided: produce safe heuristics from current log so UI has content
            caff = _num_or_0(current.get("caffeine_intake") or current.get("caffeineIntake"))
            exer = _num_or_0(current.get("exercise_minutes") or current.get("exerciseMinutes"))
            scrn = _num_or_0(current.get("screen_time_before_bed") or current.get("screenTimeBeforeBed"))
            water= _num_or_0(current.get("water_intake") or current.get("waterIntake"))
            stress=_num_or_0(current.get("stress_level") or current.get("stressLevel"))
            # simple normalized hints mapped to [-0.4, +0.4]
            def norm(x, hi): 
                return round(_clamp((x/hi)*0.4, 0, 0.4), 3)
            lifestyleCorrelations = [
                {"label":"Caffeine","value": -norm(caff, 300)},     # higher caffeine → worse
                {"label":"Exercise","value":  norm(exer, 60)},      # moderate exercise → better
                {"label":"Screen Time","value": -norm(scrn, 120)},  # more screen → worse
                {"label":"Water Intake","value":  norm(water, 2500)},
                {"label":"Stress","value": -norm(stress*10, 100)},
            ]

        response = {
            "summary": "",
            "environment_analysis": {"noise":50,"light":50,"temperature":50,"comfort":50,"overall":50,"notes":""},
            "dream_mood_forecast": {"mood":"neutral","confidence":60},
            "historical_analysis": "",
            "lifestyleCorrelations": lifestyleCorrelations
        }
        return jsonify(response), 200
    except Exception as e:
        logger.error("/insights failed", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------- Unified Report (NEW) ----------

def _add_minutes_hhmm(hhmm, minutes):
    try:
        parts = hhmm.split(':')
        h = int(parts[0]); m = int(parts[1])
        total = h*60 + m + int(minutes)
        total %= (24*60)
        nh = str(total // 60).zfill(2)
        nm = str(total % 60).zfill(2)
        return f"{nh}:{nm}"
    except Exception:
        return hhmm

@app.route("/report", methods=["POST"])
def report():
    """
    Build the five Report sections on the server and return one JSON object:
      - executiveSummary, riskAssessment, energyPlan, wakeWindows, whatIfScenarios
    Accepts:
      { "current": {...}, "history": [ {...}, ... ] }
      or compatible variants: {"current_log":...,"logs":[...]}, {"sleep_data":...}
    """
    try:
        payload = request.get_json(silent=True) or {}
        current = payload.get("current") or payload.get("current_log") or payload.get("sleep_data") or {}
        history = payload.get("history") or payload.get("logs") or payload.get("recent_logs") or []

        # 1) Lifestyle correlations (reuse insights logic)
        lifestyleCorrelations = []
        if isinstance(history, list) and len(history) >= 2:
            lifestyle_keys = {
                "caffeine_intake": ["caffeine_intake", "caffeineIntake"],
                "exercise_minutes": ["exercise_minutes", "exerciseMinutes"],
                "screen_time_before_bed": ["screen_time_before_bed", "screenTimeBeforeBed", "screen_time"],
                "water_intake": ["water_intake", "waterIntake"],
                "stress_level": ["stress_level", "stressLevel"],
                "medications": ["medications", "meds_count", "meds"]
            }
            outcome_keys = {
                "sleep_score": ["sleep_score", "sleepScore", "score", ["metrics","sleepScore"]],
                "duration_minutes": ["duration_minutes", "durationMinutes", "totalSleepMinutes", "duration"],
                "deep_sleep_minutes": ["deep_sleep_minutes", "deepSleepMinutes", ["stages","deepMinutes"]],
                "rem_sleep_minutes": ["rem_sleep_minutes", "remSleepMinutes", ["stages","remMinutes"]],
                "light_sleep_minutes": ["light_sleep_minutes", "lightSleepMinutes", ["stages","lightMinutes"]],
                "efficiency": ["sleep_efficiency", "efficiency", "efficiencyScore", "sleepEfficiency"]
            }
            series_l = {k: _series_from_logs(history, v) for k, v in lifestyle_keys.items()}
            series_o = {k: _series_from_logs(history, v) for k, v in outcome_keys.items()}
            outcome_order = ["sleep_score", "duration_minutes", "efficiency"]
            chosen_outcome = None
            for o in outcome_order:
                if any(val != 0 for val in series_o[o]):
                    chosen_outcome = o; break
            chosen_outcome = chosen_outcome or "sleep_score"
            scored = []
            for lk, x in series_l.items():
                r, _ = _pearson_corr(x, series_o[chosen_outcome])
                label = _labels_from_key(lk)
                scored.append({"label":label, "value": round(r,3)})
            scored.sort(key=lambda d: abs(d["value"]), reverse=True)
            lifestyleCorrelations = scored[:6]
        else:
            caff = _num_or_0(current.get("caffeine_intake") or current.get("caffeineIntake"))
            exer = _num_or_0(current.get("exercise_minutes") or current.get("exerciseMinutes"))
            scrn = _num_or_0(current.get("screen_time_before_bed") or current.get("screenTimeBeforeBed"))
            water= _num_or_0(current.get("water_intake") or current.get("waterIntake"))
            stress=_num_or_0(current.get("stress_level") or current.get("stressLevel"))
            def norm(x, hi): 
                return round(_clamp((x/hi)*0.4, 0, 0.4), 3)
            lifestyleCorrelations = [
                {"label":"Caffeine","value": -norm(caff, 300)},
                {"label":"Exercise","value":  norm(exer, 60)},
                {"label":"Screen Time","value": -norm(scrn, 120)},
                {"label":"Water Intake","value":  norm(water, 2500)},
                {"label":"Stress","value": -norm(stress*10, 100)},
            ]

        # 2) Highlights (try GROQ, fallback rules)
        summary = _summarize_logs_for_highlights(history if isinstance(history, list) else [])
        prompt_h = (
            "You are Silent Veil, an expert sleep coach. Given the following numeric summary of a user's recent sleep logs, "
            "produce 4–6 concise highlights strictly in JSON array format. Each item must be an object with keys:\n"
            "title (<= 40 chars), value (short stat string), change (one of: up, down, flat), insight (<= 140 chars).\n\n"
            "No markdown, no extra text.\n\n"
            f"SUMMARY_JSON = {json.dumps(summary)}"
        )
        text_h, err_h = call_groq(prompt_h) if groq_api_key else (None, "No GROQ_API_KEY")
        parsed_h = _extract_json_block(text_h or "") if not err_h else None
        highlights = []
        if isinstance(parsed_h, list):
            for item in parsed_h[:6]:
                if not isinstance(item, dict): continue
                title = extract_text(item.get("title","")).strip()[:60]
                value = extract_text(item.get("value","")).strip()[:40]
                change = extract_text(item.get("change","flat")).lower()
                if change not in ("up","down","flat"): change = "flat"
                insight = extract_text(item.get("insight","")).strip()[:160]
                if title:
                    highlights.append({"title":title,"value":value,"change":change,"insight":insight})
        if not highlights:
            lat = summary.get("latest", {}); dlt = summary.get("delta_vs_prev", {})
            def chg(k): 
                v = dlt.get(k,0.0); return "up" if v > 0.5 else ("down" if v < -0.5 else "flat")
            highlights = [
                {"title":"Sleep Duration","value":f"{int(lat.get('duration_minutes',0))} min","change":chg('duration_minutes'),
                 "insight":"Aim for 420–480 mins most nights for optimal recovery."},
                {"title":"Sleep Score","value":f"{int(lat.get('sleep_score',0))}/100","change":chg('sleep_score'),
                 "insight":"Consistent schedule & wind-down can lift your score."},
                {"title":"Deep + REM","value":f"{int(lat.get('deep',0)+lat.get('rem',0))} min","change":"flat",
                 "insight":"Protect last cycle by reducing late screens & bright light."},
                {"title":"Caffeine & Stress","value":f"{int(lat.get('caffeine',0))}mg / {int(lat.get('stress',0))}/10","change":"flat",
                 "insight":"Cut caffeine after 14:00 and try 5-min breathing pre-bed."},
            ]

        # 3) Sleep analysis text (local build like /sleep-analysis)
        def _build_sleep_analysis_text(current_obj):
            try:
                quantitative_keys = ['TST', 'TIB', 'SE', 'SOL', 'WASO', 'AHI', 'sleep_efficiency']
                is_quant = any(k in current_obj for k in quantitative_keys)
                if is_quant:
                    prompt = (
                        "You are Dr. Somnus, a board-certified sleep specialist. Analyze this quantitative sleep data:\n\n"
                        "1. Calculate sleep efficiency: (TST / TIB) × 100 (if not provided)\n"
                        "2. Assess sleep continuity metrics\n"
                        "3. Compare against AASM clinical thresholds\n"
                        "4. Identify potential sleep disorders\n"
                        "5. Provide evidence-based recommendations\n\n"
                        "Sections:\n"
                        "### Quantitative Analysis\n"
                        "### Clinical Assessment\n"
                        "### Treatment Recommendations\n\n"
                        f"Data: {json.dumps(current_obj)}"
                    )
                else:
                    symptoms = current_obj.get("symptoms", [])
                    if not symptoms:
                        symptoms = [str(v) for v in current_obj.values() if isinstance(v, (str, int, float))]
                    if not symptoms:
                        return ""
                    prompt = (
                        "You are Dr. Somnus, a board-certified sleep specialist. "
                        "Analyze these patient-reported symptoms:\n\n"
                        "1. Identify potential sleep disorders (ICSD-3)\n"
                        "2. Relate symptoms to physiological causes\n"
                        "3. Provide clinical recommendations\n\n"
                        "Sections:\n"
                        "### Symptom Analysis\n"
                        "### Clinical Assessment\n"
                        "### Personalized Recommendations\n\n"
                        f"Symptoms: {', '.join(symptoms)}"
                    )
                txt, er = call_groq(prompt)
                return txt or ""
            except Exception:
                return ""
        analysisText = _build_sleep_analysis_text(current)

        # 4) Readiness (same logic as /readiness)
        log = current if isinstance(current, dict) else {}
        duration = _num_or_0(log.get("duration_minutes") or log.get("durationMinutes"))
        if duration <= 0:
            bed_iso = log.get("bedTime") or log.get("bed_time") or log.get("sleepStart") or log.get("sleep_start")
            wake_iso = log.get("wakeTime") or log.get("wake_time") or log.get("sleepEnd") or log.get("sleep_end")
            if bed_iso and wake_iso:
                duration = float(_minutes_between(str(bed_iso), str(wake_iso)))
            else:
                bed = log.get("bedtime"); wake = log.get("wake_time")
                if isinstance(bed, str) and isinstance(wake, str) and ":" in bed and ":" in wake:
                    b = _hm_to_minutes(bed); w = _hm_to_minutes(wake)
                    duration = float(w - b) if w > b else float((24*60 - b) + w)

        quality = _num_or_0(log.get("quality") or log.get("sleepQuality"))
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
            "caffeine":  _clamp(100.0 - (caffeine / 6.0), 0, 100),
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
        advice = "Solid recovery ahead. Keep caffeine <200mg after 14:00 and aim for 7–8h sleep." if score >= 75 else \
                 "Moderate recovery. Prioritize 7.5h sleep, light cardio, and earlier wind-down." if score >= 55 else \
                 "Take it easy today. Short naps, hydration, and gentle movement recommended."
        readiness_obj = {"score": score, "components": comp, "advice": advice}

        # ---------- Compose sections ----------
        bullets = []
        for h in highlights[:4]:
            title = (h.get('title') or '').strip()
            value = (h.get('value') or '').strip()
            insight = (h.get('insight') or '').strip()
            b = ' — '.join([t for t in [title, value] if t])
            bullets.append(f"{b}. {insight}" if b else insight)
        if not bullets and analysisText:
            lines = [l for l in analysisText.splitlines() if l.strip()]
            bullets.extend(lines[:3])
        executiveSummary = {
            "title": "Executive Summary",
            "bullets": bullets,
            "rawAnalysisPreview": (analysisText[:600] + "…") if analysisText and len(analysisText) > 600 else (analysisText or ""),
        }

        level = "Low Risk" if score >= 75 else ("Moderate Risk" if score >= 55 else "High Risk")
        riskAssessment = {
            "title": "Risk Assessment",
            "score": score,
            "level": level,
            "advice": advice,
            "components": comp
        }

        morning = [
            "Hydrate on wake; natural light for 10–15 min.",
            "Gentle movement (5–10 min) to boost circadian alerting."
        ]
        afternoon = [
            "Keep caffeine <200mg total after 14:00.",
            "Short walk or light cardio (10–20 min)."
        ]
        evening = [
            "Wind-down routine 45–60 min pre-bed (dim lights, low screens).",
            "Room cool, dark, and quiet."
        ]
        if score < 55:
            evening.append("Aim for an early lights-out tonight; consider 10–20 min nap before 15:00.")
        energyPlan = {"title":"Daily Energy Plan","morning":morning,"afternoon":afternoon,"evening":evening}

        windows = []
        bed = (current.get("bedTime") or current.get("bed_time") or current.get("bedtime"))
        wake = (current.get("wakeTime") or current.get("wake_time"))
        if isinstance(wake, str) and ":" in wake:
            windows.append({"start": wake, "end": _add_minutes_hhmm(wake, 30), "why": "Maintain consistency (+30m)"})
            windows.append({"start": _add_minutes_hhmm(wake, -30), "end": wake, "why": "Keep earlier rise (-30m)"})
        elif isinstance(bed, str) and ":" in bed:
            target = _add_minutes_hhmm(bed, 7*60 + 30)
            windows.append({"start": _add_minutes_hhmm(target, -15), "end": _add_minutes_hhmm(target, 15), "why": "Target wake window (~7.5h)"})
        else:
            windows.append({"start": "06:30", "end": "07:30", "why": "Default window for 7–8h schedule"})
        wakeWindows = {"title":"Suggested Wake Windows","windows":windows,"note":"Aim for consistent wake time; adjust by ≤30 min when needed."}

        whatIf = []
        for item in (lifestyleCorrelations or [])[:3]:
            label = (item.get("label") or "").strip()
            val = str(item.get("value") or "")
            direction = "reduce" if val.startswith('-') else "increase"
            whatIf.append({
                "title": f"What if I {direction} {label}?",
                "impact": "Likely positive" if direction == "reduce" else "Potentially positive if moderate",
                "note": f"Based on your data correlation: {label} → {val}"
            })
        if not whatIf:
            whatIf.extend([
                {"title":"What if I reduce screens 1h before bed?","impact":"Likely positive","note":"Less blue light improves melatonin onset."},
                {"title":"What if I add a 10-min walk after lunch?","impact":"Positive for sleep drive","note":"Light activity supports sleep pressure."}
            ])
        whatIfScenarios = {"title":"What-If Scenarios","scenarios": whatIf}

        return jsonify({
            "executiveSummary": executiveSummary,
            "riskAssessment": riskAssessment,
            "energyPlan": energyPlan,
            "wakeWindows": wakeWindows,
            "whatIfScenarios": whatIfScenarios
        }), 200

    except Exception as e:
        logger.error("/report failed", exc_info=True)
        return jsonify({"error": str(e), "code": "ReportException"}), 500

# ---------- Health ----------

@app.route("/health", methods=["GET"])
def health_check():
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
            "/lifestyle-correlations (POST)",
            "/insights (POST)",
            "/report (POST)"
        ]
    }
    return jsonify(status)

# ===================== Main =====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    logger.info(f"Starting server on port {port} in {'debug' if debug_mode else 'production'} mode")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)








