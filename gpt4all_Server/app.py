import json
import os
import logging
import random
import re
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, Blueprint
import requests
from flask_cors import CORS

app = Flask(__name__)

# ----- Optional PDF blueprint (graceful fallback if missing) -----
try:
    from .smart_pdf_service import pdf_bp  # type: ignore
except Exception:
    pdf_bp = Blueprint("pdf", __name__)
    @pdf_bp.route("/", methods=["GET"])
    def _pdf_placeholder():
        return jsonify({"status": "ok", "details": "pdf blueprint placeholder"}), 200

app.register_blueprint(pdf_bp, url_prefix="/pdf")
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("silent-veil")

# ---------------- Config / Keys ----------------
groq_api_key = os.getenv("GROQ_API_KEY")
pixabay_api_key = os.getenv("PIXABAY_API_KEY")
default_timeout = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))

# ========== LLM provider config ==========
# Choose between:
#   - gpt4all  -> local server (OpenAI-compatible)
#   - groq     -> Groq OpenAI-compatible API
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()

# Base URLs
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GPT4ALL_BASE = (os.getenv("LLM_BASE_URL") or "http://localhost:4891/v1").rstrip("/")
GPT4ALL_CHAT_URL = f"{GPT4ALL_BASE}/chat/completions"

# Model name
# Example low-RAM choice for local server: "phi-3-mini-4k-instruct-q4_0"
# Example Groq: "gemma2-9b-it"
LLM_MODEL = (os.getenv("LLM_MODEL") or "gemma2-9b-it").strip()

# Image search (optional)
PIXABAY_SEARCH_URL = "https://pixabay.com/api/"

# ===================== Helpers (shared) =====================

def extract_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ["text", "content", "en", "value"]:
            if key in value and isinstance(value[key], str):
                return value[key]
        return json.dumps(value, ensure_ascii=False)
    return str(value) if value is not None else ""

_MD_CODE_FENCE = re.compile(r"```(?:[\w-]+)?\n([\s\S]*?)```", re.MULTILINE)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_LIST_BULLET = re.compile(r"^\s{0,3}[*\-•]\s*", re.MULTILINE)
_MD_QUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)

def sanitize_plain_text(text: str) -> str:
    """Turn any markdown-ish content into simple plain text."""
    if not text:
        return ""
    t = text
    # Replace code fences with inner content
    t = _MD_CODE_FENCE.sub(lambda m: m.group(1), t)
    # Drop headings markup
    t = _MD_HEADING.sub("", t)
    # Normalize bullets to "- "
    t = _MD_LIST_BULLET.sub("- ", t)
    # Remove block quotes
    t = _MD_QUOTE.sub("", t)
    # Trim excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _clean_llm_analysis_output(txt: str) -> str:
    """
    Remove echoed instructions and any requests for more info. Keep only analysis.
    """
    t = sanitize_plain_text(txt or "")
    # Drop initial 'You are ...' instruction paragraph
    t = re.sub(r'(?is)^\s*you are [^\n]+?\n\s*', '', t, count=1)
    # Remove 'Sections:' header block if present
    t = re.sub(r'(?ims)^\s*sections:\s*(?:.+\n)+\s*', '', t, count=1)
    # Remove 'Please provide...' or similar request paragraphs
    t = re.sub(r'(?is)^\s*please provide[\s\S]*?(?:\n\s*\n|$)', '', t)
    t = re.sub(r'(?is)^\s*i need more (?:context|information)[\s\S]*?(?:\n\s*\n|$)', '', t)
    t = re.sub(r'(?is)^\s*once i have this information[\s\S]*?(?:\n\s*\n|$)', '', t)
    # Remove generic "insufficient information" preambles
    t = re.sub(r'(?is)^\s*(unfortunately|however)[\s\S]*?(?:insufficient|lack of)\s+information[\s\S]*?(?:\n\s*\n|$)', '', t)
    # Remove trailing enumerations about what *I can do* rather than analysis
    t = re.sub(r'(?ims)^\s*(?:once i have|i can)[:\s][\s\S]*?(?:\n\s*\n|$)', '', t)
    # Normalize section headings if present (keep content)
    t = re.sub(r'(?im)^\s*(symptom analysis|clinical assessment|treatment recommendations|personalized recommendations)\s*:?\s*$', r'\1:', t)
    # Trim excessive blank lines
    t = re.sub(r'\n{3,}', '\n\n', t).strip()
    return t


# ---------- Ultra-clean text utilities (frontend-grade cleaning, now server-side) ----------

_HTML_ENTITY_DEC = re.compile(r"&(#?)(x?)(\w+);")

def _decode_html_entities(s: str) -> str:
    def _sub(m):
        lead, hexflag, word = m.groups()
        if not lead:
            named = {
                "nbsp": " ", "amp": "&", "quot": "\"", "apos": "'", "lt": "<", "gt": ">",
                "ndash": "-", "mdash": "—", "hellip": "…",
            }
            return named.get(word, m.group(0))
        try:
            if hexflag:
                code = int(word, 16)
            else:
                code = int(word, 10)
            return chr(code)
        except Exception:
            return m.group(0)
    return _HTML_ENTITY_DEC.sub(_sub, s)

_ZERO_WIDTH = "".join(["\u200B","\u200C","\u200D","\u200E","\u200F","\uFEFF"])
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

def _strip_zero_width_and_controls(s: str) -> str:
    if not s:
        return s
    for ch in _ZERO_WIDTH:
        s = s.replace(ch, "")
    return _CTRL_CHARS.sub("", s)

def _strip_markdown_and_json_cruft(s: str) -> str:
    t = s or ""
    # Remove fenced code blocks ```lang\n...\n```
    t = re.sub(r"```[\w-]*\n([\s\S]*?)```", lambda m: m.group(1), t, flags=re.MULTILINE)
    # Inline code
    t = re.sub(r"`([^`]+)`", lambda m: m.group(1), t)
    # Headings
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.MULTILINE)
    # Blockquotes
    t = re.sub(r"^\s{0,3}>\s?", "", t, flags=re.MULTILINE)
    # Tables / pipes lines
    t = re.sub(r"^\s*\|.*\|\s*$", "", t, flags=re.MULTILINE)
    # Horizontal rules
    t = re.sub(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$", "", t, flags=re.MULTILINE)
    # Normalize list markers to "- "
    t = re.sub(r"^\s{0,3}[•*\-]\s+", "- ", t, flags=re.MULTILINE)
    # Remove long inline JSON dumps on single lines
    t = re.sub(r"(?:^|\n)[\s\t]*[\{\[][^\n]{120,}[\}\]]", "", t, flags=re.MULTILINE)
    # Remove leading key: value at line start (keeps value)
    t = re.sub(r'^[ \t]*(?:"|\')?[A-Za-z0-9_ .\-]+(?:"|\')?\s*:\s*', "", t, flags=re.MULTILINE)
    # Orphan braces/brackets/commas lines
    t = re.sub(r'^[ \t]*[\}\]\[,]+[ \t]*$', "", t, flags=re.MULTILINE)
    # Collapse leftover braces/brackets
    t = re.sub(r'[\{\}\[\]]+', "", t)
    return t

def _normalize_whitespace_punct(s: str) -> str:
    t = s.replace("\r\n", "\n").replace("\r", "\n")
    # Trim trailing spaces
    t = re.sub(r"[ \t]+$", "", t, flags=re.MULTILINE)
    # Collapse 3+ newlines to 2
    t = re.sub(r"\n{3,}", "\n\n", t)
    # De-duplicate punctuation
    t = re.sub(r"([.!?…])\1{2,}", r"\1", t)
    # Collapse multiple spaces
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

def ultra_clean_text(text: str) -> str:
    """Aggressive cleaner: decode entities, strip zero-width/control, remove markdown/JSON noise, normalize spacing."""
    if not text:
        return ""
    t = text
    # Deep-unescape visible escapes like \\n, \\t
    t = t.replace("\\\\n", "\n").replace("\\\\t", "  ").replace("\\\\r", "")
    # Unicode escapes like \\u2014
    t = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), t)
    # HTML entities
    t = _decode_html_entities(t)
    # Remove zero-width + control
    t = _strip_zero_width_and_controls(t)
    # Strip markdown/json cruft
    t = _strip_markdown_and_json_cruft(t)
    t = _strip_emphasis(t)
    t = _strip_weird_quotes_around_tiny_tokens(t)
    t = _fix_weird_punctuation(t)
    # Apply existing plain-text sanitizer for any remaining MD
    t = sanitize_plain_text(t)
    # Remove garbage lines
    lines = []
    for raw in t.split("\n"):
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low in ("null", "none") or line in ("{}", "[]"):
            continue
        if re.fullmatch(r"[,.:;~`^=+_()<>|\\/]+", line):
            continue
        if re.fullmatch(r"[-•]\s*", line):
            continue
        lines.append(line)
    t = "\n".join(lines)
    # Normalize spacing
    t = _normalize_whitespace_punct(t)
    return t


def _strip_emphasis(s: str) -> str:
    s = re.sub(r'(\*\*|__)(.*?)\1', lambda m: m.group(2), s, flags=re.S)
    s = re.sub(r'(?<!\w)\*(?!\s)([^*]+)\*(?!\w)', lambda m: m.group(1), s)
    s = re.sub(r'(?<!\w)_(?!\s)([^_]+)_(?!\w)', lambda m: m.group(1), s)
    s = re.sub(r'^\*\s*', '', s, flags=re.M)
    s = re.sub(r'\*\s*\)', ')', s)
    return s

def _strip_weird_quotes_around_tiny_tokens(s: str) -> str:
    return re.sub(r'[“”"\'`]([^\w\s]{1,2})[“”"\'`]', r'\1', s)

def _fix_weird_punctuation(s: str) -> str:
    x = s
    x = re.sub(r':\.', '.', x)
    x = re.sub(r':\s*$', '', x, flags=re.M)
    x = re.sub(r'([.!?…])\s*([.!?…])+', r'\1', x)
    x = re.sub(r'\s+([,.:;!?\)])', r'\1', x)
    x = re.sub(r'([(\[])\s+', r'\1', x)
    return x
def dedupe_bullets(items):
    """Return a list of cleaned, unique bullet strings (case-insensitive)."""
    seen = set()
    out = []
    for x in items or []:
        s = ultra_clean_text(str(x))
        s = s.strip(" -•\u2022")
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        # Ensure bullet punctuation
        if len(s) >= 3 and not re.search(r"[.!?…]$", s):
            s += "."
        out.append(f"• {s}")
    return out[:6]




def _is_weak_analysis(txt: str) -> bool:
    if not txt:
        return True
    low = txt.lower()
    triggers = [
        "please provide", "need more information", "need more context",
        "once i have this information", "insufficient information",
        "cannot provide personalized recommendations", "general sleep hygiene tips"
    ]
    if any(t in low for t in triggers):
        return True
    return len(txt.strip()) < 140

def extract_json_block(text):
    """Extract the first top-level JSON object/array in text and parse it. Returns Python obj or None."""
    if not text:
        return None
    brace = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", text, re.DOTALL)
    brack = re.search(r"\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]", text, re.DOTALL)
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

def clean_json_output(json_text):
    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            content = parsed.get("content", "")
            if isinstance(content, dict):
                parsed["content"] = json.dumps(content, indent=2, ensure_ascii=False)
        return parsed
    except Exception:
        return {"raw": json_text}

# ========== Provider-agnostic LLM caller ==========
def call_llm(
    user_prompt,
    *,
    json_mode=False,
    temperature=0.6,
    max_tokens=1200,
    system_msg="You are Silent Veil, a calm sleep coach assistant."
):
    """
    Calls either GPT4All local server or Groq (OpenAI-compatible) based on env.

    - LLM_PROVIDER=gpt4all -> http://localhost:4891/v1/chat/completions
    - LLM_PROVIDER=groq    -> https://api.groq.com/openai/v1/chat/completions

    If provider is 'gpt4all' and the request fails, and GROQ_API_KEY is present,
    we automatically retry once with Groq.
    """
    try:
        # ---------- Build messages ----------
        messages = [{"role": "system", "content": system_msg}]
        if json_mode:
            user_msg = user_prompt.rstrip() + "\nReturn ONLY JSON. No prose, no markdown, no headings, no backticks."
        else:
            user_msg = user_prompt
        messages.append({"role": "user", "content": user_msg})

        # ---------- Build payload ----------
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        # ---------- Provider candidates (with failover) ----------
        candidates = []
        if LLM_PROVIDER == "groq":
            if not groq_api_key:
                return None, "Missing GROQ_API_KEY"
            candidates.append(("groq", GROQ_CHAT_URL, {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}))
        elif LLM_PROVIDER == "gpt4all":
            candidates.append(("gpt4all", GPT4ALL_CHAT_URL, {"Content-Type": "application/json"}))
            if groq_api_key:
                candidates.append(("groq", GROQ_CHAT_URL, {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}))
        else:
            # Unknown value: try groq (if key) then gpt4all
            if groq_api_key:
                candidates.append(("groq", GROQ_CHAT_URL, {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}))
            candidates.append(("gpt4all", GPT4ALL_CHAT_URL, {"Content-Type": "application/json"}))

        errors = []
        for name, url, headers in candidates:
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=default_timeout)
            except Exception as e:
                errors.append(f"{name} exception: {e}")
                continue

            if res.status_code != 200:
                errors.append(f"{name} HTTP {res.status_code}: {res.text[:300]}")
                continue

            data = res.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if not content:
                errors.append(f"{name} empty content")
                continue

            if json_mode:
                try:
                    parsed = extract_json_block(content) or json.loads(content)
                except Exception:
                    errors.append(f"{name} non-JSON content")
                    continue
                return parsed, None

            return sanitize_plain_text(content.strip()), None

        tip = " Tip: on Render, set LLM_PROVIDER=groq and add GROQ_API_KEY." if any(c[0] == "gpt4all" for c in candidates) else ""
        return None, ("; ".join(errors) or "Unknown LLM error") + tip

    except Exception as e:
        return None, f"LLM call failed: {str(e)}"

# Backward-compatible shim (existing code may call this name)
def call_groq(*args, **kwargs):
    return call_llm(*args, **kwargs)

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

# ---------- Legacy shim for old UIs ----------
def _legacy_report_shim(resp):
    """
    Produce old/snake_case + alternative shapes so any UI will render something.
    """
    legacy = {}

    # ---------- Executive Summary ----------
    es = resp.get("executiveSummary") or {}
    es_bullets = es.get("bullets") or []
    # Prefer full text if available; fallback to preview
    es_text = es.get("text") or es.get("fullText") or es.get("rawAnalysisPreview") or ""
    if not es_text and es_bullets:
        es_text = "\n".join([str(b) for b in es_bullets if str(b).strip()])
    legacy["executive_summary"] = {
        "bullets": es_bullets,
        "text": es_text
    }
    legacy["summary"] = es_text

    # ---------- Risk Assessment ----------
    ra = resp.get("riskAssessment") or {}
    comps = ra.get("components") or {}
    hotspots = []
    try:
        worst = sorted(
            [(k, float(v)) for k, v in comps.items() if isinstance(v, (int, float))],
            key=lambda kv: kv[1]
        )
        hotspots = [k.replace("_", " ").title() for k, v in worst if v < 50][:4]
    except Exception:
        pass

    legacy["risk_assessment"] = {
        "score": ra.get("score"),
        "level": ra.get("level"),
        "advice": ra.get("advice"),
        "components": comps,
        "hotspots": hotspots
    }

    # ---------- Energy Plan ----------
    ep = resp.get("energyPlan") or {}
    morning = ep.get("morning") or []
    afternoon = ep.get("afternoon") or []
    evening = ep.get("evening") or []
    flat_plan = []
    for label, items in (("morning", morning), ("afternoon", afternoon), ("evening", evening)):
        for a in items:
            flat_plan.append({
                "time_range": label,
                "timeRange": label,
                "action": str(a),
                "rationale": ""
            })
    energy_tip = ""
    for bucket in (evening, afternoon, morning):
        if bucket:
            energy_tip = str(bucket[0])
            break

    legacy["energy_plan"] = {
        "advice": energy_tip,
        "plan": flat_plan
    }

    # ---------- Wake Windows ----------
    ww = resp.get("wakeWindows") or {}
    windows = ww.get("windows") or []
    legacy["wake_windows"] = windows

    # ---------- What-If Scenarios ----------
    wis = resp.get("whatIfScenarios") or {}
    scenarios = wis.get("scenarios") or []
    legacy["what_if_scenarios"] = scenarios

    # ---------- Duplicate camelCase containers some old builds look for ----------
    legacy["executiveSummary"] = legacy["executive_summary"]
    legacy["riskAssessment"] = legacy["risk_assessment"]
    legacy["energyPlan"] = legacy["energy_plan"]
    legacy["wakeWindows"] = {"windows": legacy["wake_windows"]}
    legacy["whatIfScenarios"] = {"scenarios": legacy["what_if_scenarios"]}

    return legacy

# ===================== Routes =====================

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Silent Veil backend is online 💤",
        "llm_provider": LLM_PROVIDER,
        "model": LLM_MODEL,
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
            "/report"
        ]
    }), 200

# ---------- Chat ----------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="Missing 'prompt'"), 400
    content, err = call_llm(prompt, json_mode=False)
    if err:
        return jsonify(error=err), 502
    return jsonify(response=content)

# ---------- Generate (plain) ----------

@app.route("/generate", methods=["POST"])
def generate_story():
    data = request.get_json(silent=True) or {}
    mood = (data.get("mood") or "").strip()
    sleep_quality = (data.get("sleep_quality") or "").strip()
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400
    prompt = (
        f"You are Silent Veil, a calm sleep coach. Based on mood '{mood}' "
        f"and sleep quality '{sleep_quality}', create a calming bedtime story. "
        "Return only the story text, no JSON or formatting."
    )
    story, err = call_llm(prompt, json_mode=False)
    if err:
        return jsonify(error=err), 502
    return jsonify(story=story)

# ---------- Generate (multiple stories with images) ----------

@app.route("/generate-stories", methods=["POST"])
def generate_stories():
    data = request.get_json(silent=True) or {}
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
        parsed, err = call_llm(prompt, json_mode=True)
        if err:
            continue
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
        return jsonify(error="Failed to generate any stories"), 502
    return jsonify(stories=stories)

# ---------- Generate (single story + image) ----------

@app.route("/generate-story-and-image", methods=["POST"])
def generate_story_and_image():
    data = request.get_json(silent=True) or {}
    mood = (data.get("mood") or "").strip()
    sleep_quality = (data.get("sleep_quality") or "").strip()
    if not mood or not sleep_quality:
        return jsonify(error="Missing 'mood' or 'sleep_quality'"), 400
    prompt = (
        f"You are Silent Veil. Based on mood '{mood}' and sleep quality '{sleep_quality}', "
        "create a calming bedtime story. Respond in JSON with: title, description, content. "
        "All values must be plain strings. No markdown or nested data."
    )
    parsed, err = call_llm(prompt, json_mode=True)
    if err:
        return jsonify(error=err), 502
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
    data = request.get_json(silent=True) or {}
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
                "You are Dr. Somnus, a board-certified sleep specialist. Analyze this quantitative sleep data.\n"
                "Tasks:\n"
                "1) Calculate sleep efficiency = (TST / TIB) × 100 (if not provided);\n"
                "2) Assess sleep continuity metrics;\n"
                "3) Compare against AASM clinical thresholds;\n"
                "4) Identify potential sleep disorders;\n"
                "5) Provide evidence-based recommendations.\n\n"
                "Output: Write only your analysis (no requests for more data). "
                "If data is missing, infer with clear assumptions.\n\n"
                f"DATA_JSON = {json.dumps(sleep_data, ensure_ascii=False)}"
            )
        else:
            symptoms = sleep_data.get("symptoms", [])
            if not symptoms:
                symptoms = [str(v) for v in sleep_data.values() if isinstance(v, (str, int, float))]
            if not symptoms:
                return jsonify(error="No symptoms provided", code="SleepAnalysisException"), 400
            prompt = (
                "You are Dr. Somnus, a board-certified sleep specialist. "
                "Analyze these patient-reported symptoms.\n"
                "Tasks:\n"
                "1) Identify potential sleep disorders (ICSD-3);\n"
                "2) Relate symptoms to physiological causes;\n"
                "3) Provide clinical recommendations.\n\n"
                "Output: Write only your analysis (no requests for more data). "
                "If details are missing, infer with clear caveats.\n\n"
                f"SYMPTOMS_TXT = {', '.join(symptoms)}"
            )

        text, err = call_llm(prompt, json_mode=False)
        cleaned = _clean_llm_analysis_output(text) if text else ""
        if _is_weak_analysis(cleaned):
            # Guard-rail: return a concise rule-based analysis instead of a meta request
            se = None
            try:
                TST = float(sleep_data.get("TST", 0) or 0)
                TIB = float(sleep_data.get("TIB", 0) or 0)
                if TST > 0 and TIB > 0:
                    se = round((TST / TIB) * 100, 1)
            except Exception:
                pass
            recs = [
                "Keep a stable wake time (±30m).",
                "Avoid caffeine after 14:00 and dim screens 60m pre-bed.",
                "Add 10–20m light activity; wind-down 45–60m."
            ]
            base = "Clinical Summary: Limited fields provided; giving best-effort guidance."
            if se is not None:
                base += f" Estimated sleep efficiency ≈ {se}%."
            cleaned = base + " Recommendations: " + " ".join(recs)
        return jsonify(analysis=cleaned)
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
            m1 = {
                "sleep_score":"sleep_score",
                "duration_minutes":"duration_minutes",
                "deep":"deep_sleep_minutes",
                "rem":"rem_sleep_minutes",
                "light":"light_sleep_minutes",
                "quality":"quality",
                "stress":"stress_level",
                "caffeine":"caffeine_intake",
                "exercise":"exercise_minutes",
                "screen":"screen_time_before_bed"
            }
            m2 = {
                "sleep_score":"sleepScore",
                "duration_minutes":"durationMinutes",
                "deep":"deepSleepMinutes",
                "rem":"remSleepMinutes",
                "light":"lightSleepMinutes",
                "quality":"sleepQuality",
                "stress":"stressLevel",
                "caffeine":"caffeineIntake",
                "exercise":"exerciseMinutes",
                "screen":"screenTimeBeforeBed"
            }
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
            f"SUMMARY_JSON = {json.dumps(summary, ensure_ascii=False)}"
        )
        parsed, err = call_llm(prompt, json_mode=True)
        highlights = []
        if isinstance(parsed, list):
            for item in parsed[:6]:
                if not isinstance(item, dict): continue
                title = extract_text(item.get("title",""))
                value = extract_text(item.get("value",""))
                change = extract_text(item.get("change","flat")).lower()
                if change not in ("up","down","flat"): change = "flat"
                insight = extract_text(item.get("insight",""))
                title = title.strip()[:60]; value = value.strip()[:40]; insight = insight.strip()[:160]
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
        advice = (
            "Solid recovery ahead. Keep caffeine <200mg after 14:00 and aim for 7–8h sleep."
            if score >= 75 else
            "Moderate recovery. Prioritize 7.5h sleep, light cardio, and earlier wind-down."
            if score >= 55 else
            "Take it easy today. Short naps, hydration, and gentle movement recommended."
        )
        return jsonify({"score": score, "components": comp, "advice": advice})
    except Exception as e:
        logger.error("/readiness failed", exc_info=True)
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
                r, _ = _pearson_corr(x, series_o[chosen_outcome])
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

        # 2) Highlights (try LLM, fallback rules)
        summary = _summarize_logs_for_highlights(history if isinstance(history, list) else [])
        prompt_h = (
            "You are Silent Veil, an expert sleep coach. Given the following numeric summary of a user's recent sleep logs, "
            "produce 4–6 concise highlights strictly in JSON array format. Each item must be an object with keys:\n"
            "title (<= 40 chars), value (short stat string), change (one of: up, down, flat), insight (<= 140 chars).\n\n"
            "No markdown, no extra text.\n\n"
            f"SUMMARY_JSON = {json.dumps(summary, ensure_ascii=False)}"
        )
        parsed_h, err_h = call_llm(prompt_h, json_mode=True)
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

        # 3) Readiness FIRST (so we can synthesize analysis if needed)
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
        advice = (
            "Solid recovery ahead. Keep caffeine <200mg after 14:00 and aim for 7–8h sleep."
            if score >= 75 else
            "Moderate recovery. Prioritize 7.5h sleep, light cardio, and earlier wind-down."
            if score >= 55 else
            "Take it easy today. Short naps, hydration, and gentle movement recommended."
        )
        level = "Low Risk" if score >= 75 else ("Moderate Risk" if score >= 55 else "High Risk")

        # 4) LLM analysis (after readiness), with constraints and cleaner
        def _build_sleep_analysis_text(current_obj):
            try:
                quantitative_keys = ['TST', 'TIB', 'SE', 'SOL', 'WASO', 'AHI', 'sleep_efficiency']
                is_quant = any(k in current_obj for k in quantitative_keys)
                if is_quant:
                    prompt = (
                        "You are Dr. Somnus, a board-certified sleep specialist. Analyze this quantitative sleep data.\n"
                        "Tasks: 1) Compute sleep efficiency if missing; 2) Assess continuity (SOL, WASO); "
                        "3) Compare to AASM thresholds; 4) Identify possible disorders; 5) Give recommendations.\n"
                        "Constraints: Write only analysis; do not ask for more data. If fields are missing, infer with assumptions.\n\n"
                        f"DATA_JSON = {json.dumps(current_obj, ensure_ascii=False)}"
                    )
                else:
                    symptoms = current_obj.get("symptoms", [])
                    if not symptoms:
                        symptoms = [str(v) for v in current_obj.values() if isinstance(v, (str, int, float))]
                    if not symptoms:
                        return ""
                    prompt = (
                        "You are Dr. Somnus, a board-certified sleep specialist. Analyze these reported symptoms.\n"
                        "Tasks: 1) Likely ICSD-3 differentials; 2) Physiological links; 3) Actionable plan.\n"
                        "Constraints: Write only analysis; do not ask for more data. If details are missing, infer with caveats.\n\n"
                        f"SYMPTOMS_TXT = {', '.join(symptoms)}"
                    )
                txt, er = call_llm(prompt, json_mode=False)
                return _clean_llm_analysis_output(txt) if txt else ""
            except Exception:
                return ""

        analysisText = _build_sleep_analysis_text(current)

        # Guard: if weak, synthesize a concrete clinical summary (never meta)
        if _is_weak_analysis(analysisText):
            dur_pct = round(comp.get('duration', 0))
            eff_pct = round(comp.get('efficiency', 0))
            stress_pct = round(comp.get('stress', 0))
            caf_pct = round(comp.get('caffeine', 0))
            ex_pct = round(comp.get('exercise', 0))
            analysisText = (
                f"Clinical Summary: Readiness {score}/100 ({level}). "
                f"Key levers — duration {dur_pct}%, efficiency {eff_pct}%, stress {stress_pct}%. "
                f"Caffeine {caf_pct}% of target, exercise {ex_pct}% of goal.\n\n"
                "Recommendations:\n"
                "• Keep a stable wake time (±30m) and target 7–8h asleep.\n"
                "• Avoid caffeine after 14:00; reduce screens for 60m pre-bed.\n"
                "• Add 10–20m light activity; wind-down 45–60m with dim light."
            )

        # ---------- Compose sections ----------

        bullets = []
        # Build bullets from highlights (title — value. insight)
        for h in highlights[:4]:
            title = (h.get('title') or '').strip()
            value = (h.get('value') or '').strip()
            insight = (h.get('insight') or '').strip()
            parts = [t for t in [title, value] if t]
            merged = (' — '.join(parts) + ('. ' if parts else '')) + insight
            bullets.append(merged)
        # Fallback to first lines of analysis if empty
        if not [b for b in bullets if b.strip()] and analysisText:
            lines = [l for l in ultra_clean_text(analysisText).splitlines() if l.strip()]
            bullets.extend(lines[:3])
        bullets = dedupe_bullets(bullets)


        executiveSummary = {
            "title": "Executive Summary",
            "bullets": bullets,
            "fullText": ultra_clean_text(analysisText),
            "text": ultra_clean_text(analysisText),
            "rawAnalysisPreview": (
                ultra_clean_text(analysisText)[:600] + "…"
                if analysisText and len(analysisText) > 600
                else ultra_clean_text(analysisText)
            ),
            "highlights": highlights,
        }

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

        # ------ Build final response and add legacy shim ------
        resp = {
            "executiveSummary": executiveSummary,
            "riskAssessment": riskAssessment,
            "energyPlan": energyPlan,
            "wakeWindows": wakeWindows,
            "whatIfScenarios": whatIfScenarios,
            "lifestyleCorrelations": lifestyleCorrelations,
            "summary": ultra_clean_text(executiveSummary.get("text","")),
            "analysisText": ultra_clean_text(executiveSummary.get("text","")),
            "fullAnalysis": ultra_clean_text(executiveSummary.get("text",""))
        }

        
        
        # ------ Charts (for frontend visualization) ------
        try:
            # Helper: normalize percent if values look like 0..1
            def _normalize_percent_map(mdict):
                items = []
                for k, v in (mdict or {}).items():
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    items.append((k, val))
                if not items:
                    return []
                # If max <= 1.5 and any > 0, treat as ratio -> percent
                maxv = max(v for _, v in items)
                scale = 100.0 if (maxv > 0 and maxv <= 1.5) else 1.0
                out = [{"label": k.replace("_", " ").title(), "value": float(v) * scale} for k, v in items]
                out.sort(key=lambda d: d["value"], reverse=True)
                return out[:6]

            # 1) Components bars (0-100)
            comp_items = _normalize_percent_map(comp)

            # 2) Stages donut from current log (accept many possible keys)
            def _get_stage_minutes(m, key_variants):
                for kv in key_variants:
                    if isinstance(kv, (list, tuple)):
                        cur = m
                        ok = True
                        for seg in kv:
                            if isinstance(cur, dict) and seg in cur:
                                cur = cur[seg]
                            else:
                                ok = False; break
                        if ok:
                            try:
                                return float(cur) or 0.0
                            except Exception:
                                pass
                    else:
                        if kv in m:
                            try:
                                return float(m.get(kv) or 0.0)
                            except Exception:
                                pass
                return 0.0

            cur = current if isinstance(current, dict) else {}
            # broaden key search (minutes, mins, min, or percent that we convert using duration)
            def _stage_from_many(m, names):
                # minutes first
                val = _get_stage_minutes(m, [
                    f"{names}_sleep_minutes", f"{names}SleepMinutes",
                    ["stages", f"{names}Minutes"], ["sleep", "stages", names], names+"_minutes", names+"Minutes",
                    names+"Min", names+"_min", ["stages", names], ["sleepStages", names+"Minutes"]
                ])
                if val and val > 0:
                    return val
                # percent -> convert using duration if available
                pct = _get_stage_minutes(m, [
                    names+"_pct", names+"Pct", names+"Percent", names+"_percent", ["stages", names+"Pct"],
                ])
                if pct and duration:
                    try:
                        return float(duration) * 60.0 * (float(pct)/100.0)
                    except Exception:
                        return 0.0
                return 0.0

            deep  = _stage_from_many(cur, "deep")
            rem   = _stage_from_many(cur, "rem")
            light = _stage_from_many(cur, "light")
            stages = [{"label":"Deep","value": deep}, {"label":"REM","value": rem}, {"label":"Light","value": light}]
            # Filter zeros but keep at least one
            if sum(v["value"] for v in stages) <= 0:
                stages = [{"label":"Sleep","value": max(1.0, float(duration)*60.0 if duration else 1.0)}]

            # 3) Duration trend sparkline from history (last 7)
            series = []
            if isinstance(history, list) and history:
                def _extract_date(o):
                    for key in ("date","day","logDate","startDate","sleepDate"):
                        if isinstance(o.get(key), str):
                            return o.get(key)
                    for k2 in ("bedtime","bedTime","sleepStart","sleep_start"):
                        v = o.get(k2)
                        if isinstance(v, str):
                            return v[:5]
                    return f"#{len(series)+1}"
                for item in history[-7:]:
                    d = _num_or_0(item.get("duration_minutes") or item.get("durationMinutes") or item.get("totalSleepMinutes") or item.get("duration"))
                    if d <= 0:
                        bed_iso = item.get("bedTime") or item.get("bed_time") or item.get("sleepStart") or item.get("sleep_start")
                        wake_iso = item.get("wakeTime") or item.get("wake_time") or item.get("sleepEnd") or item.get("sleep_end")
                        if bed_iso and wake_iso:
                            d = float(_minutes_between(str(bed_iso), str(wake_iso)))
                        else:
                            bed = item.get("bedtime"); wake = item.get("wake_time")
                            if isinstance(bed, str) and isinstance(wake, str) and ":" in bed and ":" in wake:
                                b = _hm_to_minutes(bed); w = _hm_to_minutes(wake)
                                d = float(w - b) if w > b else float((24*60 - b) + w)
                    series.append({"label": _extract_date(item), "value": float(d)})
            if not series:
                series = [{"label":"", "value": float(duration)*60.0 if duration else 0.0}]

            # 4) Lifestyle bar chart (correlations -1..+1)
            lifestyle_bar = []
            for it in (lifestyleCorrelations or []):
                try:
                    lbl = (it.get("label") or str(it.get("key") or "")).strip() or "Factor"
                    val = float(it.get("value") or 0.0)
                    lifestyle_bar.append({"label": lbl, "value": val})
                except Exception:
                    continue
            lifestyle_bar = lifestyle_bar[:6]

            # 5) Doctor note (friendly, short) derived from analysis
            doctor_note = executiveSummary.get("text") or executiveSummary.get("fullText") or ""
            doctor_note = ultra_clean_text(doctor_note)[:800]

            resp_charts = {
                "componentsBars": comp_items,
                "stagesDonut": stages,
                "durationTrend": series,
                "lifestyleBar": lifestyle_bar,
                "doctorNote": doctor_note
            }
            resp["charts"] = resp_charts
        except Exception as _e:
            # non-fatal
            resp["charts"] = resp.get("charts", {})
        
        
        resp.update(_legacy_report_shim(resp))

        return jsonify(resp), 200

    except Exception as e:
        logger.error("/report failed", exc_info=True)
        return jsonify({"error": str(e), "code": "ReportException"}), 500

# ---------- Health ----------

@app.route("/health", methods=["GET"])
def health_check():
    status = {
        "status": "ok",
        "services": {
            "llm_provider": LLM_PROVIDER,
            "model": LLM_MODEL,
            "groq_key": "present" if bool(groq_api_key) else "absent",
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
    logger.info(f"Starting server on port {port} in {'debug' if debug_mode else 'production'} mode | LLM_PROVIDER={LLM_PROVIDER} | MODEL={LLM_MODEL}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)


