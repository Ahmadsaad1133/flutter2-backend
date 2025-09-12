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

def _


