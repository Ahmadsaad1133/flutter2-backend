import os
import re
import json
import base64
from datetime import datetime
from io import BytesIO
from html import escape as _escape
from typing import Any, Dict, List, Optional

import requests
from flask import Blueprint, request, jsonify, send_file
from jinja2 import Template
from weasyprint import HTML, CSS

# -------------------------------------------------
# Blueprint
# -------------------------------------------------
pdf_bp = Blueprint("pdf_bp", __name__)

# ----------------------------- Config -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GROQ_URL     = os.getenv("GROQ_URL",   "https://api.groq.com/openai/v1/chat/completions")

# ----------------------------- Small utils -----------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _num(x, default=0.0):
    try:
        if x is None: return float(default)
        return float(str(x))
    except Exception:
        return float(default)

def _clamp(v, lo, hi):
    try:
        v = float(v)
    except Exception:
        return lo
    return max(lo, min(hi, v))

def _is_jsonish(s: str) -> bool:
    sx = s.strip()
    return (sx.startswith("{") and sx.endswith("}")) or (sx.startswith("[") and sx.endswith("]"))

def _clean_text(txt: Any) -> str:
    """
    Convert any JSON-ish / markdown-ish / noisy content into clean, plain text.
    - If it looks like JSON, flatten nicely.
    - Strip braces/quotes/markdown fences; collapse whitespace.
    """
    if txt is None:
        return ""
    s = str(txt)

    # Try to flatten plain JSON for human readability
    if _is_jsonish(s):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                parts = []
                for k, v in parsed.items():
                    if isinstance(v, (dict, list)):
                        parts.append(f"{k}: " + _clean_text(json.dumps(v, ensure_ascii=False)))
                    else:
                        parts.append(f"{k}: {v}")
                return "  ".join(parts)
            if isinstance(parsed, list):
                return " • ".join(_clean_text(x) for x in parsed)
        except Exception:
            pass

    # Strip code fences
    if s.strip().startswith("```"):
        s = re.sub(r"^```[\w-]*\s*|\s*```$", "", s, flags=re.MULTILINE)

    # Drop markdown bullets/quotes/headings
    s = re.sub(r"^\s{0,3}[*\-•]\s*", "", s, flags=re.MULTILINE)  # bullets
    s = re.sub(r"^\s{0,3}>\s?", "", s, flags=re.MULTILINE)       # blockquotes
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.MULTILINE)  # headings

    # Remove braces/quotes; collapse whitespace
    s = re.sub(r"[{}\[\]\"]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _escape_lines(txt: str) -> str:
    """Escape then keep line breaks as <br/>."""
    return _escape(txt).replace("\n", "<br/>")

def _data_uri_from_bytes(content: bytes, mime: str) -> str:
    b64 = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _guess_mime(url: str) -> str:
    u = (url or "").lower()
    if u.endswith(".png"):  return "image/png"
    if u.endswith(".jpg") or u.endswith(".jpeg"): return "image/jpeg"
    if u.endswith(".svg"):  return "image/svg+xml"
    return "application/octet-stream"

def _fetch_image_as_data_uri(path_or_url: str) -> Optional[str]:
    if not path_or_url:
        return None
    if path_or_url.startswith("data:"):
        return path_or_url
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        try:
            r = requests.get(path_or_url, timeout=15)
            r.raise_for_status()
            mime = r.headers.get("Content-Type") or _guess_mime(path_or_url)
            return _data_uri_from_bytes(r.content, mime)
        except Exception:
            return None
    if os.path.exists(path_or_url):
        try:
            with open(path_or_url, "rb") as f:
                data = f.read()
            return _data_uri_from_bytes(data, _guess_mime(path_or_url))
        except Exception:
            return None
    return None

# --------------------- SVG Chart Generators ---------------------
def _svg_bar_chart(title: str, series: List[Dict[str, Any]], width: int = 700, height: int = 280) -> str:
    padding = 24
    chart_w = width - padding * 2
    chart_h = height - padding * 2 - 24
    values = [float(s.get("value", 0) or 0) for s in series] or [0.0]
    max_v = max(values) or 1.0
    bar_w = chart_w / max(1, len(series))
    x0 = padding
    y0 = padding + 20
    rects, labels = [], []
    for i, s in enumerate(series):
        v = float(s.get("value", 0) or 0)
        h = 0 if max_v <= 0 else (v / max_v) * (chart_h - 20)
        x = x0 + i * bar_w + 6
        y = y0 + (chart_h - h)
        rects.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-12:.1f}" height="{h:.1f}" rx="6" />')
        labels.append(f'<text x="{x + (bar_w-12)/2:.1f}" y="{y0 + chart_h + 16:.1f}" font-size="11" text-anchor="middle">{_escape(str(s.get("label","")))}</text>')
    grid = []
    for gy in range(5):
        y = y0 + gy * (chart_h/4)
        grid.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+chart_w}" y2="{y:.1f}" stroke="#ddd" stroke-dasharray="3,3" />')
    title_el = f'<text x="{width/2:.1f}" y="{padding+8:.1f}" font-size="14" font-weight="600" text-anchor="middle">{_escape(title)}</text>'
    return f"""    <svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#2C7BE5"/><stop offset="100%" stop-color="#6C63FF"/>
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill="white"/>
      {title_el}
      <g fill="url(#g1)" stroke="none">{''.join(grid)}{''.join(rects)}</g>
      <g fill="#333">{''.join(labels)}</g>
    </svg>"""

def _svg_donut_chart(title: str, items: List[Dict[str, Any]], width: int = 260, height: int = 260) -> str:
    total = sum(float(i.get("value", 0) or 0) for i in items) or 1.0
    cx, cy, r = width/2, height/2, min(width, height)/2 - 16
    stroke_w = 28
    start_angle = -90
    arcs, legend = [], []
    colors = ["#2C7BE5","#6C63FF","#00C9A7","#F7B924","#FF6F61","#A78BFA","#22D3EE"]
    import math
    for idx, item in enumerate(items):
        val = float(item.get("value", 0) or 0)
        pct = val / total
        ang = pct * 360.0
        end_angle = start_angle + ang
        large_arc = 1 if ang > 180 else 0
        def pt(a):
            rad = math.radians(a); return cx + r*math.cos(rad), cy + r*math.sin(rad)
        x1,y1 = pt(start_angle); x2,y2 = pt(end_angle)
        color = colors[idx % len(colors)]
        arcs.append(f'<path d="M{x1:.2f},{y1:.2f} A{r:.2f},{r:.2f} 0 {large_arc} 1 {x2:.2f},{y2:.2f}" stroke="{color}" stroke-width="{stroke_w}" fill="none" />')
        legend.append(f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;"><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;"></span><span style="font-size:12px;color:#333">{_escape(str(item.get("label","")))} — {round(pct*100)}%</span></div>')
        start_angle = end_angle
    title_el = f'<div style="text-align:center;font-size:14px;font-weight:600;margin-bottom:8px">{_escape(title)}</div>'
    center_text = f'<text x="{cx:.1f}" y="{cy+5:.1f}" font-size="14" text-anchor="middle" fill="#333">100%</text>'
    return f"""    <div style="display:flex;gap:16px;align-items:center;justify-content:center;">
      <div>
        {title_el}
        <svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
          <circle cx="{cx}" cy="{cy}" r="{r}" fill="#f9fafb"/>{''.join(arcs)}{center_text}
        </svg>
      </div>
      <div>{''.join(legend)}</div>
    </div>"""

def _svg_line_chart(title: str, series: List[Dict[str, Any]], width: int = 700, height: int = 260) -> str:
    padding = 30
    chart_w = width - padding * 2
    chart_h = height - padding * 2 - 20
    values = [float(s.get("value", 0) or 0) for s in series] or [0.0]
    max_v = max(values) or 1.0
    min_v = min(values) if max_v != min(values) else 0.0
    if max_v == min_v: max_v += 1.0
    x0 = padding; y0 = padding + 10
    pts = []
    for i, s in enumerate(series):
        x = x0 + i * (chart_w / max(1, len(series)-1))
        val = float(s.get("value", 0) or 0)
        y = y0 + (1 - (val - min_v) / (max_v - min_v)) * chart_h
        pts.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    labels = [f'<text x="{pts[i][0]:.1f}" y="{y0 + chart_h + 16:.1f}" font-size="10" text-anchor="middle">{_escape(str(s.get("label","")))}</text>' for i, s in enumerate(series)]
    grid = [f'<line x1="{x0}" y1="{y0 + gy*(chart_h/4):.1f}" x2="{x0+chart_w}" y2="{y0 + gy*(chart_h/4):.1f}" stroke="#ddd" stroke-dasharray="3,3" />' for gy in range(5)]
    title_el = f'<text x="{width/2:.1f}" y="{padding-2:.1f}" font-size="14" font-weight="600" text-anchor="middle">{_escape(title)}</text>'
    return f"""    <svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="gl" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#00C9A7"/><stop offset="100%" stop-color="#6C63FF"/>
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill="white"/>
      {title_el}
      {''.join(grid)}
      <polyline fill="none" stroke="url(#gl)" stroke-width="3" points="{polyline}"/>
      <g fill="#333">{''.join(labels)}</g>
    </svg>"""

# ----------------------------- Risk badge -----------------------------
def _risk_badge(level: str) -> str:
    lv = (level or "").lower()
    if "high" in lv:    return '<span class="pill pill-high">High Risk</span>'
    if "moderate" in lv:return '<span class="pill pill-mod">Moderate Risk</span>'
    return '<span class="pill pill-low">Low Risk</span>'

# ----------------------------- Fallback: rule-based doctor report -----------------------------
def _doctor_fallback(user_payload: dict) -> dict:
    meta = {
        "generated_at": _now_iso(),
        "model": os.getenv("GROQ_MODEL", "fallback-rule-based"),
        "title": user_payload.get("title") or "Sleep Doctor Report",
        "subtitle": user_payload.get("subtitle") or "AI-assisted clinical summary",
        "patient": user_payload.get("patient") or {},
    }

    metrics = dict(user_payload.get("metrics") or {})
    cur = user_payload.get("current") or {}
    def pick(*keys):
        for k in keys:
            if isinstance(k, (list, tuple)):
                node = cur; ok = True
                for seg in k:
                    if isinstance(node, dict) and seg in node:
                        node = node[seg]
                    else:
                        ok = False; break
                if ok: return node
        return None

    metrics.setdefault("TST",  _num(cur.get("duration_minutes") or cur.get("durationMinutes") or cur.get("totalSleepMinutes") or pick(["metrics","durationMinutes"]) , 0))
    metrics.setdefault("TIB",  _num(cur.get("time_in_bed_minutes") or cur.get("timeInBedMinutes"), 0))
    if "SE" not in metrics and metrics.get("TST",0) and metrics.get("TIB",0):
        metrics["SE"] = round((metrics["TST"] / max(1, metrics["TIB"])) * 100, 1)
    metrics.setdefault("SOL",  _num(cur.get("latency_minutes") or cur.get("latencyMinutes"), 0))
    metrics.setdefault("WASO", _num(cur.get("waso_minutes") or cur.get("WASO") or 0))
    metrics.setdefault("AHI",  _num(cur.get("AHI") or 0))
    metrics.setdefault("deep",  _num(cur.get("deep_sleep_minutes") or cur.get("deepSleepMinutes") or (cur.get("stages") or {}).get("deepMinutes"), 0))
    metrics.setdefault("rem",   _num(cur.get("rem_sleep_minutes") or cur.get("remSleepMinutes") or (cur.get("stages") or {}).get("remMinutes"), 0))
    metrics.setdefault("light", _num(cur.get("light_sleep_minutes") or cur.get("lightSleepMinutes") or (cur.get("stages") or {}).get("lightMinutes"), 0))
    metrics.setdefault("awake", _num((cur.get("stages") or {}).get("awakeMinutes") or 0))

    duration_m = _num(metrics.get("TST"), 0)
    tib_m      = _num(metrics.get("TIB"), 0)
    se         = _num(metrics.get("SE"), 0) if metrics.get("SE") is not None else (duration_m / max(1, tib_m))*100
    quality    = _num(cur.get("quality") or cur.get("sleepQuality"), 0)
    stress10   = _clamp(_num(cur.get("stress_level") or cur.get("stressLevel"), 0), 0, 10)
    caffeine   = _clamp(_num(cur.get("caffeine_intake") or cur.get("caffeineIntake"), 0), 0, 600)
    exercise   = _clamp(_num(cur.get("exercise_minutes") or cur.get("exerciseMinutes"), 0), 0, 180)

    comp = {
        "duration":  _clamp((duration_m / 480.0) * 100.0, 0, 100),
        "quality":   _clamp((quality / 10.0) * 100.0, 0, 100),
        "efficiency":_clamp(se, 0, 100),
        "stress":    _clamp(100.0 - (stress10 * 10.0), 0, 100),
        "caffeine":  _clamp(100.0 - (caffeine / 6.0), 0, 100),
        "exercise":  _clamp(min(exercise, 60) / 60.0 * 100.0, 0, 100),
    }
    risk_score = round(
        0.32*comp["duration"] + 0.20*comp["quality"] + 0.18*comp["efficiency"] +
        0.12*comp["stress"] + 0.08*comp["caffeine"] + 0.10*comp["exercise"], 1
    )
    level = "Low Risk" if risk_score >= 75 else ("Moderate Risk" if risk_score >= 55 else "High Risk")

    bullets = []
    if duration_m: bullets.append(f"Total Sleep Time ≈ {int(duration_m)} min")
    if se:         bullets.append(f"Sleep Efficiency ≈ {round(se,1)}%")
    if metrics.get("SOL"):  bullets.append(f"Sleep Latency ≈ {int(metrics['SOL'])} min")
    if metrics.get("WASO"): bullets.append(f"WASO ≈ {int(metrics['WASO'])} min")

    notes = _clean_text(user_payload.get("analysis") or
                        user_payload.get("overview") or
                        "No narrative provided. Focus on consistent schedule, light exposure in the morning, and screen reduction before bed.")

    plan = {
        "morning":   ["Natural light 10–15 min after wake", "Hydration and light mobility (5–10 min)"],
        "afternoon": ["Keep caffeine <200mg after 14:00", "10–20 min walk or easy cardio"],
        "evening":   ["Wind-down 45–60 min pre-bed (dim lights, low screens)", "Cool, dark, quiet bedroom"]
    }
    if risk_score < 55:
        plan["evening"].append("Short nap 10–20 min before 15:00 if needed")

    wake = cur.get("wakeTime") or cur.get("wake_time")
    bed  = cur.get("bedTime")  or cur.get("bed_time") or cur.get("bedtime")
    def add_minutes(hhmm, minutes):
        try:
            h, m = str(hhmm).split(":")
            total = (int(h)*60 + int(m) + int(minutes)) % (24*60)
            return f"{total//60:02d}:{total%60:02d}"
        except Exception:
            return hhmm
    windows = []
    if isinstance(wake, str) and ":" in wake:
        windows.append({"start": wake, "end": add_minutes(wake, 30), "why": "Maintain consistency (+30m)"})
        windows.append({"start": add_minutes(wake, -30), "end": wake, "why": "Allow early start (−30m)"})
    elif isinstance(bed, str) and ":" in bed:
        target = add_minutes(bed, 7*60+30)
        windows.append({"start": add_minutes(target, -15), "end": add_minutes(target, 15), "why": "Target ~7.5h"})
    else:
        windows.append({"start": "06:30", "end": "07:00", "why": "Default window"})

    what_if = [
        {"title":"Reduce screens 1h pre-bed","impact":"Likely positive","note":"Less blue light → better melatonin onset"},
        {"title":"Move caffeine earlier","impact":"Reduces latency","note":"Aim no caffeine after 14:00"},
    ]

    return {
        "meta": meta,
        "executiveSummary": {"bullets": bullets or ["Maintain a stable sleep schedule and optimize sleep hygiene."]},
        "risk": {"score": risk_score, "level": level, "rationale": "Computed from duration/quality/efficiency/stress/caffeine/exercise.", "components": comp},
        "assessment": {"diagnoses": ["Insomnia (suspected)"] if se < 85 else [], "notes": notes},
        "plan": plan,
        "wakeWindows": {"windows": windows, "note": "Adjust by ≤30 min as needed."},
        "whatIf": what_if,
        "metrics": metrics,
        "charts": user_payload.get("charts") or [],
        "images": user_payload.get("images") or [],
        "recommendations": user_payload.get("recommendations") or [],
        "sections": user_payload.get("sections") or [],
    }

# ----------------------------- Groq doctor struct (resilient) -----------------------------
def _groq_doctor_struct(user_payload: dict) -> dict:
    if not GROQ_API_KEY:
        return _doctor_fallback(user_payload)

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    system = (
        "You are Dr. Somnus, a board-certified sleep specialist. "
        "Transform the user's payload into the STRICT JSON schema described below. "
        "No markdown, no extra keys, no prose outside JSON.\n\n"
        "{"
        ' "meta":{"title":str,"subtitle":str,"generated_at":str(ISO),"model":str,"patient":{"name"?:str,"age"?:str|num,"sex"?:str}},'
        ' "executiveSummary":{"bullets":[str]},'
        ' "risk":{"score":number,"level":"Low Risk"|"Moderate Risk"|"High Risk","rationale":str,"components":{str:number}},'
        ' "assessment":{"diagnoses":[str], "notes":str},'
        ' "plan":{"morning":[str],"afternoon":[str],"evening":[str],"medications"?:[str],"behavioral"?:[str]},'
        ' "wakeWindows":{"windows":[{"start":"HH:MM","end":"HH:MM","why":str}], "note":str},'
        ' "whatIf":[{"title":str,"impact":str,"note":str}],'
        ' "metrics":{"TST"?:num,"TIB"?:num,"SE"?:num,"SOL"?:num,"WASO"?:num,"AHI"?:num,"deep"?:num,"rem"?:num,"light"?:num,"awake"?:num},'
        ' "charts":[{"type":"bar"|"line"|"donut","title":str,"data":[{"label":str,"value":num}]}],'
        ' "images":[{"title":str,"url_or_data_uri":str}],'
        ' "recommendations":[str], "sections":[{"title":str,"body":str}]'
        "}"
    )

    model_candidates = [GROQ_MODEL or "llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
    last_err_txt = None
    for model in model_candidates:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        try:
            r = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)
            if r.status_code != 200:
                last_err_txt = f"Groq {r.status_code}: {r.text[:400]}"
                continue
            data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            parsed = json.loads(content)
            parsed.setdefault("meta", {})
            parsed["meta"].setdefault("generated_at", _now_iso())
            parsed["meta"].setdefault("model", model)
            for k, v in {
                "executiveSummary":{"bullets":[]}, "risk":{"score":65,"level":"Moderate Risk","rationale":"","components":{}},
                "assessment":{"diagnoses":[],"notes":""}, "plan":{"morning":[],"afternoon":[],"evening":[]},
                "wakeWindows":{"windows":[],"note":""}, "whatIf":[], "metrics":{}, "charts":[], "images":[],
                "recommendations":[], "sections":[]
            }.items():
                parsed.setdefault(k, v)
            return parsed
        except Exception as e:
            last_err_txt = f"{type(e).__name__}: {str(e)}"

    try:
        import logging
        logging.getLogger("smart-pdf").error("Groq failed, using fallback: %s", last_err_txt)
    except Exception:
        pass
    return _doctor_fallback(user_payload)

# ----------------------------- CSS -----------------------------
BASE_CSS = CSS(string='''
@page {
  size: A4;
  margin: 16mm 14mm 20mm 14mm;
  @bottom-center {
    content: "Page " counter(page) " of " counter(pages);
    font-size: 10px;
    color: #6b7280;
  }
}
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Open Sans", sans-serif; color: #111827; }
h1, h2, h3 { margin: 0; color: #111827; }
.small { font-size: 12px; color: #6b7280; }
.kicker { color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.section { page-break-inside: avoid; margin: 10px 0 18px; }
.card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px 16px; background: white; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.hr { height: 1px; background: #e5e7eb; margin: 8px 0; }
.img { width: 100%; border-radius: 10px; border: 1px solid #e5e7eb; }
.pill { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.pill-low { background:#ECFDF5; color:#065F46; }
.pill-mod { background:#FEF3C7; color:#92400E; }
.pill-high{ background:#FEE2E2; color:#991B1B; }
.metrics { display:grid; grid-template-columns: repeat(3,1fr); gap:10px; }
.metric { border:1px dashed #e5e7eb; border-radius:10px; padding:10px; text-align:center; }
.metric .label { font-size:12px; color:#6b7280; }
.metric .value { font-size:16px; font-weight:700; margin-top:4px; }
.hero { border-radius:16px; padding:18px; background: linear-gradient(135deg,#EEF2FF,#E6FFFA); border:1px solid #e5e7eb; }
ul { margin: 6px 0 6px 20px; }
.footer-note { font-size:10px; color:#6b7280; margin-top:8px; }
''')

# ----------------------------- HTML Renderer -----------------------------
def _render_html(ctx: dict) -> str:
    meta = ctx.get("meta", {})
    title    = _escape(meta.get("title", "Sleep Doctor Report"))
    subtitle = _escape(meta.get("subtitle", "AI-assisted clinical summary"))
    gen_at   = _escape(meta.get("generated_at", ""))
    patient  = meta.get("patient") or {}
    patient_line = " • ".join([_clean_text(f"{k}: {v}") for k, v in patient.items()]) if patient else ""

    bullets = [ _clean_text(b) for b in ((ctx.get("executiveSummary") or {}).get("bullets") or []) ]

    risk = ctx.get("risk") or {}
    risk_score = risk.get("score") or 0
    risk_level = risk.get("level") or "Moderate Risk"
    risk_badge = _risk_badge(risk_level)
    risk_rat   = _clean_text(risk.get("rationale") or "")
    risk_comp  = risk.get("components") or {}

    assess = ctx.get("assessment") or {}
    diagnoses = [ _clean_text(d) for d in (assess.get("diagnoses") or []) ]
    notes     = _clean_text(assess.get("notes") or "")

    plan = ctx.get("plan") or {}
    morning   = [ _clean_text(x) for x in (plan.get("morning") or []) ]
    afternoon = [ _clean_text(x) for x in (plan.get("afternoon") or []) ]
    evening   = [ _clean_text(x) for x in (plan.get("evening") or []) ]
    meds      = [ _clean_text(x) for x in (plan.get("medications") or []) ]
    behav     = [ _clean_text(x) for x in (plan.get("behavioral") or []) ]

    ww = ctx.get("wakeWindows") or {}
    ww_windows = ww.get("windows") or []
    ww_note    = _clean_text(ww.get("note") or "")

    what_if = ctx.get("whatIf") or []

    metrics = ctx.get("metrics") or {}
    mkeys = [("TST","TST (min)"),("TIB","TIB (min)"),("SE","SE (%)"),("SOL","SOL (min)"),
             ("WASO","WASO (min)"),("AHI","AHI"),("deep","Deep (min)"),("rem","REM (min)"),
             ("light","Light (min)"),("awake","Awake (min)")]
    metric_cards = []
    for k,label in mkeys:
        if k in metrics:
            val = metrics.get(k)
            metric_cards.append(f'<div class="metric"><div class="label">{_escape(label)}</div><div class="value">{_escape(str(val))}</div></div>')

    chart_blocks = []
    for ch in ctx.get("charts", []):
        ctype = (ch.get("type") or "").lower()
        ttl   = _clean_text(ch.get("title") or "Chart")
        data  = ch.get("data") or []
        if ctype == "bar":   chart_blocks.append(_svg_bar_chart(ttl, data))
        elif ctype == "donut": chart_blocks.append(_svg_donut_chart(ttl, data))
        else:                chart_blocks.append(_svg_line_chart(ttl, data))

    image_blocks = []
    for img in ctx.get("images", []):
        src = img.get("url_or_data_uri")
        if not src: continue
        data_uri = _fetch_image_as_data_uri(src)
        if data_uri:
            ttl = _escape(_clean_text(img.get("title","")))
            image_blocks.append(f'<div class="card"><div class="small" style="margin-bottom:6px">{ttl}</div><img class="img" src="{data_uri}"/></div>')

    extra_secs = []
    for sec in ctx.get("sections", []):
        st = _escape(_clean_text(sec.get("title","Section")))
        sb = _escape(_clean_text(sec.get("body",""))).replace("\n","<br/>")
        extra_secs.append(f'<div class="card section"><div class="kicker">{st}</div><div style="margin-top:6px;font-size:13px;line-height:1.5">{sb}</div></div>')

    recs = [ _clean_text(r) for r in (ctx.get("recommendations") or []) ]

    overview_txt = ""
    if ctx.get("assessment", {}).get("notes"):
        overview_txt = _escape_lines(_clean_text(ctx["assessment"]["notes"]))

    tpl = Template('''
<html>
  <body>
    <div class="hero">
      <div style="display:flex;justify-content:space-between;align-items:flex-end">
        <div>
          <div class="kicker">Sleep Moon — AI Doctor Report</div>
          <h2 style="margin-top:4px">{{ title }}</h2>
          <div class="small">{{ subtitle }}</div>
          {% if patient_line %}<div class="small" style="margin-top:4px">{{ patient_line }}</div>{% endif %}
        </div>
        <div class="small">Generated: {{ gen_at }}</div>
      </div>
    </div>

    <!-- Executive Summary + Risk -->
    <div class="section grid-2">
      <div class="card">
        <div class="kicker">Executive Summary</div>
        <ul style="margin-top:6px">
          {% for b in bullets %}<li>{{ b }}</li>{% endfor %}
        </ul>
      </div>
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div class="kicker">Risk & Readiness</div>
          <div>{{ risk_badge | safe }}</div>
        </div>
        <div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:center">
          <div class="metric">
            <div class="label">Risk Score</div>
            <div class="value">{{ risk_score }}</div>
          </div>
          <div class="metric">
            <div class="label">Top Driver</div>
            <div class="value">{% if risk_components %}{{ (risk_components|list)[0] }}{% else %}—{% endif %}</div>
          </div>
        </div>
        {% if risk_rat %}
        <div class="hr"></div>
        <div class="small">Rationale: {{ risk_rat }}</div>
        {% endif %}
      </div>
    </div>

    {% if overview_txt %}
    <div class="section card">
      <div class="kicker">Clinical Overview</div>
      <div style="margin-top:6px;font-size:13px;line-height:1.5">{{ overview_txt | safe }}</div>
    </div>
    {% endif %}

    <!-- Assessment + Diagnoses + Plan -->
    <div class="section grid-3">
      <div class="card">
        <div class="kicker">Diagnostic Considerations</div>
        <ul style="margin-top:6px">
          {% for d in diagnoses %}<li>{{ d }}</li>{% endfor %}
        </ul>
      </div>
      <div class="card">
        <div class="kicker">Daily Plan — Morning</div>
        <ul style="margin-top:6px">{% for x in morning %}<li>{{ x }}</li>{% endfor %}</ul>
        <div class="kicker" style="margin-top:6px">Afternoon</div>
        <ul>{% for x in afternoon %}<li>{{ x }}</li>{% endfor %}</ul>
      </div>
      <div class="card">
        <div class="kicker">Evening & Sleep Hygiene</div>
        <ul style="margin-top:6px">{% for x in evening %}<li>{{ x }}</li>{% endfor %}</ul>
        {% if meds or behav %}
        <div class="hr"></div>
        {% if meds %}<div class="kicker">Medications</div><ul>{% for x in meds %}<li>{{ x }}</li>{% endfor %}</ul>{% endif %}
        {% if behav %}<div class="kicker" style="margin-top:6px">Behavioral</div><ul>{% for x in behav %}<li>{{ x }}</li>{% endfor %}</ul>{% endif %}
        {% endif %}
      </div>
    </div>

    {% if metric_cards %}
    <div class="section card">
      <div class="kicker">Key Metrics</div>
      <div class="metrics" style="margin-top:8px">
        {{ metric_cards | safe }}
      </div>
    </div>
    {% endif %}

    {% if chart_blocks %}
    <div class="section card">
      <div class="kicker">Charts</div>
      {% for c in chart_blocks %}
        <div style="margin:8px 0">{{ c | safe }}</div>
      {% endfor %}
    </div>
    {% endif %}

    {% if image_blocks %}
    <div class="section card">
      <div class="kicker">Images</div>
      <div class="grid-2" style="margin-top:8px">
        {% for im in image_blocks %}{{ im | safe }}{% endfor %}
      </div>
    </div>
    {% endif %}

    {% if ww_windows %}
    <div class="section card">
      <div class="kicker">Wake Windows</div>
      <ul style="margin-top:6px">
        {% for w in ww_windows %}
          <li>{{ w.start }} → {{ w.end }} — {{ w.why }}</li>
        {% endfor %}
      </ul>
      {% if ww_note %}<div class="small" style="margin-top:4px">{{ ww_note }}</div>{% endif %}
    </div>
    {% endif %}

    {% if what_if %}
    <div class="section card">
      <div class="kicker">What-If Scenarios</div>
      <ul style="margin-top:6px">
        {% for wi in what_if %}
          <li><strong>{{ wi.title }}</strong> — {{ wi.impact }}. <span class="small">{{ wi.note }}</span></li>
        {% endfor %}
      </ul>
    </div>
    {% endif %}

    {% if recs %}
    <div class="section card">
      <div class="kicker">General Recommendations</div>
      <ul style="margin-top:6px">{% for r in recs %}<li>{{ r }}</li>{% endfor %}</ul>
    </div>
    {% endif %}

    {% for s in extra_secs %}
      {{ s | safe }}
    {% endfor %}

    <div class="footer-note">
      This document is for educational purposes and does not replace professional medical advice. 
      If symptoms persist or worsen, consult a licensed clinician. © Sleep Moon.
    </div>
  </body>
</html>
''')

    html = tpl.render(
        title=title,
        subtitle=subtitle,
        gen_at=gen_at,
        patient_line=patient_line,

        bullets=bullets,
        risk_badge=risk_badge,
        risk_score=int(risk_score),
        risk_components=risk_comp,
        risk_rat=_escape(_clean_text(risk_rat)),

        overview_txt=overview_txt,

        diagnoses=diagnoses,
        morning=morning, afternoon=afternoon, evening=evening,
        meds=meds, behav=behav,

        metric_cards="".join(metric_cards),
        chart_blocks=chart_blocks,
        image_blocks=image_blocks,

        ww_windows=ww_windows, ww_note=_escape(_clean_text(ww_note)),

        what_if=[{"title":_escape(_clean_text(w.get("title",""))),
                  "impact":_escape(_clean_text(w.get("impact",""))),
                  "note":_escape(_clean_text(w.get("note","")))} for w in what_if],

        recs=recs,
        extra_secs=extra_secs
    )
    return html

# ----------------------------- HTTP Route -----------------------------
@pdf_bp.route("/generate", methods=["POST"])
def generate_pdf():
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON payload"}), 400

    auto_groq = bool(payload.get("auto_groq")) or not any([
        payload.get("executiveSummary"),
        payload.get("assessment"),
        payload.get("plan")
    ])

    try:
        if auto_groq:
            ctx = _groq_doctor_struct(payload)
        else:
            meta = {
                "generated_at": _now_iso(),
                "model": GROQ_MODEL,
                "title": payload.get("title") or (payload.get("meta",{})).get("title") or "Sleep Doctor Report",
                "subtitle": payload.get("subtitle") or (payload.get("meta",{})).get("subtitle") or "AI-assisted clinical summary",
                "patient": payload.get("patient") or (payload.get("meta",{})).get("patient") or {},
            }
            ctx = dict(payload)
            ctx["meta"] = meta
    except Exception as e:
        import logging
        logging.getLogger("smart-pdf").exception("Doctor struct build failed")
        return jsonify({"ok": False, "error": f"GROQ build failed (using fallback): {e}"}), 500

    try:
        html = _render_html(ctx)
    except Exception as e:
        import logging
        logging.getLogger("smart-pdf").exception("Render HTML failed. Context snapshot: %s", json.dumps(ctx, default=str)[:800])
        return jsonify({"ok": False, "error": f"Render HTML failed: {e}"}), 500

    try:
        pdf_bytes = HTML(string=html, base_url=".").write_pdf(stylesheets=[BASE_CSS])
    except Exception as e:
        import logging
        logging.getLogger("smart-pdf").exception("PDF render failed")
        return jsonify({"ok": False, "error": f"PDF render failed: {e}"}), 500

    filename = payload.get("download_filename") or (
        (ctx.get("meta",{}).get("title","report")).lower().replace(" ", "_") + ".pdf"
    )
    return send_file(BytesIO(pdf_bytes), as_attachment=True, download_name=filename, mimetype="application/pdf")

# ----------------------------- Demo -----------------------------
@pdf_bp.route("/_doctor_demo", methods=["GET"])
def doctor_demo():
    demo = {
        "title": "Sleep Doctor Report",
        "subtitle": "Sleep Moon • AI Insights",
        "patient": {"name": "A.S.", "age": 28, "sex": "M"},
        "analysis": "User reports difficulty maintaining sleep, high screen time before bed, caffeine after 4pm. Oura logs show TST ~390m, SE ~82%, SOL ~24m, WASO ~38m. Deep ~90m, REM ~110m.",
        "metrics": {"TST": 390, "TIB": 470, "SE": 83, "SOL": 24, "WASO": 38, "AHI": 2, "deep": 90, "rem": 110, "light": 160, "awake": 20},
        "charts": [
            {"type":"donut","title":"Sleep Stages","data":[
                {"label":"Deep","value":90},{"label":"REM","value":110},{"label":"Light","value":160},{"label":"Awake","value":20}
            ]},
            {"type":"bar","title":"Sleep Duration (min, last 7)","data":[
                {"label":"Mon","value":380},{"label":"Tue","value":410},{"label":"Wed","value":395},
                {"label":"Thu","value":430},{"label":"Fri","value":405},{"label":"Sat","value":450},{"label":"Sun","value":420}
            ]}
        ],
        "images":[
            {"title":"Sample Hypnogram","url_or_data_uri":"https://upload.wikimedia.org/wikipedia/commons/3/30/Hypnogram.svg"}
        ],
        "auto_groq": True,
        "download_filename": "doctor_demo_report.pdf"
    }
    with pdf_bp.test_request_context(json=demo):
        return generate_pdf()

