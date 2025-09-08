import json
from datetime import datetime
from io import BytesIO
from html import escape as _escape

from flask import Blueprint, request, jsonify, send_file
from jinja2 import Template
from weasyprint import HTML, CSS

pdf_bp = Blueprint("pdf_bp", __name__)

# ---------- helpers (same formatting engine as app.py version) ----------

def _is_scalar(x):
    return isinstance(x, (str, int, float)) or x is None

def _is_shallow_dict(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    for v in d.values():
        if isinstance(v, (dict, list)):
            return False
    return True

def _is_list_of_str(xs):
    return isinstance(xs, list) and all(isinstance(i, str) for i in xs)

def _is_list_of_shallow_dicts(xs):
    return isinstance(xs, list) and xs and all(isinstance(i, dict) and _is_shallow_dict(i) for i in xs)

def _is_list_of_dicts_with_text(xs):
    keys = {"text", "title", "description", "note"}
    return isinstance(xs, list) and xs and all(isinstance(i, dict) and any(k in i for k in keys) for i in xs)

def _to_bullets(xs):
    items = []
    for it in xs:
        if isinstance(it, dict):
            txt = it.get("text") or it.get("description") or it.get("note") or it.get("title")
            if txt is None:
                txt = "; ".join(f"{k}: {v}" for k, v in it.items())
        else:
            txt = str(it)
        items.append(f"<li>{_escape(str(txt))}</li>")
    return f"<ul>{''.join(items)}</ul>"

def _paragraphize(s: str):
    if not isinstance(s, str):
        return _escape(str(s))
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines and all(ln.startswith(("-", "•")) for ln in lines):
        stripped = [ln.lstrip("-• ").strip() for ln in lines]
        return _to_bullets(stripped)
    ps = []
    buf = []
    for ln in s.splitlines():
        if ln.strip():
            buf.append(ln.strip())
        elif buf:
            ps.append(" ".join(buf)); buf = []
    if buf:
        ps.append(" ".join(buf))
    if not ps:
        ps = [s]
    return "".join(f"<p>{_escape(p)}</p>" for p in ps)

def _kv_table(d: dict):
    rows = []
    for k, v in d.items():
        if _is_scalar(v):
            rows.append(f"<tr><td class='k'>{_escape(str(k))}</td><td class='v'>{_escape('' if v is None else str(v))}</td></tr>")
        else:
            sub_html, _ = _format_value_html(v, prefer='auto', depth=1)
            rows.append(f"<tr><td class='k'>{_escape(str(k))}</td><td class='v'>{sub_html}</td></tr>")
    return f\"\"\"
    <table class="kv">
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    \"\"\"

def _format_value_html(val, *, prefer="auto", depth=0):
    if prefer == "p" and isinstance(val, str):
        return _paragraphize(val), "p"
    if prefer == "list":
        if _is_list_of_str(val) or _is_list_of_dicts_with_text(val) or _is_list_of_shallow_dicts(val):
            return _to_bullets(val), "list"
        if isinstance(val, str) and ("\n" in val or val.strip().startswith(("-", "•"))):
            return _paragraphize(val), "p"
    if prefer == "kv" and isinstance(val, dict):
        return _kv_table(val), "kv"

    if isinstance(val, str):
        return _paragraphize(val), "p"
    if _is_list_of_str(val) or _is_list_of_dicts_with_text(val):
        return _to_bullets(val), "list"
    if _is_list_of_shallow_dicts(val):
        bullets = []
        for d in val:
            row = "; ".join(f"<b>{_escape(str(k))}:</b> {_escape(str(v))}" for k, v in d.items())
            bullets.append(f"<li>{row}</li>")
        return f"<ul>{''.join(bullets)}</ul>", "list"
    if isinstance(val, dict):
        if _is_shallow_dict(val):
            return _kv_table(val), "kv"
        return _kv_table(val), "kv"
    if isinstance(val, list):
        return _to_bullets(val), "list"

    try:
        txt = json.dumps(val, ensure_ascii=False, indent=2)
    except Exception:
        txt = str(val)
    return f"<pre>{_escape(txt)}</pre>", "pre"

def _mk_section(title: str, value, *, key=None, prefer=None, image=None):
    if value is None:
        return None
    if isinstance(value, (list, dict)) and len(value) == 0:
        return None
    html, kind = _format_value_html(value, prefer=(prefer or "auto"))
    img_src = None
    if image:
        img_src = str(image)
        if not img_src.startswith("data:"):
            img_src = f"data:image/png;base64,{img_src}"
    return {"title": title, "html": html, "kind": kind, "image": img_src, "key": key or ""}

def _first(*vals):
    for v in vals:
        if v not in (None, "", [], {}):
            return v
    return None

def _get_any(d: dict, *names):
    for n in names:
        if n in d:
            return d[n]
    for n in names:
        if "_" in n:
            camel = n.split("_")[0] + "".join([w.title() for w in n.split("_")[1:]])
            if camel in d:
                return d[camel]
    return None

def build_sections_from_analysis(analysis: dict) -> list:
    if not isinstance(analysis, dict):
        return []
    S = []
    S.append(_mk_section("Executive Summary", _first(_get_any(analysis, "executive_summary", "detailedReport", "summary", "overview")), key="executive_summary", prefer="p"))
    S.append(_mk_section("Sleep Metrics", _get_any(analysis, "sleep_metrics", "sleepMetrics"), key="sleep_metrics", prefer="kv"))
    S.append(_mk_section("Sleep Stages", _get_any(analysis, "sleep_stages", "sleepStages"), key="sleep_stages", prefer="kv"))
    S.append(_mk_section("Efficiency Analysis", _get_any(analysis, "sleep_efficiency_analysis", "efficiencyAnalysis"), key="efficiency", prefer="p"))
    S.append(_mk_section("Depth Analysis", _get_any(analysis, "sleep_depth_analysis", "depthAnalysis"), key="depth", prefer="p"))
    S.append(_mk_section("Recovery Analysis", _get_any(analysis, "recovery_analysis", "recoveryAnalysis"), key="recovery", prefer="p"))
    S.append(_mk_section("Environment", _get_any(analysis, "environment_analysis", "environmentAnalysis"), key="environment", prefer="kv"))
    S.append(_mk_section("Behavioral Factors", _get_any(analysis, "behavioral_factors", "behavioralFactors"), key="behavior", prefer="list"))
    S.append(_mk_section("Sleep Patterns", _get_any(analysis, "sleep_patterns", "sleepPatterns"), key="patterns", prefer="kv"))
    S.append(_mk_section("Sleep Trends", _get_any(analysis, "sleep_trends", "sleepTrends"), key="trends", prefer="kv"))
    S.append(_mk_section("Daily Comparison", _get_any(analysis, "daily_comparison", "dailyComparison"), key="daily", prefer="kv"))
    S.append(_mk_section("Key Insights", _get_any(analysis, "key_insights", "keyInsights"), key="insights", prefer="list"))
    S.append(_mk_section("Recommendations", _get_any(analysis, "recommendations"), key="recommendations", prefer="list"))
    S.append(_mk_section("Action Items", _get_any(analysis, "action_items", "actionItems"), key="action_items", prefer="list"))
    S.append(_mk_section("Predictive Warnings", _get_any(analysis, "predictive_warnings", "predictiveWarnings"), key="warnings", prefer="list"))
    S.append(_mk_section("Dream & Mood Forecast", _first(_get_any(analysis, "dream_mood_forecast", "dreamMoodForecast", "dreamForecast", "moodForecast")), key="dream", prefer="p"))
    S.append(_mk_section("Risk Assessment", _get_any(analysis, "risk_assessment", "riskAssessment"), key="risk", prefer="kv"))
    S.append(_mk_section("Energy Plan", _get_any(analysis, "daily_energy_plan", "energy_plan"), key="energy", prefer="p"))
    for title, key in [
        ("HRV Summary", "hrv_summary"),
        ("Respiratory Events", "respiratory_events"),
        ("Glucose Correlation", "glucose_correlation"),
        ("Nutrition Correlation", "nutrition_correlation"),
        ("Drivers", "drivers"),
        ("Causal Graph", "causal_graph"),
        ("Energy Timeline", "energy_timeline"),
        ("Next Day Forecast", "next_day_forecast"),
        ("What‑If Scenarios", "what_if_scenarios"),
    ]:
        S.append(_mk_section(title, _get_any(analysis, key), key=key, prefer="kv"))
    return [s for s in S if s]

HTML_TEMPLATE = Template(r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <style>
    @page { size: A4; margin: 18mm 15mm 20mm 15mm;
      @bottom-center { content: "Sleep Moon · " counter(page) " / " counter(pages); font-size: 10px; color: #666; } }
    :root { --ink:#0f172a; --muted:#475569; --chip:#eef2ff; --line:#e5e7eb; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "Noto Sans", "DejaVu Sans", sans-serif; color: var(--ink); }
    h1 { font-size: 28px; margin:0 0 4px 0; }
    .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
    h2 { font-size: 16px; margin: 18px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
    .chips { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0 14px; justify-content:center; }
    .chip { padding:6px 10px; background: var(--chip); border-radius: 999px; font-size: 11px; }
    .sec { margin:8px 0 14px; page-break-inside: avoid; }
    .kv { width:100%; border-collapse: collapse; }
    .kv td { border-top:1px solid var(--line); padding:6px 8px; font-size:12px; vertical-align: top; }
    .kv td.k { width: 32%; font-weight:600; color:#475569; }
    ul { margin: 0 0 0 18px; font-size: 12.5px; }
    pre { background:#f8fafc; border:1px solid var(--line); padding:10px 12px; border-radius:8px; font-size:11.5px; white-space: pre-wrap; }
    p { font-size:12.5px; margin:6px 0; }
    .imgwrap { border:1px solid var(--line); border-radius:8px; padding:6px; margin-top:8px; text-align:center; }
    .imgwrap img { max-width:100%; max-height:360px; }
    .cover { page-break-after: always; margin-top: 30mm; text-align:center; }
    .cover h1 { font-size: 34px; margin-bottom: 6px; }
    .cover .subtitle { font-size: 14px; margin-bottom: 24px; }
    .meta { color:#475569; font-size: 12px; }
  </style>
</head>
<body>
  <div class="cover">
    <h1>{{ title }}</h1>
    {% if subtitle %}<div class="subtitle">{{ subtitle }}</div>{% endif %}
    <div class="meta">Generated at {{ generated_at }}</div>
    {% if chips %}
      <div class="chips">
        {% for k, v in chips.items() %}
          <div class="chip"><strong>{{ k }}:</strong>&nbsp;{{ v }}</div>
        {% endfor %}
      </div>
    {% endif %}
  </div>
  {% for sec in sections %}
    <div class="sec">
      <h2>{{ sec.title }}</h2>
      {{ sec.html | safe }}
      {% if sec.image %}<div class="imgwrap"><img src="{{ sec.image }}"/></div>{% endif %}
    </div>
  {% endfor %}
</body>
</html>
""")

@pdf_bp.post("/api/pdf/sleep-report")
def api_sleep_report():
    try:
        data = request.get_json(force=True, silent=False) or {}
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    title = data.get("title") or "Sleep Analysis Report"
    subtitle = data.get("subtitle") or "Sleep Moon · AI Insights"
    chips = data.get("chips") or {}
    sections_in = data.get("sections")
    analysis = data.get("analysis") or {}

    sections = []
    if isinstance(sections_in, list) and sections_in:
        for s in sections_in:
            if not isinstance(s, dict): continue
            html = s.get("html")
            body = s.get("body")
            if not html and body is not None:
                html, _ = _format_value_html(body, prefer="auto")
            image = s.get("image")
            if image and not str(image).startswith("data:"):
                image = f"data:image/png;base64,{image}"
            sections.append({"title": s.get("title") or "Section", "html": html or "", "image": image})
    else:
        sections = build_sections_from_analysis(analysis)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = HTML_TEMPLATE.render(title=title, subtitle=subtitle, chips=chips, sections=sections, generated_at=generated_at)

    pdf_io = BytesIO()
    HTML(string=html).write_pdf(pdf_io, stylesheets=[CSS(string="")])
    pdf_io.seek(0)
    return send_file(pdf_io, mimetype="application/pdf", as_attachment=True, download_name="sleep_report.pdf")
