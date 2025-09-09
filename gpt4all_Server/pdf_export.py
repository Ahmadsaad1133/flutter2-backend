
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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")

# ----------------------------- Helpers ----------------------------
def _is_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float)) or x is None

def _is_shallow_dict(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    for v in d.values():
        if isinstance(v, (dict, list)):
            return False
    return True

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _clean_json_str(s: str) -> str:
    '''
    Attempt to minimally fix common JSON issues:
      - Replace single quotes with double quotes (carefully)
      - Remove trailing commas before ] or }
      - Strip code fences
    '''
    if not isinstance(s, str):
        return s  # type: ignore[return-value]
    # Remove code fences if present
    s = s.strip()
    if s.startswith('```'):
        s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s, flags=re.IGNORECASE | re.MULTILINE)

    # Extract the largest {{...}} or [...] block
    m = re.search(r'({[\s\S]*}|\[[\s\S]*\])', s)
    if m:
        s = m.group(1)

    # Replace single quotes around keys/strings with double quotes (naive but helps)
    # Avoid touching http:// or https://
    def repl_quotes(match):
        content = match.group(0)
        if content.startswith('http://') or content.startswith('https://'):
            return content
        return content.replace("'", '"')

    s = re.sub(r"(?:'[^']*')", lambda m: repl_quotes(m), s)

    # Remove trailing commas
    s = re.sub(r',\s*([}\]])', r'\1', s)

    # Remove comments (// ...)
    s = re.sub(r'//.*', '', s)

    return s.strip()

def _extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        fixed = _clean_json_str(text)
        try:
            return json.loads(fixed)
        except Exception:
            return None

def _data_uri_from_bytes(content: bytes, mime: str) -> str:
    b64 = base64.b64encode(content).decode('ascii')
    return f'data:{mime};base64,{b64}'

def _guess_mime_from_ext(url: str) -> str:
    url = url.lower()
    if url.endswith('.png'):
        return 'image/png'
    if url.endswith('.jpg') or url.endswith('.jpeg'):
        return 'image/jpeg'
    if url.endswith('.svg'):
        return 'image/svg+xml'
    return 'application/octet-stream'

def _fetch_and_embed_image(path_or_url: str) -> Optional[str]:
    '''
    Accepts http(s) URL, local path, or data URI. Returns a data URI.
    '''
    if not path_or_url:
        return None
    if path_or_url.startswith('data:'):
        return path_or_url
    if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
        try:
            r = requests.get(path_or_url, timeout=15)
            r.raise_for_status()
            mime = r.headers.get('Content-Type') or _guess_mime_from_ext(path_or_url)
            return _data_uri_from_bytes(r.content, mime)
        except Exception:
            return None
    # local file path
    if os.path.exists(path_or_url):
        try:
            with open(path_or_url, 'rb') as f:
                data = f.read()
            mime = _guess_mime_from_ext(path_or_url)
            return _data_uri_from_bytes(data, mime)
        except Exception:
            return None
    return None

# --------------------- Simple SVG Chart Generators ---------------------
def _svg_bar_chart(title: str, series: List[Dict[str, Any]], width: int = 700, height: int = 280) -> str:
    '''
    series = [{ 'label': 'Mon', 'value': 40 }, ...]
    '''
    padding = 24
    chart_w = width - padding * 2
    chart_h = height - padding * 2 - 24  # leave room for title
    values = [float(s.get('value', 0) or 0) for s in series] or [0.0]
    max_v = max(values) or 1.0
    bar_w = chart_w / max(1, len(series))
    x0 = padding
    y0 = padding + 20  # shift down for title

    rects = []
    labels = []
    for i, s in enumerate(series):
        v = float(s.get('value', 0) or 0)
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

    svg = f'''
    <svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#2C7BE5"/>
          <stop offset="100%" stop-color="#6C63FF"/>
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill="white"/>
      {title_el}
      <g fill="url(#g1)" stroke="none">
        {''.join(grid)}
        {''.join(rects)}
      </g>
      <g fill="#333">
        {''.join(labels)}
      </g>
    </svg>
    '''
    return svg

def _svg_donut_chart(title: str, items: List[Dict[str, Any]], width: int = 260, height: int = 260) -> str:
    '''
    items = [{ 'label':'Deep', 'value':30 }, ...] -> percentage donut
    '''
    total = sum(float(i.get('value', 0) or 0) for i in items) or 1.0
    cx, cy, r = width/2, height/2, min(width, height)/2 - 16
    stroke_w = 28
    start_angle = -90  # start at top
    arcs = []
    legend = []
    colors = ['#2C7BE5', '#6C63FF', '#00C9A7', '#F7B924', '#FF6F61', '#A78BFA', '#22D3EE']
    for idx, item in enumerate(items):
        val = float(item.get('value', 0) or 0)
        pct = (val / total)
        ang = pct * 360.0
        end_angle = start_angle + ang
        large_arc = 1 if ang > 180 else 0

        # polar to cart
        def pt(angle_deg):
            import math
            rad = math.radians(angle_deg)
            return cx + r * math.cos(rad), cy + r * math.sin(rad)

        x1, y1 = pt(start_angle)
        x2, y2 = pt(end_angle)
        color = colors[idx % len(colors)]
        path = (
            f'<path d="M{x1:.2f},{y1:.2f} A{r:.2f},{r:.2f} 0 {large_arc} 1 {x2:.2f},{y2:.2f}" '
            f'stroke="{color}" stroke-width="{stroke_w}" fill="none" />'
        )
        arcs.append(path)
        legend.append(f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;"><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;"></span><span style="font-size:12px;color:#333">{_escape(str(item.get("label","")))} — {round(pct*100)}%</span></div>')
        start_angle = end_angle

    title_el = f'<div style="text-align:center;font-size:14px;font-weight:600;margin-bottom:8px">{_escape(title)}</div>'
    center_text = '<text x="{:.1f}" y="{:.1f}" font-size="14" text-anchor="middle" fill="#333">100%</text>'.format(cx, cy+5)

    svg = f'''
    <div style="display:flex;gap:16px;align-items:center;justify-content:center;">
      <div>
        {title_el}
        <svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
          <circle cx="{cx}" cy="{cy}" r="{r}" fill="#f9fafb"/>
          {''.join(arcs)}
          {center_text}
        </svg>
      </div>
      <div>
        {''.join(legend)}
      </div>
    </div>
    '''
    return svg

def _svg_line_chart(title: str, series: List[Dict[str, Any]], width: int = 700, height: int = 260) -> str:
    '''
    series = [{ 'label':'2025-09-01', 'value': 72 }, ...]
    '''
    padding = 30
    chart_w = width - padding * 2
    chart_h = height - padding * 2 - 20
    values = [float(s.get('value', 0) or 0) for s in series] or [0.0]
    max_v = max(values) or 1.0
    min_v = min(values) if max_v != min(values) else 0.0
    if max_v == min_v:
        max_v += 1.0
    x0 = padding
    y0 = padding + 10

    pts = []
    for i, s in enumerate(series):
        x = x0 + i * (chart_w / max(1, len(series)-1))
        val = float(s.get('value', 0) or 0)
        y = y0 + (1 - (val - min_v) / (max_v - min_v)) * chart_h
        pts.append((x, y))

    polyline = ' '.join(f"{x:.1f},{y:.1f}" for x, y in pts)
    labels = []
    for i, s in enumerate(series):
        x, y = pts[i]
        lbl = str(s.get('label', ''))
        labels.append(f'<text x="{x:.1f}" y="{y0 + chart_h + 16:.1f}" font-size="10" text-anchor="middle">{_escape(lbl)}</text>')

    title_el = f'<text x="{width/2:.1f}" y="{padding-2:.1f}" font-size="14" font-weight="600" text-anchor="middle">{_escape(title)}</text>'
    grid = []
    for gy in range(5):
        y = y0 + gy * (chart_h/4)
        grid.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+chart_w}" y2="{y:.1f}" stroke="#ddd" stroke-dasharray="3,3" />')

    svg = f'''
    <svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="gl" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#00C9A7"/>
          <stop offset="100%" stop-color="#6C63FF"/>
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill="white"/>
      {title_el}
      {''.join(grid)}
      <polyline fill="none" stroke="url(#gl)" stroke-width="3" points="{polyline}"/>
      <g fill="#333">
        {''.join(labels)}
      </g>
    </svg>
    '''
    return svg

# ----------------------------- Groq LLM -----------------------------
def _groq_structured_analysis(user_payload: dict) -> dict:
    if not GROQ_API_KEY:
        # No key: assume payload already has what we need
        return {
            'meta': {
                'generated_at': _now_iso(),
                'model': GROQ_MODEL,
                'title': user_payload.get('title') or 'Smart Analysis Report',
                'subtitle': user_payload.get('subtitle') or 'AI‑assisted summary and insights',
            },
            'overview': user_payload.get('overview') or 'Auto mode (no GROQ_API_KEY set). Using provided payload.',
            'insights': user_payload.get('insights') or [
                'Provide a Groq API key to enable AI‑generated insights.',
                'You can still pass your own \'sections\' or \'insights\' to render.'
            ],
            'recommendations': user_payload.get('recommendations') or [
                'Connect Groq to generate personalized suggestions.',
            ],
            'charts': user_payload.get('charts') or [],
            'sections': user_payload.get('sections') or [],
            'images': user_payload.get('images') or [],
            'metrics': user_payload.get('metrics') or {},
        }

    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }

    sys_prompt = (
        'You are a data‑savvy sleep & wellness assistant. '
        "Given the user's JSON payload, produce STRICT JSON with keys: "
        'meta{title,subtitle}, overview, insights[], recommendations[], sections[], charts[], images[], metrics{}. '
        'Each section: {title, body}. '
        "charts is an array of chart specs: {type: 'bar'|'line'|'donut', title, data: [ {label, value} ]}. "
        'images is a list of {title, url_or_data_uri}. '
        'DO NOT include prose outside JSON. DO NOT include markdown fences.'
    )

    user_prompt = {
        'role': 'user',
        'content': json.dumps(user_payload, ensure_ascii=False)
    }

    body = {
        'model': GROQ_MODEL,
        'messages': [
            {'role': 'system', 'content': sys_prompt},
            user_prompt
        ],
        'temperature': 0.3
    }

    r = requests.post(GROQ_URL, headers=headers, data=json.dumps(body), timeout=60)
    r.raise_for_status()
    data = r.json()
    content = data['choices'][0]['message']['content']
    parsed = _extract_json(content)
    if not parsed:
        raise ValueError('LLM returned non‑JSON or unparseable content')
    parsed.setdefault('meta', {})
    parsed['meta'].setdefault('generated_at', _now_iso())
    parsed['meta'].setdefault('model', GROQ_MODEL)
    parsed.setdefault('overview', '')
    parsed.setdefault('insights', [])
    parsed.setdefault('recommendations', [])
    parsed.setdefault('sections', [])
    parsed.setdefault('charts', [])
    parsed.setdefault('images', [])
    parsed.setdefault('metrics', {})
    return parsed

# ----------------------------- HTML Template -----------------------------
BASE_CSS = CSS(string='''
@page {
  size: A4;
  margin: 20mm 15mm 22mm 15mm;
  @bottom-center {
    content: "Page " counter(page) " of " counter(pages);
    font-size: 10px;
    color: #666;
  }
}
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Open Sans', sans-serif;
  color: #111827;
}
h1, h2, h3 { color: #111827; margin: 0; }
.section { page-break-inside: avoid; margin: 10px 0 18px; }
.card {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 14px 16px;
  background: white;
}
.kicker { color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 10px; background: #EEF2FF; color: #4338CA; font-size: 11px;
}
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.hero {
  border-radius: 16px;
  padding: 18px;
  background: linear-gradient(135deg, #EEF2FF, #E6FFFA);
  border: 1px solid #e5e7eb;
}
.small { font-size: 12px; color: #6b7280; }
.hr { height: 1px; background: #e5e7eb; margin: 8px 0; }
.img { width: 100%; border-radius: 10px; border: 1px solid #e5e7eb; }
ul { margin: 6px 0 6px 20px; }
''')

def _render_html(ctx: dict) -> str:
    hero_svg = f'''
    <svg viewBox="0 0 820 120" width="100%" height="120" xmlns="http://www.w3.org/2000/svg" style="border-radius:12px">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#6366F1"/>
          <stop offset="100%" stop-color="#06B6D4"/>
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="820" height="120" fill="url(#bg)" rx="12"/>
      <text x="24" y="48" font-size="22" fill="white" font-weight="600">{_escape(ctx["meta"].get("title", "Smart Report"))}</text>
      <text x="24" y="76" font-size="14" fill="#E0E7FF">{_escape(ctx["meta"].get("subtitle", ""))}</text>
      <text x="24" y="100" font-size="11" fill="#E0E7FF">Generated: {_escape(ctx["meta"].get("generated_at",""))}</text>
    </svg>
    '''

    # Charts (SVG strings) from ctx["charts"]
    chart_blocks = []
    for ch in ctx.get('charts', []):
        ctype = (ch.get('type') or '').lower()
        title = ch.get('title') or 'Chart'
        data = ch.get('data') or []
        if ctype == 'bar':
            chart_blocks.append(_svg_bar_chart(title, data))
        elif ctype == 'donut':
            chart_blocks.append(_svg_donut_chart(title, data))
        else:
            chart_blocks.append(_svg_line_chart(title, data))

    # Images
    image_blocks = []
    for img in ctx.get('images', []):
        src = img.get('url_or_data_uri')
        if src:
            data_uri = _fetch_and_embed_image(src)
            if data_uri:
                ttl = _escape(img.get('title',''))
                image_blocks.append(f'<div class="card"><div class="small" style="margin-bottom:6px">{ttl}</div><img class="img" src="{data_uri}"/></div>')

    # Sections
    section_blocks = []
    for sec in ctx.get('sections', []):
        title = _escape(str(sec.get('title','Section')))
        body  = str(sec.get('body','')).replace('\n', '<br/>')
        section_blocks.append(f'<div class="card section"><div class="kicker">{title}</div><div style="margin-top:6px;font-size:13px;line-height:1.5">{body}</div></div>')

    # Insights / Recs
    insights = ctx.get('insights') or []
    recs = ctx.get('recommendations') or []

    tpl = Template('''
    <html>
      <body>
        <div class="hero">{{ hero_svg | safe }}</div>

        <div class="section grid-2">
          <div class="card">
            <div class="kicker">Overview</div>
            <div style="margin-top:6px;font-size:13px;line-height:1.5">{{ overview|safe }}</div>
          </div>
          <div class="card">
            <div class="kicker">At a glance</div>
            <ul>
              {% for i in insights %}<li>{{ i }}</li>{% endfor %}
            </ul>
            <div class="hr"></div>
            <div class="kicker" style="margin-top:6px">Recommendations</div>
            <ul>
              {% for r in recs %}<li>{{ r }}</li>{% endfor %}
            </ul>
          </div>
        </div>

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

        {% for sec in section_blocks %}
          {{ sec | safe }}
        {% endfor %}
      </body>
    </html>
    ''')
    html = tpl.render(
        hero_svg=hero_svg,
        overview=_escape(ctx.get('overview','')).replace('\n', '<br/>'),
        insights=insights,
        recs=recs,
        chart_blocks=chart_blocks,
        image_blocks=image_blocks,
        section_blocks=section_blocks,
    )
    return html

# ----------------------------- Route -----------------------------
@pdf_bp.route('/generate', methods=['POST'])
def generate_pdf():
    '''
    POST JSON:
    {
      "title": "...",
      "subtitle": "...",
      "overview": "...",
      "insights": ["...", "..."],
      "recommendations": ["...", "..."],
      "sections": [{"title":"", "body":""}, ...],
      "charts": [
        {"type":"bar","title":"Scores","data":[{"label":"Mon","value":64}, ...]},
        {"type":"donut","title":"Sleep Stages","data":[{"label":"Deep","value":90}, ...]},
        {"type":"line","title":"Trend","data":[{"label":"2025-09-01","value":72}, ...]}
      ],
      "images": [{"title":"Hypnogram","url_or_data_uri":"https://... or data:..."}],
      "metrics": {...},
      "auto_groq": true,
      "download_filename": "sleep_report.pdf"
    }
    '''
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({'ok': False, 'error': 'Invalid JSON payload'}), 400

    auto_groq = bool(payload.get('auto_groq')) or not any([
        payload.get('overview'),
        payload.get('insights'),
        payload.get('sections')
    ])

    try:
        if auto_groq:
            ctx = _groq_structured_analysis(payload)
        else:
            # Build ctx directly from payload
            ctx = {
                'meta': {
                    'generated_at': _now_iso(),
                    'model': GROQ_MODEL,
                    'title': payload.get('title') or 'Smart Analysis Report',
                    'subtitle': payload.get('subtitle') or '',
                },
                'overview': payload.get('overview') or '',
                'insights': payload.get('insights') or [],
                'recommendations': payload.get('recommendations') or [],
                'sections': payload.get('sections') or [],
                'charts': payload.get('charts') or [],
                'images': payload.get('images') or [],
                'metrics': payload.get('metrics') or {},
            }
    except Exception as e:
        return jsonify({'ok': False, 'error': f'GROQ request failed: {str(e)}'}), 500

    # Render HTML
    html = _render_html(ctx)

    # Make PDF
    try:
        pdf_bytes = HTML(string=html, base_url='.').write_pdf(stylesheets=[BASE_CSS])
    except Exception as e:
        return jsonify({'ok': False, 'error': f'PDF render failed: {str(e)}'}), 500

    # Deliver
    filename = payload.get('download_filename') or (ctx['meta'].get('title','report').lower().replace(' ', '_') + '.pdf')
    return send_file(BytesIO(pdf_bytes), as_attachment=True, download_name=filename, mimetype='application/pdf')

# ----------------------------- Quick Self-Test (optional) -----------------------------
@pdf_bp.route('/_demo', methods=['GET'])
def _demo():
    demo_ctx = {
        'title': 'Sleep AI Report',
        'subtitle': 'Personalized analysis',
        'overview': 'Your sleep quality improved this week. Deep sleep increased and wake‑after‑sleep‑onset declined.',
        'insights': ['Deep sleep up 12%', 'Sleep latency improved', 'Screen time near bedtime still high'],
        'recommendations': ['Reduce caffeine after 4PM', 'Move workouts earlier', 'Use blue‑light filter 2 hours pre‑bed'],
        'charts': [
            {'type':'bar','title':'Sleep Quality (last 7 days)','data':[
                {'label':'Mon','value':60},{'label':'Tue','value':68},{'label':'Wed','value':62},
                {'label':'Thu','value':74},{'label':'Fri','value':70},{'label':'Sat','value':78},{'label':'Sun','value':73}
            ]},
            {'type':'donut','title':'Sleep Stages','data':[
                {'label':'Deep','value':90},{'label':'REM','value':110},{'label':'Light','value':160},{'label':'Awake','value':20}
            ]},
            {'type':'line','title':'Latency (mins)','data':[
                {'label':'09-01','value':28},{'label':'09-02','value':24},{'label':'09-03','value':22},
                {'label':'09-04','value':20},{'label':'09-05','value':21},{'label':'09-06','value':19},{'label':'09-07','value':18}
            ]}
            ],
        'images': [
            {'title':'Sample Hypnogram','url_or_data_uri':'https://upload.wikimedia.org/wikipedia/commons/3/30/Hypnogram.svg'}
        ],
        'download_filename': 'demo_sleep_ai_report.pdf'
    }
    with pdf_bp.test_request_context(json=demo_ctx):
        return generate_pdf()
