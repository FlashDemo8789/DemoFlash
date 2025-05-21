"""
PDF Report Generator
────────────────────
Renders a one-page PDF summarising pillar scores, success probability
and alerts.  Uses WeasyPrint (HTML → PDF) so it stays server-side and
doesn't require LaTeX.

Install once:
    pip install weasyprint Jinja2

Directory structure:
flashcamp/
└─ reports/
   ├─ templates/
   │   └─ summary.html      # basic Jinja template (auto-created below)
   └─ pdf/
       └─ <startup_id>.pdf
"""

from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

# Determine project root relative to this file's location
# Assumes this file is in flashcamp/backend/
BACKEND_DIR = Path(__file__).resolve().parent
FLASHCAMP_DIR = BACKEND_DIR.parent
ROOT = FLASHCAMP_DIR.parent # FLASH directory

# Define paths relative to FLASHCAMP_DIR for consistency within the package
TPL_DIR = FLASHCAMP_DIR / "reports" / "templates"
PDF_DIR = FLASHCAMP_DIR / "reports" / "pdf"

# Ensure directories exist
TPL_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# Create a simple default template on first run
_default_tpl_path = TPL_DIR / "summary.html"
if not _default_tpl_path.exists():
    _default_tpl_content = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { font-family: sans-serif; padding: 2em; }
    h1   { margin-bottom: 0; color: #333; }
    h2   { margin-top: 1.5em; margin-bottom: 0.5em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; color: #555; }
    table{ border-collapse: collapse; width: 100%; margin-bottom: 1.5em; }
    th, td { padding: 8px 10px; border: 1px solid #ccc; text-align: left; }
    th   { background-color: #f2f2f2; font-weight: bold; }
    ul   { padding-left: 20px; }
    li   { margin-bottom: 0.5em; color: #d9534f; }
    .timestamp { color: #777; font-size: 0.9em; margin-bottom: 2em; }
    .probability { font-size: 28px; font-weight: bold; color: #4f46e5; }
  </style>
</head>
<body>
  <h1>FlashCAMP Report – {{ startup_id }}</h1>
  <p class="timestamp">Generated: {{ timestamp }}</p>

  <h2>Pillar Scores</h2>
  <table>
    <tr><th>Pillar</th><th>Score</th></tr>
    {% for k, v in pillars.items() %}
    <tr><td>{{ k.title() }}</td><td>{{ v }}</td></tr>
    {% endfor %}
  </table>

  <h2>Success Probability</h2>
  <p class="probability">{{ (success_prob * 100) | round(1) }}%</p>

  {% if alerts %}
    <h2>Alerts</h2>
    <ul>{% for a in alerts %}<li>{{ a }}</li>{% endfor %}</ul>
  {% endif %}
</body>
</html>"""
    _default_tpl_path.write_text(_default_tpl_content)
    print(f"✅ Default template created at: {_default_tpl_path}")

# Set up Jinja2 environment
env = Environment(
    loader=FileSystemLoader(str(TPL_DIR)),
    autoescape=select_autoescape(["html"]),
)

# ------------------------------------------------------------------ #
def make_pdf(
    startup_id: str,
    pillars: dict,
    success_prob: float,
    alerts: list[str] | None = None,
) -> Path:
    """
    Renders summary.html → PDF and returns the file path.
    """
    template = env.get_template("summary.html")
    html_content = template.render(
        startup_id=startup_id,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        pillars=pillars,
        success_prob=success_prob,
        alerts=alerts or [],
    )
    
    # Sanitize startup_id for filename
    safe_startup_id = "".join(c for c in startup_id if c.isalnum() or c in ('_', '-')).rstrip()
    if not safe_startup_id:
        safe_startup_id = "report"
        
    pdf_path = PDF_DIR / f"{safe_startup_id}.pdf"
    
    try:
        HTML(string=html_content).write_pdf(str(pdf_path))
        print(f"✅ PDF report generated: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"Error generating PDF for {startup_id}: {e}")
        # Consider raising the exception or returning None/error indicator
        raise # Re-raise the exception for the endpoint to handle 