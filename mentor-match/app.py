#!/usr/bin/env python3
"""
Flask app for the Ghost Social mentor-matching LLM arm (Condition B).

Two modes, switched by the APP_MODE env var:
  - display (default): serves the committed, anonymized results in
    display_data/results.json. ZERO Claude API calls — safe to share publicly,
    needs no API key, costs nothing per visit.
  - live: additionally allows a password-gated regenerate (real matching, which
    is the only thing that spends API budget).

Everything renders from one committed JSON; the deployed default boots and serves
it with no secrets set.
"""

import io
import json
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from analysis import cohens_kappa

BASE = Path(__file__).parent
DISPLAY_FILE = BASE / "display_data" / "results.json"

APP_MODE = os.environ.get("APP_MODE", "display").lower()      # display | live
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

app = Flask(__name__)
# In-memory override produced by a live regenerate (ephemeral; the committed file
# is always the boot default).
_override: dict | None = None


def load_results() -> dict:
    if _override is not None:
        return _override
    if DISPLAY_FILE.exists():
        return json.loads(DISPLAY_FILE.read_text())
    return {"stats": {}, "matches": [], "unmatched": [], "reconciliation": {},
            "generation_mode": "none",
            "error": "No committed results found (display_data/results.json missing)."}


# ── Pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", mode=APP_MODE, live=(APP_MODE == "live"))


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "mode": APP_MODE, "has_results": DISPLAY_FILE.exists()})


# ── Data (zero API calls) ────────────────────────────────────────────────────

@app.route("/api/results")
def api_results():
    return jsonify(load_results())


@app.route("/api/analysis")
def api_analysis():
    """Derived analysis computed from the committed results — no uploads, no API."""
    d = load_results()
    matches = d.get("matches", [])
    crit_keys = ["skill_alignment", "domain_fit", "goal_compatibility", "style_fit"]
    crit_avg = {}
    for k in crit_keys:
        vals = [m["criteria_scores"].get(k) for m in matches if m.get("criteria_scores")]
        vals = [v for v in vals if isinstance(v, (int, float))]
        crit_avg[k] = round(sum(vals) / len(vals), 2) if vals else 0
    conf_hist = {i: 0 for i in range(1, 11)}
    for m in matches:
        c = int(m.get("confidence_score", 0))
        if 1 <= c <= 10:
            conf_hist[c] += 1
    rec = d.get("reconciliation", {})
    slots = rec.get("total_mentor_slots", 0) or 0
    util = round(100 * len(matches) / slots, 1) if slots else 0
    return jsonify({
        "criteria_avg": crit_avg,
        "confidence_histogram": conf_hist,
        "slot_utilization_pct": util,
        "match_rate_pct": round(100 * len(matches) / max(1, rec.get("students_total", 1)), 1),
        "reconciliation": rec,
    })


# ── Cross-condition comparison (Cohen's kappa) — optional, no API ────────────

@app.route("/api/comparison", methods=["POST"])
def api_comparison():
    """Upload another condition's CSV (mentee_name,mentor_name[,score]) and get
    agreement vs. this tool's matches. Keys must match what's displayed."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "CSV file required"}), 400
    import csv as _csv

    def _num(v, default=5):
        try:
            return max(1, min(10, int(float(v))))
        except (TypeError, ValueError):
            return default

    rows = list(_csv.DictReader(io.StringIO(f.read().decode("utf-8", "replace"))))
    other = {(r.get("mentee_name", "").strip(), r.get("mentor_name", "").strip()):
             _num(r.get("confidence_score", r.get("score"))) for r in rows}
    mine = {(m["mentee_name"], m["mentor_name"]): int(m["confidence_score"])
            for m in load_results().get("matches", [])}
    # Cohen's kappa over the score buckets for pairs BOTH conditions proposed.
    try:
        kappa = round(cohens_kappa(mine, other), 3)
    except Exception:
        kappa = None
    return jsonify({"cohens_kappa": kappa,
                    "shared_pairs": len(set(mine) & set(other)),
                    "this_tool_pairs": len(mine), "uploaded_pairs": len(other),
                    "note": "Agreement on confidence buckets for pairs both conditions matched. "
                            "Keys must match the displayed mentee/mentor ids."})


# ── PDF export ───────────────────────────────────────────────────────────────

@app.route("/api/report")
def api_report():
    d = load_results()
    pdf = _build_pdf(d)
    return send_file(io.BytesIO(pdf), mimetype="application/pdf",
                     as_attachment=True, download_name="ghost_social_matches.pdf")


def _build_pdf(d: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=40, bottomMargin=40)
    st = getSampleStyleSheet()
    small = ParagraphStyle("s", parent=st["Normal"], fontSize=8, leading=10)
    story = [Paragraph("Ghost Social — Mentor/Mentee Matches (LLM arm)", st["Title"])]
    s = d.get("stats", {})
    story.append(Paragraph(
        f"Mode: {d.get('generation_mode','?')} &nbsp; Matched: {s.get('matched','?')} &nbsp; "
        f"Unmatched: {s.get('unmatched','?')} &nbsp; Avg confidence: {s.get('avg_confidence','?')} &nbsp; "
        f"Solver agreement: {s.get('solver_agreement','?')}", st["Normal"]))
    story.append(Spacer(1, 12))
    data = [["Mentee", "Mentor", "Conf", "Rationale"]]
    for m in d.get("matches", []):
        data.append([m["mentee_name"], Paragraph(str(m["mentor_name"]), small),
                     str(m["confidence_score"]), Paragraph(m["rationale"], small)])
    t = Table(data, colWidths=[60, 90, 28, 320], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
    ]))
    story.append(t)
    doc.build(story)
    return buf.getvalue()


# ── Gated regenerate (the ONLY path that may spend API budget) ───────────────

@app.route("/api/match", methods=["POST"])
def api_match():
    global _override
    if APP_MODE != "live":
        return jsonify({"error": "Regeneration is disabled in display mode. "
                                 "Set APP_MODE=live to enable it."}), 403
    if not ADMIN_PASSWORD or request.form.get("password", "") != ADMIN_PASSWORD:
        return jsonify({"error": "Unauthorized — admin password required."}), 401

    mentor_f = request.files.get("mentors")
    student_f = request.files.get("students")
    if not mentor_f or not student_f:
        return jsonify({"error": "Both mentor and student files are required."}), 400

    from matcher import MentorMatcher
    work = Path(tempfile.mkdtemp())
    mp = work / ("mentors" + Path(mentor_f.filename or "m.xlsx").suffix)
    sp = work / ("students" + Path(student_f.filename or "s.csv").suffix)
    mentor_f.save(str(mp)); student_f.save(str(sp))

    mode = "llm" if os.environ.get("ANTHROPIC_API_KEY") else "offline"
    disp = work / "results.json"
    try:
        mm = MentorMatcher(output_dir=str(work / "out"), mode=mode,
                           display_out=str(disp), anonymize_students=True)
        mm.run(str(mp), str(sp))
        _override = json.loads(disp.read_text())
    except Exception as exc:
        return jsonify({"error": f"Run failed: {exc}"}), 500
    return jsonify({"ok": True, "mode": mode, "stats": _override.get("stats", {})})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    print(f"\n  Ghost Social matcher — mode={APP_MODE} — http://0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
