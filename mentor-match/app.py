#!/usr/bin/env python3
"""
Flask app for the Ghost Social mentor-matching tool — one internal view.

Single view with real identities everywhere: each match shows mentor name + email
and student name + email, the confidence score, and the full rationale. Matched
and unmatched lists, search, CSV export, and a PDF all carry the real data.

Security: if SITE_PASSWORD is set, the WHOLE site is gated behind it (HTTP Basic
auth, one password). If unset, the site is open. There is no admin/blind tier.

NOTE: display_data/results.json contains real student names + emails, so the repo
must be PRIVATE.
"""

import csv
import hmac
import io
import json
import os
import tempfile
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

BASE = Path(__file__).parent
DISPLAY_FILE = BASE / "display_data" / "results.json"

APP_MODE = os.environ.get("APP_MODE", "display").lower()
SITE_PASSWORD = os.environ.get("SITE_PASSWORD") or os.environ.get("ADMIN_PASSWORD") or ""

app = Flask(__name__)
_override: dict | None = None     # from a live regenerate (ephemeral)


def load_results() -> dict:
    if _override is not None:
        return _override
    if DISPLAY_FILE.exists():
        return json.loads(DISPLAY_FILE.read_text())
    return {"matches": [], "unmatched": [], "generation_mode": "none",
            "error": "No results (display_data/results.json missing)."}


# ── Whole-site password gate (HTTP Basic auth) ───────────────────────────────

@app.before_request
def _gate():
    if not SITE_PASSWORD or request.path == "/healthz":
        return None
    auth = request.authorization
    if auth and hmac.compare_digest((auth.password or "").encode("utf-8"), SITE_PASSWORD.encode("utf-8")):
        return None
    return Response("Authentication required.", 401,
                    {"WWW-Authenticate": 'Basic realm="Ghost Social"'})


# ── Pages / health ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "mode": APP_MODE,
                    "has_results": DISPLAY_FILE.exists(), "gated": bool(SITE_PASSWORD)})


# ── Data ─────────────────────────────────────────────────────────────────────

@app.route("/api/results")
def api_results():
    d = load_results()
    return jsonify({
        "generation_mode": d.get("generation_mode"),
        "generated_at": d.get("generated_at"),
        "counts": {"matched": len(d.get("matches", [])), "unmatched": len(d.get("unmatched", []))},
        "matches": d.get("matches", []),
        "unmatched": d.get("unmatched", []),
    })


@app.route("/api/export.csv")
def api_csv():
    d = load_results()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["status", "mentor_name", "mentor_email", "student_name", "student_email", "confidence"])
    for m in d.get("matches", []):
        w.writerow(["matched", m.get("mentor_name", ""), m.get("mentor_email", ""),
                    m.get("student_name", ""), m.get("student_email", ""), m.get("confidence", "")])
    for u in d.get("unmatched", []):
        w.writerow(["unmatched", "", "", u.get("student_name", ""), u.get("student_email", ""), ""])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=ghost_social_roster.csv"})


@app.route("/api/report")
def api_report():
    return send_file(io.BytesIO(_build_pdf(load_results())), mimetype="application/pdf",
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
    matches, unmatched = d.get("matches", []), d.get("unmatched", [])
    story = [Paragraph("Ghost Social — Mentor / Mentee Matches", st["Title"]),
             Paragraph(f"Matched: {len(matches)} &nbsp; Unmatched: {len(unmatched)} &nbsp; "
                       f"Mode: {d.get('generation_mode','?')}", st["Normal"]), Spacer(1, 12)]
    data = [["Mentor", "Student", "Conf", "Rationale"]]
    for m in matches:
        data.append([
            Paragraph(f"{m.get('mentor_name','')}<br/><font size=7 color='#666666'>{m.get('mentor_email','')}</font>", small),
            Paragraph(f"{m.get('student_name','')}<br/><font size=7 color='#666666'>{m.get('student_email','')}</font>", small),
            str(m.get("confidence", "")), Paragraph(str(m.get("rationale", "")), small)])
    t = Table(data, colWidths=[110, 110, 26, 250], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C5050C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f3f3")]),
    ]))
    story.append(t)
    if unmatched:
        story.append(Spacer(1, 16))
        story.append(Paragraph("Unmatched students", st["Heading2"]))
        ud = [["Student", "Email"]] + [[u.get("student_name", ""), u.get("student_email", "")] for u in unmatched]
        ut = Table(ud, colWidths=[160, 240], repeatRows=1)
        ut.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C5050C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db"))]))
        story.append(ut)
    doc.build(story)
    return buf.getvalue()


# ── Regenerate (live mode only; the only paid action) ────────────────────────

@app.route("/api/match", methods=["POST"])
def api_match():
    global _override
    if APP_MODE != "live":
        return jsonify({"error": "Regeneration disabled (set APP_MODE=live)."}), 403
    mentor_f, student_f = request.files.get("mentors"), request.files.get("students")
    if not mentor_f or not student_f:
        return jsonify({"error": "Both mentor and student files are required."}), 400
    from matcher import MentorMatcher
    work = Path(tempfile.mkdtemp())
    mp = work / ("mentors" + Path(mentor_f.filename or "m.xlsx").suffix)
    sp = work / ("students" + Path(student_f.filename or "s.csv").suffix)
    mentor_f.save(str(mp)); student_f.save(str(sp))
    mode = "llm" if os.environ.get("ANTHROPIC_API_KEY") else "offline"
    try:
        mm = MentorMatcher(output_dir=str(work / "out"), mode=mode, display_out=str(work / "pub.json"))
        mm.run(str(mp), str(sp))
        _override = json.loads((work / "pub.json").read_text())
    except Exception as exc:
        return jsonify({"error": f"Run failed: {exc}"}), 500
    return jsonify({"ok": True, "mode": mode, "counts": {"matched": len(_override.get("matches", [])),
                                                         "unmatched": len(_override.get("unmatched", []))}})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    print(f"\n  Ghost Social — mode={APP_MODE} gated={bool(SITE_PASSWORD)} — http://0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
