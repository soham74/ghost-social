#!/usr/bin/env python3
"""
Flask app for the Ghost Social mentor-matching LLM arm (Condition B).

Two surfaces:
  - PUBLIC / blind (default): serves the committed, anonymized matches in
    display_data/results.json. ZERO Claude API calls, ZERO student PII. Safe to
    share for blind review.
  - ADMIN (password-gated): overlays the gitignored private roster to show real
    student + mentor names and emails per match, with a CSV roster export. The
    student->identity map never leaves the server and is never sent to the public
    surface.

APP_MODE=live additionally enables a password-gated regenerate (the only paid action).
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
ROSTER_FILE = BASE / "private" / "roster.json"

APP_MODE = os.environ.get("APP_MODE", "display").lower()                 # display | live
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD") or os.environ.get("SITE_PASSWORD") or ""

app = Flask(__name__)
_override_results: dict | None = None      # from a live regenerate (ephemeral)
_override_roster: dict | None = None


# ── data loaders ─────────────────────────────────────────────────────────────

def load_results() -> dict:
    if _override_results is not None:
        return _override_results
    if DISPLAY_FILE.exists():
        return json.loads(DISPLAY_FILE.read_text())
    return {"matches": [], "unmatched": [], "generation_mode": "none",
            "error": "No committed results (display_data/results.json missing)."}


def load_roster() -> dict | None:
    """Admin roster (PII). On Render set ADMIN_ROSTER (JSON); locally read the
    gitignored private/roster.json. Returns None if unavailable."""
    if _override_roster is not None:
        return _override_roster
    env = os.environ.get("ADMIN_ROSTER")
    if env:
        try:
            return json.loads(env)
        except json.JSONDecodeError:
            return None
    if ROSTER_FILE.exists():
        return json.loads(ROSTER_FILE.read_text())
    return None


def admin_ok(req) -> bool:
    supplied = req.headers.get("X-Admin-Password") or req.form.get("password") or req.args.get("password") or ""
    # Encode both sides so non-ASCII input returns a clean 401 (compare_digest on
    # str raises TypeError on non-ASCII); never opens the gate.
    return bool(ADMIN_PASSWORD) and hmac.compare_digest(supplied.encode("utf-8"), ADMIN_PASSWORD.encode("utf-8"))


# ── pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           admin_enabled=bool(ADMIN_PASSWORD),
                           live=(APP_MODE == "live"))


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "mode": APP_MODE,
                    "has_results": DISPLAY_FILE.exists(),
                    "admin_configured": bool(ADMIN_PASSWORD)})


# ── PUBLIC data (no PII, zero API) ───────────────────────────────────────────

@app.route("/api/results")
def api_results():
    d = load_results()
    # Defensive projection: only ever emit the blind fields to the public surface.
    return jsonify({
        "generation_mode": d.get("generation_mode"),
        "generated_at": d.get("generated_at"),
        "counts": {"matched": len(d.get("matches", [])), "unmatched": len(d.get("unmatched", []))},
        "matches": [{"student": m.get("student"), "mentor": m.get("mentor"),
                     "confidence": m.get("confidence"), "rationale": m.get("rationale")}
                    for m in d.get("matches", [])],
        "unmatched": list(d.get("unmatched", [])),
    })


# ── ADMIN data (PII, password-gated) ─────────────────────────────────────────

def _admin_rows():
    """Join public matches + roster into full-contact rows. Raises if no roster."""
    d = load_results()
    roster = load_roster()
    if roster is None:
        return None, None
    rmatched, runmatched = roster.get("matched", {}), roster.get("unmatched", {})
    matches = []
    for m in d.get("matches", []):
        label = m.get("student")
        info = rmatched.get(label, {})
        matches.append({
            "student_label": label,
            "student_name": info.get("student_name", ""),
            "student_email": info.get("student_email", ""),
            "mentor_name": m.get("mentor"),
            "mentor_email": info.get("mentor_email", ""),
            "confidence": m.get("confidence"),
            "rationale": m.get("rationale"),
        })
    unmatched = [{"student_label": lbl,
                  "student_name": runmatched.get(lbl, {}).get("student_name", ""),
                  "student_email": runmatched.get(lbl, {}).get("student_email", "")}
                 for lbl in d.get("unmatched", [])]
    return matches, unmatched


@app.route("/api/admin")
def api_admin():
    if not admin_ok(request):
        return jsonify({"error": "Unauthorized"}), 401
    matches, unmatched = _admin_rows()
    if matches is None:
        return jsonify({"error": "Admin roster not available on this instance. "
                                 "Set the ADMIN_ROSTER env var (contents of private/roster.json)."}), 503
    return jsonify({"matches": matches, "unmatched": unmatched})


@app.route("/api/admin/export.csv")
def api_admin_csv():
    if not admin_ok(request):
        return jsonify({"error": "Unauthorized"}), 401
    matches, unmatched = _admin_rows()
    if matches is None:
        return jsonify({"error": "Admin roster not available on this instance."}), 503
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["status", "mentor_name", "mentor_email", "student_name", "student_email", "confidence"])
    for m in matches:
        w.writerow(["matched", m["mentor_name"], m["mentor_email"], m["student_name"],
                    m["student_email"], m["confidence"]])
    for u in unmatched:
        w.writerow(["unmatched", "", "", u["student_name"], u["student_email"], ""])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=ghost_social_roster.csv"})


# ── PDF (public, no PII) ─────────────────────────────────────────────────────

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
    matches = d.get("matches", [])
    story = [Paragraph("Ghost Social — Mentor/Mentee Matches (blind)", st["Title"]),
             Paragraph(f"Matched: {len(matches)} &nbsp; Unmatched: {len(d.get('unmatched', []))} &nbsp; "
                       f"Mode: {d.get('generation_mode','?')}", st["Normal"]), Spacer(1, 12)]
    data = [["Student", "Mentor", "Conf", "Rationale"]]
    for m in matches:
        data.append([m.get("student", ""), Paragraph(str(m.get("mentor", "")), small),
                     str(m.get("confidence", "")), Paragraph(str(m.get("rationale", "")), small)])
    t = Table(data, colWidths=[60, 90, 28, 320], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
    ]))
    story.append(t)
    doc.build(story)
    return buf.getvalue()


# ── Gated regenerate (only paid action) ──────────────────────────────────────

@app.route("/api/match", methods=["POST"])
def api_match():
    global _override_results, _override_roster
    if APP_MODE != "live":
        return jsonify({"error": "Regeneration is disabled in display mode (set APP_MODE=live)."}), 403
    if not admin_ok(request):
        return jsonify({"error": "Unauthorized — admin password required."}), 401
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
        mm = MentorMatcher(output_dir=str(work / "out"), mode=mode,
                           display_out=str(work / "pub.json"),
                           roster_out=str(work / "roster.json"), anonymize_students=True)
        mm.run(str(mp), str(sp))
        _override_results = json.loads((work / "pub.json").read_text())
        _override_roster = json.loads((work / "roster.json").read_text())
    except Exception as exc:
        return jsonify({"error": f"Run failed: {exc}"}), 500
    return jsonify({"ok": True, "mode": mode,
                    "counts": {"matched": len(_override_results.get("matches", [])),
                               "unmatched": len(_override_results.get("unmatched", []))}})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    print(f"\n  Ghost Social matcher — mode={APP_MODE} — http://0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
