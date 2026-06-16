# Ghost Social — Mentor–Mentee Matching (LLM arm / Condition B)

Knowledge-graph-style matching tool for the UW Tech Exploration Lab mentor–mentee
experiment. Given the mentor interest sheet and the student application sheet, it
produces one-to-one pairings with grounded rationales and an evidence-based
confidence score, then serves them in a small Flask app.

## What it does

1. **Column mapping** (`data_mapping.py`) — maps the raw Google-Form headers (smart
   quotes, embedded newlines) to clean internal fields via an editable dict.
2. **Capacity from free text** — an editable rules dict turns "Ongoing engagement
   throughout the year" → 4 slots, "A few hours per semester" → 2, etc.; blank or
   unrecognized → `unknown` (defaults to 1, flagged and logged).
3. **Pool filter** — mentors are eligible for 1:1 matching only if their formats
   include "1:1 mentoring conversations" and `consent_to_share` ≠ No. Excluded
   mentors are reported with reasons. This is a **hard constraint** — ineligible
   mentors are not nodes in the assignment graph.
4. **Scoring** — four criteria (skill, domain, goals, style) 1–10.
   - `offline` (default): deterministic keyword-overlap on the real profile fields.
     **Zero API calls, fully reproducible.**
   - `llm`: Claude tool-use scoring + rationales (Condition B). Needs a key.
5. **Assignment** — OR-Tools **min-cost max-flow** (capacity = hard limit;
   forbidden pairs are simply absent arcs), cross-checked against scipy's Hungarian
   algorithm. Capacity shortfall is normal (85 slots, 116 students) → excess
   students are left **unmatched and surfaced explicitly**, never silently dropped.
6. **Output** — `matches.csv`, `unmatched.csv`, `metadata.json`, `score_matrix.npz`
   (in `output/`, git-ignored) and the committed `display_data/results.json` —
   a single internal view carrying **real** `mentor_name/email`, `student_name/email`,
   `confidence`, and the full `rationale` for matched records (and `student_name/email`
   for unmatched). **Contains student PII → keep the repo private.**

The Flask app is **one view** with real identities everywhere: searchable cards
(mentor + student name/email, confidence, expandable rationale), Matched/Unmatched
tabs, pagination, per-card copy-contact, CSV + PDF export.

## Run locally

```bash
cd mentor-match
python3.12 -m venv venv312 && ./venv312/bin/pip install -r requirements.txt

# Generate a run (offline, deterministic — no key needed):
./venv312/bin/python matcher.py \
  --mentors "/path/TEL Mentorship and Advisor Interest (Responses).xlsx" \
  --students "/path/2026 Application - UW Tech Exploration Lab (Responses).csv" \
  --mode offline --output-dir output --display-out display_data/results.json

# Serve the committed results (display mode, zero API):
APP_MODE=display ./venv312/bin/gunicorn app:app --bind 0.0.0.0:3000
#  → http://localhost:3000
```

For the real LLM (Condition B) run: `export ANTHROPIC_API_KEY=...` and add
`--mode llm` (default model `claude-sonnet-4-6`, `temperature=0`, seeded —
logged in `metadata.json`). To rebuild the committed real-identity display from an
existing run without recomputing: `python rebuild_display.py --mentors … --students …`.

## Deploy to Render

The repo ships a `render.yaml` blueprint. **Set `SITE_PASSWORD` in the Render
dashboard to gate the whole site** (one password, HTTP Basic auth — the only
access control). If unset, the site is open. `ANTHROPIC_API_KEY` + `APP_MODE=live`
are only needed to enable in-app regeneration (paid).

## Privacy

`.env`, `output/`, and `private/` are git-ignored. The committed
`display_data/results.json` **contains real student names and emails**, so the
**repository must be Private**. Access is controlled at the app layer by
`SITE_PASSWORD`.
