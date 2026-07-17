# Ghost Social — Mentor–Mentee Matching (LLM arm / Condition B)

Knowledge-graph-style matching tool for the UW Tech Exploration Lab mentor–mentee
experiment. Given the mentor interest sheet and the student application sheet, it
produces one-to-one pairings with grounded rationales and an evidence-based
confidence score, then serves them in a small Flask app.

## What it does

1. **Column mapping + stable ids** (`data_mapping.py`) — maps the raw Google-Form
   headers (smart quotes, embedded newlines) to clean internal fields via an
   editable dict, and assigns an immutable `student_id` / `mentor_id` from row
   order at load time. **All joins are by id** (scoring, rationales, exports,
   display) — names are display-only; this cohort has 4 same-name student pairs.
2. **Capacity from free text** — an editable rules dict turns "Ongoing engagement
   throughout the year" → 4 slots, "A few hours per semester" → 2, etc.; blank or
   unrecognized → `unknown` (defaults to 1, flagged and logged).
3. **Pool filter** — mentors are eligible for 1:1 matching only if their formats
   include "1:1 mentoring conversations" and `consent_to_share` ≠ No. Also **hard**:
   an in-person-only mentor located outside the Madison/WI area gets no arcs at
   all (edge removal, logged). Excluded mentors are reported with reasons.
4. **Scoring** — four criteria (skill, domain, goals, style) 1–10; the matching
   **utility is the full-precision mean of the 3 measured criteria** — `style_fit`
   is stored for reference but excluded (the forms collected no style data).
   - `offline` (default): deterministic keyword-overlap on the real profile fields.
     **Zero API calls, fully reproducible.**
   - `llm`: Claude tool-use scoring + rationales (Condition B). Needs a key.
     Prompts declare profile text data-not-instructions and are stripped of
     URLs/emails before the LLM sees them.
5. **Assignment** — OR-Tools **min-cost max-flow** on integer costs
   `-round(1000·utility)` plus a documented **seeded tie-break** that can never
   flip a substantive difference (`solver.py`), cross-checked against scipy's
   Hungarian algorithm. Capacity shortfall is normal (85 slots, 116 students) →
   excess students are left **unmatched and surfaced explicitly with reasons**.
6. **Robustness audit** (`robustness_audit.py`) — forbid-each-edge re-solves,
   200 seeded perturbation re-solves, and low/base/high capacity scenarios →
   `robustness_report.json` plus a per-match `robust` flag (selected in ≥80% of
   perturbations and not arbitrary-among-optima). Fragile matches get a small
   dashed marker in the UI.
7. **Output** — `matches.csv`, `unmatched.csv`, `metadata.json`,
   `robustness_report.json`, `score_matrix.npz` (future runs persist
   full-precision utilities **and the full criteria tensor**, so they can be
   re-solved without re-scoring; the June 2026 npz predates this and holds only
   the rounded overall matrix — its per-pair criteria are preserved in
   `output/june_criteria_scores.json` and in git at `6676bf2`) in
   `output/` (git-ignored), and the committed `display_data/results.json` —
   a single internal view carrying ids, **real** names/emails, `fit`, `robust`,
   and the full `rationale`. **Contains student PII → keep the repo private.**

The Flask app is **one view** with real identities everywhere: searchable cards
(mentor + student name/email, a `fit N/10` badge — a balanced-fit score, not a
confidence estimate — and expandable rationale), Matched/Unmatched tabs,
pagination, per-card copy-contact, CSV + PDF export.

`rederive_run.py` re-derives the committed 2026-06-15 run from its persisted
artifacts under the current cost model/constraints **without re-scoring** —
see its docstring for what is and isn't retroactively computable.

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
logged in `metadata.json`). To re-derive the committed display from the
persisted 2026-06-15 artifacts without re-scoring: `python rederive_run.py`.

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
