# Resume Screening Agent

An AI-assisted resume screening tool that parses a folder of resumes
(PDF / DOCX / TXT), scores each one against a job description using NLP
similarity + rule-based skill matching, and outputs a ranked, explainable
shortlist — no manual reading of every resume required.

## Overview

Given a job description and a folder of candidate resumes, this agent:

- **Parses** resumes in PDF, DOCX, or TXT format
- **Extracts** skills, years of experience, and education from each resume
- **Scores** every candidate against the JD using a blend of TF-IDF
  semantic similarity, keyword-based skill matching, and experience fit
- **Ranks** candidates and outputs a scored shortlist (CSV + JSON) with
  human-readable reasoning for every score — matched skills, missing
  skills, and how experience compares to what the JD asks for

It's built to be transparent and auditable rather than a black box: every
score is broken into sub-scores you can inspect, and the full method is
documented in [`SCORING_METHOD.md`](SCORING_METHOD.md).

## Features

- 📄 Multi-format resume parsing (PDF, DOCX, TXT)
- 🧠 TF-IDF + cosine similarity for semantic relevance scoring
- ✅ Rule-based skill taxonomy matching (required vs. preferred skills)
- 📊 Experience-fit scoring against the JD's stated experience range
- 📁 Batch processing — handles 10+ resumes in a single run
- 📤 CSV and JSON output with per-candidate reasoning
- 🔧 Fully customizable skill taxonomy and scoring weights
- 🔒 Runs entirely offline/locally — no API keys, no external calls

## Tech Stack

- Python 3.8+
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorization & cosine similarity
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF text extraction
- [python-docx](https://python-docx.readthedocs.io/) — DOCX text extraction

## Example Output

```
Shortlist (top candidates):
  # 1   63.2  Aisha Khan (aisha_khan.pdf)
  # 2   51.0  Rohan Das (rohan_das.pdf)
  # 3   41.0  Sandeep Nair (sandeep_nair.pdf)
  # 4   35.1  Aditya Rao (aditya_rao.pdf)
  # 5   34.5  Meera Pillai (meera_pillai.pdf)
```

Each candidate's full record (in the CSV/JSON) includes matched/missing
required skills, matched preferred skills, extracted experience and
education, and a plain-English reasoning summary.

## Folder Contents

```
resume_screening_agent/
├── agent.py                 # The screening agent (run this)
├── job_description.txt      # The JD to score resumes against — edit this
├── SCORING_METHOD.md         # Explanation of the scoring formula
├── README.md                 # This file
├── resumes/                  # Put candidate resumes here (.pdf/.docx/.txt)
└── output/                   # Results land here after running
    ├── ranked_candidates.csv
    └── ranked_candidates.json
```

## 1. Requirements

- Python 3.8 or later
- Three packages: `scikit-learn`, `pdfplumber`, `python-docx`

## 2. Setup

Open a terminal / Command Prompt in this folder and install the required
packages:

```bash
python -m pip install scikit-learn pdfplumber python-docx
```

(Use `python3` instead of `python` on macOS/Linux if that's how your
system is set up.)

## 3. Add your files

- **Job description:** open `job_description.txt` and replace the content
  with your real JD. Keep the structure — a `Required Skills` section and a
  `Preferred Skills` section with `-` bulleted lines, plus a line like
  `Experience Required: 4-8 years` — since the agent parses those sections
  directly.
- **Resumes:** make sure a `resumes` folder exists in this same directory,
  and place candidate resume files inside it (`.pdf`, `.docx`, or `.txt`
  are all supported). If the folder doesn't exist yet, create it first —
  the agent won't create it for you.

## 4. Run it

```bash
python agent.py --jd job_description.txt --resumes resumes/ --out output/
```

This prints a ranked shortlist to the terminal and writes full results to:
- `output/ranked_candidates.csv`
- `output/ranked_candidates.json`

Each result includes: final score, sub-scores (semantic similarity, skill
match, experience fit), matched/missing skills, extracted experience and
education, and a one-line reasoning summary.

## 5. Customize

- **Add skills the agent doesn't recognize** (e.g. `React`, `Tableau`,
  `Figma` for a non-backend role): open `agent.py` and add them to the
  `SKILL_TAXONOMY` list near the top of the file.
- **Change the scoring weights**: in `agent.py`, look for the line
  `final_score = (0.40 * semantic_score) + (0.40 * skill_score) + (0.20 * exp_score)`
  inside `score_candidates()` and adjust the weights (they should sum to 1.0).

## Troubleshooting

- **`FileNotFoundError: ... 'resumes'`** — the `resumes` folder doesn't
  exist in this directory yet, or has no resume files in it. Create the
  folder and add files, then rerun.
- **`CryptographyDeprecationWarning: Python 3.8 is no longer supported...`**
  — this is a harmless warning from a dependency, not an error. Safe to
  ignore; the script will continue and finish normally.
- **`ModuleNotFoundError`** after installing packages — pip may have
  installed into a different Python environment than the one running the
  script. Try `python -m pip install scikit-learn pdfplumber python-docx`
  (using `-m pip`) to guarantee it installs where `python agent.py` can
  find it.

## Sample Data

The `resumes/` folder in this repo ships with sample resumes covering a
range of fit levels (strong match, moderate match, wrong tech stack,
under-experienced) against the included `job_description.txt`, so you can
run the agent immediately after cloning to see how it behaves before
swapping in your own JD and resumes.

## Limitations

- Skill detection relies on a built-in taxonomy of common tech terms —
  extend `SKILL_TAXONOMY` in `agent.py` for skills outside a typical
  backend/software engineering role.
- Experience-years extraction depends on explicit phrasing in the resume
  (e.g. "X years of professional experience" or `[X years]` job-duration
  tags).
- This is a **screening aid, not an auto-reject tool** — it's meant to
  produce a ranked shortlist with visible reasoning for a human recruiter
  to review, not to make final hiring decisions on its own.

## Contributing

Issues and pull requests are welcome — in particular, extensions to the
skill taxonomy for other job families (data, design, sales, etc.) or
support for additional resume formats.

## License

MIT — see `LICENSE` for details (add one if you haven't already).
