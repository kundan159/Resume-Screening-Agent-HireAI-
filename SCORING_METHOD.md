# Scoring Method — Resume Screening Agent

Each resume gets a **Final Score (0–100)** made up of three weighted components:

| Component | Weight | What it measures |
|---|---|---|
| Semantic Similarity | 40% | Overall textual/contextual overlap between the resume and the full JD |
| Skill Match | 40% | Coverage of the JD's explicit required/preferred skills |
| Experience Fit | 20% | Whether the candidate's years of experience meet the JD's stated minimum |

`final_score = 0.40 × semantic_similarity + 0.40 × skill_match + 0.20 × experience_fit`

## 1. Semantic Similarity (TF-IDF + Cosine Similarity)

The JD text and every resume's raw text are vectorized together with a TF-IDF
vectorizer (unigrams + bigrams, English stop words removed). Cosine similarity
between the JD vector and each resume vector gives a 0–1 score, scaled to 0–100.

This captures overall contextual overlap (phrasing, responsibilities, domain
language) beyond just keyword presence — e.g. a resume that talks about
"owning services end-to-end" and "on-call rotations" will score better on
this axis even if it doesn't use the JD's exact skill vocabulary.

*Why TF-IDF instead of embeddings:* it runs fully offline with no external
model downloads, is fast on batches of resumes, and is easy to audit —
useful properties for an HR tool where explainability matters. It can be
swapped for a sentence-embedding model (e.g. `sentence-transformers`) for
better semantic nuance if network/model access is available.

## 2. Skill Match

The JD's **Required Skills** and **Preferred Skills** bullets are scanned
against a canonical taxonomy of atomic skill terms (e.g. "Python",
"PostgreSQL", "Docker", "system design") to build two clean skill lists —
this avoids treating a whole sentence like "Experience with Docker and
container-based deployments" as one giant "skill."

Each resume is then checked for which of those terms it contains
(whole-word matching, case-insensitive):

```
skill_match_score = 0.8 × (required skills matched / required skills total)
                   + 0.2 × (preferred skills matched / preferred skills total)
```

Required skills are weighted 4x more heavily than preferred skills, since
missing a "must-have" should hurt a candidate's score more than missing a
"nice-to-have."

## 3. Experience Fit

The JD's stated experience range (e.g. "4–8 years") is compared against the
candidate's total years of professional experience (extracted from the
resume). Meeting or exceeding the minimum gives full credit (1.0); falling
short gives linear partial credit (`candidate_years / min_years`), so a
2-year candidate against a 4-year minimum requirement scores 0.5 on this
axis rather than 0.

## Output

For every candidate the agent reports:
- **Final score** and the three component sub-scores (for auditability)
- **Matched required / missing required / matched preferred skills**
- **Extracted experience (years) and education**
- **A one-line reasoning summary** ready for a recruiter to scan

Results are sorted descending by final score and written to
`output/ranked_candidates.csv` and `output/ranked_candidates.json`.

## Known Limitations

- Skill detection depends on the JD/resume using recognizable terminology
  from the built-in taxonomy (`SKILL_TAXONOMY` in `agent.py`) — extend that
  list when adapting the agent to a different role family (e.g. adding
  "Figma", "React", "Tableau" for a design or data-analyst JD).
- Experience-years extraction relies on explicit phrasing in the resume
  ("Total years of professional experience: X years", or `[X years]` job
  duration tags). Resumes that only list start/end dates without any such
  phrasing will fall back to 0 years unless the regex patterns are extended
  to parse date ranges.
- This is a **screening aid, not an auto-reject tool** — it is designed to
  produce a ranked shortlist with visible reasoning for a human recruiter to
  review, not to make final hiring decisions.
