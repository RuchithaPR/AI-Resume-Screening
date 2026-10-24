# AI Resume Screening System

Ranks candidates against a job description using NLP. Built as a **decision-support tool** for recruiters:
every score comes with an explanation (matched / missing skills, experience, education) and nobody is auto-rejected.

## How it works

```
Job description ─┐                        ┌─ Text similarity (SBERT or TF-IDF) ─┐
                 ├─► Information          ├─ Skills match (required + preferred)├─► Weighted score ─► Ranking
Resumes (PDF/    │   extraction           ├─ Experience (years)                 │   (0-100)         + explanation
DOCX/TXT) ───────┘   (skills, education,  └─ Education level ───────────────────┘
                      experience, contact)
```

| Component | Default weight | How it is computed |
|---|---|---|
| Text similarity | 40% | SBERT (`all-MiniLM-L6-v2`) cosine similarity between JD and the best resume chunks; TF-IDF as baseline |
| Skills | 35% | Share of JD skills found in the resume (preferred skills count half) |
| Experience | 15% | `min(candidate years / required years, 1)`, from date ranges and "N years" statements |
| Education | 10% | PhD > Master's > Bachelor's > Diploma compared with the JD minimum |

If the JD doesn't mention a component (e.g. no years of experience), it is skipped and the other weights are rescaled.

**Fairness safeguards:** scoring runs on anonymised text (names, emails, phones, URLs, gender words and lines such as
"Gender / DOB / Marital status" are removed); a *Blind screening mode* hides identity in the UI; results are explainable.

## Setup (VS Code)

1. Install Python 3.10+ and open this folder in VS Code (`File > Open Folder`).
2. Open a terminal (`Ctrl+` `` ` ``) and run:
   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```
   In VS Code press `Ctrl+Shift+P` -> *Python: Select Interpreter* -> choose `.venv`.
3. Start the app:
   ```bash
   streamlit run app.py
   ```
   Click **Load sample job description**, tick **Also include the 5 bundled sample resumes**, then **Screen candidates**.
   The first SBERT run downloads a ~90 MB model (internet needed once).

## Command line

```bash
python cli.py --jd sample_data/job_description.txt --resumes sample_data/resumes --method sbert --csv results.csv
python evaluate.py --jd sample_data/job_description.txt --resumes sample_data/resumes --labels sample_data/labels.csv
```

`evaluate.py` compares **TF-IDF only → TF-IDF + rules → SBERT + rules** using Spearman correlation, Precision@k and NDCG@k
against your own manual labels (`file,relevance` with 0-3). Label 20-30 real resumes for a meaningful result; the 5 bundled
samples are only a smoke test.

## Project structure

```
app.py               Streamlit interface
cli.py               Command-line screening
evaluate.py          Metrics against manual labels
src/parser.py        PDF / DOCX / TXT -> text
src/skills.py        Skill vocabulary + matching (edit SKILL_DB to add skills)
src/extractor.py     Skills, education, experience, contact info, JD requirements, anonymisation
src/matcher.py       TF-IDF baseline and SBERT semantic matcher
src/screener.py      Scoring, ranking, explanations
sample_data/         Sample JD, 5 resumes, labels
```

## Known limitations (good for the report)

- Skill detection is vocabulary based: unknown skills are missed. Extend `SKILL_DB` for your domain.
- Scanned/image PDFs have no text (OCR such as `pytesseract` would be a future improvement); they are flagged.
- Experience years come from date ranges and explicit statements; unusual date formats may be missed.
- Experience and education scores can be high for an otherwise irrelevant candidate; read the score breakdown, not just the total.
- Scores are a ranking aid, not a prediction of job performance. A human should review every shortlist.
