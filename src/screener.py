"""End-to-end screening pipeline: parse -> extract -> compare -> score -> rank."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .extractor import JobRequirements, ResumeProfile, parse_job_description, parse_resume
from .matcher import SemanticMatcher, tfidf_similarity

DEFAULT_WEIGHTS = {"semantic": 0.40, "skills": 0.35, "experience": 0.15, "education": 0.10}
PREFERRED_SKILL_WEIGHT = 0.5  # a preferred skill counts half of a required one
VERDICTS = [(75, "Strong match"), (60, "Good match"), (40, "Partial match"), (0, "Weak match")]


@dataclass
class ScreeningResult:
    job: JobRequirements
    candidates: list[dict]
    method: str                      # "sbert" or "tfidf" (what was actually used)
    weights: dict                    # effective, normalised weights
    warnings: list[str] = field(default_factory=list)


def verdict(score: float) -> str:
    return next(label for threshold, label in VERDICTS if score >= threshold)


# --------------------------------------------------------------------------- #
# Components (each returns 0..1, or None when the JD doesn't specify it)
# --------------------------------------------------------------------------- #
def skills_component(candidate_skills: list[str], job: JobRequirements) -> dict | None:
    req, pref = job.required_skills, job.preferred_skills
    total = len(req) + PREFERRED_SKILL_WEIGHT * len(pref)
    if total == 0:
        return None
    have = set(candidate_skills)
    m_req = [s for s in req if s in have]
    m_pref = [s for s in pref if s in have]
    got = len(m_req) + PREFERRED_SKILL_WEIGHT * len(m_pref)
    return {
        "score": got / total,
        "matched_required": m_req, "missing_required": [s for s in req if s not in have],
        "matched_preferred": m_pref, "missing_preferred": [s for s in pref if s not in have],
    }


def experience_component(years: float, required: float | None) -> float | None:
    if not required:
        return None
    return min(years / required, 1.0)


def education_component(level: int, required_level: int) -> float | None:
    if required_level <= 0:
        return None
    if level >= required_level:
        return 1.0
    return level / required_level if level else 0.0


def _effective_weights(weights: dict, applicable: dict) -> dict:
    raw = {k: max(float(weights.get(k, 0)), 0.0) for k in DEFAULT_WEIGHTS if applicable[k]}
    total = sum(raw.values())
    if total == 0:
        raw, total = {"semantic": 1.0}, 1.0
    return {k: raw.get(k, 0.0) / total for k in DEFAULT_WEIGHTS}


def _explain(c: dict, job: JobRequirements, sk: dict | None) -> str:
    parts = []
    if sk:
        s = f"{len(sk['matched_required'])}/{len(job.required_skills)} required skills"
        if job.preferred_skills:
            s += f", {len(sk['matched_preferred'])}/{len(job.preferred_skills)} preferred"
        parts.append(s)
    if job.min_experience:
        parts.append(f"{c['experience_years']:g} yrs experience vs {job.min_experience:g} required")
    if job.education_level:
        if c["education_level"] == 0:
            parts.append("education not detected")
        elif c["education_level"] >= job.education_level:
            parts.append(f"{c['education_label']} meets the {job.education_label} requirement")
        else:
            parts.append(f"{c['education_label']} is below the {job.education_label} requirement")
    parts.append(f"text similarity {c['components']['semantic']:.0f}%")
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def screen(jd_text: str, resumes: list[tuple[str, str]], weights: dict | None = None,
           method: str = "sbert", matcher: SemanticMatcher | None = None) -> ScreeningResult:
    """Screen ``resumes`` ([(filename, text), ...]) against ``jd_text``.

    method: "sbert" (semantic, recommended) or "tfidf" (baseline).
    Scoring uses anonymised text; name / email / phone never influence the score.
    """
    warnings: list[str] = []
    job = parse_job_description(jd_text)
    profiles: list[ResumeProfile] = [parse_resume(fn, tx) for fn, tx in resumes]
    texts = [p.clean_text for p in profiles]

    tfidf = tfidf_similarity(job.text, texts)
    semantic, used = tfidf, "tfidf"
    if method == "sbert":
        try:
            matcher = matcher or SemanticMatcher()
            semantic, used = matcher.similarity(job.text, texts), "sbert"
        except Exception as exc:  # missing package, no internet for first download, ...
            warnings.append(f"SBERT model unavailable ({type(exc).__name__}: {exc}). Fell back to TF-IDF similarity.")

    applicable = {
        "semantic": True, "skills": bool(job.required_skills or job.preferred_skills),
        "experience": bool(job.min_experience), "education": job.education_level > 0,
    }
    for key, label in (("skills", "no recognised skills"), ("experience", "no minimum years of experience"),
                       ("education", "no education requirement")):
        if not applicable[key]:
            warnings.append(f"The job description lists {label}; the '{key}' score is ignored and its weight redistributed.")
    eff = _effective_weights(weights or DEFAULT_WEIGHTS, applicable)

    candidates = []
    for i, p in enumerate(profiles):
        sk = skills_component(p.skills, job)
        exp = experience_component(p.experience_years, job.min_experience)
        edu = education_component(p.education_level, job.education_level)
        comps = {"semantic": semantic[i], "skills": sk["score"] if sk else None, "experience": exp, "education": edu}
        score = sum(eff[k] * v for k, v in comps.items() if v is not None) * 100
        c = {
            "index": i + 1, "filename": p.filename, "name": p.name, "email": p.email, "phone": p.phone,
            "score": round(score, 1), "verdict": verdict(score),
            "components": {k: (round(v * 100, 1) if v is not None else None) for k, v in comps.items()},
            "contributions": {k: round(eff[k] * v * 100, 2) for k, v in comps.items() if v is not None},
            "tfidf_baseline": round(float(tfidf[i]) * 100, 1),
            "experience_years": p.experience_years, "education_level": p.education_level,
            "education_label": p.education_label, "candidate_skills": p.skills,
            "matched_required": sk["matched_required"] if sk else [], "missing_required": sk["missing_required"] if sk else [],
            "matched_preferred": sk["matched_preferred"] if sk else [], "missing_preferred": sk["missing_preferred"] if sk else [],
            "extra_skills": [s for s in p.skills if s not in job.required_skills and s not in job.preferred_skills],
            "warning": "",
        }
        if len(p.text.split()) < 30:
            c["warning"] = "Very little text extracted (scanned/image PDF?). Review manually."
        c["explanation"] = _explain(c, job, sk)
        candidates.append(c)

    candidates.sort(key=lambda c: (-c["score"], -(c["components"]["skills"] or 0), c["index"]))
    for rank, c in enumerate(candidates, 1):
        c["rank"] = rank
    return ScreeningResult(job=job, candidates=candidates, method=used, weights=eff, warnings=warnings)


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def display_name(c: dict, blind: bool = False) -> str:
    return f"Candidate #{c['index']}" if blind else c["name"]


def results_to_dataframe(candidates: list[dict], blind: bool = False) -> pd.DataFrame:
    rows = []
    for c in candidates:
        row = {"Rank": c["rank"], "Candidate": display_name(c, blind)}
        if not blind:
            row.update({"File": c["filename"], "Email": c["email"], "Phone": c["phone"]})
        row.update({
            "Match %": c["score"], "Verdict": c["verdict"],
            "Semantic %": c["components"]["semantic"], "Skills %": c["components"]["skills"],
            "Experience %": c["components"]["experience"], "Education %": c["components"]["education"],
            "TF-IDF baseline %": c["tfidf_baseline"],
            "Experience (yrs)": c["experience_years"], "Education": c["education_label"],
            "Matched skills": ", ".join(c["matched_required"] + c["matched_preferred"]),
            "Missing required skills": ", ".join(c["missing_required"]),
            "Why": c["explanation"], "Warning": c["warning"],
        })
        rows.append(row)
    return pd.DataFrame(rows)
