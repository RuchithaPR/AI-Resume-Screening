"""Structured information extraction from resumes and job descriptions.

Extracts: name, email, phone, skills, education level and years of experience
from a resume, and required / preferred skills, minimum experience and minimum
education from a job description.  Pure regex + rules: fast, offline, explainable.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

from .skills import find_skills

EDU_LABELS = {0: "Not specified", 1: "Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"}

# --------------------------------------------------------------------------- #
# Education
# --------------------------------------------------------------------------- #
# (level, case-insensitive patterns, case-sensitive patterns)
_EDU_RULES = [
    (4, [r"\bph\.?\s?d\b", r"\bdoctorate\b", r"\bdoctor of philosophy\b"], []),
    (3, [r"\bmaster(?:['’]s|s)\b", r"\bmaster\s+(?:of|in|degree)\b", r"\bm\.?\s?tech\b", r"\bm\.?\s?sc\b",
         r"\bmba\b", r"\bmca\b", r"\bpost[- ]?graduate\b", r"\bpgdm\b"],
        [r"\bM\.E\b\.?", r"\bM\.S\b\.?", r"\bMS(?=\s+(?:in|degree|\())", r"\bME(?=\s+(?:in|\())"]),
    (2, [r"\bbachelor(?:['’]s|s)?\b", r"\bb\.?\s?tech\b", r"\bb\.?\s?sc\b", r"\bb\.?\s?com\b",
         r"\bbca\b", r"\bbba\b", r"\bundergraduate\b"],
        [r"\bB\.E\b\.?", r"\bBE(?=\s+(?:in|\())"]),
    (1, [r"\bdiploma\b", r"\bpolytechnic\b"], []),
]
_EDU_COMPILED = [
    (lvl, [re.compile(p, re.I) for p in ci] + [re.compile(p) for p in cs]) for lvl, ci, cs in _EDU_RULES
]


def detect_education_levels(text: str) -> set[int]:
    return {lvl for lvl, pats in _EDU_COMPILED if any(p.search(text) for p in pats)}


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
_SECTION_PATTERNS = {
    "experience": r"(?:(?:work|professional|relevant|employment)\s+)?experience|work history|employment(?: history)?"
                  r"|professional background|internships?(?: experience)?",
    "education": r"education(?:al)?(?: background| qualifications?)?|academic(?: background| qualifications?)?|qualifications?",
    "skills": r"(?:(?:technical|key|core)\s+)?skills|technical proficiency|competencies|technologies",
    "projects": r"(?:academic |personal |key )?projects?",
    "other": r"certifications?|achievements?|awards?|publications?|summary|objective|profile|about me|languages|interests|"
             r"hobbies|references|declaration|contact",
}
_SECTION_RE = {k: re.compile(rf"^(?:{v})$", re.I) for k, v in _SECTION_PATTERNS.items()}


def split_sections(text: str) -> dict[str, str]:
    """Split resume text into sections using short heading lines."""
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in text.splitlines():
        stripped = line.strip().rstrip(":").strip()
        key = None
        if 0 < len(stripped) <= 40:
            for name, rx in _SECTION_RE.items():
                if rx.match(stripped):
                    key = name
                    break
        if key:
            current = key
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


# --------------------------------------------------------------------------- #
# Experience
# --------------------------------------------------------------------------- #
_NUM = r"(\d{1,2}(?:\.\d+)?)"
_EXP_A = re.compile(
    rf"{_NUM}(?:\s*(?:-|–|to)\s*\d{{1,2}}(?:\.\d+)?)?\s*(?:\+|plus)?\s*(?:years?|yrs?)\b(?=[^.\n]{{0,60}}?(?:experience|exp\b))", re.I)
_EXP_B = re.compile(rf"experience[^.\n]{{0,25}}?{_NUM}\s*\+?\s*(?:years?|yrs?)\b", re.I)

_MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MONTH_RX = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_RANGE = re.compile(
    rf"(?:({_MONTH_RX})\.?,?\s+)?((?:19|20)\d{{2}})\s*(?:-|–|—|to)\s*"
    rf"(?:(?:({_MONTH_RX})\.?,?\s+)?((?:19|20)\d{{2}})|(present|current|now|till date|ongoing))", re.I)


def explicit_experience(text: str) -> list[float]:
    vals = [float(m.group(1)) for m in _EXP_A.finditer(text)] + [float(m.group(1)) for m in _EXP_B.finditer(text)]
    return [v for v in vals if 0 < v <= 45]


def date_range_experience(text: str) -> float:
    """Sum of date ranges (e.g. 'Jan 2019 - Present'), merging overlaps."""
    today = dt.date.today()
    now = today.year + (today.month - 1) / 12
    spans = []
    for m in _RANGE.finditer(text):
        sm, sy, em, ey, present = m.groups()
        start = int(sy) + ((_MONTHS[sm[:3].lower()] - 1) / 12 if sm else 0)
        if present:
            end = now
        else:
            end = int(ey) + ((_MONTHS[em[:3].lower()] - 1) / 12 if em else 0)
        if 0 < end - start <= 40 and end <= now + 0.1:
            spans.append((start, end))
    spans.sort()
    total, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


# --------------------------------------------------------------------------- #
# Contact info / name
# --------------------------------------------------------------------------- #
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,16}\d")
_URL = re.compile(r"(?:https?://|www\.)\S+|(?:linkedin|github)\.com/\S*", re.I)
_PII_LINE = re.compile(r"^(?:gender|sex|marital status|date of birth|dob|nationality|religion|father'?s? name|age)\b", re.I)
_GENDER_WORDS = re.compile(r"\b(?:male|female|he|she|his|her|him|mr|mrs|ms|miss)\b\.?", re.I)
_NOT_NAME = re.compile(
    r"resume|curriculum|vitae|\bcv\b|summary|profile|objective|contact|education|experience|skills|address|email|phone|mobile|"
    r"engineer|developer|analyst|manager|scientist|intern\b|designer|consultant|architect|specialist", re.I)


def _valid_phone(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    return 10 <= len(digits) <= 13 and len(re.findall(r"(?:19|20)\d{2}", s)) < 2


def extract_email(text: str) -> str:
    m = _EMAIL.search(text)
    return m.group(0) if m else ""


def extract_phone(text: str) -> str:
    for m in _PHONE.finditer(text):
        if _valid_phone(m.group(0)):
            return m.group(0).strip()
    return ""


def extract_name(text: str, filename: str = "") -> str:
    for line in [ln.strip() for ln in text.splitlines() if ln.strip()][:6]:
        if "@" in line or any(ch.isdigit() for ch in line) or _NOT_NAME.search(line):
            continue
        words = line.replace(",", " ").split()
        if 2 <= len(words) <= 4 and re.fullmatch(r"[A-Za-z][A-Za-z.'\- ]+", line):
            return line.title() if line.isupper() else line
    stem = re.sub(r"[_\-.]+", " ", Path(filename).stem)
    stem = re.sub(r"\b(resume|cv|curriculum|vitae)\b", "", stem, flags=re.I).strip()
    return stem.title() or "Unknown"


def anonymize(text: str, name: str = "") -> str:
    """Remove personal identifiers before scoring (bias mitigation)."""
    lines = [ln for ln in text.splitlines() if not _PII_LINE.match(ln.strip())]
    out = "\n".join(lines)
    out = _EMAIL.sub(" ", out)
    out = _URL.sub(" ", out)
    out = _PHONE.sub(lambda m: " " if _valid_phone(m.group(0)) else m.group(0), out)
    for token in name.split():
        if len(token) > 2:
            out = re.sub(rf"\b{re.escape(token)}\b", " ", out, flags=re.I)
    return _GENDER_WORDS.sub(" ", out)


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #
@dataclass
class ResumeProfile:
    filename: str
    name: str
    email: str
    phone: str
    skills: list[str]
    education_level: int
    education_label: str
    experience_years: float
    text: str
    clean_text: str  # anonymised text used for similarity scoring


def parse_resume(filename: str, text: str) -> ResumeProfile:
    sections = split_sections(text)
    name = extract_name(text, filename)

    edu_text = sections.get("education") or text
    levels = detect_education_levels(edu_text) or detect_education_levels(text)
    level = max(levels) if levels else 0

    exp_text = sections.get("experience")
    if not exp_text:  # no experience heading: use everything except the education section
        exp_text = text.replace(sections.get("education", ""), "") if sections.get("education") else text
    years = max([date_range_experience(exp_text)] + explicit_experience(text))
    if re.search(r"\bfresher\b", text, re.I) and years == 0:
        years = 0.0

    return ResumeProfile(
        filename=filename, name=name, email=extract_email(text), phone=extract_phone(text),
        skills=find_skills(text), education_level=level, education_label=EDU_LABELS[level],
        experience_years=round(years, 1), text=text, clean_text=anonymize(text, name),
    )


# --------------------------------------------------------------------------- #
# Job description
# --------------------------------------------------------------------------- #
@dataclass
class JobRequirements:
    text: str
    title: str
    required_skills: list[str]
    preferred_skills: list[str]
    min_experience: float | None
    education_level: int
    education_label: str


_BULLET = re.compile(r"^[-•*·▪●◦\d]")
_PREF_HDR = re.compile(r"\b(?:preferred|nice[- ]to[- ]have|good[- ]to[- ]have|bonus|desirable|optional|added advantage)\b", re.I)
_PREF_INLINE = re.compile(r"\b(?:is a plus|are a plus|a plus|nice to have|preferred|added advantage|bonus)\b", re.I)
_REQ_HDR = re.compile(
    r"\b(?:requirements?|required|must[- ]have|qualifications?|responsibilit\w+|what you.{0,3}ll (?:need|bring)|key skills|skills)\b", re.I)
_TITLE = re.compile(r"^(?:job\s+title|position|role)\s*:\s*(.+)$", re.I)


def parse_job_description(text: str) -> JobRequirements:
    required_lines: list[str] = []
    preferred_lines: list[str] = []
    mode = "required"
    title = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not title:
            m = _TITLE.match(line)
            title = m.group(1).strip() if m else (line if len(line) <= 80 else "")
        header_like = len(line) <= 60 and not _BULLET.match(line)
        line_mode = mode
        if header_like:
            if _PREF_HDR.search(line):
                mode = line_mode = "preferred"
            elif _REQ_HDR.search(line):
                mode = line_mode = "required"
        elif _PREF_INLINE.search(line):
            line_mode = "preferred"
        (preferred_lines if line_mode == "preferred" else required_lines).append(line)

    req_text = "\n".join(required_lines)
    required = find_skills(req_text)
    preferred = [s for s in find_skills("\n".join(preferred_lines)) if s not in required]

    exp_vals = explicit_experience(req_text)
    levels = detect_education_levels(req_text)
    level = min(levels) if levels else 0
    return JobRequirements(
        text=text, title=title or "Job Description", required_skills=required, preferred_skills=preferred,
        min_experience=max(exp_vals) if exp_vals else None, education_level=level, education_label=EDU_LABELS[level],
    )
