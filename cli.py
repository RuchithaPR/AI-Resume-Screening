"""Command-line screening (no web UI needed).

Example:
    python cli.py --jd sample_data/job_description.txt --resumes sample_data/resumes --method sbert --csv results.csv
"""
import argparse
from pathlib import Path

from src.parser import SUPPORTED_EXTENSIONS, extract_text
from src.screener import results_to_dataframe, screen


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Resume Screening")
    ap.add_argument("--jd", required=True, help="job description file (txt/pdf/docx)")
    ap.add_argument("--resumes", required=True, help="folder containing resumes")
    ap.add_argument("--method", choices=["sbert", "tfidf"], default="sbert")
    ap.add_argument("--top", type=int, default=10, help="rows to print")
    ap.add_argument("--csv", help="optional path to save full results")
    args = ap.parse_args()

    jd_text = extract_text(args.jd)
    files = sorted(p for p in Path(args.resumes).iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    resumes = []
    for p in files:
        try:
            resumes.append((p.name, extract_text(p)))
        except Exception as exc:
            print(f"Skipping {p.name}: {exc}")

    result = screen(jd_text, resumes, method=args.method)
    for w in result.warnings:
        print("WARNING:", w)
    print(f"Job: {result.job.title} | method: {result.method}")
    print(f"Required skills: {', '.join(result.job.required_skills)}\n")

    df = results_to_dataframe(result.candidates)
    cols = ["Rank", "Candidate", "Match %", "Verdict", "Semantic %", "Skills %", "Experience (yrs)", "Missing required skills"]
    print(df[cols].head(args.top).to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved full results to {args.csv}")


if __name__ == "__main__":
    main()
