"""Evaluate the system against manual relevance labels.

labels.csv format (one row per resume file):
    file,relevance
    priya_sharma_resume.txt,3        # 0 = not relevant ... 3 = excellent fit

Example:
    python evaluate.py --jd sample_data/job_description.txt --resumes sample_data/resumes --labels sample_data/labels.csv
Reports Spearman correlation, Precision@k and NDCG@k for three systems so you can
show in your report that each added component improves the ranking.
"""
import argparse
import math
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.parser import SUPPORTED_EXTENSIONS, extract_text
from src.screener import screen


def ndcg_at_k(rels_in_ranked_order, k):
    def dcg(rels):
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = dcg(sorted(rels_in_ranked_order, reverse=True))
    return dcg(rels_in_ranked_order) / ideal if ideal else 0.0


def evaluate_scores(scores: dict, labels: dict, k: int, threshold: int) -> dict:
    files = [f for f in scores if f in labels]
    ranked = sorted(files, key=lambda f: -scores[f])
    rels = [labels[f] for f in ranked]
    rho = spearmanr([scores[f] for f in files], [labels[f] for f in files])[0]
    return {
        "Spearman": round(float(rho), 3) if rho == rho else float("nan"),
        f"Precision@{k}": round(sum(r >= threshold for r in rels[:k]) / max(min(k, len(rels)), 1), 3),
        f"NDCG@{k}": round(ndcg_at_k(rels, k), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", required=True)
    ap.add_argument("--resumes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--threshold", type=int, default=2, help="relevance >= threshold counts as a good match")
    args = ap.parse_args()

    labels = dict(pd.read_csv(args.labels).values)
    jd = extract_text(args.jd)
    resumes = [(p.name, extract_text(p)) for p in sorted(Path(args.resumes).iterdir()) if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    rows = {}
    tfidf = screen(jd, resumes, method="tfidf")
    rows["1. TF-IDF only (baseline)"] = {c["filename"]: c["tfidf_baseline"] for c in tfidf.candidates}
    rows["2. TF-IDF + skills/exp/edu"] = {c["filename"]: c["score"] for c in tfidf.candidates}
    sbert = screen(jd, resumes, method="sbert")
    if sbert.method == "sbert":
        rows["3. SBERT + skills/exp/edu (proposed)"] = {c["filename"]: c["score"] for c in sbert.candidates}
    else:
        print("SBERT unavailable - skipping system 3:", *sbert.warnings)

    table = pd.DataFrame({name: evaluate_scores(s, labels, args.k, args.threshold) for name, s in rows.items()}).T
    print(f"\nEvaluated on {len(labels)} labelled resumes\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
