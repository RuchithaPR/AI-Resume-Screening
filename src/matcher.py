"""Text similarity between a job description and resumes.

Two matchers are provided:
* ``tfidf_similarity``  - classic bag-of-words baseline (fast, no downloads).
* ``SemanticMatcher``   - Sentence-BERT embeddings (understands meaning, e.g.
                          "built ML models" ~ "machine learning experience").

Raw cosine similarities are rescaled to 0-1 with fixed calibration constants so
a score does not depend on which other resumes were uploaded in the same batch.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Calibration: raw cosine range -> 0..1 (tuned for typical resume/JD pairs).
TFIDF_CEIL = 0.45
SBERT_FLOOR, SBERT_CEIL = 0.10, 0.65


def _rescale(values, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.asarray(values, dtype=float) - lo) / (hi - lo), 0.0, 1.0)


def tfidf_similarity(jd_text: str, texts: list[str]) -> np.ndarray:
    """TF-IDF (1-2 grams) cosine similarity of each text with the JD, scaled to 0-1."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not texts:
        return np.array([])
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([jd_text] + list(texts))
    except ValueError:  # empty vocabulary
        return np.zeros(len(texts))
    sims = cosine_similarity(matrix[0], matrix[1:]).ravel()
    return _rescale(sims, 0.0, TFIDF_CEIL)


def chunk_text(text: str, chunk_words: int = 150, overlap: int = 30) -> list[str]:
    """Split text into overlapping word windows (SBERT models truncate long input)."""
    words = text.split()
    if not words:
        return []
    step = max(chunk_words - overlap, 1)
    chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), step)]
    # drop a tiny trailing chunk that is fully covered by the previous overlap
    if len(chunks) > 1 and len(chunks[-1].split()) <= overlap:
        chunks.pop()
    return chunks


class SemanticMatcher:
    """Sentence-BERT similarity.

    The JD is embedded as the mean of its chunk embeddings.  A resume's score is
    the mean of its top-k chunk similarities, so a long resume is judged by its
    most relevant sections rather than diluted by irrelevant ones.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, top_k: int = 3):
        from sentence_transformers import SentenceTransformer  # may raise ImportError

        self.model = SentenceTransformer(model_name)
        self.top_k = top_k

    def _embed(self, chunks: list[str]) -> np.ndarray:
        return self.model.encode(chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    def similarity(self, jd_text: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        jd_chunks = chunk_text(jd_text) or [jd_text or " "]
        jd_vec = self._embed(jd_chunks).mean(axis=0)
        jd_vec = jd_vec / (np.linalg.norm(jd_vec) or 1.0)

        all_chunks, owners = [], []
        for i, text in enumerate(texts):
            for chunk in chunk_text(text):
                all_chunks.append(chunk)
                owners.append(i)
        scores = np.zeros(len(texts))
        if all_chunks:
            sims = self._embed(all_chunks) @ jd_vec
            owners_arr = np.array(owners)
            for i in range(len(texts)):
                s = np.sort(sims[owners_arr == i])
                if len(s):
                    scores[i] = s[-min(self.top_k, len(s)):].mean()
        has_text = np.array([bool(t.strip()) for t in texts])
        return np.where(has_text, _rescale(scores, SBERT_FLOOR, SBERT_CEIL), 0.0)
