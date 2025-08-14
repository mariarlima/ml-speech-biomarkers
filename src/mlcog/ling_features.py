import pandas as pd
import spacy
import math
from ideadensity import depid as _depid
from functools import lru_cache
from collections import Counter


@lru_cache(maxsize=2)
def _load_spacy(lang="en"):
    """
    Lazy-load and cache spaCy model for faster repeated calls.
    Supports English ('en').
    """
    if lang == "en":
        return spacy.load("en_core_web_sm")
    raise ValueError("Unsupported language: %r" % lang)


def clean_and_tokenize_spacy(transcript, lang="en"):
    """
    Tokenize a transcript with spaCy and return (words, doc).
    Keeps only alphabetic tokens; lowercased.
    """
    nlp = _load_spacy(lang)
    doc = nlp(transcript or "")
    words = [t.text.lower() for t in doc if t.is_alpha]
    return words, doc


def calculate_depid(text):
    """
    Compute idea density (DEPID-R) and return the density.
    """
    density, _, _ = _depid(text or "", is_depid_r=True)
    return float(density)


def calculate_consecutive_duplicates_spacy(words):
    """
    Proportion of positions where the current word equals the previous word.
    Example: ['a','a','b'] -> 1 duplicate over 3 tokens => 1/3.
    """
    n = len(words)
    if n == 0:
        return 0.0
    dup = sum(words[i] == words[i - 1] for i in range(1, n))
    return dup / n


def calculate_ling_nlp(transcript, lang="en", alpha=0.165):
    """
    Compute linguistic features:
      - Brunet's Index
      - Honoré's Statistic
      - Corrected TTR (CTTR)
      - Idea Density (DEPID-R)
      - Proportion of consecutive duplicate words
    """
    words, _ = clean_and_tokenize_spacy(transcript, lang)

    N = len(words)
    freq = Counter(words)
    V = len(freq)
    V1 = sum(1 for c in freq.values() if c == 1)

    brunet = round(N ** (V ** (-alpha)), 2) if N > 0 and V > 0 else 0.0

    honore = (100 * math.log(N)) / (1 - (V1 / V)) if V > V1 and N > 0 else None
    honore = round(honore, 2) if honore is not None else None

    cttr = round(V / math.sqrt(2 * N), 2) if N > 0 else 0.0

    pidensity = round(calculate_depid(transcript), 5)
    dup_prop = round(calculate_consecutive_duplicates_spacy(words), 2)

    return brunet, honore, cttr, pidensity, dup_prop


def text_analysis_features(file_path, lang="en"):
    """
    Extract linguistic features from a single .txt file into a one-row DataFrame.
    """
    results = []

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            content = f.read()

        brunet, honore, cttr, pid, dup = calculate_ling_nlp(content, lang=lang)
        results.append(
            {
                "Brunet": brunet,
                "Honore": honore,
                "CTTR": cttr,
                "PIDensity": pid,
                "Duplic": dup,
            }
        )
    df = pd.DataFrame(results)
    return df
