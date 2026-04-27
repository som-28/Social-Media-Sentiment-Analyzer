import re
from typing import List

import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#(\w+)")
MULTISPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
    "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with", "you", "your", "i",
}


def normalize_text(text: str) -> str:
    """Normalize social text while preserving sentiment-carrying tokens."""
    text = text.strip()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = re.sub(r"[^a-zA-Z0-9_:\s!?]", " ", text)
    text = text.lower()
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [t for t in text.split(" ") if t]


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]


def preprocess_text(text: str) -> str:
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    clean_tokens = remove_stopwords(tokens)
    return " ".join(clean_tokens)
