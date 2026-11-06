"""Step 1 - clean the raw CSV and create the sentiment label."""
import re

import pandas as pd

from config import CLEAN_CSV, PLATFORM_TYPE, RAW_CSV

# Negations carry sentiment ("not bad"), so never drop them as stopwords.
KEEP_WORDS = {"not", "no", "nor", "never", "neither"}


def _load_stopwords():
    try:
        from nltk.corpus import stopwords
        try:
            words = set(stopwords.words("english"))
        except LookupError:
            import nltk
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("english"))
    except Exception:  # NLTK missing / offline -> sklearn fallback
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        words = set(ENGLISH_STOP_WORDS)
    return words - KEEP_WORDS


def _load_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        try:
            lem.lemmatize("tests")
        except LookupError:
            import nltk
            nltk.download("wordnet", quiet=True)
            lem.lemmatize("tests")
        return lem.lemmatize
    except Exception:
        return lambda w: w


STOPWORDS = _load_stopwords()
LEMMATIZE = _load_lemmatizer()


def clean_text(text) -> str:
    """lowercase -> strip punctuation/digits -> tokenise -> drop stopwords -> lemmatise."""
    if not isinstance(text, str):
        return ""
    tokens = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    return " ".join(LEMMATIZE(t) for t in tokens if t not in STOPWORDS and len(t) > 1)


def rating_to_sentiment(r: int) -> str:
    return "Negative" if r <= 2 else "Neutral" if r == 3 else "Positive"


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    n_start = len(df)

    df = df.dropna(subset=["review", "rating"]).copy()
    df["review"] = df["review"].astype(str).str.strip()

    # Excel exports unreadable dates as '########' -> becomes NaT (kept, not dropped).
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    df["sentiment"] = df["rating"].astype(int).map(rating_to_sentiment)
    df["word_count"] = df["review"].str.split().str.len()  # original review_length is a word count
    df["char_length"] = df["review"].str.len()
    df["platform_type"] = df["platform"].map(PLATFORM_TYPE).fillna("Other")
    df["clean_review"] = df["review"].apply(clean_text)
    df = df[df["clean_review"].str.len() > 0].reset_index(drop=True)

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"Rows: {n_start} -> {len(df)} | valid dates: {df['date'].notna().sum()} "
          f"| unique review texts: {df['review'].nunique()}")
    print(f"Saved cleaned data to {CLEAN_CSV}")
    return df


if __name__ == "__main__":
    load_and_clean()
