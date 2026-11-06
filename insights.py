"""Reusable analysis functions - used by eda.py AND the Streamlit app."""
from collections import Counter

import pandas as pd

SENT_SCORE = {"Negative": -1, "Neutral": 0, "Positive": 1}

THEMES = {
    "Performance / reliability": ["performance", "crash", "bug", "issue", "slow"],
    "Usability": ["friendly", "interface", "easy", "use"],
    "Value / expectations": ["waste", "time", "expectation", "disappointing", "meet"],
    "Needs improvement": ["improvement", "major", "need", "update"],
}


def top_words(df: pd.DataFrame, n: int = 15) -> pd.Series:
    c = Counter(w for t in df["clean_review"] for w in t.split())
    return pd.Series(dict(c.most_common(n)), dtype="int64")


def rating_dist(df: pd.DataFrame) -> pd.Series:
    return df["rating"].value_counts().sort_index()


def avg_by(df: pd.DataFrame, col: str, target: str = "rating") -> pd.Series:
    return df.groupby(col)[target].mean().sort_values(ascending=False)


def sentiment_mix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """% of Negative/Neutral/Positive per group in `col`."""
    mix = pd.crosstab(df[col], df["sentiment"], normalize="index") * 100
    return mix.reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0).round(1)


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["date"]).copy()
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    d["sent_score"] = d["sentiment"].map(SENT_SCORE)
    return d.groupby("month").agg(avg_rating=("rating", "mean"),
                                  avg_sentiment=("sent_score", "mean"),
                                  reviews=("rating", "size"))


def helpful_split(df: pd.DataFrame, threshold: int) -> pd.Series:
    return pd.Series({f"> {threshold} votes": int((df["helpful_votes"] > threshold).sum()),
                      f"<= {threshold} votes": int((df["helpful_votes"] <= threshold).sum())})


def length_by(df: pd.DataFrame, col: str = "rating") -> pd.DataFrame:
    return df.groupby(col)[["char_length", "word_count"]].mean().round(1)


def theme_counts(df_neg: pd.DataFrame) -> pd.Series:
    out = {}
    for theme, words in THEMES.items():
        out[theme] = int(df_neg["clean_review"].apply(lambda t: any(w in t.split() for w in words)).sum())
    return pd.Series(out).sort_values(ascending=False)


def nmf_topics(df_neg: pd.DataFrame, n_topics: int = 3, n_words: int = 5) -> dict:
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform(df_neg["clean_review"])
    k = min(n_topics, X.shape[1])
    nmf = NMF(n_components=k, random_state=42, init="nndsvd", max_iter=500).fit(X)
    terms = vec.get_feature_names_out()
    return {f"Topic {i+1}": [terms[j] for j in comp.argsort()[-n_words:][::-1]]
            for i, comp in enumerate(nmf.components_)}
