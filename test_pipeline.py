"""Quick sanity checks.  Run:  python test_pipeline.py"""
import pandas as pd

import insights as ins
from config import CLEAN_CSV, FIG_DIR, MODEL_PATH
from predict import predict_sentiment
from preprocess import clean_text, load_and_clean


def test_clean_text():
    assert clean_text("Not BAD!!! 123") == "not bad"
    assert clean_text(None) == ""


def test_cleaned_data():
    df = load_and_clean()
    assert len(df) == 500
    assert set(df.sentiment) == {"Negative", "Neutral", "Positive"}
    assert df.groupby("sentiment").rating.apply(lambda s: set(s)).to_dict() == {
        "Negative": {1, 2}, "Neutral": {3}, "Positive": {4, 5}}
    assert df.date.notna().sum() == 142


def test_insights():
    df = pd.read_csv(CLEAN_CSV, parse_dates=["date"])
    assert ins.rating_dist(df).sum() == 500
    assert not ins.monthly_trend(df).empty
    assert ins.sentiment_mix(df, "platform").sum(axis=1).round().eq(100).all()
    assert len(ins.nmf_topics(df[df.sentiment == "Negative"])) >= 1


def test_model():
    assert MODEL_PATH.exists()
    assert predict_sentiment("Terrible app, keeps crashing")["label"] == "Negative"
    assert predict_sentiment("It is okay, nothing special")["label"] == "Neutral"
    assert predict_sentiment("Excellent and very helpful!")["label"] == "Positive"


def test_figures():
    assert len(list(FIG_DIR.glob("*.png"))) >= 15


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    print("All tests passed")
