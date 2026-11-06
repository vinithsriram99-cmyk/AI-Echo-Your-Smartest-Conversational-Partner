"""Step 3 - generate the EDA report figures (outputs/figures/*.png)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import insights as ins
from config import CLEAN_CSV, FIG_DIR
from preprocess import load_and_clean

COLORS = {"Negative": "#d9534f", "Neutral": "#f0ad4e", "Positive": "#5cb85c"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def words_chart(df, title, name, ax_color):
    """Word cloud if the wordcloud package exists, otherwise a bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4))
    freq = ins.top_words(df, 40)
    try:
        from wordcloud import WordCloud
        ax.imshow(WordCloud(width=700, height=450, background_color="white").generate_from_frequencies(freq.to_dict()))
        ax.axis("off")
    except ImportError:
        freq.head(12).sort_values().plot.barh(ax=ax, color=ax_color)
    ax.set_title(title)
    save(fig, name)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_CSV, parse_dates=["date"]) if CLEAN_CSV.exists() else load_and_clean()

    # ---- EDA questions 1-10 ----
    fig, ax = plt.subplots(); ins.rating_dist(df).plot.bar(ax=ax, color="#4c78a8")
    ax.set_title("Q1 Rating distribution"); ax.set_xlabel("Stars"); ax.set_ylabel("Reviews"); save(fig, "q01_rating_distribution.png")

    fig, ax = plt.subplots(); s = ins.helpful_split(df, 100)
    ax.pie(s, labels=s.index, autopct="%1.0f%%", colors=["#5cb85c", "#bbbbbb"]); ax.set_title("Q2 Helpful reviews (threshold 100 votes)"); save(fig, "q02_helpful_votes.png")

    words_chart(df[df.rating >= 4], "Q3 Positive reviews (4-5 stars)", "q03_wordcloud_positive.png", "#5cb85c")
    words_chart(df[df.rating <= 2], "Q3 Negative reviews (1-2 stars)", "q03_wordcloud_negative.png", "#d9534f")

    t = ins.monthly_trend(df)
    fig, ax = plt.subplots(figsize=(8, 4)); t["avg_rating"].plot(ax=ax, marker="o")
    ax.set_title(f"Q4 Average rating over time (only {int(t['reviews'].sum())} reviews have valid dates)"); ax.set_ylim(1, 5); save(fig, "q04_rating_over_time.png")

    for q, col, title in [(5, "location", "Q5 Avg rating by location"), (6, "platform", "Q6 Avg rating by platform"),
                          (7, "verified_purchase", "Q7 Avg rating: verified vs not"), (10, "version", "Q10 Avg rating by version")]:
        fig, ax = plt.subplots(figsize=(7, 4)); ins.avg_by(df, col).plot.bar(ax=ax, color="#4c78a8")
        ax.set_title(title); ax.set_ylabel("Average rating"); ax.set_ylim(1, 5); save(fig, f"q{q:02d}_{col}.png")

    fig, ax = plt.subplots(); df.boxplot(column="char_length", by="rating", ax=ax)
    ax.set_title("Q8 Review length (characters) by rating"); fig.suptitle(""); save(fig, "q08_length_by_rating.png")

    words_chart(df[df.rating == 1], "Q9 Most frequent words in 1-star reviews", "q09_one_star_words.png", "#d9534f")

    # ---- Streamlit key questions (sentiment) ----
    fig, ax = plt.subplots(); df.sentiment.value_counts().reindex(list(COLORS)).plot.bar(ax=ax, color=list(COLORS.values()))
    ax.set_title("S1 Overall sentiment"); save(fig, "s01_overall_sentiment.png")

    for n, col in [(5, "verified_purchase"), (7, "location"), (8, "platform"), (9, "version")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ins.sentiment_mix(df, col).plot.bar(stacked=True, ax=ax, color=list(COLORS.values()))
        ax.set_ylabel("% of reviews"); ax.set_title(f"S{n} Sentiment mix by {col}"); ax.legend(loc="upper right", fontsize=7); save(fig, f"s{n:02d}_sentiment_by_{col}.png")

    print(f"EDA figures saved to {FIG_DIR}  ({len(list(FIG_DIR.glob('*.png')))} files)")


if __name__ == "__main__":
    main()
