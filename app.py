"""Step 4 - Streamlit dashboard.   Run:  streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import insights as ins
from config import CLEAN_CSV, METRICS_CSV, MODEL_PATH, REPORT_MD, FIG_DIR

st.set_page_config(page_title="AI Echo - Sentiment Dashboard", page_icon="💬", layout="wide")


@st.cache_data
def load():
    return pd.read_csv(CLEAN_CSV, parse_dates=["date"])


if not CLEAN_CSV.exists() or not MODEL_PATH.exists():
    st.error("Run these first:  `python preprocess.py`  then  `python train_model.py`")
    st.stop()

df_all = load()

# ---------------- sidebar filters ----------------
st.sidebar.header("Filters")
platforms = st.sidebar.multiselect("Platform", sorted(df_all.platform.unique()), default=sorted(df_all.platform.unique()))
versions = st.sidebar.multiselect("Version", sorted(df_all.version.unique()), default=sorted(df_all.version.unique()))
verified = st.sidebar.multiselect("Verified purchase", ["Yes", "No"], default=["Yes", "No"])
df = df_all[df_all.platform.isin(platforms) & df_all.version.isin(versions) & df_all.verified_purchase.isin(verified)]
if df.empty:
    st.warning("No reviews match the filters."); st.stop()

st.title("💬 AI Echo: Sentiment Analysis of ChatGPT-style Reviews")
st.caption(f"{len(df)} reviews shown (of {len(df_all)})")
tab_over, tab_eda, tab_sent, tab_model, tab_try = st.tabs(
    ["Overview", "EDA (10 questions)", "Sentiment insights (10 questions)", "Model performance", "Try it live"])


def words_plot(sub, title, color):
    freq = ins.top_words(sub, 40)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    try:
        from wordcloud import WordCloud
        ax.imshow(WordCloud(width=700, height=400, background_color="white").generate_from_frequencies(freq.to_dict()))
        ax.axis("off")
    except ImportError:
        freq.head(12).sort_values().plot.barh(ax=ax, color=color)
    ax.set_title(title)
    st.pyplot(fig); plt.close(fig)


# ---------------- Overview ----------------
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", len(df)); c2.metric("Avg rating", f"{df.rating.mean():.2f}")
    c3.metric("% Positive", f"{(df.sentiment == 'Positive').mean() * 100:.0f}%")
    c4.metric("% Negative", f"{(df.sentiment == 'Negative').mean() * 100:.0f}%")
    st.info("Data notes: 358 of 500 dates were corrupted ('########') in the source file, so time trends use only "
            "the 142 valid dates. The dataset contains only 15 distinct review sentences.")
    st.dataframe(df[["date", "review", "rating", "sentiment", "platform", "location", "version"]].head(50))

# ---------------- EDA ----------------
with tab_eda:
    a, b = st.columns(2)
    with a:
        st.subheader("1. Rating distribution"); st.bar_chart(ins.rating_dist(df))
        st.subheader("3. Keywords: positive (4-5★)"); words_plot(df[df.rating >= 4], "Positive", "#5cb85c")
        st.subheader("5. Rating by location"); st.bar_chart(ins.avg_by(df, "location"))
        st.subheader("7. Verified vs non-verified"); st.bar_chart(ins.avg_by(df, "verified_purchase"))
        st.subheader("9. Words in 1-star reviews"); words_plot(df[df.rating == 1], "1-star", "#d9534f")
    with b:
        thr = st.slider("2. Helpful-votes threshold", 0, 200, 100)
        st.subheader("2. Helpful reviews"); st.bar_chart(ins.helpful_split(df, thr))
        st.subheader("3. Keywords: negative (1-2★)"); words_plot(df[df.rating <= 2], "Negative", "#d9534f")
        st.subheader("4. Average rating over time"); tr = ins.monthly_trend(df)
        if tr.empty:
            st.write("No valid dates in the selection.")
        else:
            st.line_chart(tr["avg_rating"])
        st.subheader("6. Rating by platform"); st.bar_chart(ins.avg_by(df, "platform"))
        st.subheader("8. Avg review length by rating"); st.bar_chart(ins.length_by(df)["char_length"])
        st.subheader("10. Rating by version"); st.bar_chart(ins.avg_by(df, "version"))

# ---------------- Sentiment questions ----------------
with tab_sent:
    st.subheader("1. Overall sentiment")
    st.bar_chart(df.sentiment.value_counts(normalize=True).mul(100).round(1))
    st.subheader("2. Sentiment vs rating (mismatch check)")
    st.dataframe(pd.crosstab(df.rating, df.sentiment))
    st.caption("Labels are derived from rating (1-2 Negative, 3 Neutral, 4-5 Positive). In this dataset every review text "
               "matches its rating band, so there are no rating/text mismatches.")
    st.subheader("3. Top keywords per sentiment")
    cols = st.columns(3)
    for col, s in zip(cols, ["Negative", "Neutral", "Positive"]):
        col.write(f"**{s}**"); col.dataframe(ins.top_words(df[df.sentiment == s], 10).rename("count"))
    st.subheader("4. Sentiment over time")
    tr = ins.monthly_trend(df)
    if tr.empty:
        st.write("No valid dates in the selection.")
    else:
        st.line_chart(tr["avg_sentiment"])
    for n, col, title in [(5, "verified_purchase", "Verified vs non-verified"), (7, "location", "By location"),
                          (8, "platform", "By platform"), (9, "version", "By version")]:
        st.subheader(f"{n}. Sentiment mix - {title} (%)")
        st.bar_chart(ins.sentiment_mix(df, col))
    st.subheader("6. Review length vs sentiment")
    st.dataframe(ins.length_by(df, "sentiment"))
    st.subheader("10. Negative feedback themes")
    neg = df[df.sentiment == "Negative"]
    if len(neg):
        st.bar_chart(ins.theme_counts(neg))
        st.write("Topic-model (NMF) keywords:", ins.nmf_topics(neg))

# ---------------- Model ----------------
with tab_model:
    if METRICS_CSV.exists():
        st.subheader("Model comparison"); st.dataframe(pd.read_csv(METRICS_CSV))
        st.caption("'random_' = random split (optimistic: same sentences in train & test). "
                   "'unseen_' = test sentences never seen in training (realistic).")
    for img in ["confusion_matrix.png", "roc_curve.png"]:
        if (FIG_DIR / img).exists():
            st.image(str(FIG_DIR / img), width=480)
    if REPORT_MD.exists():
        with st.expander("Full model report"): st.markdown(REPORT_MD.read_text())

# ---------------- Live prediction ----------------
with tab_try:
    from predict import predict_sentiment
    txt = st.text_area("Type a review", "The app keeps crashing and the answers are useless.")
    if st.button("Predict sentiment") and txt.strip():
        r = predict_sentiment(txt)
        st.success(f"Predicted sentiment: **{r['label']}**")
        st.bar_chart(pd.Series(r["probabilities"]))
