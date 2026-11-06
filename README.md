# AI Echo: Sentiment Analysis of ChatGPT-style Reviews

NLP project that classifies app reviews as **Positive / Neutral / Negative**, explores the data (10 EDA questions),
answers 10 sentiment questions, and serves everything in a **Streamlit** dashboard with live prediction.

## Files
| File | Purpose |
| --- | --- |
| `data/chatgpt_style_reviews_dataset.csv` | Supplied dataset (500 reviews, 12 columns) |
| `data/extra_training_phrases.csv` | 45 hand-written phrases (15 per class), used for **training only** - see "Data limitations" |
| `config.py` | Paths and constants |
| `preprocess.py` | Cleaning: lowercase, strip punctuation, stop-words (negations kept), lemmatise, date fix, sentiment label |
| `insights.py` | Reusable analysis functions (used by EDA script and dashboard) |
| `eda.py` | Saves 18 EDA / model figures to `outputs/figures/` |
| `train_model.py` | TF-IDF + Naive Bayes / Logistic Regression / Linear SVM / Random Forest, evaluation, saves best model + report |
| `predict.py` | `predict_sentiment(text)` using the saved model |
| `app.py` | Streamlit dashboard (5 tabs, sidebar filters) |
| `test_pipeline.py` | Sanity tests |

## Run
```bash
python3 -m pip install -r requirements.txt
python3 -m pip install wordcloud        # optional: real word clouds (bar charts are used if missing)
python3 preprocess.py                   # data/cleaned_reviews.csv
python3 train_model.py                  # models/sentiment_model.joblib + outputs/model_report.md
python3 eda.py                          # outputs/figures/*.png
python3 test_pipeline.py                # all tests should PASS
python3 -m streamlit run app.py         # dashboard at http://localhost:8501
```

## Method
- **Label:** rating 1-2 = Negative, 3 = Neutral, 4-5 = Positive.
- **Features:** TF-IDF unigrams + bigrams. **Models:** Naive Bayes, Logistic Regression, Linear SVM (calibrated), Random Forest.
- **Metrics:** accuracy, macro precision / recall / F1, one-vs-rest ROC-AUC, confusion matrix, ROC curves.
- **Evaluation done two ways** (see below). Best model is chosen on the honest one.

## Data limitations (read this)
1. **Only 15 distinct review sentences** appear in the 500 reviews, and each sentence belongs to exactly one sentiment class.
   A random train/test split therefore puts identical sentences on both sides and every model scores **100%**, which is meaningless.
2. So models are also scored with **grouped 5-fold CV** where test sentences are never seen in training (`unseen_*` columns).
   Without help, models score 18-30% there (worse than chance). Adding 45 hand-written phrases to the *training* data
   (never to the evaluation data) lifts the best model (Linear SVM) to about **85% accuracy / 0.88 macro-F1 on unseen sentences**.
3. **358 of 500 dates** were corrupted (`########`) in the supplied CSV, so time-trend charts use only 142 reviews (Jan 2024 - May 2025).
   Re-exporting the original `.xlsx` with a wider date column would restore them.
4. `platform` contains Amazon / Flipkart / App Store / Google Play / Website (not Web vs Mobile); `review_length` is a *word count*
   (a character count is added as `char_length`); the `language` column does not match the text (all reviews are English).
5. Ratings and review text always agree here, so there are no rating/text mismatches to find, and group differences
   (platform, version, location, verified) are small (e.g. average rating 2.9 - 3.2) and not statistically meaningful at n = 500.

## Insights and recommendations
- Sentiment split is about 40% Positive / 39% Negative / 21% Neutral; average rating about 3.0 - users are polarised.
- Negative reviews centre on **crashes/bugs, slow performance, poor usability and unmet expectations ("waste of time")**.
- Positive reviews praise **ease of use, helpfulness, reliability and smooth performance**.
- Differences by version, platform and region are small: prioritise fixing stability/performance and usability issues across
  all platforms rather than targeting one segment, and collect a larger, real review sample before making release decisions.
- Retrain regularly on fresh reviews; track the `unseen_*` metrics, not the random-split ones.
