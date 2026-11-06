"""Central configuration - all paths and constants live here."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "chatgpt_style_reviews_dataset.csv"
EXTRA_CSV = BASE_DIR / "data" / "extra_training_phrases.csv"
CLEAN_CSV = BASE_DIR / "data" / "cleaned_reviews.csv"
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
METRICS_CSV = OUTPUT_DIR / "model_comparison.csv"
REPORT_MD = OUTPUT_DIR / "model_report.md"

RANDOM_STATE = 42
LABELS = ["Negative", "Neutral", "Positive"]

# Dataset 'platform' values are storefronts, not Web/Mobile - group them.
PLATFORM_TYPE = {
    "Website": "Web",
    "App Store": "Mobile",
    "Google Play": "Mobile",
    "Amazon": "Marketplace",
    "Flipkart": "Marketplace",
}
