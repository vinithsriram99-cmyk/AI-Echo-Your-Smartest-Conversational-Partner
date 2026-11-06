"""Step 2 - train, compare and save sentiment models.

Because the dataset has only a handful of distinct review texts, a normal random
split leaks identical sentences into the test set and gives ~100% accuracy.
So every model is evaluated TWO ways:
  1. Random stratified 80/20 split          (optimistic)
  2. Group 5-fold CV by review text         (honest: tests on unseen sentences)
The best model by (2) is saved.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

from config import (EXTRA_CSV, FIG_DIR, LABELS, METRICS_CSV, MODEL_PATH, RANDOM_STATE, REPORT_MD)
from preprocess import clean_text, load_and_clean


def load_extras():
    """Hand-written phrases (training only, never evaluated on) to widen the vocabulary."""
    if not EXTRA_CSV.exists():
        return pd.Series(dtype=str), pd.Series(dtype=str)
    e = pd.read_csv(EXTRA_CSV)
    return e["review"].apply(clean_text), e["sentiment"]


def oof_predict(model, X, y, groups, Xe, ye, n_splits=5):
    """Group K-fold: test sentences are unseen; extras are always added to training."""
    classes = sorted(y.unique())
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(classes)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = clone(model).fit(pd.concat([X.iloc[tr], Xe]), pd.concat([y.iloc[tr], ye]))
        pred[te] = m.predict(X.iloc[te])
        proba[te] = m.predict_proba(X.iloc[te])
    return pred, proba, classes


def make_models():
    def pipe(clf):
        return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("clf", clf)])
    return {
        "Naive Bayes": pipe(MultinomialNB()),
        "Logistic Regression": pipe(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "Linear SVM": pipe(CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE), cv=3)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    }


def metrics(y_true, y_pred, proba, classes):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(label_binarize(y_true, classes=classes), proba, average="macro", multi_class="ovr")
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f, "auc_roc": auc}


def main():
    df = load_and_clean()
    X, y, groups = df["clean_review"], df["sentiment"], df["clean_review"]
    Xe, ye = load_extras()
    print(f"Extra hand-written training phrases: {len(Xe)}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows, oof = [], {}

    for name, model in make_models().items():
        # 1) random split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        model.fit(pd.concat([Xtr, Xe]), pd.concat([ytr, ye]))
        m1 = metrics(yte, model.predict(Xte), model.predict_proba(Xte), model.classes_)
        # 2) group CV (unseen sentences)
        pred, proba, classes = oof_predict(model, X, y, groups, Xe, ye)
        m2 = metrics(y, pred, proba, classes)
        oof[name] = (pred, proba, classes)
        rows.append({"model": name,
                     **{f"random_{k}": v for k, v in m1.items()},
                     **{f"unseen_{k}": v for k, v in m2.items()}})
        print(f"{name:20s} random acc={m1['accuracy']:.3f} | unseen-text acc={m2['accuracy']:.3f} f1={m2['f1']:.3f}")

    res = pd.DataFrame(rows).round(4)
    res.to_csv(METRICS_CSV, index=False)
    best = res.sort_values(["unseen_f1", "unseen_accuracy"], ascending=False).iloc[0]["model"]
    print(f"\nBest model (unseen-text F1): {best}")

    # Plots for the best model
    pred, proba, classes = oof[best]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=LABELS, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix\n{best} (unseen texts)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    Y = label_binarize(y, classes=classes)
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC={roc_auc_score(Y[:, i], proba[:, i]):.2f})")
    ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves - {best}"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "roc_curve.png", dpi=150); plt.close(fig)

    # Final model: fit on all data and save
    final = make_models()[best].fit(pd.concat([X, Xe]), pd.concat([y, ye]))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)

    # Feature importance (words driving each class) for linear models
    top = ""
    clf = final.named_steps["clf"]
    if hasattr(clf, "coef_"):
        terms = final.named_steps["tfidf"].get_feature_names_out()
        for i, c in enumerate(clf.classes_):
            idx = np.argsort(clf.coef_[i])[-8:][::-1]
            top += f"- **{c}**: {', '.join(terms[idx])}\n"
    elif hasattr(clf, "feature_log_prob_"):
        terms = final.named_steps["tfidf"].get_feature_names_out()
        for i, c in enumerate(clf.classes_):
            idx = np.argsort(clf.feature_log_prob_[i] - clf.feature_log_prob_.mean(axis=0))[-8:][::-1]
            top += f"- **{c}**: {', '.join(terms[idx])}\n"

    REPORT_MD.write_text(
        "# Model Performance Report\n\n"
        f"Best model: **{best}** (saved to `models/sentiment_model.joblib`)\n\n"
        "## Comparison\n\n" + "```\n" + res.T.to_string() + "\n```" + "\n\n"
        "* `random_*` = stratified 80/20 split (optimistic - identical sentences appear in train and test)\n"
        "* `unseen_*` = 5-fold group CV where test sentences never appear in training (honest estimate)\n\n"
        "## Classification report (unseen texts)\n\n```\n" + classification_report(y, pred, zero_division=0) + "```\n\n"
        "## Top words per class\n\n" + (top or "n/a for this model type") + "\n")
    print(f"Saved model -> {MODEL_PATH}\nSaved report -> {REPORT_MD}")


if __name__ == "__main__":
    main()
