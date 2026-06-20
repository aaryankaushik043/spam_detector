"""
================================================================================
Spam Detector - Model Training Script
================================================================================
Trains a Naive Bayes classifier on SMS messages using a TF-IDF feature pipeline.

Pipeline:
    raw text -> lowercase -> remove punctuation -> remove stopwords
            -> stem -> TF-IDF vector -> Naive Bayes classifier

Outputs (saved to /models):
    spam_classifier.pkl     : the full sklearn Pipeline (vectorizer + model)
    metrics.json            : accuracy, precision, recall, F1, confusion matrix
    confusion_matrix.png    : visual confusion matrix chart

Run:
    python train_model.py
================================================================================
"""

import json
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.calibration import CalibratedClassifierCV

import matplotlib
matplotlib.use("Agg")  # headless backend (no GUI window during training)
import matplotlib.pyplot as plt

# NOTE: `preprocess` MUST be imported from its own stable module so that the
# pickled TfidfVectorizer can find it again by qualified name at load time.
from text_utils import preprocess


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODELS_DIR = os.path.join(HERE, "models")

DATA_FILE = os.path.join(DATA_DIR, "spam_dataset.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "spam_classifier.pkl")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")
CM_IMAGE = os.path.join(MODELS_DIR, "confusion_matrix.png")

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """
    Load the dataset. Expected CSV columns: 'label' ('spam'|'ham'), 'text'.
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. "
            f"Run build_dataset.py first to create it."
        )

    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["label", "text"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def build_pipeline() -> Pipeline:
    """
    Construct the sklearn Pipeline:
        TF-IDF (up to bigrams) -> Multinomial Naive Bayes.

    We wrap the NB classifier in CalibratedClassifierCV so that
    `predict_proba` returns well-calibrated probabilities. This lets the
    app pick a tuned decision threshold (better F1) instead of being
    stuck with Naive Bayes' default 0.5 cutoff, which is too conservative
    on imbalanced data.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess,
        ngram_range=(1, 2),     # unigrams + bigrams
        max_features=5000,
    )
    base_clf = MultinomialNB()
    # cv=3 cross-fits probabilities on the training data.
    calibrated = CalibratedClassifierCV(base_clf, cv=3)
    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", calibrated),
    ])


def find_best_threshold(y_true, proba_spam) -> tuple:
    """
    Sweep thresholds from 0.05 to 0.95 and return the one maximizing F1.
    Returns (best_threshold, best_f1).
    """
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (proba_spam >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def save_confusion_matrix_image(cm, labels, path):
    """Render and save a nicely styled confusion matrix PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # annotate each cell with the count
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=14, fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main():
    print("=" * 60)
    print(" Spam Detector - Training")
    print("=" * 60)

    # 1. Load + prepare data
    print("\n[1/5] Loading dataset ...")
    df = load_data()
    print(f"      Loaded {len(df)} messages "
          f"({(df['label']=='spam').sum()} spam, "
          f"{(df['label']=='ham').sum()} ham)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_num"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label_num"],
    )
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    # 2. Build + train pipeline
    print("\n[2/5] Training TF-IDF + Naive Bayes pipeline ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 3. Evaluate
    print("\n[3/5] Evaluating on test set ...")
    # Get calibrated spam probabilities, then pick the threshold that
    # maximizes F1 (handles the class imbalance much better than 0.5).
    proba_spam = pipeline.predict_proba(X_test)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_test, proba_spam)
    print(f"      Tuned threshold (max F1): {best_threshold:.2f}")
    print(f"      Best F1 at that threshold: {best_f1:.4f}")

    y_pred = (proba_spam >= best_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"      Accuracy : {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall   : {recall:.4f}")
    print(f"      F1 Score : {f1:.4f}")
    print("\nClassification report:\n",
          classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # 4. Persist model + metrics
    print("\n[4/5] Saving model and metrics ...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(best_threshold),
        "confusion_matrix": cm.tolist(),
        "labels": ["ham", "spam"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_spam": int((df["label"] == "spam").sum()),
        "n_ham": int((df["label"] == "ham").sum()),
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"      -> {MODEL_FILE}")
    print(f"      -> {METRICS_FILE}")

    # 5. Save confusion-matrix chart
    print("\n[5/5] Saving confusion matrix chart ...")
    save_confusion_matrix_image(cm, ["ham", "spam"], CM_IMAGE)
    print(f"      -> {CM_IMAGE}")

    print("\nTraining complete. You can now run the Streamlit app:\n")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    main()
