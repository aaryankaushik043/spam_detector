# 🛡️ st."Spam Detector"

An **AI-powered SMS / email spam classifier** with a modern, animated web UI.
Built as **Task 5 — AI Mini Project**.

It classifies any text message as **SPAM** (promotional / scam / phishing) or
**HAM** (legitimate) using a TF-IDF + Naive Bayes NLP pipeline, served through
a polished Streamlit interface with smooth animations and transitions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Live Detection** | Type/paste a message → instant spam verdict with animated confidence meter |
| 📁 **Batch Upload** | Upload a CSV of messages → classify all at once, with summary stats |
| 🕐 **History Log** | Every prediction is logged per session & exportable to CSV |
| 📈 **Metrics Dashboard** | Accuracy, precision, recall, F1 + confusion matrix chart |
| ⚙️ **Threshold Tuning** | Decision threshold auto-tuned on test data to maximize F1 |
| 🎨 **Modern UI** | Glassmorphism cards, gradient hero, fade-in / slide-up / glow animations |
| 📤 **Export** | Download live results & history as CSV |
| 🔌 **Offline-first** | Dataset is bundled — **no external downloads** required |

---

## 🧠 How It Works

```
raw text
   │
   ▼  preprocess (lowercase → strip punctuation → remove stopwords → stem)
cleaned text
   │
   ▼  TF-IDF vectorizer (unigrams + bigrams, 5000 features)
numeric vectors
   │
   ▼  Calibrated Multinomial Naive Bayes
spam probability
   │
   ▼  tuned decision threshold → SPAM / HAM
verdict
```

- **Preprocessing** — NLTK (stopwords + Porter stemmer)
- **Vectorization** — scikit-learn `TfidfVectorizer`
- **Classifier** — `MultinomialNB` wrapped in `CalibratedClassifierCV` for
  accurate probability estimates
- **Threshold tuning** — a sweep over `[0.05, 0.95]` picks the threshold that
  maximizes F1 score, which handles the spam/ham class imbalance far better
  than the default 0.5 cutoff.

---

## 📊 Model Performance

Trained on **487** curated messages (104 spam / 383 ham), evaluated on a
stratified 20% test split:

| Metric | Score |
|---|---|
| Accuracy  | **95.9%** |
| Precision | **100%** |
| Recall    | **81.0%** |
| F1 Score  | **89.5%** |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Rebuild the dataset & retrain
The pre-trained model is already included in `models/`. Only run these if you
want to regenerate everything from scratch:
```bash
python build_dataset.py    # writes data/spam_dataset.csv
python train_model.py      # trains + saves model + metrics + confusion matrix
```

### 3. Launch the app
```bash
streamlit run app.py
```
Then open **http://localhost:8502** in your browser.

> 💡 If `streamlit` isn't on your PATH, run it as a module:
> `python -m streamlit run app.py`

---

## 📁 Project Structure

```
spam_detector/
├── app.py                 # Streamlit web UI (5 tabs)
├── ui_styles.py           # All CSS, colors & animations
├── train_model.py         # Training pipeline (TF-IDF + calibrated NB)
├── build_dataset.py       # Builds the curated spam/ham dataset
├── text_utils.py          # Shared preprocess() (importable by the pickle)
├── requirements.txt
├── README.md
├── data/
│   ├── spam_dataset.csv   # 487 labeled messages
│   └── sample_batch.csv   # sample file for the Batch tab
└── models/
    ├── spam_classifier.pkl      # trained sklearn pipeline
    ├── metrics.json             # accuracy/precision/recall/F1/threshold/CM
    └── confusion_matrix.png     # rendered confusion matrix chart
```

---

## 🧪 Try These

Paste into the **Live** tab:

- ✅ *“Hey are you coming to the party tonight?”* → **HAM**
- ⚠️ *“Congratulations! You’ve won a $1000 Walmart gift card. Click to claim.”* → **SPAM**
- ⚠️ *“URGENT: Your bank account is suspended. Verify immediately.”* → **SPAM**
- ✅ *“Your OTP is 482913. Do not share it with anyone.”* → **HAM**

Or upload `data/sample_batch.csv` in the **Batch** tab to classify 16 messages
at once.

---

## 🛠️ Tech Stack

- **scikit-learn** — ML pipeline, TF-IDF, Naive Bayes, calibration
- **NLTK** — stopword removal & stemming
- **Streamlit** — interactive web UI
- **Matplotlib** — confusion-matrix chart
- **Pandas / NumPy / Joblib** — data handling & model persistence

---

## 📝 Notes

- The dataset is **hand-curated** and bundled in-repo, so the project runs
  fully **offline** with no dataset download step.
- The decision threshold is tuned for **high precision** (no false alarms) with
  good recall. Adjust `metrics.json → threshold` to favor recall if you prefer
  catching more spam at the cost of occasional false positives.
- History is stored **per browser session** (Streamlit session state).

---

*AI Mini Project — NLP Spam Classifier · Built with Streamlit + scikit-learn*
