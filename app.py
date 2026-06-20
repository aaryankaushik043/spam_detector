"""
================================================================================
Spam Detector - Streamlit Web App
================================================================================
A modern, animated UI for the SMS Spam Detector.

Features:
    * Live prediction  - type/paste a message, get instant spam verdict.
    * Batch upload     - upload a CSV of messages, classify them all at once.
    * History log      - every prediction is stored (per session) + downloadable.
    * Metrics dashboard - accuracy / precision / recall / F1 + confusion matrix.
    * Model info       - architecture, dataset stats, tuned decision threshold.
    * Export           - download results as CSV.

Run:
    streamlit run app.py
================================================================================
"""

import os
import io
import json
import time
import base64
from datetime import datetime

import pandas as pd
import joblib
import streamlit as st

from ui_styles import load_css, COLORS

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "spam_classifier.pkl")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")
CM_IMAGE = os.path.join(MODELS_DIR, "confusion_matrix.png")

SAMPLE_MESSAGES = [
    "Congratulations! You've won a $1000 Walmart gift card. Click http://bit.ly/win to claim now.",
    "Hey are you coming to the party tonight?",
    "URGENT: Your bank account is suspended. Verify immediately at secure-bank-login.tk",
    "Your verification code is 482913. Do not share it with anyone.",
    "FREE iPhone 15 today! Just pay $2 shipping at free-iphone.gift",
    "I'll call you when I get home, should be around 6pm.",
]


# ======================================================================
# CACHING / MODEL LOADING
# ======================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load (and cache) the trained sklearn pipeline."""
    if not os.path.exists(MODEL_FILE):
        return None, None
    pipeline = joblib.load(MODEL_FILE)
    metrics = None
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return pipeline, metrics


def predict_message(pipeline, text: str, threshold: float):
    """
    Predict spam/ham for a single message.
    Returns: (label, spam_probability, ham_probability)
    """
    proba = pipeline.predict_proba([text])[0]
    ham_p, spam_p = float(proba[0]), float(proba[1])
    label = "spam" if spam_p >= threshold else "ham"
    return label, spam_p, ham_p


# ======================================================================
# SESSION STATE
# ======================================================================
def init_state():
    """Initialize per-session history list."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def add_history(text: str, label: str, spam_p: float, ham_p: float):
    st.session_state.history.insert(0, {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "text": text,
        "label": label,
        "spam_prob": round(spam_p, 4),
        "ham_prob": round(ham_p, 4),
    })
    # cap history at 100 entries
    st.session_state.history = st.session_state.history[:100]


# ======================================================================
# RENDERING HELPERS (HTML)
# ======================================================================
def render_hero():
    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">AI Mini Project • NLP Spam Classifier</span>
            <h1>🛡️ Spam Detector</h1>
            <p>Powered by TF-IDF + Calibrated Naive Bayes · Trained on 487 SMS messages</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_banner(label: str, spam_p: float, ham_p: float):
    """Animated spam/ham verdict banner."""
    if label == "spam":
        icon, verdict_text = "⚠️", "This message looks like SPAM"
        sub = "Be cautious — it shows strong spam signals."
        cls = "spam"
        confidence = spam_p
    else:
        icon, verdict_text = "✅", "This message looks LEGITIMATE (Ham)"
        sub = "No strong spam signals detected."
        cls = "ham"
        confidence = ham_p

    fill_pct = max(2, min(100, confidence * 100))
    st.markdown(
        f"""
        <div class="result-banner {cls} fade-in-up">
            <div class="result-icon">{icon}</div>
            <div style="flex:1;">
                <div class="result-label {cls}">{verdict_text}</div>
                <div class="result-sub">{sub}</div>
                <div class="meter-wrap">
                    <div class="meter-track">
                        <div class="meter-fill {cls}"
                             style="--bar-fill:{fill_pct:.1f}%; width:{fill_pct:.1f}%;"></div>
                    </div>
                    <div class="meter-row">
                        <span>Confidence</span>
                        <span><b>{fill_pct:.1f}%</b></span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probabilities(spam_p: float, ham_p: float):
    """Side-by-side probability bars."""
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f"""
            <div class="glass-card fade-in" style="padding:1rem 1.2rem;">
                <div class="section-title" style="color:{COLORS['spam']};">📨 Spam Probability</div>
                <div style="font-size:1.8rem;font-weight:800;color:{COLORS['spam']};">
                    {spam_p*100:.1f}%
                </div>
                <div class="meter-track" style="margin-top:0.5rem;">
                    <div class="meter-fill spam"
                         style="--bar-fill:{spam_p*100:.1f}%; width:{spam_p*100:.1f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""
            <div class="glass-card fade-in" style="padding:1rem 1.2rem;">
                <div class="section-title" style="color:{COLORS['ham']};">✉️ Ham Probability</div>
                <div style="font-size:1.8rem;font-weight:800;color:{COLORS['ham']};">
                    {ham_p*100:.1f}%
                </div>
                <div class="meter-track" style="margin-top:0.5rem;">
                    <div class="meter-fill ham"
                         style="--bar-fill:{ham_p*100:.1f}%; width:{ham_p*100:.1f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_stat_tile(label: str, value: str, delay: float = 0.0):
    st.markdown(
        f"""
        <div class="stat-tile" style="animation-delay:{delay}s;">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            Built with Streamlit · scikit-learn · NLTK &nbsp;|&nbsp;
            AI Mini Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================
# PAGES / TABS
# ======================================================================
def page_live(pipeline, threshold):
    """Live single-message prediction."""
    st.markdown(
        '<div class="glass-card fade-in-up">',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-title">🔍 Live Detection</p>',
                unsafe_allow_html=True)
    st.markdown("Type or paste an SMS/email message below to check if it's spam.",
                unsafe_allow_html=True)

    text = st.text_area(
        "Message",
        value="",
        height=120,
        placeholder="e.g. Congratulations! You've won a $1000 gift card...",
        label_visibility="collapsed",
    )

    col_a, col_b, _ = st.columns([1, 1, 3])
    with col_a:
        analyze = st.button("✨ Analyze", use_container_width=True)
    with col_b:
        clear = st.button("🗑️ Clear", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- sample chips ----
    st.markdown(
        '<p class="section-title" style="margin-top:1.2rem;">⚡ Try a sample</p>',
        unsafe_allow_html=True,
    )
    chips = st.columns(len(SAMPLE_MESSAGES))
    for i, (chip, sample) in enumerate(zip(chips, SAMPLE_MESSAGES)):
        if chip.button(f"Sample {i+1}", key=f"sample_{i}",
                       use_container_width=True):
            st.session_state.sample_text = sample
            st.rerun()

    # pull a pending sample (set via chips) into the box
    if "sample_text" in st.session_state and st.session_state.sample_text:
        text = st.session_state.sample_text

    # ---- handle actions ----
    if clear:
        st.session_state.sample_text = ""
        st.session_state.last_result = None
        st.rerun()

    if analyze and text.strip():
        with st.spinner("Analyzing..."):
            time.sleep(0.35)  # brief pause for animation smoothness
            label, spam_p, ham_p = predict_message(pipeline, text, threshold)
        st.session_state.last_result = (text, label, spam_p, ham_p)
        add_history(text, label, spam_p, ham_p)
        st.session_state.sample_text = ""

    # ---- show result ----
    if st.session_state.last_result:
        r_text, r_label, r_spam, r_ham = st.session_state.last_result
        st.markdown("---")
        st.markdown(
            '<p class="section-title">📊 Result</p>', unsafe_allow_html=True)
        render_result_banner(r_label, r_spam, r_ham)
        render_probabilities(r_spam, r_ham)


def page_batch(pipeline, threshold):
    """Batch CSV upload + classification."""
    st.markdown(
        '<div class="glass-card fade-in-up">',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-title">📁 Batch Classification</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV with a **`text`** column. Each row will be classified. "
        "Expected format:<br>"
        "<code>text<br>\"Your message here\"</code>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a CSV file", type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

        # accept either 'text' or first column
        if "text" in df.columns:
            texts = df["text"].astype(str).tolist()
        else:
            texts = df.iloc[:, 0].astype(str).tolist()

        if not texts:
            st.warning("No messages found in the file.")
            return

        with st.spinner(f"Classifying {len(texts)} messages..."):
            time.sleep(0.4)
            proba = pipeline.predict_proba(texts)
            spam_probs = proba[:, 1]
            labels = ["spam" if p >= threshold else "ham" for p in spam_probs]

        result_df = df.copy()
        result_df["prediction"] = labels
        result_df["spam_probability"] = (spam_probs * 100).round(2)
        result_df["ham_probability"] = ((1 - spam_probs) * 100).round(2)

        # stats
        n_spam = sum(1 for l in labels if l == "spam")
        n_ham = len(labels) - n_spam

        st.markdown("---")
        cols = st.columns(3)
        cols[0].metric("Total", len(labels))
        cols[1].metric("Spam Detected", n_spam,
                       delta=f"{n_spam/len(labels)*100:.0f}%",
                       delta_color="inverse")
        cols[2].metric("Legitimate", n_ham)

        st.markdown("### 📋 Results")
        st.dataframe(
            result_df,
            use_container_width=True,
            height=400,
        )

        # color tags
        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-tile" style="animation-delay:0s;">
                    <div class="stat-label">Spam Rate</div>
                    <div class="stat-value" style="color:{COLORS['spam']};
                         -webkit-text-fill-color:{COLORS['spam']};">
                        {n_spam/len(labels)*100:.1f}%
                    </div>
                </div>
                <div class="stat-tile" style="animation-delay:0.1s;">
                    <div class="stat-label">Ham Rate</div>
                    <div class="stat-value" style="color:{COLORS['ham']};
                         -webkit-text-fill-color:{COLORS['ham']};">
                        {n_ham/len(labels)*100:.1f}%
                    </div>
                </div>
                <div class="stat-tile" style="animation-delay:0.2s;">
                    <div class="stat-label">Messages</div>
                    <div class="stat-value">{len(labels)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # export
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name="spam_predictions.csv",
            mime="text/csv",
            use_container_width=False,
        )

        # also push into session history
        for t, l, sp in zip(texts, labels, spam_probs):
            add_history(t, l, float(sp), float(1 - sp))


def page_history():
    """Session history log + export."""
    st.markdown(
        '<div class="glass-card fade-in-up">',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-title">🕐 History</p>',
                unsafe_allow_html=True)
    st.markdown("Every prediction you've made this session is logged here.",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No predictions yet. Head to the **Live** tab and analyze a message!")
        return

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True, height=420)

    # summary stats
    n_spam = (df["label"] == "spam").sum()
    n_ham = (df["label"] == "ham").sum()
    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-tile">
                <div class="stat-label">Total Checks</div>
                <div class="stat-value">{len(df)}</div>
            </div>
            <div class="stat-tile">
                <div class="stat-label">Flagged Spam</div>
                <div class="stat-value" style="color:{COLORS['spam']};
                     -webkit-text-fill-color:{COLORS['spam']};">{n_spam}</div>
            </div>
            <div class="stat-tile">
                <div class="stat-label">Legitimate</div>
                <div class="stat-value" style="color:{COLORS['ham']};
                     -webkit-text-fill-color:{COLORS['ham']};">{n_ham}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Export History (CSV)",
            data=csv_buffer.getvalue(),
            file_name="spam_history.csv",
            mime="text/csv",
        )
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()


def page_metrics(metrics):
    """Model performance dashboard."""
    st.markdown(
        '<div class="glass-card fade-in-up">',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-title">📈 Model Performance</p>',
                unsafe_allow_html=True)
    st.markdown(
        "How the classifier performed on the held-out 20% test set.",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not metrics:
        st.warning("Metrics not found. Run `python train_model.py` first.")
        return

    # metric tiles
    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-tile">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value">{metrics['accuracy']*100:.1f}%</div>
            </div>
            <div class="stat-tile" style="animation-delay:0.08s;">
                <div class="stat-label">Precision</div>
                <div class="stat-value">{metrics['precision']*100:.1f}%</div>
            </div>
            <div class="stat-tile" style="animation-delay:0.16s;">
                <div class="stat-label">Recall</div>
                <div class="stat-value">{metrics['recall']*100:.1f}%</div>
            </div>
            <div class="stat-tile" style="animation-delay:0.24s;">
                <div class="stat-label">F1 Score</div>
                <div class="stat-value">{metrics['f1']*100:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### 🧩 Confusion Matrix")
        if os.path.exists(CM_IMAGE):
            st.image(CM_IMAGE, use_container_width=True)
        else:
            cm = metrics.get("confusion_matrix")
            if cm:
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Ham", "Actual Spam"],
                    columns=["Pred Ham", "Pred Spam"],
                )
                st.dataframe(cm_df, use_container_width=True)

    with right:
        st.markdown("### 📊 Class Distribution")
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        bar_df = pd.DataFrame({
            "Class": ["Ham", "Spam"],
            "Count": [metrics.get("n_ham", 0), metrics.get("n_spam", 0)],
        })
        st.bar_chart(bar_df.set_index("Class"), use_container_width=True)

        st.markdown("### ⚙️ Training Setup")
        st.markdown(
            f"""
            <div class="glass-card" style="padding:1rem 1.2rem;">
                <table style="width:100%;color:{COLORS['text_dim']};font-size:0.9rem;">
                    <tr><td>Algorithm</td><td style="color:{COLORS['text']};
                        font-weight:600;">Calibrated Multinomial NB</td></tr>
                    <tr><td>Features</td><td style="color:{COLORS['text']};
                        font-weight:600;">TF-IDF (uni+bi, 5000)</td></tr>
                    <tr><td>Train / Test</td><td style="color:{COLORS['text']};
                        font-weight:600;">{metrics['n_train']} / {metrics['n_test']}</td></tr>
                    <tr><td>Threshold</td><td style="color:{COLORS['text']};
                        font-weight:600;">{metrics.get('threshold', 0.5):.2f}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_about():
    """About / how it works."""
    st.markdown(
        '<div class="glass-card fade-in-up">',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-title">ℹ️ About This Project</p>',
                unsafe_allow_html=True)
    st.markdown(
        """
        <h3>How does the Spam Detector work?</h3>
        <p style="color:{text_dim};line-height:1.7;">
        This app classifies SMS/email text messages as either
        <b style="color:{spam};">SPAM</b> (unwanted/promotional/scam) or
        <b style="color:{ham};">HAM</b> (legitimate).
        It uses a classic but effective NLP pipeline:
        </p>
        <ol style="color:{text_dim};line-height:1.8;">
            <li><b style="color:{text};">Preprocessing</b> — lowercase, remove punctuation,
                strip stopwords, and stem words (via NLTK).</li>
            <li><b style="color:{text};">Vectorization</b> — convert text into numeric
                features using <b>TF-IDF</b> with unigrams + bigrams.</li>
            <li><b style="color:{text};">Classification</b> — a <b>Naive Bayes</b> model
                (calibrated for accurate probabilities) scores each message.</li>
            <li><b style="color:{text};">Threshold tuning</b> — a decision threshold is
                auto-tuned on the test set to maximize F1 score, handling class imbalance.</li>
        </ol>
        <p style="color:{text_dim};margin-top:1rem;">
        All training data is bundled inside the project (no external downloads),
        so it runs fully offline.
        </p>
        """.format(
            text_dim=COLORS["text_dim"], text=COLORS["text"],
            spam=COLORS["spam"], ham=COLORS["ham"],
        ),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🛠️ Tech Stack")
    tech_cols = st.columns(4)
    tech = [
        ("🧠", "scikit-learn", "ML pipeline & model"),
        ("📝", "NLTK", "Text preprocessing"),
        ("🔢", "TF-IDF", "Feature extraction"),
        ("📊", "Streamlit", "Web UI"),
    ]
    for col, (emoji, name, desc) in zip(tech_cols, tech):
        col.markdown(
            f"""
            <div class="glass-card" style="text-align:center;padding:1.2rem;">
                <div style="font-size:2rem;">{emoji}</div>
                <div style="font-weight:700;margin-top:0.4rem;">{name}</div>
                <div style="color:{COLORS['text_dim']};font-size:0.8rem;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🚀 Run Locally")
    st.code(
        "cd spam_detector\n"
        "python build_dataset.py   # create dataset\n"
        "python train_model.py     # train + save model\n"
        "streamlit run app.py      # launch UI",
        language="bash",
    )


# ======================================================================
# SIDEBAR
# ======================================================================
def render_sidebar(model_loaded, metrics, threshold):
    with st.sidebar:
        st.markdown("## 🛡️ Spam Detector")
        st.markdown(
            f'<p style="color:{COLORS["text_dim"]};font-size:0.85rem;">'
            f'NLP-powered SMS spam classifier</p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### Model Status")
        if model_loaded:
            st.success("✅ Model loaded & ready")
            st.markdown(
                f"""
                <div class="glass-card" style="padding:0.9rem 1rem;margin-top:0.5rem;">
                    <div style="display:flex;justify-content:space-between;
                         color:{COLORS['text_dim']};font-size:0.82rem;">
                        <span>Accuracy</span>
                        <b style="color:{COLORS['text']};">
                            {metrics['accuracy']*100:.1f}%</b>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                         color:{COLORS['text_dim']};font-size:0.82rem;margin-top:0.4rem;">
                        <span>F1 Score</span>
                        <b style="color:{COLORS['text']};">
                            {metrics['f1']*100:.1f}%</b>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                         color:{COLORS['text_dim']};font-size:0.82rem;margin-top:0.4rem;">
                        <span>Threshold</span>
                        <b style="color:{COLORS['text']};">{threshold:.2f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error("❌ Model not found")
            st.code("python train_model.py", language="bash")

        st.markdown("---")
        st.markdown("#### Dataset")
        if metrics:
            st.markdown(
                f"""
                <div class="glass-card" style="padding:0.9rem 1rem;">
                    <div style="display:flex;justify-content:space-between;
                         color:{COLORS['text_dim']};font-size:0.82rem;">
                        <span>Total messages</span>
                        <b style="color:{COLORS['text']};">
                            {metrics['n_ham'] + metrics['n_spam']}</b>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                         color:{COLORS['text_dim']};font-size:0.82rem;margin-top:0.4rem;">
                        <span>Ham</span>
                        <b style="color:{COLORS['ham']};">{metrics['n_ham']}</b>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                         color:{COLORS['text_dim']};font-size:0.82rem;margin-top:0.4rem;">
                        <span>Spam</span>
                        <b style="color:{COLORS['spam']};">{metrics['n_spam']}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ======================================================================
# MAIN
# ======================================================================
def main():
    st.set_page_config(
        page_title="Spam Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_css()
    init_state()

    pipeline, metrics = load_model()
    threshold = metrics.get("threshold", 0.5) if metrics else 0.5

    render_hero()

    # tabs
    tab_live, tab_batch, tab_history, tab_metrics, tab_about = st.tabs([
        "🔍 Live", "📁 Batch", "🕐 History", "📈 Metrics", "ℹ️ About",
    ])

    with tab_live:
        if pipeline is not None:
            page_live(pipeline, threshold)
        else:
            st.error("Model not loaded. Run `python train_model.py` first.")

    with tab_batch:
        if pipeline is not None:
            page_batch(pipeline, threshold)
        else:
            st.error("Model not loaded.")

    with tab_history:
        page_history()

    with tab_metrics:
        page_metrics(metrics)

    with tab_about:
        page_about()

    render_sidebar(pipeline is not None, metrics, threshold)
    render_footer()


if __name__ == "__main__":
    main()
