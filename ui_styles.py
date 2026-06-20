"""
================================================================================
Spam Detector - UI Styling Module
================================================================================
Centralizes all CSS, color palettes, and animation keyframes for the Streamlit
app. Keeping styles separate makes the main app.py readable and lets us tweak
the look-and-feel in one place.

Design language:
    - Modern dark UI with a purple/indigo accent gradient.
    - Glassmorphism cards (frosted background + soft shadow).
    - Smooth fade-in / slide-up animations on page load.
    - Animated confidence meter bar.
    - Themed result banners (green = ham, red = spam).
================================================================================
"""

import streamlit as st

# ----------------------------------------------------------------------
# Color palette (hex)
# ----------------------------------------------------------------------
COLORS = {
    "bg":            "#0f0f1a",
    "bg_alt":        "#16162a",
    "card":          "rgba(255, 255, 255, 0.04)",
    "card_border":   "rgba(255, 255, 255, 0.08)",
    "text":          "#e8e8f0",
    "text_dim":      "#9a9ab0",
    "accent":        "#7c5cff",   # indigo/purple
    "accent2":       "#00d4ff",   # cyan
    "ham":           "#22c55e",   # green
    "spam":          "#ef4444",   # red
    "warn":          "#f59e0b",   # amber
}


def load_css() -> None:
    """Inject the global stylesheet into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* ============================================================
           ROOT + GLOBAL
           ============================================================ */
        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background: radial-gradient(ellipse at top left,
                          {COLORS['bg_alt']} 0%, {COLORS['bg']} 55%),
                        radial-gradient(ellipse at bottom right,
                          rgba(124, 92, 255, 0.15) 0%, transparent 50%);
            background-attachment: fixed;
            color: {COLORS['text']};
            font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
        }}

        /* hide default top padding & streamline layout */
        .block-container {{
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        /* ============================================================
           ANIMATIONS
           ============================================================ */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(24px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to   {{ opacity: 1; }}
        }}
        @keyframes pulseGlow {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(124, 92, 255, 0.4); }}
            50%      {{ box-shadow: 0 0 32px 6px rgba(124, 92, 255, 0.25); }}
        }}
        @keyframes shimmer {{
            0%   {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}
        @keyframes growBar {{
            from {{ width: 0%; }}
            to   {{ width: var(--bar-fill, 50%); }}
        }}
        @keyframes spinIn {{
            from {{ opacity: 0; transform: scale(0.5) rotate(-180deg); }}
            to   {{ opacity: 1; transform: scale(1) rotate(0deg); }}
        }}

        .fade-in-up {{ animation: fadeInUp 0.6s cubic-bezier(0.22, 1, 0.36, 1) both; }}
        .fade-in    {{ animation: fadeIn 0.8s ease both; }}

        /* ============================================================
           HERO TITLE
           ============================================================ """
        + _hero_css() + f"""
        /* ============================================================
           GLASS CARD
           ============================================================ """
        + _card_css() + f"""
        /* ============================================================
           RESULT BANNERS
           ============================================================ """
        + _result_banner_css() + f"""
        /* ============================================================
           CONFIDENCE METER
           ============================================================ """
        + _meter_css() + f"""
        /* ============================================================
           STAT TILES
           ============================================================ """
        + _stat_css() + f"""
        /* ============================================================
           BUTTONS, INPUTS, TABS, SIDEBAR TWEAKS
           ============================================================ """
        + _controls_css() + f"""
        /* subtle scrollbar */
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(124, 92, 255, 0.4);
            border-radius: 8px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(124, 92, 255, 0.7);
        }}

        /* footer */
        .app-footer {{
            text-align: center;
            color: {COLORS['text_dim']};
            font-size: 0.8rem;
            margin-top: 3rem;
            opacity: 0.7;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero_css() -> str:
    return f"""
        .hero {{
            text-align: center;
            padding: 1.5rem 0 2rem;
            animation: fadeInUp 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
        }}
        .hero-badge {{
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border: 1px solid rgba(124, 92, 255, 0.4);
            background: rgba(124, 92, 255, 0.08);
            border-radius: 999px;
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {COLORS['accent2']};
            margin-bottom: 1rem;
        }}
        .hero h1 {{
            font-size: 2.6rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(120deg, #ffffff 0%, {COLORS['accent']} 50%, {COLORS['accent2']} 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }}
        .hero p {{
            color: {COLORS['text_dim']};
            font-size: 1.05rem;
            margin-top: 0.6rem;
        }}
    """


def _card_css() -> str:
    return f"""
        .glass-card {{
            background: {COLORS['card']};
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid {COLORS['card_border']};
            border-radius: 18px;
            padding: 1.6rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(124, 92, 255, 0.35);
            box-shadow: 0 16px 48px rgba(124, 92, 255, 0.15), 0 8px 32px rgba(0,0,0,0.4);
        }}
        .section-title {{
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {COLORS['text_dim']};
            margin: 0 0 0.9rem;
        }}
    """


def _result_banner_css() -> str:
    return f"""
        .result-banner {{
            display: flex;
            align-items: center;
            gap: 1.2rem;
            padding: 1.4rem 1.6rem;
            border-radius: 16px;
            margin: 1rem 0;
            animation: spinIn 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
            transition: all 0.3s ease;
        }}
        .result-banner.ham {{
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.05));
            border: 1px solid rgba(34, 197, 94, 0.4);
        }}
        .result-banner.spam {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
            border: 1px solid rgba(239, 68, 68, 0.4);
            animation: spinIn 0.55s cubic-bezier(0.22, 1, 0.36, 1) both,
                       pulseGlow 2.4s ease-in-out infinite 0.55s;
        }}
        .result-icon {{
            font-size: 2.6rem;
            line-height: 1;
        }}
        .result-label {{
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: 0.02em;
        }}
        .result-label.ham  {{ color: {COLORS['ham']}; }}
        .result-label.spam {{ color: {COLORS['spam']}; }}
        .result-sub {{
            color: {COLORS['text_dim']};
            font-size: 0.9rem;
            margin-top: 0.15rem;
        }}
    """


def _meter_css() -> str:
    return f"""
        .meter-wrap {{
            margin-top: 0.5rem;
            animation: fadeIn 0.9s ease 0.2s both;
        }}
        .meter-track {{
            width: 100%;
            height: 14px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid {COLORS['card_border']};
        }}
        .meter-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, {COLORS['accent']} 0%, {COLORS['accent2']} 100%);
            animation: growBar 1.1s cubic-bezier(0.22, 1, 0.36, 1) both;
            box-shadow: 0 0 12px rgba(124, 92, 255, 0.5);
        }}
        .meter-fill.spam {{
            background: linear-gradient(90deg, {COLORS['spam']} 0%, #f97316 100%);
            box-shadow: 0 0 12px rgba(239, 68, 68, 0.5);
        }}
        .meter-fill.ham {{
            background: linear-gradient(90deg, {COLORS['ham']} 0%, #16a34a 100%);
            box-shadow: 0 0 12px rgba(34, 197, 94, 0.5);
        }}
        .meter-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            color: {COLORS['text_dim']};
            margin-top: 0.5rem;
        }}
    """


def _stat_css() -> str:
    return f"""
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .stat-tile {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['card_border']};
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            transition: transform 0.3s ease, border-color 0.3s ease;
            animation: fadeInUp 0.5s ease both;
        }}
        .stat-tile:hover {{
            transform: translateY(-3px);
            border-color: rgba(124, 92, 255, 0.35);
        }}
        .stat-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {COLORS['text_dim']};
        }}
        .stat-value {{
            font-size: 1.7rem;
            font-weight: 800;
            margin-top: 0.3rem;
            background: linear-gradient(120deg, #ffffff, {COLORS['accent']});
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
    """


def _controls_css() -> str:
    return f"""
        /* primary buttons */
        .stButton > button {{
            background: linear-gradient(120deg, {COLORS['accent']}, {COLORS['accent2']});
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.6rem 1.4rem;
            transition: all 0.25s ease;
            box-shadow: 0 4px 16px rgba(124, 92, 255, 0.35);
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(124, 92, 255, 0.55);
            filter: brightness(1.1);
        }}
        .stButton > button:active {{
            transform: translateY(0);
        }}

        /* text area + text input */
        .stTextArea textarea, .stTextInput input {{
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid {COLORS['card_border']} !important;
            border-radius: 12px !important;
            color: {COLORS['text']} !important;
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }}
        .stTextArea textarea:focus, .stTextInput input:focus {{
            border-color: {COLORS['accent']} !important;
            box-shadow: 0 0 0 3px rgba(124, 92, 255, 0.2) !important;
        }}
        .stTextArea textarea::placeholder, .stTextInput input::placeholder {{
            color: {COLORS['text_dim']} !important;
        }}

        /* tabs */
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px 10px 0 0;
            color: {COLORS['text_dim']} !important;
            transition: all 0.25s ease;
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(124, 92, 255, 0.15) !important;
            color: #ffffff !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background: linear-gradient(90deg, {COLORS['accent']}, {COLORS['accent2']}) !important;
            height: 3px !important;
        }}

        /* sidebar */
        section[data-testid="stSidebar"] {{
            background: {COLORS['bg_alt']};
            border-right: 1px solid {COLORS['card_border']};
        }}
        section[data-testid="stSidebar"] .stMarkdown {{
            color: {COLORS['text']};
        }}

        /* metric delta / streamlit metric styling */
        [data-testid="stMetric"] {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['card_border']};
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }}

        /* headings */
        h2, h3, h4 {{ color: {COLORS['text']} !important; }}

        /* dataframes / tables */
        .stDataFrame, .stTable {{
            border-radius: 12px;
            overflow: hidden;
        }}
    """
