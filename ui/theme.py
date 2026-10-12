"""Theme + custom CSS injection (dark/light, LTR/RTL).
"""

import textwrap

import streamlit as st

from localrag.ui.i18n import current_lang, is_rtl_lang


# --------------------------------------------------------------------------
# UI theming
# --------------------------------------------------------------------------
def inject_custom_css():
    rtl = is_rtl_lang(current_lang())
    direction = "rtl" if rtl else "ltr"
    text_align = "right" if rtl else "left"
    text_align_opp = "left" if rtl else "right"
    # System fonts only (nothing is fetched from the web). Tahoma / Segoe UI /
    # Noto Sans Arabic cover Persian on Windows, macOS and Linux; Vazirmatn
    # and Inter are used if they happen to be installed locally.
    font_family = (
        "'Vazirmatn', 'Segoe UI', Tahoma, 'Noto Sans Arabic', 'Inter', system-ui, sans-serif"
        if rtl else
        "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Tahoma, 'Noto Sans Arabic', 'Vazirmatn', sans-serif"
    )

    theme = st.session_state.get("theme", "dark")
    if theme == "light":
        palette = {
            "brand1": "#6D5BD0", "brand2": "#8E7CFB",
            "bg_grad_start": "#F7F6FC", "bg_grad_end": "#FFFFFF",
            "surface": "#FFFFFF", "surface_muted": "#F1EFFB",
            "border": "#E1DEF2", "text": "#241F3D", "text_muted": "#6B6683",
            "code_bg": "#F3F2FA", "code_text": "#241F3D",
            "empty_title": "#241F3D",
        }
    else:
        palette = {
            "brand1": "#8E7CFB", "brand2": "#B4A6FF",
            "bg_grad_start": "#14121F", "bg_grad_end": "#17142A",
            "surface": "#1E1B33", "surface_muted": "#262244",
            "border": "#35304F", "text": "#EDEBFA", "text_muted": "#A9A4C4",
            "code_bg": "#0F0D1A", "code_text": "#EDEBFA",
            "empty_title": "#EDEBFA",
        }

    st.markdown(
        textwrap.dedent(f"""\
        <style>
        html, body, [class*="css"]  {{
            font-family: {font_family};
        }}

        :root {{
            --brand-1: {palette['brand1']};
            --brand-2: {palette['brand2']};
            --bg-0: {palette['bg_grad_start']};
            --bg-1: {palette['bg_grad_end']};
            --surface: {palette['surface']};
            --surface-muted: {palette['surface_muted']};
            --border: {palette['border']};
            --text: {palette['text']};
            --text-muted: {palette['text_muted']};
        }}

        /* App background */
        .stApp {{
            background: linear-gradient(180deg, {palette['bg_grad_start']} 0%, {palette['bg_grad_end']} 100%);
            color: var(--text);
            direction: {direction};
        }}

        .stApp, .stApp * {{
            font-family: {font_family};
        }}

        /* Base text alignment follows the active language direction.
           Numbers, code and inputs stay LTR internally (handled below)
           so mixed Persian/English/number content still reads correctly. */
        .stMarkdown, .stMarkdown p, .stMarkdown li, h1, h2, h3, h4, h5, h6,
        label, .stCaption, .stAlert, .app-card, .empty-state {{
            text-align: {text_align};
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #4A3FBF 0%, #2C2560 100%);
            direction: {direction};
        }}
        section[data-testid="stSidebar"] * {{
            color: #F2F0FF !important;
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            font-weight: 500;
        }}
        section[data-testid="stSidebar"] [data-testid="stMetric"] {{
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 8px 12px;
            margin-bottom: 6px;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.2);
        }}

        /* Headings */
        h1, h2, h3 {{
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            color: var(--text);
        }}
        p, span, label, div {{
            color: var(--text);
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
            border: 1px solid var(--border);
            background: var(--surface);
            color: var(--text);
            transition: all 0.15s ease;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, var(--brand-1), var(--brand-2));
            border: none;
            color: #14121F;
            box-shadow: 0 4px 14px rgba(142, 124, 251, 0.35);
        }}
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(142, 124, 251, 0.5);
        }}

        /* Text inputs, number inputs, selects, text areas — direction
           follows the UI language, but individual fields (e.g. file paths)
           can still be typed in either script. */
        .stTextInput input, .stNumberInput input, .stTextArea textarea,
        div[data-baseweb="select"] > div, div[data-baseweb="input"] {{
            background-color: var(--surface) !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
            direction: {direction};
            text-align: {text_align};
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            color: var(--text-muted);
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--brand-2);
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: var(--brand-1) !important;
        }}
        div[data-baseweb="tab-border"] {{
            background-color: var(--border) !important;
        }}

        /* Expanders */
        details {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            direction: {direction};
        }}

        /* File uploader */
        [data-testid="stFileUploaderDropzone"] {{
            background: var(--surface);
            border-color: var(--border) !important;
        }}

        /* Dataframes / tables */
        [data-testid="stDataFrame"] {{
            background: var(--surface);
        }}

        /* Code blocks always stay LTR regardless of UI language — code,
           paths and stack traces are not Persian text. */
        .stCodeBlock, pre, code {{
            background-color: {palette['code_bg']} !important;
            color: {palette['code_text']} !important;
            direction: ltr;
            text-align: left;
        }}

        /* Cards */
        .app-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
        }}
        .app-card-title {{
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text);
        }}
        .app-card-sub {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        .app-pill {{
            display: inline-block;
            background: var(--surface-muted);
            color: var(--brand-2);
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.72rem;
            font-weight: 600;
        }}

        /* Empty state */
        .empty-state {{
            text-align: center;
            padding: 48px 24px;
            background: var(--surface);
            border: 1px dashed var(--border);
            border-radius: 16px;
            color: var(--text-muted);
        }}
        .empty-state .emoji {{
            font-size: 2.4rem;
            margin-bottom: 8px;
        }}

        /* Chat bubbles — container follows UI direction; individual
           messages get dir="auto" applied inline per-message so a Persian
           question and an English answer in the same thread each align
           correctly (see render_chat_message). */
        [data-testid="stChatMessage"] {{
            border-radius: 14px;
            padding: 4px 6px;
            background: var(--surface);
        }}

        /* Metric styling on main area */
        [data-testid="stMetric"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 14px;
            text-align: {text_align};
        }}
        [data-testid="stMetricValue"] {{
            color: var(--text) !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--text-muted) !important;
        }}

        /* Header banner */
        .hero-banner {{
            background: linear-gradient(135deg, #5B4BD6, var(--brand-1));
            border-radius: 18px;
            padding: 22px 26px;
            color: white;
            margin-bottom: 18px;
            text-align: {text_align};
        }}
        .hero-banner h1 {{
            color: white !important;
            margin: 0 0 4px 0;
            font-size: 1.5rem;
        }}
        .hero-banner p {{
            color: rgba(255,255,255,0.9);
            margin: 0;
            font-size: 0.9rem;
        }}
        </style>
        <script>
        (function() {{
            try {{
                document.documentElement.setAttribute('dir', '{direction}');
                document.documentElement.setAttribute('lang', '{current_lang()}');
            }} catch (e) {{}}
        }})();
        </script>
        """),
        unsafe_allow_html=True,
    )
