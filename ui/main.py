"""App shell: startup, sidebar (network / language / navigation / overview), page routing.
"""

import traceback

import streamlit as st

from localrag.core.application import RAGApplication
from localrag.core.config import Config
from localrag.core.network import apply_network_mode, is_offline_mode
from localrag.ui.components import api_status_label, show_import_errors_banner
from localrag.ui.i18n import t
from localrag.ui.pages.chat import show_chat_page
from localrag.ui.pages.documents import show_documents_page
from localrag.ui.pages.settings import show_settings_page
from localrag.ui.pages.status import show_status_page
from localrag.ui.theme import inject_custom_css


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Local Document Q&A", page_icon="📚", layout="wide"
    )

    try:
        if "app" not in st.session_state:
            st.session_state.app = RAGApplication()
    except Exception:
        # Language may not be resolvable yet if RAGApplication itself failed,
        # so this failure path is always shown in English.
        inject_custom_css()
        show_import_errors_banner()
        st.error(
            "❌ The app failed to start. This is almost always a missing or "
            "mismatched dependency — see the full error below and the fix "
            "commands in the import banner above."
        )
        st.code(traceback.format_exc(), language="text")
        if is_offline_mode():
            st.info(
                "Offline mode is on, so missing models can't be downloaded. If the error above is about "
                "a missing embedding model, copy that model into the app's models/embeddings folder — "
                "or switch to Online mode and let HuggingFace download it."
            )
            if st.button("🌐 Switch to Online mode and retry"):
                apply_network_mode(False)
                try:
                    _cfg = Config()
                    _cfg.config["network_mode"] = "online"
                    _cfg.save_config()
                except Exception:
                    pass
                st.rerun()
        st.stop()

    if "language" not in st.session_state:
        st.session_state.language = st.session_state.app.config.config.get("language", "en")
    if "theme" not in st.session_state:
        st.session_state.theme = st.session_state.app.config.config.get("theme", "dark")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_loaded" not in st.session_state:
        st.session_state.llm_loaded = False
    if "confirm_delete_all" not in st.session_state:
        st.session_state.confirm_delete_all = False

    inject_custom_css()
    show_import_errors_banner()

    app = st.session_state.app

    with st.sidebar:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
            "<span style='font-size:1.8rem;'>📚</span>"
            f"<span style='font-size:1.25rem;font-weight:700;'>{t('app_title')}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(t("app_tagline"))

        net_options = {"offline": "🔒 " + t("net_offline"), "online": "🌐 " + t("net_online")}
        current_net = "offline" if is_offline_mode() else "online"
        chosen_net = st.radio(
            t("network_label"),
            options=list(net_options.keys()),
            format_func=lambda k: net_options[k],
            index=list(net_options.keys()).index(current_net),
            horizontal=True,
            key="network_mode_radio",
            help=t("network_help"),
        )
        if chosen_net != current_net:
            apply_network_mode(chosen_net == "offline")
            app.config.config["network_mode"] = chosen_net
            app.config.save_config()
            # Re-evaluate the API backend under the new mode (remote endpoints
            # are only allowed online), then refresh the "LLM loaded" status.
            app.llm_manager.openai_client = None
            app.llm_manager.api_verified = None
            if app.llm_manager._openai_enabled():
                app.llm_manager.initialize_openai()
            st.session_state.llm_loaded = app.llm_manager.is_ready()
            st.rerun()
        st.caption(("🔒 " + t("offline_badge")) if is_offline_mode() else ("🌐 " + t("online_badge")))

        lang_options = {"en": "🇬🇧 English", "fa": "🇮🇷 فارسی"}
        selected_lang = st.selectbox(
            t("language_label"),
            options=list(lang_options.keys()),
            format_func=lambda k: lang_options[k],
            index=list(lang_options.keys()).index(st.session_state.language),
            key="language_selector",
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            app.config.config["language"] = selected_lang
            app.config.save_config()
            st.rerun()

        st.markdown("---")

        page_keys = ["nav_chat", "nav_documents", "nav_settings", "nav_status"]
        page = st.radio(
            "Navigation",
            [t(k) for k in page_keys],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(f"**{t('overview')}**")

        doc_count = app.vector_manager.get_document_count()
        st.metric(t("indexed_chunks"), doc_count)

        total_size = app.doc_manager.get_total_size()
        st.metric(t("documents_size"), app.doc_manager.format_size(total_size))

        llm_status = t("loaded") if st.session_state.llm_loaded else t("not_loaded")
        st.metric(t("llm_status"), llm_status)
        st.metric(t("api_status_label"), api_status_label(app))

    if page == t("nav_chat"):
        show_chat_page(app)
    elif page == t("nav_documents"):
        show_documents_page(app)
    elif page == t("nav_settings"):
        show_settings_page(app)
    elif page == t("nav_status"):
        show_status_page(app)
