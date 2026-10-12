"""Reusable UI building blocks: hero banner, empty state, chat bubbles, sources, scope selector.
"""

import textwrap
from datetime import datetime
from typing import List, Optional

import streamlit as st

from localrag.core.application import RAGApplication
from localrag.core.compat import IMPORT_ERRORS
from localrag.ui.i18n import current_lang, detect_text_direction, is_rtl_lang, t


def api_status_label(app) -> str:
    """Short, translated label for the OpenAI-compatible backend status."""
    code, _detail = app.llm_manager.api_status()
    labels = {
        "not_used": "⚪ " + t("api_not_used"),
        "no_key": "🔴 " + t("api_no_key"),
        "blocked": "🔒 " + t("api_blocked"),
        "not_ready": "🟠 " + t("api_not_ready"),
        "ready": "🟡 " + t("api_ready"),
        "connected": "🟢 " + t("api_connected"),
        "failed": "🔴 " + t("api_failed"),
    }
    return labels.get(code, code)


def hero(title: str, subtitle: str):
    rtl = is_rtl_lang(current_lang())
    st.markdown(
        textwrap.dedent(f"""\
        <div class="hero-banner" dir="{'rtl' if rtl else 'ltr'}">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """),
        unsafe_allow_html=True,
    )


def empty_state(emoji: str, title: str, subtitle: str):
    rtl = is_rtl_lang(current_lang())
    st.markdown(
        textwrap.dedent(f"""\
        <div class="empty-state" dir="{'rtl' if rtl else 'ltr'}">
            <div class="emoji">{emoji}</div>
            <div style="font-weight:600; color:var(--text); margin-bottom:4px;">{title}</div>
            <div>{subtitle}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_chat_message(role: str, content: str, avatar: str):
    """Render one chat bubble with per-message direction detection, so a
    Persian message and an English message in the same thread each get
    correct alignment instead of the whole chat being forced one way."""
    direction = detect_text_direction(content)
    align = "right" if direction == "rtl" else "left"
    with st.chat_message(role, avatar=avatar):
        st.markdown(
            f'<div dir="{direction}" style="text-align:{align};">{content}</div>',
            unsafe_allow_html=True,
        )


def show_import_errors_banner():
    """Show any dependency import failures as a real, readable error instead
    of letting the script crash silently into a blank page. Kept in English
    /LTR regardless of UI language since it surfaces raw package/exception
    names that are not translatable."""
    if not IMPORT_ERRORS:
        return
    critical = [e for e in IMPORT_ERRORS if e[0] in (
        "RecursiveCharacterTextSplitter", "HuggingFaceEmbeddings", "Chroma",
    )]
    st.error(
        f"⚠️ {len(IMPORT_ERRORS)} optional package(s) failed to import. "
        f"Missing file-type support is skipped automatically; anything listed "
        f"under **Core (required)** below will stop the app from working."
    )
    with st.expander("Show import details", expanded=bool(critical)):
        if critical:
            st.markdown("**Core (required):**")
            for label, hint, err in critical:
                st.code(f"{label}\n  {err}\n  fix: {hint}", language="text")
        others = [e for e in IMPORT_ERRORS if e not in critical]
        if others:
            st.markdown("**Optional (file-type support only):**")
            for label, hint, err in others:
                st.code(f"{label}\n  {err}\n  fix: {hint}", language="text")


def _format_source_label(source) -> str:
    """sources are now dicts ({file_name, page, snippet, ocr}); this renders
    a short label for contexts (exports, the sidebar) that just need the
    filename+page, not the full expander UI."""
    if isinstance(source, dict):
        label = source.get("file_name", "Unknown")
        page = source.get("page", "")
        if page != "":
            label += f" (page {page})"
        if source.get("ocr"):
            label += " [OCR]"
        return label
    return str(source)


def chat_history_to_markdown(history: List[dict]) -> str:
    """Serialize the chat transcript to a Markdown document, preserving
    per-message direction hints as HTML comments-free plain text so it
    still reads correctly when opened in any editor or viewer."""
    lines = ["# Chat Export", f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_", ""]
    for msg in history:
        who = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"{who}:")
        lines.append("")
        lines.append(msg["content"])
        if msg.get("sources"):
            lines.append("")
            lines.append("Sources: " + ", ".join(_format_source_label(s) for s in msg["sources"]))
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def render_sources_expander(sources: list):
    """Render source entries as an expander with the matched excerpt shown
    under each filename (with query terms bolded), not just a bare
    filename — lets someone actually check an answer against the text it
    came from instead of having to go find the original document."""
    with st.expander(f"{t('sources')} ({len(sources)})"):
        for source in sources:
            if isinstance(source, dict):
                label = source.get("file_name", "Unknown")
                page = source.get("page", "")
                if page != "":
                    label += f" (page {page})"
                if source.get("ocr"):
                    label += " 🔎 OCR"
                st.markdown(f"**{label}**")
                snippet = source.get("snippet", "")
                if snippet:
                    st.caption(snippet)
            else:
                st.markdown(f"- {source}")


def render_scope_selector(app: RAGApplication) -> Optional[List[str]]:
    """Let the user choose between answering from ALL documents or only from
    a SELECTED subset. Returns None for 'all documents', otherwise the list
    of selected file names (possibly empty). The choice is saved in
    config.json so it survives restarts."""
    cfg = app.config.config
    available = app.persistent_store.list_file_names()

    mode = cfg.get("doc_scope_mode", "all")
    if mode not in ("all", "selected"):
        mode = "all"
    # Drop files that no longer exist in the index (e.g. deleted documents).
    saved = [f for f in cfg.get("doc_scope_files", []) if f in available]

    if mode == "selected":
        summary = t("scope_active_selected").format(m=len(saved), n=len(available))
    else:
        summary = t("scope_active_all").format(n=len(available))

    with st.expander(f"📂 {t('scope_title')} — {summary}", expanded=False):
        options = {"all": t("scope_all"), "selected": t("scope_selected")}
        new_mode = st.radio(
            t("scope_title"),
            options=list(options.keys()),
            format_func=lambda k: options[k],
            index=list(options.keys()).index(mode),
            horizontal=True,
            label_visibility="collapsed",
            key="doc_scope_mode_radio",
        )
        new_files = saved
        if new_mode == "selected":
            new_files = st.multiselect(
                t("scope_pick"),
                options=available,
                default=saved,
                key="doc_scope_files_multiselect",
            )

        if new_mode != mode or (new_mode == "selected" and new_files != cfg.get("doc_scope_files", [])):
            cfg["doc_scope_mode"] = new_mode
            cfg["doc_scope_files"] = list(new_files)
            app.config.save_config()
            st.rerun()

    if mode == "selected":
        return list(saved)
    return None
