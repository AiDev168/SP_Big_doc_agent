"""Chat page.
"""

from datetime import datetime

import streamlit as st

from localrag.core.application import RAGApplication
from localrag.ui.components import (
    chat_history_to_markdown,
    empty_state,
    hero,
    render_chat_message,
    render_scope_selector,
    render_sources_expander,
)
from localrag.ui.i18n import current_lang, detect_text_direction, t


def show_chat_page(app: RAGApplication):
    hero(t("chat_hero_title"), t("chat_hero_sub"))
    lang = current_lang()

    if not st.session_state.llm_loaded:
        empty_state("🧠", t("no_model_title"), t("no_model_sub"))
        st.write("")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button(t("load_default_model"), type="primary", use_container_width=True):
                with st.spinner("Loading LLM..."):
                    if app.initialize_llm():
                        st.session_state.llm_loaded = True
                        st.success(t("llm_loaded_success"))
                        st.rerun()
                    else:
                        st.error(t("llm_load_failed"))
        return

    if app.vector_manager.get_document_count() == 0:
        empty_state("📄", t("no_docs_title"), t("no_docs_sub"))
        return

    st.markdown(
        f"<span class='app-pill'>🔍 {app.retriever.describe_active_pipeline()}</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Which documents may be used to answer: None = all, list = only those.
    scope_files = render_scope_selector(app)
    scope_empty = scope_files is not None and len(scope_files) == 0
    if scope_empty:
        st.warning(t("scope_none_selected"))

    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            empty_state("💡", t("ask_first_title"), t("ask_first_sub"))

        for message in st.session_state.chat_history:
            avatar = "🧑" if message["role"] == "user" else "📚"
            render_chat_message(message["role"], message["content"], avatar)
            if message["role"] == "assistant" and message.get("sources"):
                render_sources_expander(message["sources"])

    if prompt := st.chat_input(t("chat_placeholder"), disabled=scope_empty):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        render_chat_message("user", prompt, "🧑")

        with st.chat_message("assistant", avatar="📚"):
            placeholder = st.empty()
            direction = "rtl" if lang == "fa" else "ltr"
            align = "right" if direction == "rtl" else "left"
            collected = []
            sources = []
            retrieval_time = None
            generation_time = None
            from_cache = False
            backend_used = None
            error = None

            # Streamed token-by-token: the placeholder is rewritten on every
            # chunk so the answer appears progressively instead of the app
            # sitting on a spinner for the whole generation — this is the
            # single biggest fix for a local CPU model feeling slow, since
            # the person sees it working immediately rather than waiting in
            # silence for the full response before anything shows up.
            # Prior turns (excluding the just-appended current prompt) are
            # passed along so follow-up questions can be resolved in context.
            prior_history = st.session_state.chat_history[:-1]
            with st.spinner(t("thinking")):
                for event in app.query_stream(
                    prompt, language=lang, chat_history=prior_history, file_names=scope_files
                ):
                    if event["type"] == "meta":
                        sources = event["sources"]
                        retrieval_time = event["retrieval_time"]
                        from_cache = event.get("from_cache", False)
                    elif event["type"] == "token":
                        collected.append(event["text"])
                        current_dir = detect_text_direction("".join(collected)) or direction
                        current_align = "right" if current_dir == "rtl" else "left"
                        placeholder.markdown(
                            f'<div dir="{current_dir}" style="text-align:{current_align};">'
                            f'{"".join(collected)}▌</div>',
                            unsafe_allow_html=True,
                        )
                    elif event["type"] == "done":
                        generation_time = event["generation_time"]
                        backend_used = event.get("backend")
                    elif event["type"] == "error":
                        error = event["message"]

            if error:
                placeholder.empty()
                st.error(f"Error: {error}")
            else:
                answer = "".join(collected).strip()
                final_dir = detect_text_direction(answer)
                final_align = "right" if final_dir == "rtl" else "left"
                placeholder.markdown(
                    f'<div dir="{final_dir}" style="text-align:{final_align};">{answer}</div>',
                    unsafe_allow_html=True,
                )

                if sources:
                    render_sources_expander(sources)

                if retrieval_time is not None and generation_time is not None:
                    if from_cache:
                        st.caption(f"⚡ {t('cached_answer_label')}")
                    else:
                        st.caption(
                            f"⏱️ {t('retrieval_time_label')}: {retrieval_time:.1f}s · "
                            f"{t('generation_time_label')}: {generation_time:.1f}s"
                        )

                if backend_used == "openai":
                    ep = app.llm_manager.describe_openai_endpoint()
                    if ep["local"]:
                        st.caption(f"🖥️ {t('local_endpoint_notice')} ({ep['host']})")
                    else:
                        st.caption(f"☁️ {t('cloud_answer_notice')} ({ep['host']})")

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

    st.write("")
    if st.session_state.chat_history:
        c1, c2 = st.columns(2)
        with c1:
            if st.button(t("clear_chat"), use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with c2:
            st.download_button(
                t("export_chat"),
                data=chat_history_to_markdown(st.session_state.chat_history),
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
