"""Documents page: upload, library, bulk folder import, danger zone.
"""

from pathlib import Path

import streamlit as st

from localrag.core.application import RAGApplication
from localrag.core.compat import SUPPORTED_EXTENSIONS
from localrag.core.constants import icon_for
from localrag.core.indexer import (
    FolderIndexer,
    is_folder_indexing_running,
    start_folder_indexing,
    stop_folder_indexing,
)
from localrag.ui.components import empty_state, hero
from localrag.ui.i18n import t


def show_documents_page(app: RAGApplication):
    hero(t("docs_hero_title"), t("docs_hero_sub"))

    tab_upload, tab_library, tab_bulk, tab_danger = st.tabs(
        [t("tab_upload"), t("tab_library"), t("tab_bulk"), t("tab_danger")]
    )

    with tab_upload:
        available_exts = sorted(SUPPORTED_EXTENSIONS.keys())
        if available_exts:
            st.caption(t("supported") + ": " + " · ".join(e.lstrip(".").upper() for e in available_exts))
        else:
            st.error(t("no_loaders"))

        uploaded_files = st.file_uploader(
            t("upload_drop"),
            accept_multiple_files=True,
            type=[e.lstrip(".") for e in available_exts] or None,
            label_visibility="collapsed",
            disabled=not available_exts,
        )

        if uploaded_files:
            total_size = sum(f.size for f in uploaded_files)
            c1, c2 = st.columns(2)
            c1.metric(t("files_selected"), len(uploaded_files))
            c2.metric(t("total_size"), app.doc_manager.format_size(total_size))

            if st.button(t("add_to_kb"), type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_callback(current, total, message):
                    progress_bar.progress(min(current / max(total, 1), 1.0))
                    status_text.text(message)

                with st.spinner(t("processing_docs")):
                    success, message, failures = app.add_documents(uploaded_files, progress_callback)

                    if success:
                        st.success(f"✅ {message}")
                        if not failures:
                            st.balloons()
                    else:
                        st.error(f"❌ {message}")

                    if failures:
                        with st.expander(f"⚠️ {len(failures)} file(s) skipped — see why", expanded=not success):
                            for fname, reason in failures:
                                st.markdown(f"**{fname}**")
                                st.caption(reason)

    with tab_library:
        documents = app.doc_manager.get_all_documents()

        if documents:
            st.markdown(
                f"<span class='app-pill'>{len(documents)} files</span>&nbsp;"
                f"<span class='app-pill'>{app.doc_manager.format_size(app.doc_manager.get_total_size())}</span>",
                unsafe_allow_html=True,
            )
            st.write("")

            search = st.text_input(t("filter_filename"), "")
            filtered = [d for d in documents if search.lower() in d.name.lower()] if search else documents

            for doc in filtered:
                col1, col2, col3 = st.columns([5, 2, 1])
                with col1:
                    st.markdown(
                        f"<div class='app-card' style='margin-bottom:6px;padding:10px 16px;'>"
                        f"<span style='font-size:1.2rem;'>{icon_for(doc)}</span>&nbsp;"
                        f"<span class='app-card-title'>{doc.name}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f"<div style='padding-top:14px;color:#6B6B7B;font-size:0.85rem;'>"
                        f"{app.doc_manager.format_size(doc.stat().st_size)}</div>",
                        unsafe_allow_html=True,
                    )
                with col3:
                    if st.button("🗑️", key=f"delete_{doc.name}", help=f"Delete {doc.name}"):
                        if app.delete_document_full(doc):
                            st.success(f"Deleted {doc.name} and its indexed chunks")
                            st.rerun()
                        else:
                            st.error(f"Failed to fully delete {doc.name}")

            if search and not filtered:
                st.info(t("no_match_filter"))
        else:
            empty_state("📥", t("no_docs_yet_title"), t("no_docs_yet_sub"))

    with tab_bulk:
        st.caption(t("bulk_intro"))

        default_folder = st.session_state.get(
            "bulk_folder_path", app.config.config.get("last_bulk_folder_path", "")
        )
        folder_input = st.text_input(t("bulk_folder_label"), value=default_folder)

        if folder_input:
            folder_path = Path(folder_input).expanduser()
            st.session_state.bulk_folder_path = folder_input
            if app.config.config.get("last_bulk_folder_path") != folder_input:
                app.config.config["last_bulk_folder_path"] = folder_input
                app.config.save_config()
            indexer = FolderIndexer(app, folder_path)
            running = is_folder_indexing_running(folder_path)

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button(t("bulk_start"), type="primary", use_container_width=True, disabled=running):
                    if not folder_path.exists() or not folder_path.is_dir():
                        st.error(t("bulk_folder_missing"))
                    else:
                        start_folder_indexing(app, folder_path)
                        st.rerun()
            with c2:
                if st.button(t("bulk_stop"), use_container_width=True, disabled=not running):
                    stop_folder_indexing(folder_path)
                    st.rerun()
            with c3:
                if st.button(t("bulk_refresh"), use_container_width=True):
                    st.rerun()

            status = indexer.read_status()
            status_state = status.get("status", "idle")
            status_messages = {
                "idle": t("bulk_status_idle"),
                "running": t("bulk_status_running"),
                "stopped": t("bulk_status_stopped"),
                "done": t("bulk_status_done"),
                "error": t("bulk_status_error"),
            }
            if status_state == "running" or running:
                st.info(status_messages.get("running"))
            elif status_state == "done":
                st.success(status_messages.get("done"))
            elif status_state == "error":
                st.error(f"{status_messages.get('error')} {status.get('error', '')}")
            elif status_state == "stopped":
                st.warning(status_messages.get("stopped"))
            else:
                st.caption(status_messages.get("idle"))

            if status_state != "idle":
                total_files = status.get("total_files", 0)
                done_files = status.get("done_files", 0)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(t("bulk_files_done"), done_files)
                m2.metric(t("bulk_files_total"), total_files)
                m3.metric(t("bulk_chunks"), status.get("chunks_indexed", 0))
                m4.metric(t("bulk_errors"), status.get("errors", 0))

                if total_files:
                    st.progress(min(done_files / total_files, 1.0))

                if status.get("current_file"):
                    st.caption(f"{t('bulk_current_file')}: `{status['current_file']}`")

                if status.get("errors", 0):
                    with st.expander(t("bulk_view_errors")):
                        for item in app.persistent_store.failed_files():
                            st.markdown(f"**{item['path']}**")
                            st.caption(item["error"])

    with tab_danger:
        st.warning(t("danger_warning"))

        if not st.session_state.confirm_delete_all:
            if st.button(t("clear_all"), type="secondary"):
                st.session_state.confirm_delete_all = True
                st.rerun()
        else:
            st.error(t("confirm_delete_all_msg"))
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("confirm_delete_all_btn"), type="primary", use_container_width=True):
                    if app.clear_knowledge_base():
                        st.success(t("kb_cleared"))
                    else:
                        st.error(t("kb_clear_failed"))
                    st.session_state.confirm_delete_all = False
                    st.rerun()
            with col2:
                if st.button(t("cancel"), use_container_width=True):
                    st.session_state.confirm_delete_all = False
                    st.rerun()
