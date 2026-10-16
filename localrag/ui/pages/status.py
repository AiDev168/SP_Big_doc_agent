"""Status page: counts, per-type statistics, system information.
"""

import streamlit as st

from localrag.core.application import RAGApplication
from localrag.core.constants import EXT_ICONS
from localrag.ui.components import api_status_label, empty_state, hero
from localrag.ui.i18n import t


def show_status_page(app: RAGApplication):
    hero(t("status_hero_title"), t("status_hero_sub"))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(t("indexed_chunks"), app.vector_manager.get_document_count())

    with col2:
        st.metric(t("total_documents_size"), app.doc_manager.format_size(app.doc_manager.get_total_size()))

    with col3:
        llm_status = t("loaded") if st.session_state.llm_loaded else t("not_loaded")
        st.metric(t("llm_status"), llm_status)

    st.metric(t("api_status_label"), api_status_label(app))

    st.markdown("---")

    st.subheader(t("doc_stats"))

    documents = app.doc_manager.get_all_documents()

    if documents:
        ext_counts = {}
        ext_sizes = {}

        for doc in documents:
            ext = doc.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            ext_sizes[ext] = ext_sizes.get(ext, 0) + doc.stat().st_size

        import pandas as pd

        df = pd.DataFrame(
            {
                "Type": [f"{EXT_ICONS.get(e, '📄')} {e}" for e in ext_counts.keys()],
                "Count": list(ext_counts.values()),
                "Size": [app.doc_manager.format_size(s) for s in ext_sizes.values()],
            }
        )

        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(df, use_container_width=True, hide_index=True)
        with c2:
            st.bar_chart(
                {row["Type"]: row["Count"] for _, row in df.iterrows()},
                use_container_width=True,
            )
    else:
        empty_state("📈", t("no_stats_title"), t("no_stats_sub"))

    st.markdown("---")

    st.subheader(t("system_info"))

    import platform

    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"<div class='app-card' dir='ltr' style='text-align:left;'>"
            f"<div class='app-card-sub'>OS</div><div class='app-card-title'>{platform.system()} {platform.release()}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='app-card' dir='ltr' style='text-align:left;'>"
            f"<div class='app-card-sub'>Python</div><div class='app-card-title'>{platform.python_version()}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='app-card' dir='ltr' style='text-align:left;'>"
            f"<div class='app-card-sub'>CPU</div><div class='app-card-title'>{platform.processor() or 'Unknown'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col2:
        if has_psutil:
            memory = psutil.virtual_memory()
            st.markdown(
                f"<div class='app-card' dir='ltr' style='text-align:left;'>"
                f"<div class='app-card-sub'>Total RAM</div><div class='app-card-title'>{app.doc_manager.format_size(memory.total)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='app-card' dir='ltr' style='text-align:left;'>"
                f"<div class='app-card-sub'>Available RAM</div><div class='app-card-title'>{app.doc_manager.format_size(memory.available)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(memory.percent / 100.0, text=f"RAM Usage: {memory.percent}%")
        else:
            empty_state("🧩", t("no_psutil_title"), t("no_psutil_sub"))
