"""Settings page: LLM / OpenAI API, retrieval, embeddings & chunking, chat & cache, appearance, storage, language.
"""

import json
import os
from pathlib import Path

import streamlit as st

from localrag.core.application import RAGApplication
from localrag.core.compat import (
    BM25Retriever,
    CrossEncoder,
    EnsembleRetriever,
    OCR_AVAILABLE,
    OpenAIModule,
    Pytesseract,
)
from localrag.core.config import resolve_local_model
from localrag.core.constants import LLM_BACKEND_MODES, OPENAI_PROVIDER_PRESETS
from localrag.core.llm import LLMManager
from localrag.core.network import is_offline_mode
from localrag.ui.components import api_status_label, hero
from localrag.ui.i18n import current_lang, t


def show_settings_page(app: RAGApplication):
    hero(t("settings_hero_title"), t("settings_hero_sub"))

    tab_llm, tab_retrieval, tab_embed, tab_chat, tab_appearance, tab_storage, tab_language = st.tabs(
        [t("tab_llm"), t("tab_retrieval"), t("tab_embed"), t("tab_chat"), t("tab_appearance"), t("tab_storage"), t("tab_language")]
    )

    with tab_llm:
        st.caption(
            "💡 Answer speed on a local model is dominated by three things: **model size** "
            "(a 7B model is much slower per token than a 3B one), **GPU Layers** (0 = pure "
            "CPU; if you have a GPU, raising this offloads work to it and is usually the "
            "single biggest speedup available), and **Context Length / Max Tokens** (larger "
            "values mean more work per answer). If generation feels slow, try a smaller "
            "quantized model or raising GPU Layers before anything else."
        )
        st.markdown("##### 🖥️ Offline GGUF model")
        model_path = st.text_input(
            "Model Path (GGUF format)",
            value=app.config.config["llm_model_path"],
            help="Path to your local GGUF model file",
        )

        col1, col2 = st.columns(2)

        with col1:
            context_length = st.number_input(
                "Context Length",
                min_value=512,
                max_value=32768,
                value=app.config.config["llm_context_length"],
                step=512,
            )

            max_tokens = st.number_input(
                "Max Tokens",
                min_value=64,
                max_value=4096,
                value=app.config.config["llm_max_tokens"],
                step=64,
            )

        with col2:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=app.config.config["llm_temperature"],
                step=0.1,
            )

            gpu_layers = st.number_input(
                "GPU Layers (0 for CPU only)",
                min_value=0,
                max_value=100,
                value=app.config.config["llm_gpu_layers"],
                step=1,
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save LLM Settings", use_container_width=True):
                app.config.config["llm_model_path"] = model_path
                app.config.config["llm_context_length"] = context_length
                app.config.config["llm_max_tokens"] = max_tokens
                app.config.config["llm_temperature"] = temperature
                app.config.config["llm_gpu_layers"] = gpu_layers
                app.config.save_config()
                st.success("Settings saved! Reload the model for changes to take effect.")
        with c2:
            if st.button("🚀 Load Model", type="primary", use_container_width=True):
                with st.spinner("Loading LLM... this may take a few minutes..."):
                    if app.initialize_llm(model_path):
                        st.session_state.llm_loaded = True
                        st.success("✅ LLM loaded successfully!")
                    else:
                        st.error("❌ Failed to load LLM. Check the GGUF model path or the OpenAI API key.")

        st.markdown("---")
        st.markdown("##### 📂 Local GGUF models")
        st.caption(
            f"The app doesn't download GGUF models itself. Copy your `.gguf` files into "
            f"`{app.config.models_dir}` (or type any path above)."
        )
        found_ggufs = sorted(p.name for p in app.config.models_dir.glob("*.gguf"))
        if found_ggufs:
            gc1, gc2 = st.columns([3, 1])
            with gc1:
                picked_gguf = st.selectbox("Found in the models folder", found_ggufs, key="found_gguf_select")
            with gc2:
                st.write("")
                if st.button("Use this file", use_container_width=True, key="use_found_gguf_btn"):
                    app.config.config["llm_model_path"] = str(app.config.models_dir / picked_gguf)
                    app.config.save_config()
                    st.rerun()
        else:
            st.caption("No .gguf files found in the models folder yet.")

        # ------------------------------------------------------------------
        # OpenAI / OpenAI-compatible API backend — sits behind the offline
        # GGUF model. Works with OpenAI itself and with any server that
        # speaks the OpenAI chat-completions protocol.
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("##### ☁️ OpenAI / OpenAI-compatible API (optional — sits behind the offline GGUF model)")
        if OpenAIModule is None:
            st.info("Install the SDK to enable this: `pip install openai --break-system-packages`")

        cfg = app.config.config
        saved_key = cfg.get("openai_api_key", "")
        env_key = os.environ.get("OPENAI_API_KEY", "")

        backend_keys = list(LLM_BACKEND_MODES.keys())
        current_mode = cfg.get("llm_backend", "gguf_only")
        backend_mode = st.selectbox(
            "LLM backend",
            options=backend_keys,
            format_func=lambda k: LLM_BACKEND_MODES[k],
            index=backend_keys.index(current_mode) if current_mode in backend_keys else 0,
            key="llm_backend_select",
            help="GGUF only: never uses the API. GGUF first: the API is used only if the "
                 "GGUF model isn't loaded or fails before producing any output. "
                 "API only: always uses the API.",
        )

        # Offline mode hides the remote providers; local servers stay available.
        preset_keys = [
            k for k, v in OPENAI_PROVIDER_PRESETS.items()
            if not is_offline_mode() or k == "custom" or LLMManager.is_local_url(v["base_url"])
        ]
        current_provider = cfg.get("openai_provider", "openai")
        if current_provider not in preset_keys:
            current_provider = "custom"
        provider = st.selectbox(
            "Provider preset",
            options=preset_keys,
            format_func=lambda k: OPENAI_PROVIDER_PRESETS[k]["label"],
            index=preset_keys.index(current_provider),
            key="openai_provider_select",
            help="Fills in a typical base URL and model. Any OpenAI-compatible server works — "
                 "pick 'Custom' and enter your own URL.",
        )
        preset = OPENAI_PROVIDER_PRESETS[provider]
        same_provider = provider == current_provider

        # Widget keys include the provider, so switching presets resets the
        # fields to that preset's defaults instead of keeping stale values.
        openai_base_url = st.text_input(
            "Base URL",
            value=cfg.get("openai_base_url", "") if same_provider else preset["base_url"],
            placeholder="https://api.openai.com/v1",
            help="Include the /v1 part, e.g. http://localhost:1234/v1. Leave empty for api.openai.com.",
            key=f"openai_base_url_input_{provider}",
        )

        endpoint_blocked = is_offline_mode() and not LLMManager.is_local_url(openai_base_url)
        if endpoint_blocked:
            st.error(
                "🔒 Offline mode is on: only local / LAN endpoints are allowed "
                "(e.g. http://localhost:11434/v1 for Ollama). Enter a local URL, or switch to "
                "Online mode in the sidebar to use a remote API."
            )
        elif LLMManager.is_local_url(openai_base_url):
            st.info("🖥️ This endpoint looks local / on your LAN — documents stay on your own network.")
        else:
            st.warning(
                "This is a remote service: your question **and the retrieved document excerpts** "
                "are sent to the provider when it answers. Keep **Offline GGUF only** selected "
                "if your documents must never leave this machine.",
                icon="⚠️",
            )

        if saved_key:
            st.caption(f"🔑 A key is saved in config (…{saved_key[-4:]}).")
        elif env_key:
            st.caption(f"🔑 Using the OPENAI_API_KEY environment variable (…{env_key[-4:]}).")
        elif preset["key_required"]:
            st.caption("No key set yet — this provider requires one.")
        else:
            st.caption("No key set. That's fine for most local/self-hosted servers (a placeholder is sent automatically).")

        key_input = st.text_input(
            "API key",
            type="password",
            value="",
            placeholder="leave blank to keep the saved key" if saved_key else (
                "sk-…" if preset["key_required"] else "optional for local servers"
            ),
            key="openai_key_input",
        )

        default_model = cfg.get("openai_model", "gpt-4o-mini") if same_provider else preset["model"]
        fetched_models = st.session_state.get("openai_fetched_models", {}).get(provider)
        if fetched_models:
            model_options = fetched_models if default_model in fetched_models else ([default_model] + fetched_models if default_model else fetched_models)
            openai_model = st.selectbox(
                f"Model ({len(fetched_models)} available on this server)",
                options=model_options,
                index=model_options.index(default_model) if default_model in model_options else 0,
                key=f"openai_model_select_{provider}",
            )
        else:
            openai_model = st.text_input(
                "Model",
                value=default_model,
                help="Use 'Fetch models' below to list what the server offers.",
                key=f"openai_model_input_{provider}",
            )

        with st.expander("Advanced"):
            extra_headers_text = st.text_area(
                "Extra HTTP headers (JSON object, optional)",
                value=cfg.get("openai_extra_headers", "") if same_provider else "",
                placeholder='{"HTTP-Referer": "https://example.com", "X-Title": "My App"}',
                height=80,
                key=f"openai_headers_input_{provider}",
            )
            use_streaming = st.toggle(
                "Stream tokens",
                value=bool(cfg.get("openai_use_streaming", True)),
                help="Turn off for servers without streaming support. Even when on, the app "
                     "automatically retries without streaming if streaming fails before any text arrives.",
                key="openai_streaming_toggle",
            )

        def _parse_headers(text):
            text = (text or "").strip()
            if not text:
                return {}, None
            try:
                data = json.loads(text)
            except Exception as e:
                return None, f"not valid JSON ({e})"
            if not isinstance(data, dict):
                return None, "must be a JSON object like {\"Header\": \"value\"}"
            return {str(k): str(v) for k, v in data.items()}, None

        parsed_headers, header_error = _parse_headers(extra_headers_text)
        if header_error:
            st.error(f"Extra headers: {header_error}")

        def _client_from_form():
            return app.llm_manager.build_openai_client(
                api_key=key_input.strip() or None,
                base_url=openai_base_url.strip(),
                extra_headers=parsed_headers,
            )

        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            if st.button("💾 Save", use_container_width=True, key="save_openai_btn", disabled=bool(header_error) or endpoint_blocked):
                cfg["llm_backend"] = backend_mode
                cfg["openai_provider"] = provider
                cfg["openai_base_url"] = openai_base_url.strip()
                if key_input.strip():
                    cfg["openai_api_key"] = key_input.strip()
                cfg["openai_model"] = (openai_model or "").strip() or preset["model"] or "gpt-4o-mini"
                cfg["openai_extra_headers"] = extra_headers_text.strip()
                cfg["openai_use_streaming"] = bool(use_streaming)
                app.config.save_config()
                if backend_mode in ("gguf_then_openai", "openai_only"):
                    app.llm_manager.initialize_openai()
                st.session_state.llm_loaded = app.llm_manager.is_ready()
                st.success(
                    "Saved. " + (
                        "Backend ready." if st.session_state.llm_loaded
                        else "Backend not ready yet — load the GGUF model or check the endpoint settings."
                    )
                )
        with bc2:
            if st.button("📋 Fetch models", use_container_width=True, key="fetch_models_btn", disabled=bool(header_error) or endpoint_blocked):
                try:
                    client = _client_from_form()
                    ids = sorted({m.id for m in client.models.list()})
                    if not ids:
                        st.warning("The server returned an empty model list. Type the model name manually.")
                    else:
                        st.session_state.setdefault("openai_fetched_models", {})[provider] = ids
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Could not list models ({type(e).__name__}: {e}). "
                             f"Some servers don't expose /models — type the model name manually.")
        with bc3:
            if st.button("🧪 Test", use_container_width=True, key="test_openai_btn", disabled=bool(header_error) or endpoint_blocked):
                try:
                    client = _client_from_form()
                    test_kwargs = dict(
                        model=(openai_model or "").strip() or preset["model"] or "gpt-4o-mini",
                        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                    )
                    try:
                        resp = client.chat.completions.create(max_tokens=16, **test_kwargs)
                    except Exception:
                        resp = client.chat.completions.create(max_completion_tokens=256, **test_kwargs)
                    reply = (resp.choices[0].message.content or "").strip()
                    app.llm_manager.api_verified = True
                    app.llm_manager.api_verify_error = ""
                    st.success(f"✅ Endpoint works — model replied: {reply!r}")
                except Exception as e:
                    app.llm_manager.api_verified = False
                    app.llm_manager.api_verify_error = f"{type(e).__name__}: {e}"
                    st.error(f"❌ {type(e).__name__}: {e}")
        with bc4:
            if st.button(
                "🗑️ Remove key", use_container_width=True,
                disabled=not saved_key, key="remove_openai_key_btn",
            ):
                cfg["openai_api_key"] = ""
                app.config.save_config()
                app.llm_manager.openai_client = None
                app.llm_manager.api_verified = None
                st.session_state.llm_loaded = app.llm_manager.is_ready()
                st.rerun()

        # ---- API connection status badge --------------------------------
        api_code, api_detail = app.llm_manager.api_status()
        badge = f"**API status:** {api_status_label(app)}"
        if api_detail and api_code in ("connected", "ready"):
            badge += f" — `{api_detail}`"
        if api_code == "connected":
            st.success(badge)
        elif api_code in ("no_key", "failed", "blocked"):
            st.error(badge + (f"\n\n{api_detail}" if api_code == "failed" and api_detail else ""))
        elif api_code in ("ready", "not_ready"):
            st.warning(badge + ("  \nPress 🧪 Test to verify it." if api_code == "ready" else ""))
        else:
            st.info(badge + "  \nSwitch the LLM backend above to enable it.")

    with tab_retrieval:
        st.caption(f"Currently active: **{app.retriever.describe_active_pipeline()}**")

        st.markdown("##### 🔀 Hybrid search")
        hybrid_available = BM25Retriever is not None and EnsembleRetriever is not None
        if not hybrid_available:
            st.info(
                "Hybrid keyword search isn't available — install `rank_bm25` "
                "(and `langchain-classic` or `langchain`) to enable it. See the "
                "import banner at the top of the app for exact commands."
            )
        use_hybrid = st.toggle(
            "Blend BM25 keyword search with vector search",
            value=app.config.config.get("use_hybrid_search", True) and hybrid_available,
            disabled=not hybrid_available,
            help="Catches exact terms, names, codes, and acronyms that pure semantic "
                 "similarity can miss. Recommended when your documents mix scripts "
                 "(e.g. Persian text with English product codes or numbers).",
        )
        hybrid_weight = st.slider(
            "Vector weight (remainder goes to BM25)",
            min_value=0.0, max_value=1.0,
            value=float(app.config.config.get("hybrid_vector_weight", 0.6)),
            step=0.05,
            disabled=not (use_hybrid and hybrid_available),
        )

        st.markdown("##### 🧩 Multi-query expansion")
        multi_query = st.toggle(
            "Ask the LLM to rephrase the question before retrieving",
            value=app.config.config.get("use_multi_query", False),
            disabled=not st.session_state.get("llm_loaded", False),
            help="Retrieves for several phrasings of the same question and merges the "
                 "results — helps when the wording of a question doesn't match the "
                 "wording in the documents. Requires the LLM to be loaded.",
        )
        if not st.session_state.get("llm_loaded", False):
            st.caption("Load the LLM (in the LLM tab) to enable multi-query expansion.")
        if multi_query:
            st.warning(
                "⚠️ This runs a full extra LLM generation *before* retrieval even starts, "
                "on top of the answer generation itself — with a local CPU model this can "
                "roughly double how long each answer takes (with OpenAI it costs one extra "
                "API call per question). If answers feel slow, try turning this off first.",
                icon="⚠️",
            )
        multi_query_count = st.number_input(
            "Number of alternate phrasings to generate",
            min_value=1, max_value=6,
            value=int(app.config.config.get("multi_query_count", 3)),
            disabled=not multi_query,
            help="Each additional phrasing means more retrieval work per question, but "
                 "does not add another LLM call (only the rephrasing step above does).",
        )

        st.markdown("##### 🎯 Cross-encoder reranking")
        reranker_available = CrossEncoder is not None
        if not reranker_available:
            st.info(
                "Reranking isn't available — install `sentence-transformers` to "
                "enable it. See the import banner at the top of the app."
            )
        use_reranker = st.toggle(
            "Rerank candidates with a cross-encoder before answering",
            value=app.config.config.get("use_reranker", True) and reranker_available,
            disabled=not reranker_available,
            help="Scores every retrieved chunk directly against the question with a "
                 "small, more precise model, and keeps only the best ones.",
        )
        reranker_model = st.text_input(
            "Reranker model (HuggingFace cross-encoder)",
            value=app.config.config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            disabled=not use_reranker,
            help="For multilingual/Persian corpora, consider a multilingual cross-encoder "
                 "such as 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'. Offline: enter a local "
                 "folder path, or copy the model folder into the app's models/rerankers folder.",
        )

        col1, col2 = st.columns(2)
        with col1:
            rerank_candidates = st.number_input(
                "Candidates fetched per query",
                min_value=5, max_value=100,
                value=int(app.config.config.get("rerank_candidates", 20)),
                step=5,
                help="How many chunks each retrieval stage pulls before merging/reranking.",
            )
        with col2:
            final_k = st.number_input(
                "Final chunks used to answer",
                min_value=1, max_value=20,
                value=int(app.config.config.get("final_k", 5)),
            )

        if st.button("💾 Save Retrieval Settings", type="primary", use_container_width=True):
            app.config.config["use_hybrid_search"] = use_hybrid and hybrid_available
            app.config.config["hybrid_vector_weight"] = hybrid_weight
            app.config.config["use_multi_query"] = multi_query
            app.config.config["multi_query_count"] = multi_query_count
            app.config.config["use_reranker"] = use_reranker and reranker_available
            app.config.config["reranker_model"] = reranker_model
            app.config.config["rerank_candidates"] = rerank_candidates
            app.config.config["final_k"] = final_k
            app.config.save_config()
            st.success(f"Saved. Active pipeline: {app.retriever.describe_active_pipeline()}")

    with tab_embed:
        # Multilingual / Persian-capable embedding models are listed alongside
        # the original English-optimized ones, since all-MiniLM-L6-v2 and
        # mpnet are trained primarily on English and perform poorly on Persian.
        embedding_options = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "intfloat/multilingual-e5-base",
            "intfloat/multilingual-e5-large",
            "heydariAI/persian-embeddings",
        ]
        multilingual_models = {
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "intfloat/multilingual-e5-base",
            "intfloat/multilingual-e5-large",
            "heydariAI/persian-embeddings",
        }
        current_embedding = app.config.config["embedding_model"]
        default_index = (
            embedding_options.index(current_embedding)
            if current_embedding in embedding_options
            else 0
        )

        if current_lang() == "fa" and current_embedding not in multilingual_models:
            st.info(
                "برای بهترین نتیجه با اسناد فارسی، یکی از مدل‌های چندزبانه (مثل "
                "paraphrase-multilingual-mpnet-base-v2 یا multilingual-e5-base) را انتخاب کنید."
            )

        embedding_model = st.selectbox(
            "Embedding Model", embedding_options, index=default_index,
            help="Models marked multilingual below support Persian and other "
                 "non-English scripts; the plain English ones do not.",
        )
        local_embedding_path = st.text_input(
            "Local embedding model folder (optional — overrides the list above)",
            value=current_embedding if Path(current_embedding).expanduser().is_dir() else "",
            placeholder=str(app.config.models_dir / "embeddings" / "my-model"),
            help="Offline mode never downloads models; online mode downloads names from the list "
                 "automatically. Copy a sentence-transformers model folder "
                 f"into {app.config.models_dir / 'embeddings'} (named after the model, e.g. "
                 "'paraphrase-multilingual-mpnet-base-v2') or point to any folder here.",
        )
        if local_embedding_path.strip():
            embedding_model = local_embedding_path.strip()
        _resolved_embedding = resolve_local_model(embedding_model, app.config.models_dir, "embeddings")
        if Path(_resolved_embedding).is_dir():
            st.caption(f"✅ Found locally: `{_resolved_embedding}`")
        elif is_offline_mode():
            st.caption(
                "⚠️ Not found locally. In offline mode it only loads if it is already in the HuggingFace "
                f"cache; otherwise copy the model folder into `{app.config.models_dir / 'embeddings'}` "
                "or switch to Online mode."
            )
        else:
            st.caption("🌐 Not found locally — it will be downloaded from HuggingFace the next time it is loaded (online mode).")
        if embedding_model in multilingual_models:
            st.caption("🌐 Multilingual model — supports Persian and other non-English scripts.")
        if embedding_model != current_embedding:
            st.caption(
                "⚠️ Changing the embedding model does not re-embed existing chunks. "
                "Clear the knowledge base and re-add documents after switching."
            )

        col1, col2 = st.columns(2)

        with col1:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=4000,
                value=app.config.config["chunk_size"],
                step=100,
            )

        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=app.config.config["chunk_overlap"],
                step=50,
            )

        st.markdown("##### 🧱 Chunking strategy")
        chunking_options = {"fixed": "Fixed size (character count)", "structural": "Structure-aware (headings & paragraphs first)"}
        chunking_strategy = st.selectbox(
            "How documents are split into chunks",
            options=list(chunking_options.keys()),
            format_func=lambda k: chunking_options[k],
            index=list(chunking_options.keys()).index(app.config.config.get("chunking_strategy", "fixed")),
            help="Structure-aware tries to split on headings and paragraph breaks before "
                 "falling back to plain character counts, so a chunk is more likely to be "
                 "a coherent section instead of an arbitrary slice through a sentence. "
                 "Best for well-formatted documents (Markdown, docx, structured PDFs).",
        )

        st.markdown("##### 🔎 OCR for scanned PDFs")
        if not OCR_AVAILABLE:
            st.info(
                "OCR isn't available — install `pymupdf` and `pytesseract` "
                "(plus the `tesseract-ocr` system package) to enable it. "
                "Without it, scanned/image-only PDFs will fail to index with "
                "'no usable text' instead of being read via OCR."
            )
        else:
            if st.button("🩺 Run OCR diagnostic check"):
                # Importing pytesseract only confirms the *Python package* is
                # present — it says nothing about whether the tesseract-ocr
                # *system binary* it wraps is actually installed and on
                # PATH, which is the far more common way OCR silently fails.
                # get_tesseract_version() is the one call that actually
                # invokes the binary, so it's the real test.
                try:
                    version = Pytesseract.get_tesseract_version()
                    st.success(f"✅ tesseract-ocr engine found (version {version}).")
                    langs_configured = app.config.config.get("ocr_languages", "eng")
                    try:
                        available_langs = set(Pytesseract.get_languages())
                        for lang in langs_configured.split("+"):
                            lang = lang.strip()
                            if lang and lang not in available_langs:
                                st.warning(
                                    f"⚠️ Language pack '{lang}' is not installed for tesseract "
                                    f"(available: {', '.join(sorted(available_langs)) or 'none detected'}). "
                                    f"Install it, e.g. 'apt install tesseract-ocr-{lang}'."
                                )
                            elif lang:
                                st.caption(f"✅ Language pack '{lang}' is installed.")
                    except Exception:
                        pass
                except Exception as e:
                    st.error(
                        f"❌ The tesseract-ocr Python package (pytesseract) imported fine, but "
                        f"calling the actual OCR engine failed: {type(e).__name__}: {e}\n\n"
                        f"This almost always means the tesseract-ocr **system package** isn't "
                        f"installed or isn't on PATH — install it separately from the Python "
                        f"packages (e.g. 'apt install tesseract-ocr' on Debian/Ubuntu, "
                        f"'brew install tesseract' on macOS), then restart the app."
                    )
        ocr_enabled = st.toggle(
            "Fall back to OCR when a PDF has no extractable text layer",
            value=app.config.config.get("ocr_enabled", True) and OCR_AVAILABLE,
            disabled=not OCR_AVAILABLE,
            help="Only used when normal text extraction finds nothing — most PDFs have a "
                 "real text layer and never touch OCR, so this costs nothing for them.",
        )
        ocr_languages = st.text_input(
            "OCR language(s) (tesseract codes, + separated)",
            value=app.config.config.get("ocr_languages", "eng"),
            disabled=not (ocr_enabled and OCR_AVAILABLE),
            help="e.g. 'eng' for English, 'fas' for Persian, or 'eng+fas' for both. "
                 "Requires the matching tesseract language pack to be installed.",
        )

        if st.button("💾 Save Embedding Settings", type="primary", use_container_width=True):
            app.config.config["embedding_model"] = embedding_model
            app.config.config["chunk_size"] = chunk_size
            app.config.config["chunk_overlap"] = chunk_overlap
            app.config.config["chunking_strategy"] = chunking_strategy
            app.config.config["ocr_enabled"] = ocr_enabled and OCR_AVAILABLE
            app.config.config["ocr_languages"] = ocr_languages
            app.config.save_config()
            # Rebuild the splitter immediately so new chunk settings actually apply.
            app.doc_processor._build_splitter()
            st.success("Settings saved and applied to new documents.")

    with tab_chat:
        st.markdown(f"##### 🧠 {t('tab_chat')}")
        use_chat_memory = st.toggle(
            "Include recent conversation when answering",
            value=app.config.config.get("use_chat_memory", True),
            help="Lets follow-up questions like \"what about its capital?\" be understood "
                 "in context. The recent turns are only used to interpret the question — "
                 "the answer itself is still grounded only in the retrieved documents.",
        )
        memory_turns = st.number_input(
            "Number of recent exchanges to remember",
            min_value=1, max_value=10,
            value=int(app.config.config.get("memory_turns", 3)),
            disabled=not use_chat_memory,
        )

        st.markdown("---")
        st.markdown("##### ⚡ Answer cache")
        use_answer_cache = st.toggle(
            "Cache answers and reuse them instantly for repeated questions",
            value=app.config.config.get("use_answer_cache", True),
            help="A question you've asked before is answered instantly from cache — no "
                 "retrieval, no LLM call — as long as your documents, document selection "
                 "and retrieval settings haven't changed since. Automatically invalidated "
                 "when you add, remove, or clear documents, or change retrieval or LLM "
                 "backend settings.",
        )
        cache_count = app.persistent_store.answer_cache_count()
        st.caption(f"{cache_count} cached answer(s) stored.")
        if st.button("🗑️ Clear answer cache"):
            app.persistent_store.clear_answer_cache()
            st.success("Answer cache cleared.")
            st.rerun()

        if st.button("💾 Save Chat Settings", type="primary", use_container_width=True):
            app.config.config["use_chat_memory"] = use_chat_memory
            app.config.config["memory_turns"] = memory_turns
            app.config.config["use_answer_cache"] = use_answer_cache
            app.config.save_config()
            st.success("Settings saved.")

    with tab_appearance:
        st.markdown(f"##### 🎨 {t('tab_appearance')}")
        theme_options = {"dark": t("theme_dark"), "light": t("theme_light")}
        chosen_theme = st.radio(
            t("theme_label"),
            options=list(theme_options.keys()),
            format_func=lambda k: theme_options[k],
            index=list(theme_options.keys()).index(app.config.config.get("theme", "dark")),
            key="theme_settings_radio",
        )
        if chosen_theme != st.session_state.get("theme", "dark"):
            st.session_state.theme = chosen_theme
            app.config.config["theme"] = chosen_theme
            app.config.save_config()
            st.rerun()

    with tab_storage:
        for label, path in [
            ("Base Directory", app.config.base_dir),
            ("Documents", app.config.docs_dir),
            ("Vector DB", app.config.db_dir),
            ("Models", app.config.models_dir),
        ]:
            st.markdown(
                f"<div class='app-card' dir='ltr' style='text-align:left;'>"
                f"<div class='app-card-title'>{label}</div>"
                f"<div class='app-card-sub'>{path}</div></div>",
                unsafe_allow_html=True,
            )

    with tab_language:
        st.markdown(f"##### {t('language_label')}")
        st.caption(
            "Switching language also switches the app's text direction (RTL for "
            "Persian) and the language the assistant is asked to answer in."
            if current_lang() != "fa" else
            "تغییر زبان، جهت متن برنامه را نیز تغییر می‌دهد (راست‌به‌چپ برای فارسی) "
            "و زبانی که از دستیار خواسته می‌شود پاسخ دهد."
        )
        lang_options = {"en": "🇬🇧 English", "fa": "🇮🇷 فارسی (Persian)"}
        chosen = st.radio(
            t("language_label"),
            options=list(lang_options.keys()),
            format_func=lambda k: lang_options[k],
            index=list(lang_options.keys()).index(current_lang()),
            key="language_settings_radio",
        )
        if chosen != st.session_state.language:
            st.session_state.language = chosen
            app.config.config["language"] = chosen
            app.config.save_config()
            st.rerun()

        st.markdown("---")
        st.caption(
            "Tip: for Persian documents, also pick a multilingual embedding model "
            "in the Embeddings & Chunking tab and keep hybrid (BM25) search on in "
            "the Retrieval tab — it helps a lot with exact Persian terms and names."
            if current_lang() != "fa" else
            "نکته: برای اسناد فارسی، در تب امبدینگ و تکه‌بندی یک مدل چندزبانه انتخاب کنید "
            "و در تب بازیابی، جستجوی ترکیبی (BM25) را روشن نگه دارید — این کار به تشخیص "
            "دقیق اصطلاحات و نام‌های فارسی کمک زیادی می‌کند."
        )
