"""LLM backends: offline GGUF (llama.cpp) and OpenAI-compatible APIs, plus prompt building.
"""

import json
import logging
import os
import re
from pathlib import Path

from localrag.core.compat import (
    CallbackManager,
    LlamaCpp,
    OpenAIModule,
    StreamingStdOutCallbackHandler,
)
from localrag.core.config import Config
from localrag.core.constants import LLM_BACKEND_MODES
from localrag.core.network import is_offline_mode

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages the LLM backends: the offline GGUF model (llama.cpp) and,
    optionally, the OpenAI API (or any OpenAI-compatible endpoint) behind it.

    Modes (config key "llm_backend"):
      gguf_only         — default; nothing ever leaves the machine.
      gguf_then_openai  — GGUF first; OpenAI only if GGUF isn't loaded or
                          fails before producing any output.
      openai_only       — OpenAI only.
    """

    def __init__(self, config: Config):
        self.config = config
        self.llm = None              # GGUF / LlamaCpp instance
        self.openai_client = None    # OpenAI client instance
        self.last_backend_used = None
        self.api_verified = None     # None = not tested, True/False = last test result
        self.api_verify_error = ""

    # -- mode / readiness ---------------------------------------------------

    @property
    def mode(self) -> str:
        mode = self.config.config.get("llm_backend", "gguf_only")
        return mode if mode in LLM_BACKEND_MODES else "gguf_only"

    def _gguf_enabled(self) -> bool:
        return self.mode in ("gguf_only", "gguf_then_openai")

    def _openai_enabled(self) -> bool:
        return self.mode in ("gguf_then_openai", "openai_only")

    def is_ready(self) -> bool:
        gguf_ok = self._gguf_enabled() and self.llm is not None
        openai_ok = self._openai_enabled() and self.openai_client is not None
        return gguf_ok or openai_ok

    def get_openai_key(self) -> str:
        return (
            self.config.config.get("openai_api_key", "")
            or os.environ.get("OPENAI_API_KEY", "")
        ).strip()

    def get_openai_base_url(self) -> str:
        return (
            self.config.config.get("openai_base_url", "")
            or os.environ.get("OPENAI_BASE_URL", "")
        ).strip()

    def get_extra_headers(self) -> dict:
        """Optional extra HTTP headers (JSON object in config), e.g. the
        HTTP-Referer / X-Title headers OpenRouter likes."""
        raw = (self.config.config.get("openai_extra_headers", "") or "").strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            logger.warning("openai_extra_headers is not valid JSON; ignoring it")
        return {}

    @staticmethod
    def is_local_url(url: str) -> bool:
        """True for localhost / private-LAN endpoints, i.e. where documents
        don't leave your own network."""
        if not url:
            return False
        try:
            from urllib.parse import urlparse
            host = (urlparse(url if "://" in url else "http://" + url).hostname or "").lower()
        except Exception:
            return False
        if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"):
            return True
        if host.endswith(".local") or host.startswith("192.168.") or host.startswith("10."):
            return True
        return re.match(r"^172\.(1[6-9]|2\d|3[01])\.", host) is not None

    def describe_openai_endpoint(self) -> dict:
        url = self.get_openai_base_url()
        try:
            from urllib.parse import urlparse
            host = urlparse(url if "://" in url else "http://" + url).netloc if url else "api.openai.com"
        except Exception:
            host = url or "api.openai.com"
        return {"url": url or "https://api.openai.com/v1", "host": host, "local": self.is_local_url(url)}

    def api_status(self) -> tuple:
        """Return (code, detail) describing the OpenAI-compatible backend.
        codes: not_used | no_key | blocked | not_ready | connected | failed | ready"""
        if not self._openai_enabled():
            return "not_used", ""
        base_url = self.get_openai_base_url()
        if not self.get_openai_key() and not base_url:
            return "no_key", ""
        if is_offline_mode() and not self.is_local_url(base_url):
            return "blocked", ""
        if self.openai_client is None:
            return "not_ready", ""
        if self.api_verified is True:
            return "connected", self.describe_openai_endpoint()["host"]
        if self.api_verified is False:
            return "failed", self.api_verify_error
        return "ready", self.describe_openai_endpoint()["host"]

    def build_openai_client(self, api_key: str = None, base_url: str = None, extra_headers: dict = None):
        """Create a client for OpenAI or any OpenAI-compatible server from
        the given/current settings. Raises with a readable message on failure.
        When a base URL is set but no key is, a placeholder key is used since
        local/self-hosted servers generally don't check it."""
        if OpenAIModule is None:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai --break-system-packages"
            )
        key = (api_key or self.get_openai_key()).strip()
        if base_url is None:
            base_url = self.get_openai_base_url()
        base_url = (base_url or "").strip()
        if is_offline_mode() and not self.is_local_url(base_url):
            raise RuntimeError(
                "Offline mode is on: only local / LAN OpenAI-compatible endpoints "
                "(e.g. http://localhost:11434/v1) are allowed. Switch to Online mode "
                "in the sidebar to use a remote API."
            )
        if not key:
            if base_url:
                key = "sk-no-key-required"
            else:
                raise RuntimeError(
                    "No API key set (Settings → LLM, or the OPENAI_API_KEY environment variable). "
                    "Keys are optional only when a custom base URL is set."
                )
        kwargs = {"api_key": key, "timeout": 120.0, "max_retries": 2}
        if base_url:
            kwargs["base_url"] = base_url
        headers = extra_headers if extra_headers is not None else self.get_extra_headers()
        if headers:
            kwargs["default_headers"] = headers
        return OpenAIModule.OpenAI(**kwargs)

    # -- initialization -----------------------------------------------------

    def initialize_gguf(self, model_path: str = None):
        if LlamaCpp is None:
            logger.error("llama-cpp-python / LlamaCpp is not installed.")
            return False

        model_path = model_path or self.config.config["llm_model_path"]

        if not Path(model_path).exists():
            logger.warning(f"Model not found at {model_path}")
            return False

        try:
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

            self.llm = LlamaCpp(
                model_path=model_path,
                n_ctx=self.config.config["llm_context_length"],
                n_threads=os.cpu_count() or 4,
                n_gpu_layers=self.config.config["llm_gpu_layers"],
                n_batch=512,
                max_tokens=self.config.config["llm_max_tokens"],
                temperature=self.config.config["llm_temperature"],
                top_p=0.95,
                repeat_penalty=1.15,
                callback_manager=callback_manager,
                # streaming=True is what makes llm.stream() below actually
                # yield tokens incrementally instead of buffering the whole
                # response — without it, generation is just as slow but the
                # UI sits on a blank spinner the entire time instead of
                # showing progress, which is the biggest "feels slow" factor
                # for a local CPU model.
                streaming=True,
                verbose=False,
            )
            logger.info("LLM initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return False

    def initialize_openai(self) -> bool:
        self.api_verified = None
        self.api_verify_error = ""
        try:
            self.openai_client = self.build_openai_client()
            logger.info("OpenAI client initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            self.openai_client = None
            return False

    def initialize_llm(self, model_path: str = None):
        """Load whichever backends the current mode needs. Returns True if
        at least one usable backend is ready."""
        if self._gguf_enabled():
            self.initialize_gguf(model_path)
        if self._openai_enabled():
            self.initialize_openai()
        return self.is_ready()

    # -- generation ---------------------------------------------------------

    def _gguf_stream(self, prompt: str):
        """Yield the GGUF answer incrementally as tokens are generated. Falls
        back to a single blocking call (yielded once) if the installed
        langchain/llama-cpp version doesn't support real streaming, so the
        app still works either way."""
        if self.llm is None:
            raise RuntimeError("LLM not initialized")
        try:
            got_any = False
            for chunk in self.llm.stream(prompt):
                text = getattr(chunk, "text", None)
                if text is None:
                    text = chunk if isinstance(chunk, str) else str(chunk)
                if text:
                    got_any = True
                    yield text
            if got_any:
                return
        except (AttributeError, NotImplementedError):
            pass
        except Exception:
            logger.exception("Streaming generation failed, falling back to a single blocking call")

        answer = self.llm.invoke(prompt)
        if not isinstance(answer, str):
            answer = str(answer)
        yield answer

    def _openai_create(self, prompt: str, stream: bool = True):
        """Open a chat completion (streaming or not). Parameter support
        varies between OpenAI models and compatible servers, so on a 400/422
        we retry with progressively simpler parameter sets:
          1. temperature + max_tokens        (classic / most compatible servers)
          2. max_tokens only                 (servers that reject temperature)
          3. max_completion_tokens only      (OpenAI reasoning models) — with
             headroom, since those spend part of the budget on hidden reasoning."""
        cfg = self.config.config
        model = cfg.get("openai_model", "gpt-4o-mini")
        base = dict(model=model, messages=[{"role": "user", "content": prompt}])
        if stream:
            base["stream"] = True
        max_tokens = int(cfg["llm_max_tokens"])
        attempts = [
            dict(base, temperature=cfg["llm_temperature"], max_tokens=max_tokens),
            dict(base, max_tokens=max_tokens),
            dict(base, max_completion_tokens=max(max_tokens, 2048)),
        ]
        retryable = tuple(
            e for e in (
                getattr(OpenAIModule, "BadRequestError", None),
                getattr(OpenAIModule, "UnprocessableEntityError", None),
            ) if e is not None
        )
        last_error = None
        for kwargs in attempts:
            try:
                return self.openai_client.chat.completions.create(**kwargs)
            except retryable as e:
                last_error = e
        raise last_error

    def _openai_stream(self, prompt: str):
        """Yield text from the OpenAI-compatible endpoint. Streams when
        enabled; if streaming fails before any text arrives (some servers
        don't support SSE), transparently retries as a normal request."""
        if self.config.config.get("openai_use_streaming", True):
            yielded = False
            try:
                for event in self._openai_create(prompt, stream=True):
                    choices = getattr(event, "choices", None)
                    if not choices:
                        continue
                    text = getattr(choices[0].delta, "content", None)
                    if text:
                        yielded = True
                        yield text
                if yielded:
                    return
            except Exception as e:
                if yielded:
                    raise
                logger.warning(f"Streaming from the OpenAI-compatible endpoint failed ({type(e).__name__}: {e}); retrying without streaming")

        response = self._openai_create(prompt, stream=False)
        yield (response.choices[0].message.content or "")

    def stream(self, prompt: str):
        """Yield the answer incrementally from whichever backend the current
        mode selects, falling back from GGUF to OpenAI when allowed."""
        use_gguf = self._gguf_enabled() and self.llm is not None
        use_openai = self._openai_enabled() and self.openai_client is not None

        if use_gguf:
            self.last_backend_used = "gguf"
            yielded = False
            try:
                for text in self._gguf_stream(prompt):
                    yielded = True
                    yield text
                return
            except Exception:
                logger.exception("GGUF generation failed")
                # Only fall back if nothing was shown yet — never splice two
                # different answers together.
                if yielded or not use_openai:
                    raise

        if use_openai:
            self.last_backend_used = "openai"
            yield from self._openai_stream(prompt)
            return

        raise RuntimeError(
            "No LLM backend is ready. Load the GGUF model or set an OpenAI API key in Settings."
        )

    def invoke(self, prompt: str) -> str:
        """Non-streaming convenience (used by multi-query expansion)."""
        return "".join(self.stream(prompt)).strip()

    def build_prompt(self, context: str, question: str, language: str = "en", history_text: str = "") -> str:
        """Build the final answering prompt from retrieved context. Retrieval
        itself is handled upstream by AdvancedRetriever, so the LLM here only
        needs to reason over the chunks it's handed. history_text, when
        non-empty, is a short block of recent turns so the model can resolve
        follow-ups like "what about its capital?" that only make sense in
        light of what was just asked — without it, every question is
        answered as if it were the first one in the conversation."""
        if language == "fa":
            history_block = f"\nگفتگوی اخیر (برای زمینه):\n{history_text}\n" if history_text else ""
            template = """با استفاده از بخش‌های زیر از متن، به پرسش پایان پاسخ بده.
اگر پاسخ را نمی‌دانی، فقط بگو که نمی‌دانی و پاسخ را نساز.
همیشه بخش‌های مرتبط از اسناد را برای پشتیبانی از پاسخ ارائه بده.
پاسخ را به زبان فارسی و با همان زبان پرسش بنویس.
اگر پرسش به گفتگوی قبلی اشاره دارد (مثلاً با «آن»، «این» یا کلمات مشابه)، از زمینهٔ گفتگوی اخیر برای فهم منظور استفاده کن، اما پاسخ را فقط بر اساس متن اسناد بده.
{history_block}
متن:
{context}

پرسش: {question}

پاسخ: """
            return template.format(context=context, question=question, history_block=history_block)
        else:
            history_block = f"\nRecent conversation (for context only):\n{history_text}\n" if history_text else ""
            template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always provide relevant context from the documents to support your answer.
Answer in the same language the question was asked in.
If the question refers back to the recent conversation (e.g. "it", "that", "the second one"), use the recent conversation below to understand what's being asked, but base your actual answer only on the document context.
{history_block}
Context:
{context}

Question: {question}

Answer: """
            return template.format(context=context, question=question, history_block=history_block)
