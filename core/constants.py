"""Static constants: LLM backend modes, OpenAI-compatible provider presets, file icons.
"""

from pathlib import Path


LLM_BACKEND_MODES = {
    "gguf_only": "Offline GGUF only (default, fully private)",
    "gguf_then_openai": "Offline GGUF first, OpenAI-compatible API as fallback",
    "openai_only": "OpenAI-compatible API only",
}

# One-click presets for the OpenAI-compatible backend. Anything that speaks
# the OpenAI /v1/chat/completions protocol works — hosted services and
# self-hosted servers alike. "key_required" only affects the hint shown in
# Settings: when no key is set and a base URL is given, a placeholder key is
# sent automatically, because local servers generally ignore it.
OPENAI_PROVIDER_PRESETS = {
    "openai": {"label": "OpenAI (api.openai.com)", "base_url": "", "model": "gpt-4o-mini", "key_required": True},
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o-mini", "key_required": True},
    "groq": {"label": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "key_required": True},
    "together": {"label": "Together AI", "base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "key_required": True},
    "ollama": {"label": "Ollama (local)", "base_url": "http://localhost:11434/v1", "model": "llama3.2", "key_required": False},
    "lmstudio": {"label": "LM Studio (local)", "base_url": "http://localhost:1234/v1", "model": "local-model", "key_required": False},
    "vllm": {"label": "vLLM (self-hosted)", "base_url": "http://localhost:8000/v1", "model": "", "key_required": False},
    "llamacpp": {"label": "llama.cpp server (local)", "base_url": "http://localhost:8080/v1", "model": "local-model", "key_required": False},
    "custom": {"label": "Custom OpenAI-compatible endpoint", "base_url": "", "model": "", "key_required": False},
}


EXT_ICONS = {
    ".pdf": "📕", ".txt": "📃", ".docx": "📘", ".doc": "📘",
    ".csv": "📊", ".xlsx": "📗", ".xls": "📗",
    ".pptx": "📙", ".ppt": "📙", ".md": "📝",
    ".html": "🌐", ".htm": "🌐", ".epub": "📖", ".odt": "📘",
    ".rtf": "📄", ".eml": "✉️", ".msg": "✉️", ".json": "🔧",
}


def icon_for(path: Path) -> str:
    return EXT_ICONS.get(path.suffix.lower(), "📄")
