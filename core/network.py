"""Network mode (Offline / Online) and the socket-level offline guard.

State lives on the `socket` module so it survives Streamlit reruns.
Importing this module sets telemetry-off env vars and installs the guard.
"""


import os
import sys

# --------------------------------------------------------------------------
# NETWORK MODE — Offline (default) or Online. Chosen in the sidebar and
# remembered in config.json ("network_mode").
#   Offline: HuggingFace downloads are disabled and a socket-level guard
#            refuses every connection to a non-local address, so nothing can
#            reach the internet even by accident. Local-network addresses
#            (localhost, 127.x, 10.x, 192.168.x, 172.16-31.x, *.local,
#            single-label LAN hostnames) stay allowed, so a local
#            OpenAI-compatible server (Ollama, LM Studio, llama.cpp...) works.
#   Online:  the guard lets everything through; models can be downloaded by
#            name from HuggingFace and remote OpenAI-compatible APIs can be used.
# The state lives on the `socket` module, not in a module global, because
# Streamlit re-executes this script on every interaction and a global would
# reset each time. First-run default is offline (or online if the environment
# variable RAG_OFFLINE=0 is set). Telemetry is disabled in both modes.
# --------------------------------------------------------------------------
for _k, _v in {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "ANONYMIZED_TELEMETRY": "False",              # chromadb
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    "DO_NOT_TRACK": "1",
}.items():
    os.environ[_k] = _v


def _initial_offline_state() -> bool:
    env = os.environ.get("RAG_OFFLINE", "").strip().lower()
    default_offline = env not in ("0", "false", "no", "off")
    try:
        import json as _json
        from pathlib import Path as _Path
        cfg_file = _Path.home() / ".local_rag_app" / "config.json"
        if cfg_file.exists():
            saved = _json.loads(cfg_file.read_text(encoding="utf-8")).get("network_mode")
            if saved in ("offline", "online"):
                return saved == "offline"
    except Exception:
        pass
    return default_offline


def _net_state() -> dict:
    import socket
    state = getattr(socket, "_rag_net_state", None)
    if state is None:
        state = {"offline": _initial_offline_state()}
        socket._rag_net_state = state
    return state


def is_offline_mode() -> bool:
    return _net_state()["offline"]


def apply_network_mode(offline: bool) -> None:
    """Switch between Offline and Online at runtime. The socket guard reads
    the state on every connection, so it takes effect immediately. The
    HuggingFace offline flags are normally read when those libraries are
    imported, so the already-loaded modules are patched too (best effort —
    restarting the app always applies the saved mode from the very start)."""
    offline = bool(offline)
    _net_state()["offline"] = offline
    flag = "1" if offline else "0"
    for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[_name] = flag
    hf_constants = sys.modules.get("huggingface_hub.constants")
    if hf_constants is not None and hasattr(hf_constants, "HF_HUB_OFFLINE"):
        hf_constants.HF_HUB_OFFLINE = offline
    tf_hub = sys.modules.get("transformers.utils.hub")
    if tf_hub is not None and hasattr(tf_hub, "_is_offline_mode"):
        tf_hub._is_offline_mode = offline


def _install_offline_guard() -> None:
    import socket
    import ipaddress
    import errno

    if getattr(socket, "_rag_offline_guard", False):
        return  # patch only once per process

    def _is_local_host(host) -> bool:
        if host is None:
            return True
        if isinstance(host, bytes):
            host = host.decode("ascii", "ignore")
        h = str(host).strip().lower().strip("[]")
        if h in ("", "localhost", "ip6-localhost", "broadcasthost"):
            return True
        try:
            ip = ipaddress.ip_address(h.split("%")[0])
            return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified
        except ValueError:
            pass
        return h.endswith((".local", ".localdomain", ".lan", ".internal")) or "." not in h

    _real_connect = socket.socket.connect
    _real_connect_ex = socket.socket.connect_ex
    _real_getaddrinfo = socket.getaddrinfo

    def _msg(host) -> str:
        return (f"Offline mode: blocked network access to '{host}'. "
                f"Switch to Online mode in the sidebar to allow internet access.")

    def _connect(self, address):
        if (_net_state()["offline"] and self.family in (socket.AF_INET, socket.AF_INET6)
                and isinstance(address, tuple) and not _is_local_host(address[0])):
            raise OSError(errno.ENETUNREACH, _msg(address[0]))
        return _real_connect(self, address)

    def _connect_ex(self, address):
        if (_net_state()["offline"] and self.family in (socket.AF_INET, socket.AF_INET6)
                and isinstance(address, tuple) and not _is_local_host(address[0])):
            return errno.ENETUNREACH
        return _real_connect_ex(self, address)

    def _getaddrinfo(host, *args, **kwargs):
        if _net_state()["offline"] and not _is_local_host(host):
            raise socket.gaierror(socket.EAI_NONAME, _msg(host))
        return _real_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    socket.getaddrinfo = _getaddrinfo
    socket._rag_offline_guard = True


_install_offline_guard()
apply_network_mode(is_offline_mode())
