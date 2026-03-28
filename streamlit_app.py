#!/usr/bin/env python3
from __future__ import annotations

import base64
import gzip
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import requests
import streamlit as st


ROOT = Path(__file__).resolve().parent
TOOLS_DIR = ROOT / "tools"
CHUNK_FILES = [TOOLS_DIR / f"oura_streamlit_dashboard.py.gz.b64.part0{i}" for i in range(4)]
DECODED_APP_PATH = Path(tempfile.gettempdir()) / "oura_streamlit_dashboard_streamlit_cloud.py"
REMOTE_CHUNK_BASE = "https://raw.githubusercontent.com/pythonoptic-sketch/oura-biomarker-dashboard/main/tools"
SECRET_KEYS = (
    "OURA_BOOTSTRAP_CODE",
    "OURA_CLIENT_ID",
    "OURA_CLIENT_SECRET",
    "OURA_OAUTH_REDIRECT_URI",
    "OURA_PUBLIC_INVITE_ONLY",
    "OURA_ENABLE_BROWSER_OAUTH",
    "OURA_OAUTH_SCOPES",
)


def apply_streamlit_secrets_to_env() -> None:
    for key in SECRET_KEYS:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])


def decode_dashboard_source() -> Path:
    try:
        encoded = b"".join(path.read_bytes() for path in CHUNK_FILES)
        decoded = gzip.decompress(base64.b64decode(encoded))
    except Exception:
        remote_chunks = []
        for i in range(4):
            url = f"{REMOTE_CHUNK_BASE}/oura_streamlit_dashboard.py.gz.b64.part0{i}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            remote_chunks.append(response.content)
        encoded = b"".join(remote_chunks)
        decoded = gzip.decompress(base64.b64decode(encoded))
    DECODED_APP_PATH.write_bytes(decoded)
    return DECODED_APP_PATH


def load_dashboard_module():
    source_path = decode_dashboard_source()
    spec = importlib.util.spec_from_file_location("oura_streamlit_dashboard_cloud", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load decoded Oura dashboard module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def boot() -> None:
    apply_streamlit_secrets_to_env()
    module = load_dashboard_module()
    module.main()


if __name__ == "__main__":
    boot()
