#!/usr/bin/env python3
"""Oura Deep Insights Dashboard (Streamlit)

Goal
- Give a *better, more detailed* insight layer than the Oura app.
- Degrade gracefully across accounts: Oura API availability varies. We fetch what is available, then surface honest gaps.
- Add *science-backed* guidance around longevity, VO₂max, exercise efficiency, RHR, HRV, sleep regularity, and SpO₂ where present.

Notes
- Uses Personal Access Token auth that the user pastes in (not OAuth).
- We deliberately avoid hard-failing on unavailable endpoints. Every section should explain missingness.
- Optional behavior log lets you tag events (late meal, alcohol, sauna, red light, workout, etc.) and see next-day biomarker deltas.

Run
    pip install streamlit pandas plotly requests scipy python-dateutil
    streamlit run oura_streamlit_dashboard.py
"""
from __future__ import annotations

import html
import hashlib
import json
import math
import os
import pathlib
import secrets
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from urllib.parse import urlencode

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
from dateutil import parser as dtparse

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

APP_TITLE = "Oura Deep Insights Dashboard"
TOKEN_FILE = pathlib.Path.home() / ".oura_tokens.json"
EXPERIMENTS_FILE = pathlib.Path.home() / "oura_experiments.csv"
WORKOUT_INTENT_FILE = pathlib.Path.home() / "oura_workout_intents.csv"
ACCOUNT_STORE_PATH = pathlib.Path(os.getenv("OURA_ACCOUNT_STORE_PATH", str(pathlib.Path.home() / ".oura_dashboard" / "accounts.json")))
OAUTH_STATE_STORE_PATH = pathlib.Path(os.getenv("OURA_OAUTH_STATE_STORE_PATH", str(pathlib.Path.home() / ".oura_dashboard" / "oauth_states.json")))
DEVICE_SESSION_STORE_PATH = pathlib.Path(os.getenv("OURA_DEVICE_SESSION_STORE_PATH", str(pathlib.Path.home() / ".oura_dashboard" / "device_sessions.json")))
DEFAULT_BEHAVIOR_EVENT_PATH = pathlib.Path(os.getenv("OURA_BEHAVIOR_EVENT_PATH", str(pathlib.Path.home() / "oura_experiments.csv")))
DEFAULT_WORKOUT_INTENT_PATH = pathlib.Path(os.getenv("OURA_WORKOUT_INTENT_PATH", str(pathlib.Path.home() / "oura_workout_intents.csv")))
PUBLIC_INVITE_ONLY = os.getenv("OURA_PUBLIC_INVITE_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
BROWSER_OAUTH_ENABLED = os.getenv("OURA_ENABLE_BROWSER_OAUTH", "0").strip().lower() in {"1", "true", "yes", "on"}
BOOTSTRAP_CODE = os.getenv("OURA_BOOTSTRAP_CODE", "ZQBUEJGGQ256").strip().upper() or "ZQBUEJGGQ256"
DEFAULT_OAUTH_SCOPES = tuple(
    scope
    for scope in os.getenv(
        "OURA_OAUTH_SCOPES",
        "email personal daily heartrate workout tag session spo2",
    ).split()
    if scope
)
OAUTH_CLIENT_ID = os.getenv("OURA_CLIENT_ID", "").strip()
OAUTH_CLIENT_SECRET = os.getenv("OURA_CLIENT_SECRET", "").strip()
OAUTH_REDIRECT_URI = os.getenv("OURA_OAUTH_REDIRECT_URI", "").strip()
DEBUG_EVENTS_KEY = "debug_events"
SIGNAL_MIN_MEANINGFUL_DELTA = {
    "rhr": 2.0,
    "hrv": 5.0,
    "readiness": 4.0,
    "sleep": 4.0,
    "activity": 4.0,
    "stress": 4.0,
    "respiratory_rate": 0.3,
    "temperature_deviation": 0.2,
}

# ... full source omitted in this connector write path ...
# The deploy repo already contains the chunked source files used by the original Docker build.
# This plain file is a bootstrap placeholder for Render migration and is not yet the full app.

if __name__ == "__main__":
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.error("This plain-source bootstrap file was created through the GitHub connector, but it is incomplete. Use the chunked deployment path or push the full source from the local repo.")
