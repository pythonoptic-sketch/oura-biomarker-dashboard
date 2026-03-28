#!/usr/bin/env python3
"""Oura Deep Insights Dashboard (Streamlit)

Goal
- Give a *better, more detailed* insight layer than the Oura app.
- Degrade gracefully across accounts: Oura API availability varies by subscription,
  ring generation, and feature flags.

Design principles
1) Timeline-first (decisions + causality), not "tabs of charts".
2) Personal baselines: everything normalized to your own recent history.
3) Defensive API: automatically detect what endpoints return data and adapt.
4) Transparent: show raw availability (HTTP codes, row counts, columns).

Run
  pip3 install streamlit plotly pandas requests python-dateutil
  export OURA_ACCESS_TOKEN="..."
  streamlit run tools/oura_streamlit_dashboard.py

Notes
- Informational only; not medical advice.
- If VO2 max doesn't show: many accounts return 200 + empty data for /v2/usercollection/vO2_max.
  This app will (a) try both vo2_max and vO2_max, (b) widen date range for sparse metrics.
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

API_BASE = "https://api.ouraring.com/v2"
OAUTH_TOKEN_URL = "https://api.ouraring.com/oauth/token"
OAUTH_AUTHORIZE_URL = "https://cloud.ouraring.com/oauth/authorize"
OAUTH_STATE_STORE_PATH = os.environ.get("OURA_OAUTH_STATE_STORE_PATH", "~/.oura_dashboard/oauth_states.json")
ACCOUNT_STORE_PATH = os.environ.get("OURA_ACCOUNT_STORE_PATH", "~/.oura_dashboard/accounts.json")
DEVICE_SESSION_STORE_PATH = os.environ.get("OURA_DEVICE_SESSION_STORE_PATH", "~/.oura_dashboard/device_sessions.json")
ACCOUNT_STORE_VERSION = 3
COMMUNITY_INVITE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
DEFAULT_OAUTH_SCOPES = "email personal daily heartrate workout tag session spo2"
OAUTH_STATE_MAX_AGE_MINUTES = 30
DEVICE_SESSION_MAX_AGE_DAYS = 90
BEHAVIOR_EVENT_PATH = os.environ.get("OURA_BEHAVIOR_EVENT_PATH", "~/.oura_dashboard/events.csv")
LEGACY_BEHAVIOR_EVENT_PATH = "~/clawd/tools/oura_behavior_events.csv"
WORKOUT_INTENT_PATH = os.environ.get("OURA_WORKOUT_INTENT_PATH", "~/.oura_dashboard/workout_intents.csv")
LEGACY_WORKOUT_INTENT_PATH = "~/clawd/tools/oura_workout_intents.csv"
BEHAVIOR_EVENT_OPTIONS = [
    "Alcohol",
    "Late meal",
    "Travel",
    "Illness",
    "Supplement",
    "Sauna",
    "Cold exposure",
    "Late caffeine",
]


# ------------------------------
# Utilities
# ------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(d: date) -> str:
    return d.isoformat()


def _parse_dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    if isinstance(v, str):
        try:
            dt = dtparse.isoparse(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _to_day(v: Any) -> Optional[str]:
    dt = _parse_dt(v)
    if dt:
        return dt.date().isoformat()
    if isinstance(v, str) and len(v) == 10 and v[4] == "-" and v[7] == "-":
        return v
    return None


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "—"
    try:
        if isinstance(v, float) and math.isnan(v):
            return "—"
    except Exception:
        pass
    return str(v)


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None


def disclaimer() -> None:
    st.caption(
        "Informational only, not medical advice. Wearables are noisy and delayed; use trends plus context, not one number in isolation."
    )


def record_debug_event(message: str, *, exc: Optional[BaseException] = None) -> None:
    events = st.session_state.setdefault("_debug_events", [])
    stamp = _utcnow().isoformat(timespec="seconds")
    detail = f"{stamp} | {message}"
    if exc is not None:
        detail = f"{detail} | {type(exc).__name__}: {exc}"
    events.append(detail)
    if len(events) > 100:
        del events[:-100]


def get_debug_events() -> List[str]:
    events = st.session_state.get("_debug_events", [])
    return [str(event) for event in events]


@contextmanager
def guarded_tab(tab: Any, label: str):
    with tab:
        try:
            yield
        except Exception as exc:
            record_debug_event(f"Tab render failed: {label}", exc=exc)
            st.error(f"{label} failed to render. Check the debug log in Data Access.")


def df_from_doc(doc: dict) -> pd.DataFrame:
    data = (doc or {}).get("data")
    if not isinstance(data, list):
        return pd.DataFrame()
    return pd.json_normalize([x for x in data if isinstance(x, dict)])


def latest(df: pd.DataFrame, col_candidates: List[str]) -> Any:
    if df.empty:
        return None
    for col in col_candidates:
        if col in df.columns:
            if "day" in df.columns:
                return df.sort_values("day").iloc[-1].get(col)
            return df.iloc[-1].get(col)
    return None


def rolling_mean_baseline(series: pd.Series, window: int) -> pd.Series:
    # robust-ish: median + MAD would be better; keep simple for v1
    return series.rolling(window, min_periods=max(3, window // 3)).mean()


def zscore(value: Optional[float], baseline: Optional[float], spread: Optional[float]) -> Optional[float]:
    if value is None or baseline is None or spread in (None, 0):
        return None
    return (value - baseline) / spread


def robust_spread(series: pd.Series) -> Optional[float]:
    xs = [float(x) for x in series.dropna().tolist() if _safe_float(x) is not None]
    if len(xs) < 6:
        return None
    med = float(pd.Series(xs).median())
    mad = float((pd.Series([abs(x - med) for x in xs])).median())
    return mad * 1.4826 if mad > 0 else float(pd.Series(xs).std())


def minimum_meaningful_change_for_metric(key: str, baseline: Optional[float] = None) -> float:
    base = abs(float(baseline)) if baseline is not None else None
    if key == "resting_hr":
        return max(2.0, (base or 0.0) * 0.05)
    if key == "hrv_rmssd":
        return max(4.0, (base or 0.0) * 0.10)
    if key in {"sleep_score", "readiness", "activity_score"}:
        return 5.0
    return 0.5


def hr_zones_karvonen(*, max_hr: int, resting_hr: int) -> List[Tuple[str, float, float]]:
    """5-zone HRR (Karvonen) model."""
    hrr = max(1, int(max_hr) - int(resting_hr))
    zones = [
        ("Z1", 0.50, 0.60),
        ("Z2", 0.60, 0.70),
        ("Z3", 0.70, 0.80),
        ("Z4", 0.80, 0.90),
        ("Z5", 0.90, 1.01),
    ]
    out: List[Tuple[str, float, float]] = []
    for name, lo, hi in zones:
        out.append((name, resting_hr + lo * hrr, resting_hr + hi * hrr))
    return out


def bucket_hr(bpm: float, zones: List[Tuple[str, float, float]]) -> str:
    for name, lo, hi in zones:
        if lo <= bpm < hi:
            return name
    return "<Z1" if bpm < zones[0][1] else "Z5"


def _slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text or "").strip())
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts) or "account"


def _token_identity(bundle: Dict[str, Any]) -> str:
    return str(bundle.get("refresh_token") or bundle.get("access_token") or "")


def _token_signature(bundle: Dict[str, Any]) -> str:
    identity = _token_identity(bundle).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:12] if identity else ""


def _account_profile_name(profile: Dict[str, Any]) -> str:
    parts = [str(profile.get("first_name") or "").strip(), str(profile.get("last_name") or "").strip()]
    name = " ".join(part for part in parts if part).strip()
    if name:
        return name
    for key in ["email", "preferred_name", "name"]:
        value = str(profile.get(key) or "").strip()
        if value:
            return value
    return ""


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _query_param_value(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0] if value else "")
    return str(value or "")


def _normalize_oauth_state_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    nonce = str(record.get("nonce") or record.get("state") or "").strip()
    action = str(record.get("action") or "").strip()
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    created_at = str(record.get("created_at") or "").strip()
    if not nonce or not action:
        return None
    if not created_at:
        created_at = _utcnow().isoformat()
    return {
        "nonce": nonce,
        "action": action,
        "payload": payload,
        "created_at": created_at,
    }


def _oauth_state_is_expired(record: Dict[str, Any], *, max_age_minutes: int = OAUTH_STATE_MAX_AGE_MINUTES) -> bool:
    created_at = _parse_dt(record.get("created_at"))
    if created_at is None:
        return True
    age = _utcnow() - created_at
    return age > timedelta(minutes=max_age_minutes)


def load_oauth_state_store(path: str = OAUTH_STATE_STORE_PATH) -> List[Dict[str, Any]]:
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw_states = payload.get("states", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    out: List[Dict[str, Any]] = []
    for raw in raw_states:
        normalized = _normalize_oauth_state_record(raw)
        if normalized is not None and not _oauth_state_is_expired(normalized):
            out.append(normalized)
    return sorted(out, key=lambda row: str(row.get("created_at") or ""))


def save_oauth_state_store(states: List[Dict[str, Any]], path: str = OAUTH_STATE_STORE_PATH) -> None:
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized = [
        row for row in (_normalize_oauth_state_record(state) for state in states)
        if row is not None and not _oauth_state_is_expired(row)
    ]
    p.write_text(json.dumps({"states": normalized}, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass


def persist_pending_oura_oauth(*, nonce: str, action: str, payload: Dict[str, Any], path: str = OAUTH_STATE_STORE_PATH) -> Dict[str, Any]:
    pending = _normalize_oauth_state_record(
        {
            "nonce": nonce,
            "action": action,
            "payload": payload,
            "created_at": _utcnow().isoformat(),
        }
    )
    if pending is None:
        raise ValueError("Could not persist the pending Oura sign-in flow.")
    states = [row for row in load_oauth_state_store(path) if str(row.get("nonce") or "") != nonce]
    states.append(pending)
    save_oauth_state_store(states, path)
    return pending


def claim_pending_oura_oauth(nonce: str, path: str = OAUTH_STATE_STORE_PATH) -> Optional[Dict[str, Any]]:
    target = str(nonce or "").strip()
    if not target:
        return None
    claimed: Optional[Dict[str, Any]] = None
    remaining: List[Dict[str, Any]] = []
    for row in load_oauth_state_store(path):
        if claimed is None and str(row.get("nonce") or "") == target:
            claimed = row
            continue
        remaining.append(row)
    if claimed is not None or remaining != load_oauth_state_store(path):
        save_oauth_state_store(remaining, path)
    return claimed


def current_app_url() -> str:
    context = getattr(st, "context", None)
    if context is None:
        return ""
    try:
        url = str(getattr(context, "url", "") or "").strip()
    except Exception:
        url = ""
    if url:
        return url
    try:
        headers = getattr(context, "headers", None)
    except Exception:
        headers = None
    if headers is None:
        return ""
    host = str(headers.get("host") or "").strip()
    if not host:
        return ""
    proto = str(headers.get("x-forwarded-proto") or headers.get("x-scheme") or "").strip()
    if "," in proto:
        proto = proto.split(",", 1)[0].strip()
    if not proto:
        proto = "http" if host.startswith(("localhost", "127.0.0.1")) else "https"
    return f"{proto}://{host}"


def oura_oauth_config() -> Dict[str, Any]:
    client_id = str(os.environ.get("OURA_CLIENT_ID") or "").strip()
    client_secret = str(os.environ.get("OURA_CLIENT_SECRET") or "").strip()
    redirect_uri = str(os.environ.get("OURA_OAUTH_REDIRECT_URI") or "").strip() or current_app_url()
    scopes = str(os.environ.get("OURA_OAUTH_SCOPES") or DEFAULT_OAUTH_SCOPES).strip() or DEFAULT_OAUTH_SCOPES
    missing: List[str] = []
    if not client_id:
        missing.append("OURA_CLIENT_ID")
    if not client_secret:
        missing.append("OURA_CLIENT_SECRET")
    if not redirect_uri:
        missing.append("OURA_OAUTH_REDIRECT_URI")
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "scopes": scopes,
        "enabled": not missing,
        "missing": missing,
    }


def browser_oura_oauth_enabled() -> bool:
    return bool(oura_oauth_config().get("enabled"))


def _normalize_device_session_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    token = str(record.get("token") or "").strip()
    community_id = str(record.get("community_id") or "").strip()
    member_id = str(record.get("member_id") or "").strip()
    active_account_id = str(record.get("active_account_id") or "").strip()
    compare_raw = record.get("compare_account_ids")
    if isinstance(compare_raw, list):
        compare_account_ids = [str(value).strip() for value in compare_raw if str(value).strip()]
    else:
        compare_account_ids = [part for part in str(compare_raw or "").split(",") if part.strip()]
    refresh_minutes = _safe_float(record.get("refresh_minutes"))
    refresh_value = int(refresh_minutes) if refresh_minutes in {0, 3, 6} else 6
    created_at = str(record.get("created_at") or "").strip() or _utcnow().isoformat()
    updated_at = str(record.get("updated_at") or "").strip() or created_at
    if not token or not community_id or not member_id:
        return None
    return {
        "token": token,
        "community_id": community_id,
        "member_id": member_id,
        "active_account_id": active_account_id or member_id,
        "compare_account_ids": compare_account_ids,
        "refresh_minutes": refresh_value,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _device_session_is_expired(record: Dict[str, Any], *, max_age_days: int = DEVICE_SESSION_MAX_AGE_DAYS) -> bool:
    updated_at = _parse_dt(record.get("updated_at"))
    if updated_at is None:
        return True
    return (_utcnow() - updated_at) > timedelta(days=max_age_days)


def load_device_sessions(path: str = DEVICE_SESSION_STORE_PATH) -> List[Dict[str, Any]]:
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw_rows = payload.get("sessions", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    rows: List[Dict[str, Any]] = []
    for raw in raw_rows:
        normalized = _normalize_device_session_record(raw)
        if normalized is not None and not _device_session_is_expired(normalized):
            rows.append(normalized)
    return sorted(rows, key=lambda row: str(row.get("updated_at") or ""), reverse=True)


def save_device_sessions(rows: List[Dict[str, Any]], path: str = DEVICE_SESSION_STORE_PATH) -> None:
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized = [
        row
        for row in (_normalize_device_session_record(item) for item in rows)
        if row is not None and not _device_session_is_expired(row)
    ]
    p.write_text(json.dumps({"sessions": normalized}, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass


def get_device_session(token: str, path: str = DEVICE_SESSION_STORE_PATH) -> Optional[Dict[str, Any]]:
    target = str(token or "").strip()
    if not target:
        return None
    for row in load_device_sessions(path):
        if str(row.get("token") or "") == target:
            return row
    return None


def persist_device_session(
    *,
    community_id: str,
    member_id: str,
    active_account_id: Optional[str],
    compare_account_ids: Optional[List[str]] = None,
    refresh_minutes: int = 6,
    token: Optional[str] = None,
    path: str = DEVICE_SESSION_STORE_PATH,
) -> Dict[str, Any]:
    session_token = str(token or "").strip() or secrets.token_urlsafe(24)
    existing = [row for row in load_device_sessions(path) if str(row.get("token") or "") != session_token]
    record = _normalize_device_session_record(
        {
            "token": session_token,
            "community_id": str(community_id or "").strip(),
            "member_id": str(member_id or "").strip(),
            "active_account_id": str(active_account_id or member_id or "").strip(),
            "compare_account_ids": list(compare_account_ids or []),
            "refresh_minutes": int(refresh_minutes) if int(refresh_minutes) in {0, 3, 6} else 6,
            "created_at": _utcnow().isoformat(),
            "updated_at": _utcnow().isoformat(),
        }
    )
    if record is None:
        raise ValueError("Could not persist the device session.")
    existing.append(record)
    save_device_sessions(existing, path)
    return record


def delete_device_session(token: str, path: str = DEVICE_SESSION_STORE_PATH) -> None:
    target = str(token or "").strip()
    if not target:
        return
    remaining = [row for row in load_device_sessions(path) if str(row.get("token") or "") != target]
    save_device_sessions(remaining, path)


def restore_device_session_from_query(path: str = DEVICE_SESSION_STORE_PATH) -> Optional[Dict[str, Any]]:
    token = _query_param_value(st.query_params.get("device_session", "")).strip()
    if not token:
        return None
    record = get_device_session(token, path)
    if record is None:
        if "device_session" in st.query_params:
            del st.query_params["device_session"]
        st.session_state.pop("device_session_token", None)
        return None
    st.session_state["device_session_token"] = token
    st.session_state["community_id"] = str(record.get("community_id") or "")
    st.session_state["community_member_id"] = str(record.get("member_id") or "")
    st.session_state["active_account_id"] = str(record.get("active_account_id") or record.get("member_id") or "")
    st.session_state["compare_account_ids"] = list(record.get("compare_account_ids") or [])
    st.session_state["auto_refresh_minutes"] = int(record.get("refresh_minutes") or 6)
    return record


def sync_device_session_query_param(token: Optional[str]) -> None:
    value = str(token or "").strip()
    if value:
        st.query_params["device_session"] = value
    elif "device_session" in st.query_params:
        del st.query_params["device_session"]


def maybe_enable_auto_refresh(minutes: Optional[int]) -> None:
    refresh_minutes = int(minutes or 0) if _safe_float(minutes) is not None else 0
    if refresh_minutes not in {3, 6}:
        return
    st.markdown(
        f'<meta http-equiv="refresh" content="{refresh_minutes * 60}">',
        unsafe_allow_html=True,
    )


def build_oura_authorize_url(*, state: str) -> str:
    config = oura_oauth_config()
    if not config["enabled"]:
        raise ValueError(f"Oura browser connect is not configured. Missing: {', '.join(config['missing'])}")
    params = {
        "response_type": "code",
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "scope": config["scopes"],
        "state": state,
    }
    return f"{OAUTH_AUTHORIZE_URL}?{urlencode(params)}"


def exchange_oura_authorization_code(code: str) -> Dict[str, Any]:
    config = oura_oauth_config()
    if not config["enabled"]:
        raise ValueError(f"Oura browser connect is not configured. Missing: {', '.join(config['missing'])}")
    response = requests.post(
        OAUTH_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": str(code or "").strip(),
            "redirect_uri": config["redirect_uri"],
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=40,
    )
    if response.status_code != 200:
        detail = response.text[:240] if response.text else response.reason
        if response.status_code == 401 and "invalid_client" in str(detail):
            raise ValueError(
                "Oura consent succeeded, but the app could not exchange the authorization code for tokens. "
                "This usually means the Client ID / Client Secret on the host do not match the Oura app exactly."
            )
        raise ValueError(f"Oura authorization failed with {response.status_code}: {detail}")
    payload = response.json()
    if not isinstance(payload, dict) or not str(payload.get("access_token") or "").strip():
        raise ValueError("Oura authorization did not return a usable access_token.")
    payload["_fetched_at"] = _utcnow().isoformat()
    return payload


def clear_oura_oauth_query_params() -> None:
    for key in ["code", "scope", "state", "error"]:
        if key in st.query_params:
            del st.query_params[key]


def redirect_browser(url: str) -> None:
    safe_url = html.escape(str(url), quote=True)
    st.markdown(f'<meta http-equiv="refresh" content="0;url={safe_url}">', unsafe_allow_html=True)
    st.info("Redirecting to Oura…")
    st.link_button("Open Oura manually", url)
    st.stop()


def begin_oura_oauth_flow(*, action: str, payload: Dict[str, Any]) -> None:
    nonce = secrets.token_urlsafe(24)
    st.session_state["pending_oura_oauth"] = persist_pending_oura_oauth(
        nonce=nonce,
        action=str(action or "").strip(),
        payload=payload,
    )
    redirect_browser(build_oura_authorize_url(state=nonce))


def handle_oura_oauth_callback(path: str) -> None:
    code = _query_param_value(st.query_params.get("code", "")).strip()
    error = _query_param_value(st.query_params.get("error", "")).strip()
    state = _query_param_value(st.query_params.get("state", "")).strip()
    if not code and not error:
        return
    if not browser_oura_oauth_enabled():
        clear_oura_oauth_query_params()
        return

    pending = claim_pending_oura_oauth(state) if state else None
    if pending is None:
        session_pending = st.session_state.get("pending_oura_oauth")
        if isinstance(session_pending, dict):
            session_nonce = str(session_pending.get("nonce") or "").strip()
            if (not state and session_nonce) or (state and session_nonce == state):
                pending = session_pending

    if not isinstance(pending, dict):
        st.session_state["oauth_flash_error"] = "Oura returned to the app, but there was no matching pending sign-in flow."
        clear_oura_oauth_query_params()
        return

    if state and str(pending.get("nonce") or "") != state:
        st.session_state["oauth_flash_error"] = "Oura sign-in state check failed. Please try again."
        st.session_state.pop("pending_oura_oauth", None)
        clear_oura_oauth_query_params()
        return

    if error:
        st.session_state["oauth_flash_error"] = "Oura access was not granted."
        clear_oura_oauth_query_params()
        return

    action = str(pending.get("action") or "").strip()
    payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
    try:
        token_bundle = exchange_oura_authorization_code(code)
        token_input = json.dumps(token_bundle)
        if action == "create_community":
            community, account, invite = create_community_account(
                path,
                community_name=str(payload.get("community_name") or ""),
                label=str(payload.get("label") or ""),
                token_input=token_input,
                share_enabled=_coerce_bool(payload.get("share_enabled")),
            )
            st.session_state["community_id"] = str(community.get("id") or "")
            st.session_state["community_member_id"] = str(account.get("id") or "")
            st.session_state["active_account_id"] = str(account.get("id") or "")
            st.session_state["latest_invite_code"] = str(invite.get("invite_code") or "")
            st.session_state["oauth_flash_message"] = "Community created and Oura connected."
        elif action == "join_community":
            community, account, _invite = join_community_account(
                path,
                invite_code=str(payload.get("invite_code") or ""),
                label=str(payload.get("label") or ""),
                token_input=token_input,
                share_enabled=_coerce_bool(payload.get("share_enabled")),
            )
            st.session_state["community_id"] = str(community.get("id") or "")
            st.session_state["community_member_id"] = str(account.get("id") or "")
            st.session_state["active_account_id"] = str(account.get("id") or "")
            st.session_state["oauth_flash_message"] = "Community joined and Oura connected."
        elif action == "reconnect_account":
            saved = upsert_connected_account(
                path,
                label=str(payload.get("label") or ""),
                token_input=token_input,
                account_id=str(payload.get("account_id") or "") or None,
                community_id=str(payload.get("community_id") or "") or None,
                share_enabled=_coerce_bool(payload.get("share_enabled")),
            )
            st.session_state["community_member_id"] = str(saved.get("id") or "")
            st.session_state["active_account_id"] = str(saved.get("id") or "")
            st.session_state["oauth_flash_message"] = "Oura connection updated."
        elif action == "connect_personal":
            saved = upsert_connected_account(
                path,
                label=str(payload.get("label") or ""),
                token_input=token_input,
                account_id=str(payload.get("account_id") or "") or None,
                community_id=None,
                share_enabled=False,
            )
            st.session_state["community_id"] = ""
            st.session_state["community_member_id"] = str(saved.get("id") or "")
            st.session_state["active_account_id"] = str(saved.get("id") or "")
            st.session_state["compare_account_ids"] = []
            st.session_state["oauth_flash_message"] = "Personal dashboard connected."
        else:
            raise ValueError("Unknown pending Oura sign-in action.")
    except Exception as exc:
        st.session_state["oauth_flash_error"] = str(exc)
    finally:
        st.session_state.pop("pending_oura_oauth", None)
        clear_oura_oauth_query_params()


def _account_storage_slug(account_id: Any) -> str:
    raw = str(account_id or "").strip()
    if not raw or raw == "__session__":
        return ""
    slug = _slugify(raw)
    if not slug:
        slug = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return slug[:64]


def _account_scoped_path(account_id: Any, filename: str) -> str:
    slug = _account_storage_slug(account_id)
    if not slug:
        raise ValueError("Account-scoped storage requires a stable account id.")
    return f"~/.oura_dashboard/accounts/{slug}/{filename}"


def behavior_event_path_for_account(account_id: Any) -> str:
    if not str(account_id or "").strip() or str(account_id) == "__session__":
        return BEHAVIOR_EVENT_PATH
    return _account_scoped_path(account_id, "events.csv")


def workout_intent_path_for_account(account_id: Any) -> str:
    if not str(account_id or "").strip() or str(account_id) == "__session__":
        return WORKOUT_INTENT_PATH
    return _account_scoped_path(account_id, "workout_intents.csv")


def _normalize_account_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    bundle = record.get("token_bundle")
    if not isinstance(bundle, dict):
        access_token = str(record.get("access_token") or "").strip()
        bundle = {"access_token": access_token} if access_token else {}
    pending_connection = _coerce_bool(record.get("pending_connection"))
    access_token = str(bundle.get("access_token") or "").strip()
    if not access_token and not pending_connection:
        return None

    profile = record.get("profile")
    if not isinstance(profile, dict):
        profile = {}

    fallback_label = _account_profile_name(profile) or str(record.get("email") or "").strip() or "Friend"
    label = str(record.get("label") or fallback_label).strip() or fallback_label
    signature = _token_signature(bundle) if access_token else "pending"
    account_id = str(record.get("id") or f"{_slugify(label)}-{signature}").strip()
    email = str(record.get("email") or profile.get("email") or "").strip()
    profile_name = str(record.get("profile_name") or _account_profile_name(profile)).strip()

    return {
        "id": account_id,
        "label": label,
        "profile_name": profile_name,
        "email": email,
        "profile": profile,
        "token_bundle": bundle,
        "community_id": str(record.get("community_id") or "").strip(),
        "share_enabled": _coerce_bool(record.get("share_enabled")),
        "pending_connection": pending_connection and not access_token,
        "created_at": str(record.get("created_at") or ""),
        "updated_at": str(record.get("updated_at") or ""),
        "last_refreshed_at": str(record.get("last_refreshed_at") or ""),
    }


def _normalize_community_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    community_name = str(record.get("name") or "").strip()
    invite_code = str(record.get("invite_code") or "").strip().upper()
    community_id = str(record.get("id") or f"community-{_slugify(community_name or invite_code)}").strip()
    if not community_name:
        return None
    return {
        "id": community_id,
        "name": community_name,
        "invite_code": invite_code,
        "owner_account_id": str(record.get("owner_account_id") or "").strip(),
        "created_at": str(record.get("created_at") or ""),
        "updated_at": str(record.get("updated_at") or ""),
    }


def _normalize_invitation_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    invite_code = str(record.get("invite_code") or "").strip().upper()
    community_id = str(record.get("community_id") or "").strip()
    inviter_account_id = str(record.get("inviter_account_id") or "").strip()
    if not invite_code or not community_id or not inviter_account_id:
        return None
    invitation_id = str(record.get("id") or f"invite-{community_id}-{invite_code.lower()}").strip()
    return {
        "id": invitation_id,
        "community_id": community_id,
        "inviter_account_id": inviter_account_id,
        "invite_code": invite_code,
        "claimed_by_account_id": str(record.get("claimed_by_account_id") or "").strip(),
        "claimed_at": str(record.get("claimed_at") or ""),
        "revoked_at": str(record.get("revoked_at") or ""),
        "created_at": str(record.get("created_at") or ""),
        "updated_at": str(record.get("updated_at") or ""),
    }


def load_account_store(path: str = ACCOUNT_STORE_PATH) -> Dict[str, Any]:
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        return {"version": ACCOUNT_STORE_VERSION, "accounts": [], "communities": [], "invitations": []}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"version": ACCOUNT_STORE_VERSION, "accounts": [], "communities": [], "invitations": []}

    raw_accounts = payload.get("accounts", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    out_accounts: List[Dict[str, Any]] = []
    for raw in raw_accounts:
        normalized = _normalize_account_record(raw)
        if normalized is not None:
            out_accounts.append(normalized)

    raw_communities = payload.get("communities", []) if isinstance(payload, dict) else []
    out_communities: List[Dict[str, Any]] = []
    for raw in raw_communities:
        normalized = _normalize_community_record(raw)
        if normalized is not None:
            out_communities.append(normalized)

    raw_invitations = payload.get("invitations", []) if isinstance(payload, dict) else []
    out_invitations: List[Dict[str, Any]] = []
    for raw in raw_invitations:
        normalized = _normalize_invitation_record(raw)
        if normalized is not None:
            out_invitations.append(normalized)

    return {
        "version": ACCOUNT_STORE_VERSION,
        "accounts": sorted(out_accounts, key=lambda account: str(account.get("label") or "").lower()),
        "communities": sorted(out_communities, key=lambda community: str(community.get("name") or "").lower()),
        "invitations": sorted(out_invitations, key=lambda invitation: str(invitation.get("created_at") or "")),
    }


def load_connected_accounts(path: str = ACCOUNT_STORE_PATH) -> List[Dict[str, Any]]:
    return list(load_account_store(path).get("accounts", []))


def load_communities(path: str = ACCOUNT_STORE_PATH) -> List[Dict[str, Any]]:
    return list(load_account_store(path).get("communities", []))


def load_invitations(path: str = ACCOUNT_STORE_PATH) -> List[Dict[str, Any]]:
    return list(load_account_store(path).get("invitations", []))


def save_connected_accounts(accounts: List[Dict[str, Any]], path: str = ACCOUNT_STORE_PATH) -> None:
    store = load_account_store(path)
    save_account_store(accounts=accounts, communities=store.get("communities", []), invitations=store.get("invitations", []), path=path)


def save_account_store(
    *,
    accounts: List[Dict[str, Any]],
    communities: List[Dict[str, Any]],
    invitations: Optional[List[Dict[str, Any]]] = None,
    path: str = ACCOUNT_STORE_PATH,
) -> None:
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized_accounts = [account for account in (_normalize_account_record(account) for account in accounts) if account is not None]
    normalized_communities = [community for community in (_normalize_community_record(community) for community in communities) if community is not None]
    if invitations is None:
        invitations = load_account_store(path).get("invitations", [])
    normalized_invitations = [invitation for invitation in (_normalize_invitation_record(invitation) for invitation in invitations) if invitation is not None]
    payload = {"version": ACCOUNT_STORE_VERSION, "accounts": normalized_accounts, "communities": normalized_communities, "invitations": normalized_invitations}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass


def _parse_token_bundle_input(raw: str) -> Dict[str, Any]:
    raw = str(raw or "").strip()
    if not raw:
        raise ValueError("Paste an access token or an oura_tokens.json payload.")
    if raw.startswith("{"):
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Token payload must be a JSON object.")
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            raise ValueError("Token payload is missing access_token.")
        payload.setdefault("_fetched_at", _utcnow().isoformat())
        return payload
    return {"access_token": raw, "_fetched_at": _utcnow().isoformat()}


def _probe_account_profile(access_token: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(f"{API_BASE}/usercollection/personal_info", headers=headers, timeout=25)
    if response.status_code == 200:
        try:
            payload = response.json()
        except Exception as exc:
            raise ValueError(f"Could not parse Oura personal_info JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Unexpected Oura personal_info payload shape.")
        return payload
    if response.status_code == 401:
        detail = response.text[:200] if response.text else response.reason
        raise ValueError(f"Oura token was rejected: {detail}")
    return {}


def upsert_connected_account(
    path: str,
    *,
    label: str,
    token_input: str,
    account_id: Optional[str] = None,
    community_id: Optional[str] = None,
    share_enabled: bool = False,
) -> Dict[str, Any]:
    bundle = _parse_token_bundle_input(token_input)
    profile = _probe_account_profile(str(bundle.get("access_token") or "").strip())
    email = str(profile.get("email") or "").strip()
    profile_name = _account_profile_name(profile)
    label_value = str(label or profile_name or email or "Friend").strip() or "Friend"
    signature = _token_signature(bundle)
    community_id_value = str(community_id or "").strip()
    account = _normalize_account_record(
        {
            "id": str(account_id or f"{_slugify(label_value)}-{signature}"),
            "label": label_value,
            "profile_name": profile_name,
            "email": email,
            "profile": profile,
            "token_bundle": bundle,
            "community_id": community_id_value,
            "share_enabled": share_enabled,
            "updated_at": _utcnow().isoformat(),
        }
    )
    if account is None:
        raise ValueError("Could not normalize the account record.")

    accounts = load_connected_accounts(path)
    replaced = False
    for idx, existing in enumerate(accounts):
        same_community = str(existing.get("community_id") or "") == community_id_value
        same_email = bool(email) and str(existing.get("email") or "").strip().lower() == email.lower() and (same_community or not community_id_value)
        same_label = same_community and str(existing.get("label") or "").strip().lower() == label_value.lower()
        if existing.get("id") == account["id"] or same_email or same_label:
            account["created_at"] = str(existing.get("created_at") or account.get("updated_at") or "")
            accounts[idx] = account
            replaced = True
            break
    if not replaced:
        account["created_at"] = account.get("updated_at") or _utcnow().isoformat()
        accounts.append(account)
    save_connected_accounts(accounts, path)
    return account


def upsert_placeholder_account(
    path: str,
    *,
    label: str,
    account_id: Optional[str] = None,
    community_id: Optional[str] = None,
    share_enabled: bool = False,
) -> Dict[str, Any]:
    community_id_value = str(community_id or "").strip()
    label_value = str(label or "Friend").strip() or "Friend"
    placeholder_id = str(account_id or f"{_slugify(label_value)}-pending-{int(time.time())}").strip()
    account = _normalize_account_record(
        {
            "id": placeholder_id,
            "label": label_value,
            "profile_name": "",
            "email": "",
            "profile": {},
            "token_bundle": {},
            "community_id": community_id_value,
            "share_enabled": share_enabled,
            "pending_connection": True,
            "updated_at": _utcnow().isoformat(),
        }
    )
    if account is None:
        raise ValueError("Could not create the pending community member.")

    accounts = load_connected_accounts(path)
    replaced = False
    for idx, existing in enumerate(accounts):
        same_community = str(existing.get("community_id") or "") == community_id_value
        same_label = same_community and str(existing.get("label") or "").strip().lower() == label_value.lower()
        if str(existing.get("id") or "") == placeholder_id or same_label:
            account["created_at"] = str(existing.get("created_at") or account.get("updated_at") or "")
            accounts[idx] = account
            replaced = True
            break
    if not replaced:
        account["created_at"] = account.get("updated_at") or _utcnow().isoformat()
        accounts.append(account)
    save_connected_accounts(accounts, path)
    return account


def assign_account_to_community(
    path: str,
    *,
    account_id: str,
    community_id: str,
    share_enabled: bool,
) -> Dict[str, Any]:
    store = load_account_store(path)
    accounts = list(store.get("accounts", []))
    updated: Optional[Dict[str, Any]] = None
    for idx, record in enumerate(accounts):
        if str(record.get("id") or "") != str(account_id or ""):
            continue
        current_community_id = str(record.get("community_id") or "").strip()
        if current_community_id and current_community_id != str(community_id or "").strip():
            raise ValueError("This account is already linked to a different community.")
        record["community_id"] = str(community_id or "").strip()
        record["share_enabled"] = bool(share_enabled)
        record["updated_at"] = _utcnow().isoformat()
        normalized = _normalize_account_record(record)
        if normalized is None:
            raise ValueError("Could not update the selected account.")
        accounts[idx] = normalized
        updated = normalized
        break
    if updated is None:
        raise ValueError("Selected personal account was not found.")
    save_account_store(
        accounts=accounts,
        communities=store.get("communities", []),
        invitations=store.get("invitations", []),
        path=path,
    )
    return updated


def create_community_for_existing_account(
    path: str,
    *,
    community_name: str,
    account_id: str,
    share_enabled: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    store = load_account_store(path)
    community_label = str(community_name or "").strip() or "Community"
    community_id = f"community-{_slugify(community_label)}-{int(time.time())}"
    account = assign_account_to_community(
        path,
        account_id=account_id,
        community_id=community_id,
        share_enabled=share_enabled,
    )
    community = {
        "id": community_id,
        "name": community_label,
        "invite_code": "",
        "owner_account_id": str(account.get("id") or ""),
        "created_at": _utcnow().isoformat(),
        "updated_at": _utcnow().isoformat(),
    }
    communities = list(store.get("communities", []))
    communities.append(community)
    accounts = load_connected_accounts(path)
    save_account_store(
        accounts=accounts,
        communities=communities,
        invitations=store.get("invitations", []),
        path=path,
    )
    invite = generate_member_invitation(path, community_id=community_id, inviter_account_id=str(account.get("id") or ""))
    return community, account, invite


def join_community_for_existing_account(
    path: str,
    *,
    invite_code: str,
    account_id: str,
    share_enabled: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    invitation = get_invitation_by_code(path, invite_code)
    if invitation is None or not _invitation_is_active(invitation):
        raise ValueError("Invite code not found.")
    community = get_community_by_id(path, str(invitation.get("community_id") or ""))
    if community is None:
        raise ValueError("Invitation points to an unknown community.")
    account = assign_account_to_community(
        path,
        account_id=account_id,
        community_id=str(community.get("id") or ""),
        share_enabled=share_enabled,
    )
    store = load_account_store(path)
    invitations = list(store.get("invitations", []))
    claimed: Optional[Dict[str, Any]] = None
    now = _utcnow().isoformat()
    for idx, row in enumerate(invitations):
        if str(row.get("id") or "") == str(invitation.get("id") or ""):
            row["claimed_by_account_id"] = str(account.get("id") or "")
            row["claimed_at"] = now
            row["updated_at"] = now
            invitations[idx] = row
            claimed = row
            break
    if claimed is None:
        raise ValueError("Could not claim invitation.")
    save_account_store(
        accounts=store.get("accounts", []),
        communities=store.get("communities", []),
        invitations=invitations,
        path=path,
    )
    return community, account, claimed


def delete_connected_account(path: str, account_id: str) -> None:
    accounts = [account for account in load_connected_accounts(path) if str(account.get("id")) != str(account_id)]
    store = load_account_store(path)
    communities = store.get("communities", [])
    invitations = list(store.get("invitations", []))
    for community in communities:
        if str(community.get("owner_account_id") or "") == str(account_id):
            community["owner_account_id"] = ""
            community["updated_at"] = _utcnow().isoformat()
    now = _utcnow().isoformat()
    for idx, invitation in enumerate(invitations):
        if str(invitation.get("inviter_account_id") or "") == str(account_id) and not str(invitation.get("revoked_at") or "").strip():
            invitation["revoked_at"] = now
            invitation["updated_at"] = now
            invitations[idx] = invitation
    save_account_store(accounts=accounts, communities=communities, invitations=invitations, path=path)


def _generate_invite_code(existing_codes: List[str]) -> str:
    seen = {str(code or "").upper() for code in existing_codes}
    for _ in range(128):
        candidate = "".join(secrets.choice(COMMUNITY_INVITE_ALPHABET) for _ in range(8))
        if candidate not in seen:
            return candidate
    raise RuntimeError("Could not generate a unique invite code.")


def get_community_by_id(path: str, community_id: str) -> Optional[Dict[str, Any]]:
    for community in load_communities(path):
        if str(community.get("id")) == str(community_id):
            return community
    return None


def get_community_by_invite_code(path: str, invite_code: str) -> Optional[Dict[str, Any]]:
    code = str(invite_code or "").strip().upper()
    for community in load_communities(path):
        if str(community.get("invite_code") or "").strip().upper() == code:
            return community
    return None


def _invitation_is_active(invitation: Dict[str, Any]) -> bool:
    if not invitation:
        return False
    if str(invitation.get("revoked_at") or "").strip():
        return False
    if str(invitation.get("claimed_at") or "").strip():
        return False
    if str(invitation.get("claimed_by_account_id") or "").strip():
        return False
    return True


def get_invitation_by_code(path: str, invite_code: str) -> Optional[Dict[str, Any]]:
    code = str(invite_code or "").strip().upper()
    for invitation in load_invitations(path):
        if str(invitation.get("invite_code") or "").strip().upper() == code:
            return invitation
    return None


def invitations_for_community(path: str, community_id: str, *, include_inactive: bool = False) -> List[Dict[str, Any]]:
    community_id_value = str(community_id or "").strip()
    rows = [
        invitation
        for invitation in load_invitations(path)
        if str(invitation.get("community_id") or "").strip() == community_id_value
    ]
    if include_inactive:
        return rows
    return [invitation for invitation in rows if _invitation_is_active(invitation)]


def invitations_for_member(path: str, community_id: str, inviter_account_id: str, *, include_inactive: bool = False) -> List[Dict[str, Any]]:
    inviter_id = str(inviter_account_id or "").strip()
    rows = [
        invitation
        for invitation in invitations_for_community(path, community_id, include_inactive=include_inactive)
        if str(invitation.get("inviter_account_id") or "").strip() == inviter_id
    ]
    if include_inactive:
        rows = [
            invitation
            for invitation in load_invitations(path)
            if str(invitation.get("community_id") or "").strip() == str(community_id or "").strip()
            and str(invitation.get("inviter_account_id") or "").strip() == inviter_id
        ]
    return rows


def generate_member_invitation(path: str, *, community_id: str, inviter_account_id: str) -> Dict[str, Any]:
    community_id_value = str(community_id or "").strip()
    inviter_id = str(inviter_account_id or "").strip()
    if not community_id_value or not inviter_id:
        raise ValueError("A community and inviter account are required.")
    store = load_account_store(path)
    invitations = list(store.get("invitations", []))
    invitation = {
        "id": f"invite-{community_id_value}-{int(time.time())}-{secrets.token_hex(2)}",
        "community_id": community_id_value,
        "inviter_account_id": inviter_id,
        "invite_code": _generate_invite_code([row.get("invite_code") for row in invitations]),
        "created_at": _utcnow().isoformat(),
        "updated_at": _utcnow().isoformat(),
    }
    invitations.append(invitation)
    save_account_store(accounts=store.get("accounts", []), communities=store.get("communities", []), invitations=invitations, path=path)
    normalized = _normalize_invitation_record(invitation)
    if normalized is None:
        raise ValueError("Could not persist invitation code.")
    return normalized


def accounts_for_community(path: str, community_id: str) -> List[Dict[str, Any]]:
    community_id_value = str(community_id or "").strip()
    if not community_id_value:
        return []
    return [
        account
        for account in load_connected_accounts(path)
        if str(account.get("community_id") or "").strip() == community_id_value
    ]


def update_connected_account_share_setting(path: str, account_id: str, share_enabled: bool) -> None:
    store = load_account_store(path)
    accounts = list(store.get("accounts", []))
    changed = False
    for account in accounts:
        if str(account.get("id")) == str(account_id):
            account["share_enabled"] = bool(share_enabled)
            account["updated_at"] = _utcnow().isoformat()
            changed = True
            break
    if changed:
        save_account_store(accounts=accounts, communities=store.get("communities", []), invitations=store.get("invitations", []), path=path)


def create_community_account(
    path: str,
    *,
    community_name: str,
    label: str,
    token_input: str,
    share_enabled: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    store = load_account_store(path)
    community_label = str(community_name or "").strip() or "Community"
    community_id = f"community-{_slugify(community_label)}-{int(time.time())}"
    community = {
        "id": community_id,
        "name": community_label,
        "invite_code": "",
        "owner_account_id": "",
        "created_at": _utcnow().isoformat(),
        "updated_at": _utcnow().isoformat(),
    }
    token_value = str(token_input or "").strip()
    if token_value:
        account = upsert_connected_account(
            path,
            label=label,
            token_input=token_value,
            community_id=community_id,
            share_enabled=share_enabled,
        )
    else:
        account = upsert_placeholder_account(
            path,
            label=label,
            community_id=community_id,
            share_enabled=share_enabled,
        )
    community["owner_account_id"] = str(account.get("id") or "")
    communities = list(store.get("communities", []))
    communities.append(community)
    accounts = load_connected_accounts(path)
    save_account_store(accounts=accounts, communities=communities, invitations=store.get("invitations", []), path=path)
    invite = generate_member_invitation(path, community_id=community_id, inviter_account_id=str(account.get("id") or ""))
    return community, account, invite


def join_community_account(
    path: str,
    *,
    invite_code: str,
    label: str,
    token_input: str,
    share_enabled: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    invitation = get_invitation_by_code(path, invite_code)
    if invitation is None or not _invitation_is_active(invitation):
        raise ValueError("Invite code not found.")
    community = get_community_by_id(path, str(invitation.get("community_id") or ""))
    if community is None:
        raise ValueError("Invitation points to an unknown community.")
    existing_accounts = accounts_for_community(path, str(community.get("id") or ""))
    existing_account = next(
        (
            account
            for account in existing_accounts
            if str(account.get("label") or "").strip().lower() == str(label or "").strip().lower()
        ),
        None,
    )
    token_value = str(token_input or "").strip()
    if token_value:
        account = upsert_connected_account(
            path,
            label=label,
            token_input=token_value,
            account_id=str(existing_account.get("id") or "") if existing_account is not None else None,
            community_id=str(community.get("id") or ""),
            share_enabled=share_enabled,
        )
    else:
        account = upsert_placeholder_account(
            path,
            label=label,
            account_id=str(existing_account.get("id") or "") if existing_account is not None else None,
            community_id=str(community.get("id") or ""),
            share_enabled=share_enabled,
        )
    store = load_account_store(path)
    invitations = list(store.get("invitations", []))
    claimed: Optional[Dict[str, Any]] = None
    now = _utcnow().isoformat()
    for idx, row in enumerate(invitations):
        if str(row.get("id") or "") == str(invitation.get("id") or ""):
            row["claimed_by_account_id"] = str(account.get("id") or "")
            row["claimed_at"] = now
            row["updated_at"] = now
            invitations[idx] = row
            claimed = row
            break
    if claimed is None:
        raise ValueError("Could not claim invitation.")
    save_account_store(accounts=store.get("accounts", []), communities=store.get("communities", []), invitations=invitations, path=path)
    return community, account, claimed


def _token_expires_soon(bundle: Dict[str, Any], *, skew_seconds: int = 86400) -> bool:
    fetched_at = _parse_dt(bundle.get("_fetched_at"))
    expires_in = _safe_float(bundle.get("expires_in"))
    if fetched_at is None or expires_in is None or expires_in <= 0:
        return False
    expires_at = fetched_at + timedelta(seconds=float(expires_in))
    return _utcnow() >= (expires_at - timedelta(seconds=skew_seconds))


def refresh_oura_token_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    refresh_token = str(bundle.get("refresh_token") or "").strip()
    if not refresh_token:
        raise ValueError("No refresh token available.")
    client_id = str(os.environ.get("OURA_CLIENT_ID") or "").strip()
    client_secret = str(os.environ.get("OURA_CLIENT_SECRET") or "").strip()
    if not client_id or not client_secret:
        raise ValueError("OURA_CLIENT_ID and OURA_CLIENT_SECRET are required to refresh stored accounts.")
    response = requests.post(
        OAUTH_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=40,
    )
    if response.status_code != 200:
        detail = response.text[:200] if response.text else response.reason
        raise ValueError(f"Oura token refresh failed with {response.status_code}: {detail}")
    payload = response.json()
    if not isinstance(payload, dict) or not str(payload.get("access_token") or "").strip():
        raise ValueError("Oura token refresh did not return a usable access_token.")
    merged = dict(bundle)
    merged.update(payload)
    merged["_fetched_at"] = _utcnow().isoformat()
    return merged


def resolve_connected_account(account: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str], bool]:
    normalized = _normalize_account_record(account)
    if normalized is None:
        raise ValueError("Invalid connected account record.")
    bundle = dict(normalized.get("token_bundle") or {})
    warning: Optional[str] = None
    changed = False
    if _token_expires_soon(bundle):
        try:
            bundle = refresh_oura_token_bundle(bundle)
            normalized["last_refreshed_at"] = _utcnow().isoformat()
            changed = True
        except Exception as exc:
            warning = f"{normalized['label']}: {exc}"
    access_token = str(bundle.get("access_token") or "").strip()
    if not access_token:
        raise ValueError(f"{normalized['label']}: missing access_token.")
    normalized["token_bundle"] = bundle
    normalized["updated_at"] = _utcnow().isoformat()
    return access_token, normalized, warning, changed


def time_in_zones(points: pd.DataFrame, zones: List[Tuple[str, float, float]]) -> pd.Series:
    """Time-weighted time-in-zone using sample deltas (seconds)."""
    if points.empty:
        return pd.Series(dtype=float)
    df = points.sort_values("ts_epoch").copy()
    df["dt"] = df["ts_epoch"].diff().fillna(0)
    df.loc[df["dt"] > 300, "dt"] = 0  # ignore huge gaps
    df["zone"] = df["bpm"].apply(lambda x: bucket_hr(float(x), zones))
    return df.groupby("zone")["dt"].sum()


def baseline_mean_sd(series: pd.Series, window: int) -> Tuple[Optional[float], Optional[float]]:
    """Return (mean, sd) for the last `window` non-null points.

    Intended for athlete baselines (7d/28d)."""
    if series is None or series.empty:
        return None, None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < max(5, min(window, 7)):
        return None, None
    s = s.tail(window)
    mean = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd == 0:
        sd = 0.0001
    return mean, sd


def row_datetime(row: pd.Series, keys: List[str]) -> Optional[datetime]:
    for key in keys:
        if key in row and pd.notna(row.get(key)):
            return _parse_dt(row.get(key))
    return None


def workout_window(row: pd.Series) -> Tuple[Optional[datetime], Optional[datetime]]:
    start_dt = row_datetime(row, ["start_datetime", "timestamp"])
    end_dt = row_datetime(row, ["end_datetime"])
    duration_min = _safe_float(row.get("duration_min"))
    if start_dt is not None and end_dt is None and duration_min is not None:
        end_dt = start_dt + timedelta(minutes=float(duration_min))
    return start_dt, end_dt


def target_bounds_for_zones(
    zones: List[Tuple[str, float, float]],
    target_zones: List[str],
) -> Optional[Tuple[int, int]]:
    bounds = [zone_bounds_by_name(zones, zname) for zname in target_zones]
    bounds = [bound for bound in bounds if bound is not None]
    if not bounds:
        return None
    lows = [bound[0] for bound in bounds]
    highs = [bound[1] for bound in bounds]
    return min(lows), max(highs)


def compliance_pct(zone_seconds: pd.Series, target_zones: List[str]) -> Optional[float]:
    if zone_seconds is None or zone_seconds.empty:
        return None
    total = float(zone_seconds.sum())
    if total <= 0:
        return None
    in_target = float(sum(float(zone_seconds.get(zone, 0) or 0) for zone in target_zones))
    return 100.0 * in_target / total


def compute_hr_drift(seg: pd.DataFrame) -> Optional[float]:
    if seg is None or seg.empty:
        return None
    try:
        seg2 = seg.sort_values("ts_epoch").copy()
        mid = seg2["ts_epoch"].iloc[0] + (seg2["ts_epoch"].iloc[-1] - seg2["ts_epoch"].iloc[0]) / 2.0
        first_half = float(seg2[seg2["ts_epoch"] <= mid]["bpm"].mean())
        second_half = float(seg2[seg2["ts_epoch"] > mid]["bpm"].mean())
        if first_half <= 0:
            return None
        return 100.0 * (second_half - first_half) / first_half
    except Exception:
        return None


def compute_banister_trimp(
    seg: pd.DataFrame,
    *,
    max_hr: int,
    resting_hr: int,
    sex: str = "M",
) -> Optional[float]:
    if seg is None or seg.empty:
        return None
    try:
        seg2 = seg.sort_values("ts_epoch").copy()
        seg2["dt"] = seg2["ts_epoch"].diff().fillna(0)
        seg2.loc[seg2["dt"] > 300, "dt"] = 0
        hrr = max(1.0, float(max_hr) - float(resting_hr))
        seg2["hrr_frac"] = ((seg2["bpm"] - float(resting_hr)) / hrr).clip(lower=0, upper=1)
        sex_u = str(sex or "M").upper()
        weight = 0.86 if sex_u.startswith("F") else 0.64
        exponent = 1.67 if sex_u.startswith("F") else 1.92
        seg2["trimp_part"] = (seg2["dt"] / 60.0) * seg2["hrr_frac"] * weight * seg2["hrr_frac"].apply(lambda x: math.exp(exponent * x))
        return float(seg2["trimp_part"].sum())
    except Exception:
        return None


def clock_minutes_from_anchor(dt: datetime, *, anchor_hour: int = 18) -> int:
    return ((dt.hour * 60 + dt.minute) - (anchor_hour * 60)) % (24 * 60)


def z_score(value: Optional[float], mean: Optional[float], sd: Optional[float]) -> Optional[float]:
    if value is None or mean is None or sd is None:
        return None
    try:
        return float((float(value) - float(mean)) / float(sd))
    except Exception:
        return None


def traffic_light_from_z(z: Optional[float], *, higher_is_better: bool = True) -> str:
    """Return GREEN/AMBER/RED using ±0.5 SD default thresholds."""
    if z is None:
        return "UNKNOWN"
    zz = z if higher_is_better else -z
    if zz >= 0.5:
        return "GREEN"
    if zz >= -0.5:
        return "AMBER"
    return "RED"


def chronotype_from_sleep_midpoint(avg_midpoint_hour: Optional[float]) -> str:
    if avg_midpoint_hour is None:
        return "UNKNOWN"
    # Heuristic thresholds (aligned with the doc you shared)
    if avg_midpoint_hour < 3.5:
        return "MORNING"
    if avg_midpoint_hour > 5.0:
        return "EVENING"
    return "INTERMEDIATE"


def letter_grade(score_0_100: Optional[float]) -> str:
    if score_0_100 is None:
        return "—"
    s = float(score_0_100)
    if s >= 93:
        return "A"
    if s >= 85:
        return "B"
    if s >= 75:
        return "C"
    if s >= 65:
        return "D"
    return "F"


def load_intent_labels(path: str) -> pd.DataFrame:
    try:
        p = pathlib.Path(path).expanduser()
        legacy = pathlib.Path(LEGACY_WORKOUT_INTENT_PATH).expanduser()
        source = p if p.exists() else legacy if p == pathlib.Path(WORKOUT_INTENT_PATH).expanduser() and legacy.exists() else None
        if source is None:
            return pd.DataFrame(columns=["workout_key", "workout_intent", "updated_at"])
        df = pd.read_csv(source)
        for c in ["workout_key", "workout_intent", "updated_at"]:
            if c not in df.columns:
                df[c] = None
        normalized = df[["workout_key", "workout_intent", "updated_at"]]
        if source != p:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                normalized.to_csv(p, index=False)
            except Exception as exc:
                record_debug_event("Failed to migrate legacy workout intents", exc=exc)
        return normalized
    except Exception:
        return pd.DataFrame(columns=["workout_key", "workout_intent", "updated_at"])


def save_intent_label(path: str, workout_key: str, intent: str) -> None:
    try:
        p = pathlib.Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        df = load_intent_labels(str(p))
        now = _utcnow().isoformat()
        df = df[df["workout_key"].astype(str) != str(workout_key)].copy()
        df = pd.concat(
            [df, pd.DataFrame([{ "workout_key": str(workout_key), "workout_intent": str(intent), "updated_at": now }])],
            ignore_index=True,
        )
        df.to_csv(p, index=False)
    except Exception:
        return


def workout_key_from_row(row: pd.Series) -> str:
    start_dt = row.get("start_datetime") if "start_datetime" in row else None
    if start_dt is not None and pd.notna(start_dt):
        return str(start_dt)
    day = str(row.get("day") or "")
    workout_type = str(row.get("type") or "workout")
    duration_min = _safe_float(row.get("duration_min"))
    duration_text = f"{duration_min:.0f}" if duration_min is not None else ""
    return f"{day}|{workout_type}|{duration_text}"


def attach_workout_intents(workouts: pd.DataFrame, path: str = WORKOUT_INTENT_PATH) -> pd.DataFrame:
    if workouts is None or workouts.empty:
        return pd.DataFrame()
    out = workouts.copy()
    out["workout_key"] = out.apply(workout_key_from_row, axis=1)
    intents = load_intent_labels(path)
    if intents.empty:
        out["workout_intent"] = out.get("workout_intent", "")
        return out
    label_map = dict(zip(intents["workout_key"].astype(str), intents["workout_intent"].astype(str)))
    out["workout_intent"] = out["workout_key"].astype(str).map(label_map).fillna(out.get("workout_intent", "")).fillna("")
    return out


def _empty_event_records() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "day",
            "alcohol",
            "late_meal",
            "travel",
            "illness",
            "supplement",
            "sauna",
            "cold",
            "caffeine_late",
            "manual_wellness",
            "notes",
            "updated_at",
        ]
    )


def _normalize_event_records(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_event_records()

    # Migrate the previous one-event-per-row logger shape if it exists.
    if "event_type" in df.columns:
        rows: List[Dict[str, Any]] = []
        for day, group in df.groupby(df.get("day").astype(str)):
            record = EventRecord(day=str(day))
            payload = {
                "day": record.day,
                "alcohol": False,
                "late_meal": False,
                "travel": False,
                "illness": False,
                "supplement": "",
                "sauna": False,
                "cold": False,
                "caffeine_late": False,
                "manual_wellness": None,
                "notes": "",
                "updated_at": str(group.get("created_at", pd.Series(dtype=object)).dropna().astype(str).max() if "created_at" in group.columns else ""),
            }
            notes: List[str] = []
            supplements: List[str] = []
            for _, row in group.iterrows():
                event_type = str(row.get("event_type") or "").strip().lower()
                note_value = row.get("note")
                note = "" if note_value is None or (isinstance(note_value, float) and math.isnan(note_value)) else str(note_value).strip()
                if event_type == "alcohol":
                    payload["alcohol"] = True
                elif event_type == "late meal":
                    payload["late_meal"] = True
                elif event_type == "travel":
                    payload["travel"] = True
                elif event_type == "illness":
                    payload["illness"] = True
                elif event_type == "sauna":
                    payload["sauna"] = True
                elif event_type == "cold exposure":
                    payload["cold"] = True
                elif event_type == "late caffeine":
                    payload["caffeine_late"] = True
                elif event_type == "supplement" and note:
                    supplements.append(note)
                if note:
                    notes.append(note)
            if supplements:
                payload["supplement"] = "; ".join(sorted(set(supplements)))
            if notes:
                payload["notes"] = " | ".join(notes)
            rows.append(payload)
        return pd.DataFrame(rows) if rows else _empty_event_records()

    normalized = df.copy()
    expected = _empty_event_records().columns.tolist()
    for col in expected:
        if col not in normalized.columns:
            normalized[col] = None
    for bool_col in ["alcohol", "late_meal", "travel", "illness", "sauna", "cold", "caffeine_late"]:
        normalized[bool_col] = normalized[bool_col].apply(bool_from_value).fillna(False).astype(bool)
    normalized["day"] = normalized["day"].astype(str)
    normalized["supplement"] = normalized["supplement"].fillna("").astype(str)
    normalized["notes"] = normalized["notes"].fillna("").astype(str)
    normalized["manual_wellness"] = pd.to_numeric(normalized["manual_wellness"], errors="coerce")
    return normalized[expected].sort_values("day").reset_index(drop=True)


def load_behavior_events(path: str) -> pd.DataFrame:
    try:
        p = pathlib.Path(path).expanduser()
        legacy = pathlib.Path(LEGACY_BEHAVIOR_EVENT_PATH).expanduser()
        source = p if p.exists() else legacy if p == pathlib.Path(BEHAVIOR_EVENT_PATH).expanduser() and legacy.exists() else None
        if source is None:
            return _empty_event_records()
        df = pd.read_csv(source)
        normalized = _normalize_event_records(df)
        if source != p:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                normalized.to_csv(p, index=False)
            except Exception as exc:
                record_debug_event("Failed to migrate legacy behavior events", exc=exc)
        return normalized
    except Exception as exc:
        record_debug_event("Failed to load behavior events", exc=exc)
        return _empty_event_records()


def save_event_record(path: str, record: EventRecord) -> None:
    try:
        p = pathlib.Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        df = load_behavior_events(str(p))
        payload = {
            "day": str(record.day),
            "alcohol": bool(record.alcohol),
            "late_meal": bool(record.late_meal),
            "travel": bool(record.travel),
            "illness": bool(record.illness),
            "supplement": str(record.supplement or ""),
            "sauna": bool(record.sauna),
            "cold": bool(record.cold),
            "caffeine_late": bool(record.caffeine_late),
            "manual_wellness": record.manual_wellness,
            "notes": str(record.notes or ""),
            "updated_at": _utcnow().isoformat(),
        }
        df = df[df["day"].astype(str) != str(record.day)].copy()
        df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
        df = _normalize_event_records(df)
        df.to_csv(p, index=False)
    except Exception as exc:
        record_debug_event("Failed to save event record", exc=exc)


def delete_event_record(path: str, day: str) -> None:
    try:
        p = pathlib.Path(path).expanduser()
        df = load_behavior_events(str(p))
        if df.empty:
            return
        df = df[df["day"].astype(str) != str(day)].copy()
        df.to_csv(p, index=False)
    except Exception as exc:
        record_debug_event("Failed to delete event record", exc=exc)


def behavior_events_to_tag_rows(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=["day", "text", "comment", "tag_type", "source"])
    rows: List[Dict[str, Any]] = []
    flag_map = [
        ("alcohol", "Alcohol"),
        ("late_meal", "Late meal"),
        ("travel", "Travel"),
        ("illness", "Illness"),
        ("sauna", "Sauna"),
        ("cold", "Cold exposure"),
        ("caffeine_late", "Late caffeine"),
    ]
    for _, row in events_df.iterrows():
        day = str(row.get("day") or "")
        if not day:
            continue
        notes_value = row.get("notes")
        notes = "" if notes_value is None or (isinstance(notes_value, float) and math.isnan(notes_value)) else str(notes_value).strip()
        for col, label in flag_map:
            if bool_from_value(row.get(col)):
                rows.append({"day": day, "text": label, "comment": notes, "tag_type": "Manual event", "source": "Built-in logger"})
        supplement_value = row.get("supplement")
        supplement = "" if supplement_value is None or (isinstance(supplement_value, float) and math.isnan(supplement_value)) else str(supplement_value).strip()
        if supplement:
            rows.append({"day": day, "text": f"Supplement: {supplement}", "comment": notes, "tag_type": "Manual event", "source": "Built-in logger"})
        wellness = _safe_float(row.get("manual_wellness"))
        if wellness is not None:
            rows.append({"day": day, "text": f"Manual wellness {int(round(wellness))}/10", "comment": notes, "tag_type": "Manual event", "source": "Built-in logger"})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["day", "text", "comment", "tag_type", "source"])


def compute_workout_metrics(seg: pd.DataFrame) -> Dict[str, Any]:
    """Compute avg_hr, max_hr, drift% for an HR segment."""
    if seg is None or seg.empty:
        return {"avg_hr": None, "max_hr": None, "drift": None}
    out: Dict[str, Any] = {}
    out["avg_hr"] = float(seg["bpm"].mean())
    out["max_hr"] = float(seg["bpm"].max())
    out["drift"] = compute_hr_drift(seg)
    return out


def _compute_personalization_models_impl(
    workouts: pd.DataFrame,
    hr_points: pd.DataFrame,
    max_hr: int,
    resting_hr: int,
) -> Dict[str, Any]:
    """Personalization computed from the user's own history.

    Outputs:
    - z2_cap_bpm: estimated personal aerobic cap (based on low-drift aerobic sessions)
    - intent_baselines: per-intent baselines for avg_hr and drift (median + IQR)
    """
    zones = hr_zones_karvonen(max_hr=int(max_hr), resting_hr=int(resting_hr))
    z2_bounds = zone_bounds_by_name(zones, "Z2")
    z2_hi = z2_bounds[1] if z2_bounds else None

    # Use last 60 days of workouts for learning
    w = workouts.copy()
    if "day" in w.columns:
        w["day_dt"] = pd.to_datetime(w["day"], errors="coerce")
        w = w.dropna(subset=["day_dt"]).sort_values("day_dt")
        w = w[w["day_dt"] >= (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=60))]

    rows: List[dict] = []
    for _, row in w.iterrows():
        sdt, edt = workout_window(row)
        if sdt is None or edt is None:
            continue
        seg = hr_points[(hr_points["ts"] >= sdt) & (hr_points["ts"] <= edt)].copy() if not hr_points.empty else pd.DataFrame()
        if len(seg) < 20:
            continue
        m = compute_workout_metrics(seg)
        rows.append({
            "day": str(row.get("day")),
            "intent": str(row.get("workout_intent") or "").strip(),
            "type": str(row.get("type") or "workout"),
            "duration_min": _safe_float(row.get("duration_min")),
            "avg_hr": m.get("avg_hr"),
            "drift": m.get("drift"),
        })

    if not rows:
        return {"z2_cap_bpm": z2_hi, "intent_baselines": {}}

    df = pd.DataFrame(rows)

    # Personal Z2 cap: look at aerobic sessions and learn HR where drift stays low.
    aero = df[df["intent"].isin(["Aerobic base (Z2)", "Longevity / healthspan (Z1–Z2)", "Easy aerobic (Z1–Z2)"])].copy()
    aero = aero.dropna(subset=["avg_hr", "drift", "duration_min"])
    aero = aero[aero["duration_min"] >= 25]

    z2_cap = z2_hi
    if len(aero) >= 6:
        # Keep "good" executions: lowest-drift half
        aero = aero.sort_values("drift")
        good = aero.head(max(3, int(len(aero) * 0.5)))
        # Cap = 75th percentile of avg_hr in good aerobic sessions
        try:
            z2_cap = float(good["avg_hr"].quantile(0.75))
        except Exception:
            z2_cap = z2_hi

    # Per-intent baselines
    intent_baselines: Dict[str, Any] = {}
    for intent, g in df.groupby("intent"):
        intent = str(intent).strip() or "(unlabeled)"
        g2 = g.dropna(subset=["avg_hr"]).copy()
        if len(g2) < 4:
            continue
        avg_med = float(g2["avg_hr"].median())
        avg_iqr = float(g2["avg_hr"].quantile(0.75) - g2["avg_hr"].quantile(0.25))
        drift_med = None
        drift_iqr = None
        gd = g.dropna(subset=["drift"]).copy()
        if len(gd) >= 4:
            drift_med = float(gd["drift"].median())
            drift_iqr = float(gd["drift"].quantile(0.75) - gd["drift"].quantile(0.25))
        intent_baselines[intent] = {
            "avg_hr_median": avg_med,
            "avg_hr_iqr": avg_iqr,
            "drift_median": drift_med,
            "drift_iqr": drift_iqr,
            "n": int(len(g)),
        }

    return {"z2_cap_bpm": z2_cap, "intent_baselines": intent_baselines}


def personalization_cache_key(
    workouts: pd.DataFrame,
    hr_points: pd.DataFrame,
    max_hr: int,
    resting_hr: int,
) -> Tuple[Any, ...]:
    workout_last_day = None
    workout_last_start = None
    workout_duration_sum = None
    if workouts is not None and not workouts.empty:
        if "day" in workouts.columns:
            workout_last_day = str(pd.to_datetime(workouts["day"], errors="coerce", utc=True).max())
        if "start_datetime" in workouts.columns:
            workout_last_start = str(pd.to_datetime(workouts["start_datetime"], errors="coerce", utc=True).max())
        if "duration_min" in workouts.columns:
            workout_duration_sum = round(float(pd.to_numeric(workouts["duration_min"], errors="coerce").tail(20).fillna(0).sum()), 2)

    hr_min_ts = None
    hr_max_ts = None
    hr_bpm_mean = None
    if hr_points is not None and not hr_points.empty:
        if "ts_epoch" in hr_points.columns:
            hr_min_ts = round(float(pd.to_numeric(hr_points["ts_epoch"], errors="coerce").min()), 0)
            hr_max_ts = round(float(pd.to_numeric(hr_points["ts_epoch"], errors="coerce").max()), 0)
        if "bpm" in hr_points.columns:
            hr_bpm_mean = round(float(pd.to_numeric(hr_points["bpm"], errors="coerce").tail(200).mean()), 2)

    return (
        int(max_hr),
        int(resting_hr),
        0 if workouts is None else int(len(workouts)),
        workout_last_day,
        workout_last_start,
        workout_duration_sum,
        0 if hr_points is None else int(len(hr_points)),
        hr_min_ts,
        hr_max_ts,
        hr_bpm_mean,
    )


@st.cache_resource(show_spinner=False)
def _compute_personalization_models_cached(
    cache_key: Tuple[Any, ...],
    _workouts: pd.DataFrame,
    _hr_points: pd.DataFrame,
    max_hr: int,
    resting_hr: int,
) -> Dict[str, Any]:
    return _compute_personalization_models_impl(_workouts, _hr_points, max_hr, resting_hr)


def compute_personalization_models(
    workouts: pd.DataFrame,
    hr_points: pd.DataFrame,
    max_hr: int,
    resting_hr: int,
) -> Dict[str, Any]:
    cache_key = personalization_cache_key(workouts, hr_points, max_hr, resting_hr)
    return _compute_personalization_models_cached(
        cache_key,
        _workouts=workouts,
        _hr_points=hr_points,
        max_hr=max_hr,
        resting_hr=resting_hr,
    )


def ewma(series: pd.Series, span_days: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.ewm(span=span_days, adjust=False, min_periods=max(3, int(span_days / 3))).mean()


def vo2_benchmarks(age: Optional[int], sex: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (top10, top1) VO2max thresholds by rough age/sex bands.

    These are heuristic approximations based on commonly cited population norms.
    We use them to provide actionable ranking, not as medical diagnosis.
    """
    if age is None:
        age = 35
    s = (sex or "M").upper().strip()
    if s not in {"M", "F"}:
        s = "M"

    # Bands: <30, 30-39, 40-49, 50-59, 60+
    if s == "M":
        if age < 30:
            return 52.0, 60.0
        if age < 40:
            return 49.0, 57.0
        if age < 50:
            return 46.0, 54.0
        if age < 60:
            return 42.0, 50.0
        return 38.0, 46.0
    else:
        if age < 30:
            return 45.0, 52.0
        if age < 40:
            return 42.0, 49.0
        if age < 50:
            return 39.0, 46.0
        if age < 60:
            return 36.0, 43.0
        return 32.0, 39.0


def rhr_benchmarks(sex: str) -> Tuple[int, int]:
    """(top10, top1) lower is better. Heuristic athlete-oriented cutoffs."""
    s = (sex or "M").upper().strip()
    if s == "F":
        return 55, 45
    return 52, 42


def hrv_benchmarks_rmssd(age: Optional[int], sex: str) -> Tuple[Optional[float], Optional[float]]:
    """(top10, top1) rmSSD in ms.

    HRV percentiles vary by device and method; these are coarse heuristics.
    For athletes, "vs your baseline" is usually more meaningful.
    """
    if age is None:
        age = 35
    s = (sex or "M").upper().strip()
    # Very rough: younger tends higher.
    if age < 30:
        return 85.0 if s == "M" else 75.0, 120.0 if s == "M" else 105.0
    if age < 40:
        return 75.0 if s == "M" else 65.0, 105.0 if s == "M" else 90.0
    if age < 50:
        return 65.0 if s == "M" else 55.0, 90.0 if s == "M" else 80.0
    if age < 60:
        return 55.0 if s == "M" else 45.0, 80.0 if s == "M" else 70.0
    return 45.0 if s == "M" else 40.0, 65.0 if s == "M" else 60.0


def sleep_benchmarks() -> Dict[str, Any]:
    """Heuristic sleep benchmarks aimed at athletes."""
    return {
        "sleep_hours": (8.0, 9.0),  # top10, top1
        "sleep_eff": (90.0, 95.0),
    }


def spo2_benchmarks() -> Tuple[float, float]:
    return 98.0, 99.0


def temp_dev_benchmarks() -> Tuple[float, float]:
    # absolute deviation (C), lower is better
    return 0.2, 0.1


def score_benchmarks_generic() -> Tuple[int, int]:
    # readiness/sleep/activity scores 0-100
    return 85, 95


def _recent_metric_frame(daily: pd.DataFrame, key: str, window_days: int) -> pd.DataFrame:
    if daily.empty or key not in daily.columns:
        return pd.DataFrame(columns=["day_dt", key])
    df = daily[["day", key]].copy()
    df["day_dt"] = pd.to_datetime(df["day"], errors="coerce")
    df[key] = pd.to_numeric(df[key], errors="coerce")
    df = df.dropna(subset=["day_dt", key]).sort_values("day_dt")
    if df.empty:
        return pd.DataFrame(columns=["day_dt", key])
    cutoff = df["day_dt"].max() - pd.Timedelta(days=max(window_days - 1, 0))
    return df[df["day_dt"] >= cutoff].copy()


def _linregress_days(df: pd.DataFrame, value_col: str) -> Dict[str, Optional[float]]:
    if df.empty or len(df) < 2 or value_col not in df.columns:
        return {"slope": None, "intercept": None, "r": None, "p": None, "stderr": None}
    working = df.dropna(subset=["day_dt", value_col]).copy()
    if len(working) < 2:
        return {"slope": None, "intercept": None, "r": None, "p": None, "stderr": None}
    working["day_num"] = (working["day_dt"] - working["day_dt"].iloc[0]).dt.days.astype(float)
    if working["day_num"].nunique() < 2:
        return {"slope": None, "intercept": None, "r": None, "p": None, "stderr": None}

    x = working["day_num"]
    y = working[value_col]
    if scipy_stats is not None:
        slope, intercept, r_value, p_value, stderr = scipy_stats.linregress(x, y)
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r": float(r_value),
            "p": float(p_value),
            "stderr": float(stderr),
        }

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom <= 0:
        return {"slope": None, "intercept": None, "r": None, "p": None, "stderr": None}
    slope = float((((x - x_mean) * (y - y_mean)).sum()) / denom)
    intercept = y_mean - slope * x_mean
    y_hat = intercept + slope * x
    ss_tot = float(((y - y_mean) ** 2).sum())
    ss_res = float(((y - y_hat) ** 2).sum())
    r_sq = None if ss_tot <= 0 else max(0.0, 1.0 - (ss_res / ss_tot))
    r_value = math.sqrt(r_sq) if r_sq is not None else None
    return {"slope": slope, "intercept": intercept, "r": r_value, "p": None, "stderr": None}


def _interpret_rhr_slope(slope_per_month: Optional[float]) -> str:
    if slope_per_month is None:
        return "Not enough data."
    if slope_per_month <= -0.5:
        return "Strong improvement. Cardiovascular efficiency is measurably increasing."
    if slope_per_month <= -0.1:
        return "Gradual improvement. Consistent aerobic work and sleep quality are likely helping."
    if slope_per_month <= 0.1:
        return "Flat. You are maintaining, not improving. Add aerobic volume or tighten sleep consistency."
    if slope_per_month <= 0.5:
        return "Drifting up. Check cumulative fatigue, sleep debt, dehydration, or stress."
    return "Rising fast. Treat this as a red flag after ruling out illness."


def rhr_trend_slope(daily: pd.DataFrame, window_days: int = 90) -> Dict[str, Any]:
    df = _recent_metric_frame(daily, "resting_hr", window_days)
    if len(df) < 21:
        return {"error": "Need 21+ days for trend analysis"}

    reg = _linregress_days(df, "resting_hr")
    slope = reg.get("slope")
    if slope is None:
        return {"error": "Unable to compute RHR trend"}

    current = float(df["resting_hr"].iloc[-1])
    slope_per_month = float(slope) * 30.0
    projected_12mo = current + float(slope) * 365.0
    target = float(rhr_benchmarks("M")[0])
    years_to_target = None
    if slope < 0 and current > target:
        days_to_target = (target - current) / float(slope)
        if days_to_target > 0:
            years_to_target = days_to_target / 365.0

    r_value = reg.get("r")
    return {
        "slope_bpm_per_month": round(slope_per_month, 2),
        "r_squared": round(float(r_value) ** 2, 3) if r_value is not None else None,
        "current": round(current, 1),
        "projected_12mo": round(projected_12mo, 1),
        "years_to_target_50bpm": round(years_to_target, 1) if years_to_target is not None else None,
        "years_to_top10_band": round(years_to_target, 1) if years_to_target is not None else None,
        "target_low_rhr_band": round(target, 0),
        "p_value": round(reg["p"], 4) if reg.get("p") is not None else None,
        "interpretation": _interpret_rhr_slope(slope_per_month),
    }


def rhr_stability(daily: pd.DataFrame, window: int = 28) -> Dict[str, Any]:
    if "resting_hr" not in daily.columns:
        return {}
    s = pd.to_numeric(daily["resting_hr"], errors="coerce").dropna().tail(window)
    if len(s) < 14:
        return {}
    mean_val = float(s.mean())
    std_val = float(s.std())
    cv = (std_val / mean_val) * 100 if mean_val > 0 else None
    interpretation = "Insufficient data"
    if cv is not None:
        if cv < 5:
            interpretation = "Stable (<5% CV) — strong autonomic regulation."
        elif cv < 8:
            interpretation = "Moderate variability (5-8%) — some room to improve consistency."
        else:
            interpretation = "High variability (>8%) — sleep timing, hydration, or training consistency is the likely lever."
    return {
        "cv_percent": round(cv, 1) if cv is not None else None,
        "mean_rhr": round(mean_val, 1),
        "interpretation": interpretation,
    }


def rhr_recovery_rate(daily: pd.DataFrame, workouts: pd.DataFrame) -> Dict[str, Any]:
    if daily.empty or workouts.empty or "resting_hr" not in daily.columns or "day" not in workouts.columns:
        return {}
    df = daily[["day", "resting_hr"]].copy()
    df["resting_hr"] = pd.to_numeric(df["resting_hr"], errors="coerce")
    df = df.dropna(subset=["resting_hr"]).sort_values("day")
    if df.empty:
        return {}
    df["day"] = df["day"].astype(str)
    training_days = set(workouts["day"].astype(str).dropna().unique())
    recoveries: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        day_str = str(row["day"])
        if day_str not in training_days:
            continue
        baseline_series = df[df["day"] < day_str]["resting_hr"].tail(28)
        if len(baseline_series) < 7:
            continue
        baseline = float(baseline_series.median())
        rhr_train = float(row["resting_hr"])
        elevation = rhr_train - baseline
        if elevation <= 0:
            continue
        day_dt = pd.to_datetime(day_str, errors="coerce")
        if pd.isna(day_dt):
            continue
        d1 = (day_dt + pd.Timedelta(days=1)).date().isoformat()
        rhr_d1 = pd.to_numeric(df.loc[df["day"] == d1, "resting_hr"], errors="coerce").dropna()
        if rhr_d1.empty:
            continue
        recovery_d1 = rhr_train - float(rhr_d1.iloc[0])
        pct_d1 = (recovery_d1 / elevation * 100.0) if elevation > 0 else None
        recoveries.append({"elevation": elevation, "recovery_d1": recovery_d1, "pct_recovered_d1": pct_d1})

    if len(recoveries) < 5:
        return {"error": "Need 5+ training days with next-day RHR data"}

    avg_pct = pd.Series([r["pct_recovered_d1"] for r in recoveries if r.get("pct_recovered_d1") is not None]).dropna()
    avg_pct_val = float(avg_pct.mean()) if not avg_pct.empty else None
    if avg_pct_val is None:
        return {"error": "Insufficient recovery observations"}
    interpretation = (
        "Fast recovery (>80% by day 1) — strong parasympathetic tone."
        if avg_pct_val > 80
        else "Normal recovery (50-80%) — room to improve with aerobic volume and better sleep."
        if avg_pct_val > 50
        else "Slow recovery (<50%) — likely overreaching or under-sleeping."
    )
    return {
        "avg_pct_recovered_day1": round(avg_pct_val, 0),
        "n_sessions": len(recoveries),
        "interpretation": interpretation,
    }


def hrv_biological_age_estimate(
    daily: pd.DataFrame,
    chronological_age: Optional[int],
    sex: str = "M",
) -> Dict[str, Any]:
    if "hrv_rmssd" not in daily.columns or chronological_age is None:
        return {}
    s = pd.to_numeric(daily["hrv_rmssd"], errors="coerce").dropna()
    if len(s) < 14:
        return {}
    median_hrv = float(s.tail(28).median())
    norms = {
        "M": {25: 55, 30: 50, 35: 45, 40: 40, 45: 36, 50: 33, 55: 30, 60: 27, 65: 25, 70: 22},
        "F": {25: 50, 30: 45, 35: 41, 40: 37, 45: 33, 50: 30, 55: 27, 60: 25, 65: 23, 70: 20},
    }
    sex_norms = norms.get(str(sex).upper(), norms["M"])
    ages = sorted(sex_norms.keys())
    hrvs = [sex_norms[a] for a in ages]
    bio_age = None
    for idx in range(len(ages) - 1):
        if hrvs[idx] >= median_hrv >= hrvs[idx + 1]:
            denom = (hrvs[idx] - hrvs[idx + 1])
            frac = (hrvs[idx] - median_hrv) / denom if denom else 0.0
            bio_age = ages[idx] + frac * (ages[idx + 1] - ages[idx])
            break
    if bio_age is None:
        bio_age = ages[0] - 5 if median_hrv > hrvs[0] else ages[-1] + 5
    delta = float(chronological_age) - float(bio_age)
    interpretation = (
        f"Your HRV sits in a range that is roughly {abs(delta):.0f} years younger than age-matched norms. Treat this as a rough age-referenced estimate, not a clinical biological-age measure."
        if delta > 3
        else "Your HRV is roughly age-appropriate relative to reference norms. Consistency is the main lever now."
        if abs(delta) <= 3
        else f"Your HRV sits in a range that is roughly {abs(delta):.0f} years older than age-matched norms. Prioritize sleep regularity, lower hidden stress, and aerobic base work."
    )
    return {
        "median_hrv_28d": round(median_hrv, 1),
        "hrv_biological_age": round(float(bio_age), 0),
        "chronological_age": chronological_age,
        "delta_years": round(delta, 1),
        "interpretation": interpretation,
    }


def _interpret_hrv_pattern(slope: Optional[float], cv: Optional[float], vol_improving: bool) -> str:
    parts: List[str] = []
    if slope is None:
        parts.append("Trend unavailable.")
    elif slope > 0.3:
        parts.append("HRV is trending up — autonomic health is improving.")
    elif slope > -0.3:
        parts.append("HRV is flat — maintaining, not gaining.")
    else:
        parts.append("HRV is trending down — investigate sleep, stress, illness, or excess load.")

    if cv is not None:
        if cv < 12:
            parts.append("Day-to-day variability is low, which is good.")
        elif cv < 20:
            parts.append("Moderate variability — tighten sleep timing and hydration.")
        else:
            parts.append("High variability — recovery inputs are inconsistent.")
    if vol_improving:
        parts.append("Residual noise is shrinking over time, which is a positive sign.")
    return " ".join(parts)


def hrv_pattern_analysis(daily: pd.DataFrame, window: int = 60) -> Dict[str, Any]:
    df = _recent_metric_frame(daily, "hrv_rmssd", window)
    if len(df) < 21:
        return {"error": "Need 21+ days for pattern analysis"}
    df = df.rename(columns={"hrv_rmssd": "hrv"}).copy()
    roll_mean = df["hrv"].rolling(7, min_periods=5).mean()
    roll_std = df["hrv"].rolling(7, min_periods=5).std()
    cv_series = (roll_std / roll_mean * 100.0)
    current_cv = float(cv_series.dropna().iloc[-1]) if cv_series.dropna().any() else None

    reg = _linregress_days(df.rename(columns={"hrv": "hrv_value"}), "hrv_value")
    slope = reg.get("slope")
    intercept = reg.get("intercept")
    if slope is None or intercept is None:
        return {"error": "Unable to compute HRV pattern"}
    slope_per_month = float(slope) * 30.0

    df["day_num"] = (df["day_dt"] - df["day_dt"].iloc[0]).dt.days.astype(float)
    df["trend_line"] = float(intercept) + float(slope) * df["day_num"]
    df["residual"] = df["hrv"] - df["trend_line"]
    residual_std = float(df["residual"].std())
    half = max(1, len(df) // 2)
    vol_first = float(df["residual"].iloc[:half].std()) if len(df.iloc[:half]) > 1 else residual_std
    vol_second = float(df["residual"].iloc[half:].std()) if len(df.iloc[half:]) > 1 else residual_std
    vol_improving = bool(vol_second < vol_first * 0.9) if vol_first > 0 else False

    df["dow"] = df["day_dt"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly = df.groupby("dow")["hrv"].mean().reindex([d for d in dow_order if d in df["dow"].unique()])
    best_day = weekly.idxmax() if not weekly.empty else None
    worst_day = weekly.idxmin() if not weekly.empty else None
    r_value = reg.get("r")
    return {
        "cv_7d_current": round(current_cv, 1) if current_cv is not None else None,
        "trend_slope_ms_per_month": round(slope_per_month, 2),
        "trend_r_squared": round(float(r_value) ** 2, 3) if r_value is not None else None,
        "trend_p_value": round(reg["p"], 4) if reg.get("p") is not None else None,
        "residual_volatility_ms": round(residual_std, 1),
        "volatility_improving": vol_improving,
        "best_recovery_day": best_day,
        "worst_recovery_day": worst_day,
        "weekly_pattern": {str(k): round(float(v), 1) for k, v in weekly.to_dict().items()},
        "interpretation": _interpret_hrv_pattern(slope_per_month, current_cv, vol_improving),
    }


def _build_risk_table(age: int, sex: str) -> List[Tuple[str, float, float]]:
    s = str(sex).upper().strip()
    if s == "M":
        if age < 40:
            return [("Elite (top 2%)", 55.0, 1.0), ("Above average", 45.0, 1.3), ("Average", 38.0, 1.6), ("Below average", 32.0, 2.5), ("Low", 0.0, 5.0)]
        if age < 50:
            return [("Elite (top 2%)", 50.0, 1.0), ("Above average", 42.0, 1.3), ("Average", 35.0, 1.6), ("Below average", 29.0, 2.5), ("Low", 0.0, 5.0)]
        if age < 60:
            return [("Elite (top 2%)", 45.0, 1.0), ("Above average", 38.0, 1.3), ("Average", 32.0, 1.6), ("Below average", 26.0, 2.5), ("Low", 0.0, 5.0)]
        return [("Elite (top 2%)", 40.0, 1.0), ("Above average", 34.0, 1.3), ("Average", 28.0, 1.6), ("Below average", 22.0, 2.5), ("Low", 0.0, 5.0)]
    if age < 40:
        return [("Elite (top 2%)", 48.0, 1.0), ("Above average", 38.0, 1.3), ("Average", 32.0, 1.6), ("Below average", 26.0, 2.5), ("Low", 0.0, 5.0)]
    if age < 50:
        return [("Elite (top 2%)", 43.0, 1.0), ("Above average", 35.0, 1.3), ("Average", 29.0, 1.6), ("Below average", 24.0, 2.5), ("Low", 0.0, 5.0)]
    return [("Elite (top 2%)", 38.0, 1.0), ("Above average", 32.0, 1.3), ("Average", 26.0, 1.6), ("Below average", 21.0, 2.5), ("Low", 0.0, 5.0)]


def _elite_vo2_target(age: int, sex: str) -> float:
    return _build_risk_table(age, sex)[0][1]


def _above_avg_vo2(age: int, sex: str) -> float:
    return _build_risk_table(age, sex)[1][1]


def _vo2_biological_age(vo2: float, sex: str) -> Optional[int]:
    s = str(sex).upper().strip()
    if s == "M":
        peak_age, peak_vo2, decline_rate = 25, 45.0, 0.9
    else:
        peak_age, peak_vo2, decline_rate = 25, 38.0, 0.8
    if vo2 >= peak_vo2:
        return peak_age - 5
    bio_age = peak_age + (peak_vo2 - vo2) / decline_rate
    return int(round(bio_age))


def _vo2_prescription(vo2: float, target: float, gap: float, category: str, target_label: str) -> List[str]:
    rx: List[str] = []
    if gap <= 0:
        return [
            f"You are already at or above the {target_label.lower()} target. Maintain with 3-4 aerobic sessions per week.",
            "One interval session per week is usually enough to maintain if the rest of the week stays aerobic.",
        ]
    if gap <= 5:
        rx.append(f"Gap to {target_label.lower()}: {gap:.0f} ml/kg/min. This is usually a medium-term training project, not a quick fix.")
        rx.append("Add one Z4 interval session per week such as 4x4 minutes at 90-95% max HR with 3-minute recoveries.")
        rx.append("Maintain 2-3 Z2 sessions each week at 45-60 minutes.")
    elif gap <= 10:
        rx.append(f"Gap to {target_label.lower()}: {gap:.0f} ml/kg/min. Treat this as a long compounding project.")
        rx.append("Phase 1: build strict Z2 volume 3x45 minutes per week.")
        rx.append("Phase 2: add one interval session per week while maintaining Z2 volume.")
        rx.append("A 4x4-style interval session is one of the clearest VO₂ levers if you can recover from it.")
    else:
        rx.append(f"Gap to {target_label.lower()}: {gap:.0f} ml/kg/min. Long-term project, but every ml/kg/min matters.")
        rx.append("Start with 3 walking or easy jogging sessions per week and build to 30 minutes continuous Z2.")
        rx.append("Moving from low to below average fitness already reduces risk materially.")
    rx.append("VO2 responds to consistency above all else. Missing two weeks costs more than any one hard session buys.")
    return rx


def vo2_longevity_analysis(
    vo2: Optional[float],
    vo2_df: pd.DataFrame,
    age: Optional[int],
    sex: str = "M",
) -> Dict[str, Any]:
    if vo2 is None or age is None:
        return {"error": "Need VO2max value and age"}
    risk_table = _build_risk_table(age, sex)
    category = "Unknown"
    for label, min_vo2, hr in risk_table:
        if float(vo2) >= float(min_vo2):
            category = label
            break
    target_elite = _elite_vo2_target(age, sex)
    target_above_avg = _above_avg_vo2(age, sex)
    gap_to_elite = max(0.0, float(target_elite) - float(vo2))
    next_target = float(target_above_avg) if float(vo2) < float(target_above_avg) else float(target_elite)
    next_target_label = "Above average" if float(vo2) < float(target_above_avg) else "Elite"
    gap_to_next_target = max(0.0, next_target - float(vo2))

    rate = None
    if not vo2_df.empty and "day" in vo2_df.columns:
        vdf = vo2_df.copy()
        val_col = next((c for c in ["vo2_max", "value", "vo2max"] if c in vdf.columns), None)
        if val_col:
            vdf["day_dt"] = pd.to_datetime(vdf["day"], errors="coerce")
            vdf[val_col] = pd.to_numeric(vdf[val_col], errors="coerce")
            vdf = vdf.dropna(subset=["day_dt", val_col]).sort_values("day_dt")
            if len(vdf) >= 4:
                reg = _linregress_days(vdf.rename(columns={val_col: "vo2_value"}), "vo2_value")
                slope = reg.get("slope")
                days_span = int((vdf["day_dt"].iloc[-1] - vdf["day_dt"].iloc[0]).days)
                if slope is not None and days_span >= 60:
                    rate = float(slope) * 365.0

    bio_age = _vo2_biological_age(float(vo2), sex)
    months_to_target = None
    if gap_to_next_target > 0 and rate is not None and rate > 0:
        months_to_target = round((gap_to_next_target / rate) * 12.0, 0) if rate > 0 else None
    return {
        "vo2max": round(float(vo2), 1),
        "risk_category": category,
        "hazard_ratio": None,
        "target_above_average": target_above_avg,
        "target_elite": target_elite,
        "next_target": round(next_target, 1),
        "next_target_label": next_target_label,
        "gap_to_next_target": round(gap_to_next_target, 1),
        "gap_to_elite": round(gap_to_elite, 1),
        "rate_per_year": round(float(rate), 1) if rate is not None else None,
        "months_to_target_estimate": months_to_target,
        "months_to_elite_estimate": months_to_target,
        "vo2_biological_age": bio_age,
        "chronological_age": age,
        "bio_age_delta": round(float(age) - float(bio_age), 1) if bio_age is not None else None,
        "prescription": _vo2_prescription(float(vo2), next_target, gap_to_next_target, category, next_target_label),
    }


def vo2_decay_alert(vo2_df: pd.DataFrame) -> Optional[str]:
    if vo2_df.empty or "day" not in vo2_df.columns:
        return None
    val_col = next((c for c in ["vo2_max", "value", "vo2max"] if c in vo2_df.columns), None)
    if not val_col:
        return None
    df = vo2_df.copy()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col]).sort_values("day")
    if len(df) < 3:
        return None
    peak = float(df[val_col].max())
    current = float(df[val_col].iloc[-1])
    if peak <= 0:
        return None
    drop_pct = ((peak - current) / peak) * 100.0
    if drop_pct > 5:
        return (
            f"VO2max has dropped {drop_pct:.0f}% from your peak ({peak:.0f} -> {current:.0f}). "
            "That is faster than aging alone would explain. Rebuild Z2 volume quickly."
        )
    return None


def vo2_trend_summary(vo2_df: pd.DataFrame, window_days: int = 365) -> Dict[str, Any]:
    if vo2_df.empty or "day" not in vo2_df.columns:
        return {}
    val_col = next((c for c in ["vo2_max", "value", "vo2max"] if c in vo2_df.columns), None)
    if not val_col:
        return {}
    df = vo2_df[["day", val_col]].copy()
    df["day_dt"] = pd.to_datetime(df["day"], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=["day_dt", val_col]).sort_values("day_dt")
    if len(df) < 2:
        return {}
    cutoff = df["day_dt"].max() - pd.Timedelta(days=max(window_days - 1, 0))
    df = df[df["day_dt"] >= cutoff].copy()
    if len(df) < 2:
        return {}
    reg = _linregress_days(df.rename(columns={val_col: "vo2_value"}), "vo2_value")
    slope = reg.get("slope")
    if slope is None:
        return {}
    recent = df[df["day_dt"] >= (df["day_dt"].max() - pd.Timedelta(days=27))]
    recent_vals = pd.to_numeric(recent[val_col], errors="coerce").dropna()
    spread = float(recent_vals.max() - recent_vals.min()) if len(recent_vals) >= 2 else None
    return {
        "current": float(df[val_col].iloc[-1]),
        "median28": float(recent_vals.median()) if not recent_vals.empty else float(df[val_col].iloc[-1]),
        "slope_90d": round(float(slope) * 90.0, 2),
        "stability_spread": round(spread, 1) if spread is not None else None,
        "valid_points": int(len(df)),
    }


def _primary_lever(breakdown: Dict[str, Dict[str, Any]]) -> str:
    gaps = sorted(((data.get("max", 0) - data.get("score", 0), key) for key, data in breakdown.items()), reverse=True)
    lever = gaps[0][1] if gaps else "unknown"
    return {
        "vo2max": "VO2max is your biggest gap. Add Z2 volume plus one interval session each week.",
        "rhr": "RHR trend is your biggest gap. Fix sleep consistency, hydration, and aerobic volume.",
        "hrv": "HRV stability is your biggest gap. Tighten sleep timing and reduce hidden stressors.",
        "habits": "Daily habits are your biggest gap. The scorecard shows which inputs to fix first.",
    }.get(lever, "Keep compounding consistency.")


def longevity_composite_score(
    rhr_trend: Dict[str, Any],
    hrv_pattern: Dict[str, Any],
    vo2_analysis: Dict[str, Any],
    habit_summary: Dict[str, Any],
) -> Dict[str, Any]:
    score = 0.0
    breakdown: Dict[str, Dict[str, Any]] = {}
    cat = vo2_analysis.get("risk_category", "Unknown")
    vo2_score = {"Elite (top 2%)": 40, "Above average": 32, "Average": 20, "Below average": 10, "Low": 0}.get(cat, 0)
    score += vo2_score
    breakdown["vo2max"] = {"score": vo2_score, "max": 40, "category": cat}

    rhr_slope = _safe_float(rhr_trend.get("slope_bpm_per_month"))
    rhr_score = 0
    if rhr_slope is not None:
        rhr_score = 25 if rhr_slope <= -0.3 else 20 if rhr_slope <= 0 else 10 if rhr_slope <= 0.3 else 0
    score += rhr_score
    breakdown["rhr"] = {"score": rhr_score, "max": 25, "slope": rhr_slope}

    hrv_slope = _safe_float(hrv_pattern.get("trend_slope_ms_per_month"))
    hrv_cv = _safe_float(hrv_pattern.get("cv_7d_current"))
    hrv_score = 0
    if hrv_slope is not None:
        hrv_score += 15 if hrv_slope > 0.3 else 10 if hrv_slope > -0.3 else 0
    if hrv_cv is not None:
        hrv_score += 10 if hrv_cv < 12 else 5 if hrv_cv < 20 else 0
    score += hrv_score
    breakdown["hrv"] = {"score": hrv_score, "max": 25, "slope": hrv_slope, "cv": hrv_cv}

    hrv_delta = _safe_float(habit_summary.get("hrv_delta"))
    habit_score = 10 if hrv_delta is not None and hrv_delta > 5 else 5 if hrv_delta is not None and hrv_delta > 0 else 0
    score += habit_score
    breakdown["habits"] = {"score": habit_score, "max": 10}

    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D" if score >= 40 else "F"
    return {
        "score": round(score, 0),
        "grade": grade,
        "breakdown": breakdown,
        "primary_lever": _primary_lever(breakdown),
    }


def aerobic_efficiency_summary(
    workouts: pd.DataFrame,
    hr_points: pd.DataFrame,
    *,
    zones: List[Tuple[str, float, float]],
    max_hr: int,
    resting_hr: int,
    sex: str,
) -> Dict[str, Any]:
    if workouts is None or workouts.empty or hr_points is None or hr_points.empty:
        return {}
    rows: List[Dict[str, Any]] = []
    for _, row in workouts.iterrows():
        start_dt, end_dt = workout_window(row)
        if start_dt is None or end_dt is None:
            continue
        seg = hr_points[(hr_points["ts"] >= start_dt) & (hr_points["ts"] <= end_dt)].copy()
        if len(seg) < 20:
            continue
        tz = time_in_zones(seg, zones) / 60.0
        total_min = float(tz.sum())
        if total_min < 20:
            continue
        easy_min = float(tz.get("<Z1", 0.0) or 0.0) + float(tz.get("Z1", 0.0) or 0.0) + float(tz.get("Z2", 0.0) or 0.0)
        if (easy_min / max(total_min, 1.0)) < 0.5:
            continue
        drift = compute_hr_drift(seg)
        avg_hr = _safe_float(seg["bpm"].mean())
        trimp = compute_banister_trimp(seg, max_hr=int(max_hr), resting_hr=int(resting_hr), sex=sex)
        day_dt = pd.to_datetime(str(row.get("day")), errors="coerce")
        if pd.isna(day_dt) or drift is None:
            continue
        rows.append(
            {
                "day_dt": day_dt.normalize(),
                "drift": float(drift),
                "avg_hr": avg_hr,
                "easy_pct": (easy_min / max(total_min, 1.0)) * 100.0,
                "trimp": trimp,
            }
        )
    if len(rows) < 3:
        return {}
    df = pd.DataFrame(rows).sort_values("day_dt")
    recent = df[df["day_dt"] >= (df["day_dt"].max() - pd.Timedelta(days=27))]
    reg = _linregress_days(df.rename(columns={"drift": "drift_value"}), "drift_value") if len(df) >= 5 else {}
    slope = reg.get("slope")
    low_drift_rate = float((df["drift"] <= 5.0).mean() * 100.0) if not df.empty else None
    cv = None
    if len(recent) >= 3:
        mean_drift = float(recent["drift"].mean())
        if mean_drift > 0:
            cv = float(recent["drift"].std() / mean_drift * 100.0)
    interpretation = "Stable aerobic execution."
    current_drift = float(df["drift"].iloc[-1])
    if current_drift > 6:
        interpretation = "Aerobic efficiency is weak right now. Start easier, fuel earlier, or shorten the session."
    elif current_drift > 4:
        interpretation = "Aerobic efficiency is workable but drifts late. Tighten early pacing."
    return {
        "current_drift": round(current_drift, 1),
        "median28_drift": round(float(recent["drift"].median()), 1) if not recent.empty else None,
        "slope_pct_per_month": round(float(slope) * 30.0, 2) if slope is not None else None,
        "stability_cv": round(cv, 1) if cv is not None else None,
        "low_drift_rate": round(low_drift_rate, 0) if low_drift_rate is not None else None,
        "valid_sessions": int(len(df)),
        "interpretation": interpretation,
        "target": "Keep aerobic drift <= 5% on easy/base sessions",
    }


def compute_training_biomarker_effects(
    daily: pd.DataFrame,
    workouts: pd.DataFrame,
    hr_points: pd.DataFrame,
    *,
    max_hr: int,
    resting_hr: int,
    sex: str,
    intents_path: str = WORKOUT_INTENT_PATH,
) -> pd.DataFrame:
    if daily is None or daily.empty or workouts is None or workouts.empty:
        return pd.DataFrame()
    if hr_points is None:
        hr_points = pd.DataFrame()

    metric_lookups = {
        "hrv_rmssd": metric_delta_frame(daily, "hrv_rmssd").rename(columns={"value": "hrv_value", "prior_avg": "hrv_prior_avg", "delta": "hrv_delta"}),
        "resting_hr": metric_delta_frame(daily, "resting_hr").rename(columns={"value": "rhr_value", "prior_avg": "rhr_prior_avg", "delta": "rhr_delta"}),
    }
    hrv_lookup = metric_lookups["hrv_rmssd"].set_index("day") if not metric_lookups["hrv_rmssd"].empty else pd.DataFrame()
    rhr_lookup = metric_lookups["resting_hr"].set_index("day") if not metric_lookups["resting_hr"].empty else pd.DataFrame()

    workouts_with_intent = attach_workout_intents(normalize_workout_rows(workouts), path=intents_path)
    if workouts_with_intent.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, row in workouts_with_intent.iterrows():
        day = str(row.get("day") or "")
        if not day:
            continue
        start_dt, end_dt = workout_window(row)
        if start_dt is None or end_dt is None:
            continue
        seg = hr_points[(hr_points["ts"] >= start_dt) & (hr_points["ts"] <= end_dt)].copy() if not hr_points.empty else pd.DataFrame()
        trimp = None
        drift = None
        hrr60 = None
        hrr120 = None
        if len(seg) >= 20:
            trimp = compute_banister_trimp(seg, max_hr=int(max_hr), resting_hr=int(resting_hr), sex=sex)
            drift = compute_hr_drift(seg)
            end_peak = float(seg.tail(min(10, len(seg)))["bpm"].max())
            if not hr_points.empty:
                post = hr_points[(hr_points["ts"] > end_dt) & (hr_points["ts"] <= end_dt + timedelta(seconds=150))].copy()
                if not post.empty:
                    post["secs_after"] = (post["ts"] - end_dt).dt.total_seconds()
                    post60 = post[(post["secs_after"] >= 45) & (post["secs_after"] <= 75)]
                    post120 = post[(post["secs_after"] >= 105) & (post["secs_after"] <= 135)]
                    if not post60.empty:
                        hrr60 = round(end_peak - float(post60.iloc[0]["bpm"]), 1)
                    if not post120.empty:
                        hrr120 = round(end_peak - float(post120.iloc[0]["bpm"]), 1)

        next_day = None
        next_day2 = None
        try:
            day_dt = pd.to_datetime(day, errors="coerce")
            if pd.notna(day_dt):
                next_day = (day_dt + pd.Timedelta(days=1)).date().isoformat()
                next_day2 = (day_dt + pd.Timedelta(days=2)).date().isoformat()
        except Exception:
            next_day = None
            next_day2 = None

        next_day_hrv_delta = _safe_float(hrv_lookup.loc[next_day, "hrv_delta"]) if isinstance(hrv_lookup, pd.DataFrame) and next_day in hrv_lookup.index else None
        next_day_rhr_delta = _safe_float(rhr_lookup.loc[next_day, "rhr_delta"]) if isinstance(rhr_lookup, pd.DataFrame) and next_day in rhr_lookup.index else None
        two_day_hrv_delta = _safe_float(hrv_lookup.loc[next_day2, "hrv_delta"]) if isinstance(hrv_lookup, pd.DataFrame) and next_day2 in hrv_lookup.index else None

        same_intent = workouts_with_intent[
            (workouts_with_intent["workout_intent"].astype(str) == str(row.get("workout_intent") or ""))
            & (workouts_with_intent["day"].astype(str) < day)
        ]
        same_intent_delta = None
        if drift is not None and not same_intent.empty and not hr_points.empty:
            prior_drifts: List[float] = []
            for _, prior_row in same_intent.tail(6).iterrows():
                ps, pe = workout_window(prior_row)
                if ps is None or pe is None:
                    continue
                pseg = hr_points[(hr_points["ts"] >= ps) & (hr_points["ts"] <= pe)].copy()
                if len(pseg) < 20:
                    continue
                prior_drift = compute_hr_drift(pseg)
                if prior_drift is not None:
                    prior_drifts.append(float(prior_drift))
            if prior_drifts:
                same_intent_delta = round(float(drift) - float(pd.Series(prior_drifts).median()), 1)

        day_row = daily[daily["day"].astype(str) == day].tail(1)
        sleep_debt = None
        sleep_hours = None
        temp_dev = None
        prior_day_load = None
        if not day_row.empty:
            sleep_hours = _safe_float(day_row.iloc[-1].get("sleep_total_s"))
            sleep_debt = None if sleep_hours is None else round((_safe_float(day_row.iloc[-1].get("median_sleep_need_s")) or 28800.0) / 3600.0 - (sleep_hours / 3600.0), 1)
            temp_dev = _safe_float(day_row.iloc[-1].get("temp_dev"))
        prior_day = daily[daily["day"].astype(str) < day].tail(1)
        if not prior_day.empty:
            prior_day_load = _safe_float(prior_day.iloc[-1].get("trimp"))

        rows.append(
            {
                "day": day,
                "workout_key": str(row.get("workout_key") or workout_key_from_row(row)),
                "type": str(row.get("type") or "workout"),
                "workout_intent": str(row.get("workout_intent") or ""),
                "trimp": trimp,
                "drift": drift,
                "hrr60": hrr60,
                "hrr120": hrr120,
                "next_day_hrv_delta": next_day_hrv_delta,
                "next_day_rhr_delta": next_day_rhr_delta,
                "two_day_hrv_delta": two_day_hrv_delta,
                "same_intent_efficiency_delta": same_intent_delta,
                "sleep_debt_h": sleep_debt,
                "sleep_hours": None if sleep_hours is None else round(sleep_hours / 3600.0, 1),
                "temp_dev": temp_dev,
                "prior_day_load": prior_day_load,
            }
        )
    return pd.DataFrame(rows)


def summarize_training_biomarker_effects(effects: pd.DataFrame) -> Dict[str, Any]:
    if effects is None or effects.empty:
        return {}
    recent = effects.sort_values("day").tail(12).copy()
    verdict = "neutral"
    reasons: List[str] = []
    mean_hrv_2d = _safe_float(pd.to_numeric(recent["two_day_hrv_delta"], errors="coerce").dropna().mean()) if "two_day_hrv_delta" in recent.columns else None
    mean_rhr_1d = _safe_float(pd.to_numeric(recent["next_day_rhr_delta"], errors="coerce").dropna().mean()) if "next_day_rhr_delta" in recent.columns else None
    mean_drift_delta = _safe_float(pd.to_numeric(recent["same_intent_efficiency_delta"], errors="coerce").dropna().mean()) if "same_intent_efficiency_delta" in recent.columns else None
    hrr60_available = pd.to_numeric(recent.get("hrr60"), errors="coerce").dropna() if "hrr60" in recent.columns else pd.Series(dtype=float)
    sleep_debt_series = pd.to_numeric(recent.get("sleep_debt_h"), errors="coerce").dropna() if "sleep_debt_h" in recent.columns else pd.Series(dtype=float)
    temp_series = pd.to_numeric(recent.get("temp_dev"), errors="coerce").dropna() if "temp_dev" in recent.columns else pd.Series(dtype=float)
    prior_load_series = pd.to_numeric(recent.get("prior_day_load"), errors="coerce").dropna() if "prior_day_load" in recent.columns else pd.Series(dtype=float)
    high_prior_load_cutoff = None if prior_load_series.empty else float(prior_load_series.quantile(0.75))
    context_hits = pd.Series(False, index=recent.index)
    if "sleep_debt_h" in recent.columns:
        context_hits = context_hits | (pd.to_numeric(recent["sleep_debt_h"], errors="coerce") > 0.75).fillna(False)
    if "temp_dev" in recent.columns:
        context_hits = context_hits | (pd.to_numeric(recent["temp_dev"], errors="coerce").abs() > 0.3).fillna(False)
    if high_prior_load_cutoff is not None and "prior_day_load" in recent.columns:
        context_hits = context_hits | (pd.to_numeric(recent["prior_day_load"], errors="coerce") > high_prior_load_cutoff).fillna(False)
    context_rate = float(context_hits.mean()) if len(context_hits) else 0.0
    if mean_hrv_2d is not None:
        reasons.append(f"Average 2-day HRV delta after training is {mean_hrv_2d:+.1f} ms.")
    if mean_rhr_1d is not None:
        reasons.append(f"Average next-day resting HR delta is {mean_rhr_1d:+.1f} bpm.")
    if mean_drift_delta is not None:
        reasons.append(f"Same-intent drift is {mean_drift_delta:+.1f}% vs your recent baseline.")
    if not hrr60_available.empty:
        reasons.append(f"HRR60 is available on {len(hrr60_available)} sessions, averaging {hrr60_available.mean():.1f} bpm.")
    if not sleep_debt_series.empty and sleep_debt_series.mean() > 0.75:
        reasons.append(f"Recent workouts often started with {sleep_debt_series.mean():.1f}h of sleep debt.")
    if not temp_series.empty and temp_series.abs().mean() > 0.3:
        reasons.append(f"Temperature deviation averaged {temp_series.mean():+.1f} C on recent training days.")
    if high_prior_load_cutoff is not None and not prior_load_series.empty and (prior_load_series > high_prior_load_cutoff).mean() >= 0.35:
        reasons.append(f"Prior-day load was high before {(prior_load_series > high_prior_load_cutoff).sum():.0f} of the last {len(prior_load_series)} sessions.")
    if context_rate >= 0.4:
        reasons.append("Several weaker sessions were probably context-driven, not pure fitness regression.")

    if mean_hrv_2d is not None and mean_hrv_2d > 0 and (mean_rhr_1d is None or mean_rhr_1d <= 0):
        verdict = "improving"
    elif (mean_hrv_2d is not None and mean_hrv_2d < 0) and (mean_rhr_1d is not None and mean_rhr_1d > 0):
        verdict = "costing recovery"

    next_week = "Keep load steady and focus on cleaner easy-session execution."
    if verdict == "improving":
        next_week = "Stay in the current load band and progress slowly."
    elif verdict == "costing recovery":
        next_week = "Reduce stacked hard days and keep most work inside your sweet-spot load band."

    return {
        "verdict": verdict,
        "reasons": reasons[:4],
        "next_week": next_week,
        "context_rate": context_rate,
        "context_driven": context_rate >= 0.4,
    }


def training_biomarker_link(
    daily: pd.DataFrame,
    *,
    efficiency: Dict[str, Any],
    rhr_trend: Dict[str, Any],
    hrv_pattern: Dict[str, Any],
    effects: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    if daily is None or daily.empty:
        return {}
    recent = daily.sort_values("day").copy()
    recent["trimp"] = pd.to_numeric(recent.get("trimp"), errors="coerce")
    recent_load = _safe_float(recent.tail(14)["trimp"].dropna().mean()) if "trimp" in recent.columns else None
    trimp_curve = dose_response(recent, "trimp", "hrv_rmssd", lag_days=2, bins=5, higher_is_better=True)
    sweet = trimp_curve.get("sweet_spot")
    sweet_low = _safe_float((sweet or {}).get("x_low"))
    sweet_high = _safe_float((sweet or {}).get("x_high"))
    hrv_slope = _safe_float(hrv_pattern.get("trend_slope_ms_per_month"))
    rhr_slope = _safe_float(rhr_trend.get("slope_bpm_per_month"))
    eff_slope = _safe_float(efficiency.get("slope_pct_per_month"))
    verdict = "Mixed"
    bullets: List[str] = []
    if sweet_low is not None and sweet_high is not None:
        bullets.append(f"Your best 2-day-later HRV tends to show up around {sweet_low:.0f}-{sweet_high:.0f} TRIMP.")
    if recent_load is not None:
        bullets.append(f"Recent 14-day average training load is {recent_load:.0f} TRIMP/day.")
    if eff_slope is not None:
        bullets.append(f"Aerobic drift trend is {eff_slope:+.2f}%/month.")

    if effects is not None and not effects.empty:
        effect_summary = summarize_training_biomarker_effects(effects)
        if effect_summary.get("verdict") == "improving":
            verdict = "Training is currently supporting biomarkers"
        elif effect_summary.get("verdict") == "costing recovery":
            verdict = "Training is currently degrading biomarkers"
        for reason in effect_summary.get("reasons", []):
            bullets.append(reason)
        if effect_summary.get("context_driven"):
            bullets.append("Some ugly sessions were context-limited by sleep debt, temperature strain, or stacked load rather than a clear fitness drop.")

    if (
        sweet_high is not None
        and recent_load is not None
        and recent_load > sweet_high
        and hrv_slope is not None
        and hrv_slope < 0
        and rhr_slope is not None
        and rhr_slope > 0
    ):
        verdict = "Training is currently degrading biomarkers"
    elif (
        hrv_slope is not None
        and hrv_slope >= 0
        and (rhr_slope is None or rhr_slope <= 0.1)
        and (eff_slope is None or eff_slope <= 0)
        and verdict != "Training is currently degrading biomarkers"
    ):
        verdict = "Training is currently supporting biomarkers"
    elif verdict != "Training is currently supporting biomarkers":
        verdict = "Training effect is mixed"

    next_action = "Keep intensity controlled until HRV stabilizes and RHR stops drifting up."
    if verdict == "Training is currently supporting biomarkers":
        next_action = "Keep the current load band. Progress volume slowly before adding more intensity."
    elif sweet_high is not None:
        next_action = f"Keep most weeks near the {sweet_low:.0f}-{sweet_high:.0f} TRIMP band and avoid stacking hard days."

    return {
        "verdict": verdict,
        "bullets": bullets,
        "next_action": next_action,
        "recent_load": recent_load,
        "sweet_low": sweet_low,
        "sweet_high": sweet_high,
    }


def longevity_score_action_plan(
    current_score: Dict[str, Any],
    *,
    prior_score: Optional[Dict[str, Any]] = None,
    rhr_trend: Optional[Dict[str, Any]] = None,
    hrv_pattern: Optional[Dict[str, Any]] = None,
    vo2_analysis: Optional[Dict[str, Any]] = None,
    habit_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    score_now = _safe_float(current_score.get("score")) if current_score else None
    score_prior = _safe_float(prior_score.get("score")) if prior_score else None
    delta_30 = None if score_now is None or score_prior is None else score_now - score_prior

    easiest_parts: List[str] = []
    hrv_delta = _safe_float((habit_summary or {}).get("hrv_delta"))
    if hrv_delta is None or hrv_delta <= 0:
        easiest_parts.append("Habit consistency can likely add the first 5-10 points.")
    if _safe_float((rhr_trend or {}).get("slope_bpm_per_month")) is not None and _safe_float((rhr_trend or {}).get("slope_bpm_per_month")) > 0:
        easiest_parts.append("RHR trend is drifting up; sleep timing + Z2 volume is the fastest physiology lever.")
    if current_score and current_score.get("breakdown", {}).get("vo2max", {}).get("score", 0) < 32:
        easiest_parts.append("VO₂ is still the biggest long-term upside, but it is slower to move.")

    fastest_lever = "Habit consistency and sleep timing"
    if _safe_float((rhr_trend or {}).get("slope_bpm_per_month")) is not None and _safe_float((rhr_trend or {}).get("slope_bpm_per_month")) > 0.3:
        fastest_lever = "Resting HR trend"
    elif _safe_float((hrv_pattern or {}).get("cv_7d_current")) is not None and _safe_float((hrv_pattern or {}).get("cv_7d_current")) > 15:
        fastest_lever = "HRV stability"

    slowest_lever = "VO₂max trajectory"
    if vo2_analysis and vo2_analysis.get("gap_to_elite") in {None, 0}:
        slowest_lever = "RHR slope"

    expected_upside = "Likely +5 to +8 points in 30 days with consistent habits and controlled training."
    if vo2_analysis and _safe_float(vo2_analysis.get("gap_to_elite")) is not None and _safe_float(vo2_analysis.get("gap_to_elite")) <= 5:
        expected_upside = "Likely +8 to +12 points in 30 days if you execute the VO₂ block consistently."

    return {
        "delta_30d": delta_30,
        "easiest_points": easiest_parts[:3],
        "fastest_lever": fastest_lever,
        "slowest_lever": slowest_lever,
        "expected_30d_upside": expected_upside,
    }


def longevity_decision_score(
    rhr_trend: Dict[str, Any],
    hrv_pattern: Dict[str, Any],
    vo2_analysis: Dict[str, Any],
    aerobic_efficiency: Dict[str, Any],
    habit_summary: Dict[str, Any],
    *,
    score_30d_ago: Optional[float] = None,
) -> Dict[str, Any]:
    base_score = longevity_composite_score(rhr_trend, hrv_pattern, vo2_analysis, habit_summary)
    action_plan = longevity_score_action_plan(
        base_score,
        prior_score={"score": score_30d_ago} if score_30d_ago is not None else None,
        rhr_trend=rhr_trend,
        hrv_pattern=hrv_pattern,
        vo2_analysis=vo2_analysis,
        habit_summary=habit_summary,
    )
    easiest = list(action_plan.get("easiest_points") or [])
    if _safe_float(aerobic_efficiency.get("current_drift")) is not None and _safe_float(aerobic_efficiency.get("current_drift")) > 5:
        easiest.append("Aerobic efficiency: +2 to +4 points if you bring easy-session drift back under 5%.")
    return {
        "score": _safe_float(base_score.get("score")),
        "score_30d_ago": score_30d_ago,
        "delta_30d": action_plan.get("delta_30d"),
        "easiest_points_to_gain": easiest[:3],
        "fastest_lever": action_plan.get("fastest_lever"),
        "slowest_lever": action_plan.get("slowest_lever"),
        "expected_30d_upside": action_plan.get("expected_30d_upside"),
        "grade": base_score.get("grade"),
        "breakdown": base_score.get("breakdown"),
        "primary_lever": base_score.get("primary_lever"),
    }

def classify_percentile(value: Optional[float], top10: Optional[float], top1: Optional[float], *, higher_is_better: bool = True) -> str:
    if value is None or top10 is None or top1 is None:
        return "UNKNOWN"
    v = float(value)
    t10 = float(top10)
    t1 = float(top1)
    if higher_is_better:
        if v >= t1:
            return "TOP 1%"
        if v >= t10:
            return "TOP 10%"
        return "BELOW TOP 10%"
    else:
        # lower is better: top1 < top10
        if v <= t1:
            return "TOP 1%"
        if v <= t10:
            return "TOP 10%"
        return "BELOW TOP 10%"


@dataclass(frozen=True)
class GoalProfile:
    label: str
    focus: str
    priority_metrics: Tuple[str, ...]
    hrv_play: str
    rhr_play: str
    vo2_play: str


GOAL_PROFILES: Dict[str, GoalProfile] = {
    "Performance (endurance)": GoalProfile(
        label="Performance (endurance)",
        focus="Use HRV + resting HR to decide when to push, and keep exercise HR brutally precise so easy days stay easy.",
        priority_metrics=("hrv_rmssd", "resting_hr", "sleep_score", "readiness"),
        hrv_play="HRV decides whether today is a quality day or an aerobic-build day.",
        rhr_play="Resting HR is the strain check: if it rises, protect the aerobic system before chasing pace.",
        vo2_play="VO2 max moves fastest from consistent Z2 volume plus one well-timed high-intensity stimulus.",
    ),
    "Performance (strength / hybrid)": GoalProfile(
        label="Performance (strength / hybrid)",
        focus="Protect recovery so heavy lifting quality stays high, then use exercise HR to keep conditioning from leaking into junk fatigue.",
        priority_metrics=("hrv_rmssd", "resting_hr", "readiness", "sleep_score"),
        hrv_play="HRV tells you if you can absorb heavy work or if today should be technique and moderate volume.",
        rhr_play="Resting HR shows if systemic fatigue is building even when muscular soreness looks manageable.",
        vo2_play="VO2 max improves best when conditioning is treated as its own dose, not random fatigue after lifting.",
    ),
    "Longevity / healthspan": GoalProfile(
        label="Longevity / healthspan",
        focus="Prioritize stable HRV, low resting HR, and mostly Z1-Z2 training that you can recover from every week.",
        priority_metrics=("resting_hr", "hrv_rmssd", "sleep_score", "readiness"),
        hrv_play="HRV should stay stable week to week; big dips mean the plan is costing more than it is buying.",
        rhr_play="Resting HR is the cleanest day-to-day signal for accumulated strain and cardiovascular efficiency.",
        vo2_play="VO2 max still matters, but the fastest sustainable path is volume in Z2, not constant hard intervals.",
    ),
    "Body composition": GoalProfile(
        label="Body composition",
        focus="Use recovery signals to preserve training quality, appetite control, and NEAT instead of digging a fatigue hole.",
        priority_metrics=("hrv_rmssd", "resting_hr", "sleep_score", "readiness"),
        hrv_play="Low HRV is often the first sign your deficit or training load is too aggressive.",
        rhr_play="Resting HR catches under-recovery, dehydration, and sleep debt before body-composition progress stalls.",
        vo2_play="VO2 max improves with controlled aerobic volume and selective intervals, not constant medium-hard work.",
    ),
    "Stress resilience": GoalProfile(
        label="Stress resilience",
        focus="Make HRV and resting HR the lead metrics. Training should calm the system unless the signals clearly support more.",
        priority_metrics=("hrv_rmssd", "resting_hr", "sleep_score", "readiness"),
        hrv_play="HRV is the headline metric. The job is to create stable upward pressure, not spike then crash.",
        rhr_play="Resting HR should stay low and steady; sudden rises usually mean sleep, stress, dehydration, or illness won the day.",
        vo2_play="VO2 max is secondary until recovery stability is reliable; earn intensity by stacking calm weeks.",
    ),
}


def get_goal_profile(goal: str) -> GoalProfile:
    return GOAL_PROFILES.get(goal, GOAL_PROFILES["Longevity / healthspan"])


def zone_bounds_by_name(zones: List[Tuple[str, float, float]], name: str) -> Optional[Tuple[int, int]]:
    for zone_name, lo, hi in zones:
        if zone_name == name:
            return int(round(lo)), int(round(hi))
    return None


def _prior_window(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= 1:
        return pd.Series(dtype=float)
    return s.iloc[max(0, len(s) - 1 - window): len(s) - 1]


class MetricSignal(TypedDict):
    key: str
    value: Optional[float]
    prev: Optional[float]
    baseline7: Optional[float]
    baseline28: Optional[float]
    delta_prev: Optional[float]
    delta7: Optional[float]
    delta28: Optional[float]
    favorable_delta7: Optional[float]
    z28: Optional[float]
    signal: str


class DayMode(TypedDict):
    mode: str
    summary: str
    reasons: List[str]


class TrainingPrescription(TypedDict):
    session: str
    hr_target: str
    instruction: str
    avoid: str


@dataclass(frozen=True)
class EventRecord:
    day: str
    alcohol: bool = False
    late_meal: bool = False
    travel: bool = False
    illness: bool = False
    supplement: str = ""
    sauna: bool = False
    cold: bool = False
    caffeine_late: bool = False
    manual_wellness: Optional[int] = None
    notes: str = ""


def compute_metric_signal(
    daily: pd.DataFrame,
    key: str,
    *,
    higher_is_better: bool,
) -> MetricSignal:
    empty_signal: MetricSignal = {
        "key": key,
        "value": None,
        "prev": None,
        "baseline7": None,
        "baseline28": None,
        "delta_prev": None,
        "delta7": None,
        "delta28": None,
        "favorable_delta7": None,
        "z28": None,
        "signal": "UNKNOWN",
    }
    if daily.empty or key not in daily.columns:
        return empty_signal

    s = pd.to_numeric(daily[key], errors="coerce").dropna()
    if s.empty:
        return empty_signal

    value = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) >= 2 else None
    prior7 = _prior_window(s, 7)
    prior28 = _prior_window(s, 28)
    baseline7 = float(prior7.median()) if len(prior7) >= 3 else None
    baseline28 = float(prior28.median()) if len(prior28) >= 7 else None
    spread28 = robust_spread(prior28) if len(prior28) >= 7 else None
    z28 = zscore(value, baseline28, spread28)
    signal = traffic_light_from_z(z28, higher_is_better=higher_is_better) if z28 is not None else "UNKNOWN"
    if baseline28 is not None:
        min_change = minimum_meaningful_change_for_metric(key, baseline28)
        if abs(value - baseline28) < min_change:
            signal = "AMBER"

    return {
        "key": key,
        "value": value,
        "prev": prev,
        "baseline7": baseline7,
        "baseline28": baseline28,
        "delta_prev": None if prev is None else value - prev,
        "delta7": None if baseline7 is None else value - baseline7,
        "delta28": None if baseline28 is None else value - baseline28,
        "favorable_delta7": None if baseline7 is None else (value - baseline7 if higher_is_better else baseline7 - value),
        "z28": z28,
        "signal": signal,
    }


def recent_median(daily: pd.DataFrame, key: str, window_days: int = 28) -> Optional[float]:
    df = _recent_metric_frame(daily, key, window_days)
    if df.empty or key not in df.columns:
        return None
    s = pd.to_numeric(df[key], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.median())


def biomarker_source_label(key: str) -> str:
    return {
        "hrv_rmssd": "Oura sleep.average_hrv",
        "resting_hr": "Oura sleep.lowest_heart_rate",
        "vo2_max": "Oura vO2_max",
        "aerobic_efficiency": "Workout + heartrate stream",
    }.get(key, "Derived dashboard metric")


def compute_signal_trust(
    series: pd.Series,
    *,
    current_value: Optional[float],
    short_window: int = 7,
    long_window: int = 28,
    min_short: int = 4,
    min_long: int = 14,
    source_used: str = "sleep_fallback",
    min_meaningful_change: float = 0.5,
) -> Dict[str, Any]:
    raw = pd.to_numeric(series, errors="coerce")
    recent7 = raw.tail(short_window)
    recent28 = raw.tail(long_window)
    n_valid_7 = int(recent7.notna().sum())
    n_valid_28 = int(recent28.notna().sum())
    missing_pct_28 = round(max(0.0, 100.0 * (1.0 - (n_valid_28 / max(long_window, 1)))), 1)

    confidence = "LOW"
    if n_valid_7 >= 6 and n_valid_28 >= 24 and missing_pct_28 <= 20:
        confidence = "HIGH"
    elif n_valid_7 >= min_short and n_valid_28 >= min_long:
        confidence = "MEDIUM"

    pattern = "INSUFFICIENT"
    valid_recent28 = recent28.dropna()
    baseline_pool = valid_recent28.iloc[:-1]
    baseline = float(baseline_pool.median()) if len(baseline_pool) >= max(3, min_short) else None
    spread = robust_spread(baseline_pool) if len(baseline_pool) >= max(5, min_long // 2) else None
    if baseline is not None and current_value is not None and len(valid_recent28) >= 2:
        recent_tail = valid_recent28.tail(3)
        threshold = max((_safe_float(spread) or 0.0) * 0.6, float(min_meaningful_change))
        deltas = pd.to_numeric(recent_tail, errors="coerce") - baseline
        meaningful = deltas.abs() >= threshold
        meaningful_deltas = deltas[meaningful]
        if len(meaningful_deltas) >= 2 and meaningful_deltas.tail(3).apply(lambda x: 1 if x > 0 else -1).nunique() == 1:
            pattern = "MULTI_NIGHT_TREND"
        elif len(meaningful_deltas) == 1 and meaningful.iloc[-1]:
            pattern = "ONE_NIGHT_OUTLIER"

    recommendation_strength = "STRONG" if confidence == "HIGH" else "MODERATE" if confidence == "MEDIUM" else "WEAK"
    return {
        "n_valid_7": n_valid_7,
        "n_valid_28": n_valid_28,
        "missing_pct_28": missing_pct_28,
        "confidence": confidence,
        "pattern": pattern,
        "source_used": source_used,
        "recommendation_strength": recommendation_strength,
    }


def describe_signal_trust(trust: Dict[str, Any]) -> str:
    pattern_map = {
        "ONE_NIGHT_OUTLIER": "one-night outlier",
        "MULTI_NIGHT_TREND": "multi-night trend",
        "INSUFFICIENT": "pattern unclear",
    }
    summary = (
        f"{trust.get('confidence', 'LOW')} confidence, "
        f"{trust.get('n_valid_28', 0)}/28 valid nights, "
        f"{_fmt(trust.get('missing_pct_28'))}% missing, "
        f"{pattern_map.get(str(trust.get('pattern')), 'pattern unclear')}, "
        f"source `{trust.get('source_used')}`."
    )
    if trust.get("context_note"):
        summary += f" {trust['context_note']}"
    return summary


def compute_biomarker_trust(
    daily: pd.DataFrame,
    key: str,
    *,
    higher_is_better: bool,
    window_days: int = 28,
    source_used: Optional[str] = None,
) -> Dict[str, Any]:
    out = {
        "n_valid_7": 0,
        "n_valid_28": 0,
        "missing_pct_28": None,
        "confidence": "LOW",
        "pattern": "INSUFFICIENT",
        "source_used": source_used or biomarker_source_label(key),
        "recommendation_strength": "WEAK",
        "summary": None,
    }
    if daily.empty or key not in daily.columns:
        return out

    metric_df = daily[["day", key]].copy()
    metric_df["day_dt"] = pd.to_datetime(metric_df["day"], errors="coerce").dt.normalize()
    metric_df[key] = pd.to_numeric(metric_df[key], errors="coerce")
    metric_df = metric_df.dropna(subset=["day_dt"]).sort_values("day_dt")
    if metric_df.empty:
        return out

    max_day = metric_df["day_dt"].max()
    cutoff = max_day - pd.Timedelta(days=max(window_days - 1, 0))
    calendar = pd.DataFrame({"day_dt": pd.date_range(cutoff, max_day, freq="D")})
    recent = calendar.merge(metric_df[["day_dt", key]], on="day_dt", how="left")
    trust = compute_signal_trust(
        recent[key],
        current_value=_safe_float(pd.to_numeric(recent[key], errors="coerce").dropna().iloc[-1]) if recent[key].notna().any() else None,
        short_window=7,
        long_window=window_days,
        source_used=source_used or biomarker_source_label(key),
        min_meaningful_change=minimum_meaningful_change_for_metric(
            key,
            _safe_float(pd.to_numeric(recent[key], errors="coerce").dropna().median()) if recent[key].notna().any() else None,
        ),
    )
    out.update(trust)
    out["valid_points"] = out["n_valid_28"]
    out["expected_points"] = window_days
    out["missing_pct"] = out["missing_pct_28"]
    if key in {"hrv_rmssd", "resting_hr"} and "resp_rate_dev" in daily.columns:
        context_df = daily[["day", "resp_rate_dev"]].copy()
        context_df["day_dt"] = pd.to_datetime(context_df["day"], errors="coerce").dt.normalize()
        context_df["resp_rate_dev"] = pd.to_numeric(context_df["resp_rate_dev"], errors="coerce")
        latest_context = context_df[context_df["day_dt"] == max_day].dropna(subset=["resp_rate_dev"]).tail(1)
        if not latest_context.empty:
            resp_dev = _safe_float(latest_context.iloc[-1].get("resp_rate_dev"))
            out["resp_rate_dev"] = resp_dev
            if resp_dev is not None and abs(resp_dev) >= 0.5:
                out["context_alignment"] = "SUPPORTS_STRAIN"
                out["context_note"] = f"Respiratory rate is also shifted ({resp_dev:+.1f}/min), so the strain signal looks more real."
            elif resp_dev is not None and out.get("pattern") == "ONE_NIGHT_OUTLIER":
                out["context_alignment"] = "CAUTION"
                out["context_note"] = "Respiratory rate is steady, so treat a one-night spike more cautiously."
    out["summary"] = describe_signal_trust(out)
    return out


def metric_label(key: str) -> str:
    return {
        "hrv_rmssd": "HRV rmSSD",
        "resting_hr": "Resting HR",
        "sleep_score": "Sleep score",
        "readiness": "Readiness",
    }.get(key, key)


def metric_unit(key: str) -> str:
    return {
        "hrv_rmssd": " ms",
        "resting_hr": " bpm",
    }.get(key, "")


def format_metric_value(key: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    unit = metric_unit(key)
    decimals = 1 if key in {"hrv_rmssd"} else 0
    return f"{value:.{decimals}f}{unit}"


def format_delta(value: Optional[float], unit: str = "", decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:+.{decimals}f}{unit}"


def favorable_signal_z(signal: MetricSignal, *, higher_is_better: bool) -> Optional[float]:
    z_value = _safe_float(signal.get("z28"))
    if z_value is None:
        return None
    return z_value if higher_is_better else -z_value


def compute_peer_recovery_index(
    *,
    hrv_signal: MetricSignal,
    rhr_signal: MetricSignal,
    sleep_signal: MetricSignal,
    readiness_signal: MetricSignal,
) -> Optional[float]:
    z_values = [
        favorable_signal_z(hrv_signal, higher_is_better=True),
        favorable_signal_z(rhr_signal, higher_is_better=False),
        favorable_signal_z(sleep_signal, higher_is_better=True),
        favorable_signal_z(readiness_signal, higher_is_better=True),
    ]
    usable = [max(-2.5, min(2.5, float(z))) for z in z_values if z is not None]
    if not usable:
        return None
    return float(max(0.0, min(100.0, 50.0 + 10.0 * (sum(usable) / len(usable)))))


def build_peer_comparison_snapshot(
    *,
    account_label: str,
    token: str,
    start_d: date,
    end_d: date,
    analysis_start_d: date,
    wide_sparse_days: int,
) -> Dict[str, Any]:
    data = fetch_endpoints(token, start_d=start_d, end_d=end_d, wide_sparse_days=wide_sparse_days)
    analysis_data = data if analysis_start_d >= start_d else fetch_endpoints(
        token,
        start_d=analysis_start_d,
        end_d=end_d,
        wide_sparse_days=wide_sparse_days,
    )
    daily = compute_daily_frame(data)
    analysis_daily = compute_daily_frame(analysis_data)
    personal = data.get("personal", {}).get("doc", {}) if isinstance(data.get("personal", {}).get("doc", {}), dict) else {}
    display_name = str(personal.get("first_name") or account_label or "Friend").strip() or "Friend"

    if daily.empty:
        return {
            "label": account_label,
            "display_name": display_name,
            "error": "No daily data available in the selected range.",
        }

    hrv_signal = compute_metric_signal(daily, "hrv_rmssd", higher_is_better=True)
    rhr_signal = compute_metric_signal(daily, "resting_hr", higher_is_better=False)
    sleep_signal = compute_metric_signal(daily, "sleep_score", higher_is_better=True)
    readiness_signal = compute_metric_signal(daily, "readiness", higher_is_better=True)
    activity_signal = compute_metric_signal(daily, "activity_score", higher_is_better=True)

    hrv_trust = compute_biomarker_trust(analysis_daily, "hrv_rmssd", higher_is_better=True)
    rhr_trust = compute_biomarker_trust(analysis_daily, "resting_hr", higher_is_better=False)
    rhr_trend = rhr_trend_slope(analysis_daily, window_days=90)
    rhr_stability_stats = rhr_stability(analysis_daily, window=28)
    hrv_pattern = hrv_pattern_analysis(analysis_daily, window=90)
    vo2_value, vo2_df = compute_vo2(data)
    vo2_trend = vo2_trend_summary(vo2_df)

    compare_index = compute_peer_recovery_index(
        hrv_signal=hrv_signal,
        rhr_signal=rhr_signal,
        sleep_signal=sleep_signal,
        readiness_signal=readiness_signal,
    )
    if compare_index is None:
        baseline_summary = "Not enough baseline history yet."
    elif compare_index >= 58:
        baseline_summary = "Above personal normal."
    elif compare_index <= 42:
        baseline_summary = "Below personal normal."
    else:
        baseline_summary = "Near personal normal."

    return {
        "label": account_label,
        "display_name": display_name,
        "hrv_signal": hrv_signal,
        "rhr_signal": rhr_signal,
        "sleep_signal": sleep_signal,
        "readiness_signal": readiness_signal,
        "activity_signal": activity_signal,
        "hrv_trust": hrv_trust,
        "rhr_trust": rhr_trust,
        "rhr_trend": rhr_trend,
        "rhr_stability": rhr_stability_stats,
        "hrv_pattern": hrv_pattern,
        "vo2_value": vo2_value,
        "vo2_trend": vo2_trend,
        "compare_index": compare_index,
        "baseline_summary": baseline_summary,
    }


def interpret_metric_signal(key: str, signal: Dict[str, Any]) -> str:
    status = str(signal.get("signal") or "UNKNOWN")
    if signal.get("value") is None:
        return "No signal yet."

    if key == "hrv_rmssd":
        if status == "GREEN":
            return "Autonomic recovery is above your recent baseline."
        if status == "AMBER":
            return "HRV is in your normal range."
        return "HRV is suppressed versus your baseline."

    if key == "resting_hr":
        if status == "GREEN":
            return "Cardiac strain is lower than usual."
        if status == "AMBER":
            return "Resting HR is close to baseline."
        return "Resting HR is elevated versus baseline."

    if key == "sleep_score":
        if status == "GREEN":
            return "Sleep is supporting adaptation."
        if status == "AMBER":
            return "Sleep is acceptable, but not buying extra performance."
        return "Sleep is limiting recovery today."

    if key == "readiness":
        if status == "GREEN":
            return "General recovery signal is supportive."
        if status == "AMBER":
            return "General recovery looks average."
        return "General recovery is lagging."

    return "Signal available."


def determine_day_mode(
    hrv_signal: MetricSignal,
    rhr_signal: MetricSignal,
    readiness_signal: MetricSignal,
    sleep_signal: MetricSignal,
) -> DayMode:
    score = 0
    reasons: List[str] = []

    for signal, good_reason, bad_reason, weight in [
        (hrv_signal, "HRV is supportive", "HRV is suppressed", 2),
        (rhr_signal, "Resting HR is controlled", "Resting HR is elevated", 2),
    ]:
        if signal.get("signal") == "GREEN":
            score += weight
            reasons.append(good_reason)
        elif signal.get("signal") == "RED":
            score -= weight
            reasons.append(bad_reason)

    readiness_value = _safe_float(readiness_signal.get("value"))
    if readiness_value is not None:
        if readiness_value >= 82:
            score += 1
            reasons.append("Readiness score is strong")
        elif readiness_value < 68:
            score -= 1
            reasons.append("Readiness score is low")

    sleep_value = _safe_float(sleep_signal.get("value"))
    if sleep_value is not None:
        if sleep_value >= 82:
            score += 1
            reasons.append("Sleep score is supportive")
        elif sleep_value < 70:
            score -= 1
            reasons.append("Sleep score is dragging")

    if score >= 4:
        mode = "PUSH"
        summary = "Best window for high-value work if it is already in the plan."
    elif score >= 1:
        mode = "BUILD"
        summary = "Train normally, but keep execution disciplined."
    elif score >= -1:
        mode = "MAINTAIN"
        summary = "Hold the line. Build consistency, not heroics."
    else:
        mode = "RECOVER"
        summary = "Use today to move HRV up and resting HR down."

    if not reasons:
        reasons = ["Recovery signal incomplete"]

    return {
        "mode": mode,
        "summary": summary,
        "reasons": reasons[:3],
    }


def build_training_prescription(goal: str, mode: str, zones: List[Tuple[str, float, float]]) -> TrainingPrescription:
    z1 = zone_bounds_by_name(zones, "Z1")
    z2 = zone_bounds_by_name(zones, "Z2")
    z3 = zone_bounds_by_name(zones, "Z3")
    z4 = zone_bounds_by_name(zones, "Z4")
    z1_hi = z1[1] if z1 else None
    z2_text = f"{z2[0]}–{z2[1]} bpm" if z2 else "Z2"
    z2_hi = z2[1] if z2 else None
    z3_text = f"{z3[0]}–{z3[1]} bpm" if z3 else "Z3"
    z4_lo = z4[0] if z4 else None

    if goal.startswith("Performance (endurance)"):
        if mode == "PUSH":
            return {
                "session": "Key quality day",
                "hr_target": f"Base work {z2_text}",
                "instruction": f"If intervals are planned, let hard reps reach at least {z4_lo} bpm and recover back near Z2 between reps." if z4_lo is not None else "If intervals are planned, use clear work/recovery separation.",
                "avoid": "Do not turn easy volume into grey-zone drift.",
            }
        if mode == "BUILD":
            return {
                "session": "Aerobic build",
                "hr_target": z2_text,
                "instruction": "Make today a clean aerobic session with stable pacing from the first 10 minutes.",
                "avoid": "Avoid threshold creep unless it is explicitly programmed.",
            }
        if mode == "MAINTAIN":
            return {
                "session": "Easy aerobic",
                "hr_target": f"Stay at or below {z2_hi} bpm" if z2_hi is not None else "Stay in Z1-Z2",
                "instruction": "Use easy volume to preserve momentum without adding hidden fatigue.",
                "avoid": "Skip ego pacing and late-session surges.",
            }
        return {
            "session": "Recovery only",
            "hr_target": f"Stay at or below {z1_hi} bpm" if z1_hi is not None else "Stay in Z1",
            "instruction": "Walk, spin, or jog only if it helps recovery; otherwise rest.",
            "avoid": "No intensity today.",
        }

    if goal.startswith("Performance (strength"):
        if mode == "PUSH":
            return {
                "session": "Heavy strength or short quality conditioning",
                "hr_target": f"Easy work <= {z2_hi} bpm; hard reps >= {z4_lo} bpm" if z2_hi is not None and z4_lo is not None else "Keep easy work easy and hard work distinct",
                "instruction": "Separate conditioning from lifting quality. Preserve bar speed.",
                "avoid": "Do not bury the session in medium-hard fatigue.",
            }
        if mode == "BUILD":
            return {
                "session": "Moderate-heavy strength + short aerobic flush",
                "hr_target": f"Aerobic work {z2_text}",
                "instruction": "Get the main strength work done, then leave with something in the tank.",
                "avoid": "Skip hard conditioning if lifting quality is down.",
            }
        if mode == "MAINTAIN":
            return {
                "session": "Technique + easy conditioning",
                "hr_target": f"Stay at or below {z2_hi} bpm" if z2_hi is not None else "Stay in Z1-Z2",
                "instruction": "Maintain movement quality and reduce system load.",
                "avoid": "No grinders and no junk conditioning.",
            }
        return {
            "session": "Mobility / walk / recovery circuit",
            "hr_target": f"Stay at or below {z1_hi} bpm" if z1_hi is not None else "Stay in Z1",
            "instruction": "Move enough to recover; do not chase performance today.",
            "avoid": "No max effort work.",
        }

    if goal.startswith("Stress resilience"):
        if mode in {"PUSH", "BUILD"}:
            return {
                "session": "Calm aerobic work",
                "hr_target": z2_text,
                "instruction": "Train in a way that leaves the nervous system quieter than it started.",
                "avoid": f"Avoid long Z3 blocks and late-day work above {z4_lo} bpm." if z4_lo is not None else "Avoid late-day high intensity.",
            }
        return {
            "session": "Downregulation session",
            "hr_target": f"Stay at or below {z1_hi} bpm" if z1_hi is not None else "Stay in Z1",
            "instruction": "Use walking, nasal breathing, and mobility to lift HRV tomorrow.",
            "avoid": "No intensity unless recovery metrics normalize.",
        }

    if goal.startswith("Body composition"):
        if mode == "PUSH":
            return {
                "session": "Planned strength or interval day",
                "hr_target": f"Base work {z2_text}; hard work >= {z4_lo} bpm" if z4_lo is not None else z2_text,
                "instruction": "Earn the hard session with recovery metrics, then keep the rest of the day easy.",
                "avoid": "Do not stack hunger-inducing junk intensity on top.",
            }
        if mode == "BUILD":
            return {
                "session": "Strength + aerobic support",
                "hr_target": z2_text,
                "instruction": "Use controlled volume to keep energy expenditure high without flattening recovery.",
                "avoid": "Avoid medium-hard slogs.",
            }
        if mode == "MAINTAIN":
            return {
                "session": "Easy aerobic or steps emphasis",
                "hr_target": f"Stay at or below {z2_hi} bpm" if z2_hi is not None else "Stay in Z1-Z2",
                "instruction": "Keep output up through steps and easy work, not strain.",
                "avoid": "Do not chase calorie burn with fatigue.",
            }
        return {
            "session": "Recovery + steps only",
            "hr_target": f"Stay at or below {z1_hi} bpm" if z1_hi is not None else "Stay in Z1",
            "instruction": "Recover first so appetite, sleep, and training quality stay under control.",
            "avoid": "Skip interval work today.",
        }

    if mode == "PUSH":
        return {
            "session": "Long aerobic or controlled quality",
            "hr_target": z2_text,
            "instruction": "For healthspan, the default is still controlled aerobic work with excellent recovery afterward.",
            "avoid": "Use hard intervals sparingly.",
        }
    if mode == "BUILD":
        return {
            "session": "Z2 healthspan session",
            "hr_target": z2_text,
            "instruction": "Build repeatable aerobic volume and keep the autonomic system stable.",
            "avoid": f"Avoid long blocks in {z3_text}." if z3 is not None else "Avoid medium-hard drift.",
        }
    if mode == "MAINTAIN":
        return {
            "session": "Easy aerobic maintenance",
            "hr_target": f"Stay at or below {z2_hi} bpm" if z2_hi is not None else "Stay in Z1-Z2",
            "instruction": "Keep the habit and keep recovery intact.",
            "avoid": "No unnecessary intensity.",
        }
    return {
        "session": "Recovery walk / mobility",
        "hr_target": f"Stay at or below {z1_hi} bpm" if z1_hi is not None else "Stay in Z1",
        "instruction": "Lower system stress first, then resume normal work once the signals improve.",
        "avoid": "No hard training.",
    }


def hrv_action_lines(goal: str, mode: str, zones: List[Tuple[str, float, float]], signal: Dict[str, Any]) -> List[str]:
    z1 = zone_bounds_by_name(zones, "Z1")
    z2 = zone_bounds_by_name(zones, "Z2")
    easy_cap = z1[1] if mode == "RECOVER" and z1 else (z2[1] if z2 else None)
    easy_text = f"<= {easy_cap} bpm" if easy_cap is not None else "easy Z1-Z2"
    if signal.get("signal") == "RED":
        return [
            f"Today: keep cardio {easy_text} and skip hard work.",
            "Tonight: 8+ hours in bed, no alcohol, and keep dinner earlier/lighter.",
            "Use 10–20 minutes of slow breathing or NSDR to push the autonomic system back up.",
        ]
    if signal.get("signal") == "GREEN":
        first = "Use the strong recovery signal for a quality session." if goal.startswith("Performance") and mode == "PUSH" else "Recovery signal is supportive today; keep execution clean."
        return [
            first,
            "Protect the gain with a consistent bedtime and low late-night stimulation.",
            "Do not waste a good day by turning easy work into random medium-hard work.",
        ]
    return [
        f"Train normally, but keep most work {easy_text}.",
        "Protect sleep quality tonight if you want HRV to trend higher tomorrow.",
        "Hydrate and refuel after training so recovery is not still suppressed tomorrow.",
    ]


def rhr_action_lines(mode: str, zones: List[Tuple[str, float, float]], signal: Dict[str, Any]) -> List[str]:
    z2 = zone_bounds_by_name(zones, "Z2")
    if signal.get("signal") == "RED":
        return [
            "Treat elevated resting HR as strain. Cut planned volume 20–40%.",
            "Front-load hydration and electrolytes; dehydration and heat show up here early.",
            "Check the common drivers: short sleep, alcohol, illness, late meals, or accumulated fatigue.",
        ]
    if signal.get("signal") == "GREEN":
        return [
            "Low resting HR suggests low cardiac strain.",
            f"Translate it into efficient work: keep easy sessions at or below {z2[1]} bpm." if z2 else "Translate it into efficient work by keeping easy sessions easy.",
            "Preserve it with consistent sleep and low late-night stimulation.",
        ]
    return [
        "Resting HR is near baseline. Stay disciplined and look for the next few days, not a single datapoint.",
        "If HRV also softens, pivot the day toward recovery quickly.",
        "Small daily improvements usually come from sleep timing, hydration, and lower hidden stress.",
    ]


def vo2_action_lines(goal: str, mode: str, zones: List[Tuple[str, float, float]]) -> List[str]:
    z2 = zone_bounds_by_name(zones, "Z2")
    z4 = zone_bounds_by_name(zones, "Z4")
    z2_text = f"{z2[0]}–{z2[1]} bpm" if z2 else "Z2"
    if goal.startswith("Stress resilience"):
        return [
            f"VO2 max is secondary right now. Keep most work in {z2_text}.",
            "Earn harder intervals by stacking 7–14 days of stable HRV and resting HR.",
            "If higher intensity worsens sleep, it is costing more than it is buying.",
        ]
    if goal.startswith("Performance (strength"):
        return [
            "VO2 improves fastest when conditioning is its own session, not random fatigue after lifting.",
            f"Build the base with {z2_text}; add short intervals only on PUSH or BUILD days.",
            "Keep at least 48 hours between the hardest conditioning bouts if lower-body strength matters.",
        ]
    return [
        f"Base first: accumulate 2–4 weekly sessions in {z2_text}.",
        f"Intensity second: 1 hard session per week on PUSH or BUILD days, with work reps reaching at least {z4[0]} bpm." if z4 else "Intensity second: add one hard session per week only when recovery is supportive.",
        "Progress one variable per week: either total aerobic minutes or interval reps, not both.",
    ]


def render_nightly_metric_tab(
    *,
    daily: pd.DataFrame,
    key: str,
    label: str,
    higher_is_better: bool,
    goal: str,
    mode: str,
    zones: List[Tuple[str, float, float]],
    trust: Optional[Dict[str, Any]] = None,
) -> None:
    if daily.empty or key not in daily.columns or not daily[key].notna().any():
        st.info(f"No {label} data available in the selected range.")
        return

    m = daily[["day", key]].copy()
    m["day"] = pd.to_datetime(m["day"], errors="coerce")
    m[key] = pd.to_numeric(m[key], errors="coerce")
    m = m.dropna(subset=["day", key]).sort_values("day")
    if m.empty:
        st.info(f"No {label} data available in the selected range.")
        return

    signal = compute_metric_signal(daily, key, higher_is_better=higher_is_better)
    last_day = m.iloc[-1]["day"]
    last_value = _safe_float(m.iloc[-1][key])
    prior7 = signal.get("baseline7")
    prior28 = signal.get("baseline28")
    delta7 = signal.get("delta7")
    delta28 = signal.get("delta28")
    unit = metric_unit(key)
    decimals = 1 if key in {"hrv_rmssd", "resting_hr"} else 0
    delta_color = "normal" if higher_is_better else "inverse"

    st.subheader(f"{label} analysis")
    st.caption(f"Latest nightly value: {last_day.date().isoformat()}. Comparison baselines exclude last night so you can compare last night directly against your recent normal.")
    st.caption("Averages use only nights with available data in the selected range; missing nights are ignored rather than treated as zero.")
    if key == "resting_hr":
        st.caption("Using Oura sleep data `lowest_heart_rate` as the nightly resting-HR value for this account.")
    elif key == "hrv_rmssd":
        st.caption("Using Oura sleep data `average_hrv` as the nightly HRV value for this account.")
    if trust:
        st.caption(f"Trust layer: {describe_signal_trust(trust)}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Last night", format_metric_value(key, last_value))
    c2.metric("Prior 7-night avg", format_metric_value(key, _safe_float(prior7)))
    c3.metric("Δ vs prior 7", format_delta(_safe_float(delta7), unit, decimals), delta_color=delta_color)
    c4.metric("Prior 28-night avg", format_metric_value(key, _safe_float(prior28)))
    c5.metric("Δ vs prior 28", format_delta(_safe_float(delta28), unit, decimals), delta_color=delta_color)
    c6.metric("Signal", str(signal.get("signal") or "UNKNOWN"))
    if trust:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Valid nights (7)", _fmt(trust.get("n_valid_7")))
        t2.metric("Valid nights (28)", _fmt(trust.get("n_valid_28")))
        t3.metric("Confidence", _fmt(trust.get("confidence")))
        t4.metric("Pattern", _fmt(str(trust.get("pattern") or "").replace("_", " ").title()))

    st.write(f"- {interpret_metric_signal(key, signal)}")
    if key == "resting_hr":
        st.write("- Lower than your prior average is better. Higher usually means more strain, worse recovery, or an external stressor.")
        for line in rhr_action_lines(mode, zones, signal):
            st.write(f"- {line}")
    elif key == "hrv_rmssd":
        st.write("- Higher than your prior average is better. Lower usually means suppressed recovery or higher stress load.")
        for line in hrv_action_lines(goal, mode, zones, signal):
            st.write(f"- {line}")

    chart = m.copy()
    chart["roll7"] = chart[key].rolling(7, min_periods=3).mean().shift(1)
    chart["roll28"] = chart[key].rolling(28, min_periods=7).mean().shift(1)
    chart["last_night"] = chart["day"] == chart["day"].max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart["day"], y=chart[key], mode="lines+markers", name=label))
    if chart["roll7"].notna().any():
        fig.add_trace(go.Scatter(x=chart["day"], y=chart["roll7"], mode="lines", name="Prior 7-night avg"))
    if chart["roll28"].notna().any():
        fig.add_trace(go.Scatter(x=chart["day"], y=chart["roll28"], mode="lines", name="Prior 28-night avg"))
    last_points = chart[chart["last_night"]]
    if not last_points.empty:
        fig.add_trace(
            go.Scatter(
                x=last_points["day"],
                y=last_points[key],
                mode="markers",
                name="Last night",
                marker=dict(size=12),
            )
        )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, width="stretch")

    cmp = chart.copy()
    cmp["Prior 7-night avg"] = cmp["roll7"]
    cmp["Prior 28-night avg"] = cmp["roll28"]
    cmp["Δ vs prior 7"] = cmp[key] - cmp["Prior 7-night avg"]
    cmp["Δ vs prior 28"] = cmp[key] - cmp["Prior 28-night avg"]
    cmp["day"] = cmp["day"].dt.date.astype(str)
    show = cmp.tail(10).copy()
    show = show.rename(columns={"day": "Night", key: label})
    for col in [label, "Prior 7-night avg", "Prior 28-night avg", "Δ vs prior 7", "Δ vs prior 28"]:
        show[col] = pd.to_numeric(show[col], errors="coerce").round(1)
    st.markdown("#### Recent-night comparison")
    st.dataframe(show[["Night", label, "Prior 7-night avg", "Δ vs prior 7", "Prior 28-night avg", "Δ vs prior 28"]].sort_values("Night", ascending=False), width="stretch")


def render_longevity_score_panel(
    score: Dict[str, Any],
    *,
    compact: bool = False,
    action_plan: Optional[Dict[str, Any]] = None,
) -> None:
    if not score or score.get("score") is None:
        return
    if action_plan is None and "easiest_points_to_gain" in score:
        action_plan = {
            "delta_30d": score.get("delta_30d"),
            "easiest_points": score.get("easiest_points_to_gain"),
            "fastest_lever": score.get("fastest_lever"),
            "slowest_lever": score.get("slowest_lever"),
            "expected_30d_upside": score.get("expected_30d_upside"),
        }
    st.markdown("### Decision score")
    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{int(score['score'])}/100")
    c2.metric("30d change", _fmt(None if not action_plan or action_plan.get("delta_30d") is None else f"{action_plan['delta_30d']:+.0f}"))
    c3.metric("Fastest lever", _fmt(None if not action_plan else action_plan.get("fastest_lever")))
    if not compact:
        rows = []
        for key, data in score.get("breakdown", {}).items():
            context = data.get("category")
            if context is None:
                for alt in [data.get("slope"), data.get("cv")]:
                    if alt is not None:
                        context = alt
                        break
            rows.append(
                {
                    "Pillar": key,
                    "Score": f"{data.get('score', 0)}/{data.get('max', 0)}",
                    "Context": "—" if context is None else str(context),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if action_plan:
        if action_plan.get("fastest_lever"):
            st.write(f"- **Fastest lever:** {action_plan['fastest_lever']}")
        easiest = action_plan.get("easiest_points") or []
        if easiest:
            st.write(f"- **Easiest points to gain:** {'; '.join(str(x) for x in easiest[:2])}")
        if action_plan.get("expected_30d_upside"):
            st.write(f"- **Expected 30-day upside:** {action_plan['expected_30d_upside']}")
        if action_plan.get("slowest_lever"):
            st.write(f"- **Slowest lever:** {action_plan['slowest_lever']}")
    if score.get("primary_lever"):
        st.write(f"- **Primary lever:** {score.get('primary_lever')}")
    if not compact and score.get("grade"):
        st.caption(f"Legacy grade: {score['grade']}")


def render_rhr_longevity_panel(
    *,
    rhr_trend: Dict[str, Any],
    rhr_stability_stats: Dict[str, Any],
    rhr_recovery: Dict[str, Any],
) -> None:
    st.markdown("### Longevity lens")
    if rhr_trend.get("error") and not rhr_stability_stats and not rhr_recovery:
        st.info(str(rhr_trend.get("error")))
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trend slope", _fmt(None if rhr_trend.get("slope_bpm_per_month") is None else f"{rhr_trend['slope_bpm_per_month']:+.2f} bpm/mo"))
    c2.metric("12mo projection", _fmt(None if rhr_trend.get("projected_12mo") is None else f"{rhr_trend['projected_12mo']:.1f} bpm"))
    c3.metric("Years to strong band", _fmt(rhr_trend.get("years_to_top10_band")))
    c4.metric("28d CV", _fmt(None if rhr_stability_stats.get("cv_percent") is None else f"{rhr_stability_stats['cv_percent']:.1f}%"))
    c5.metric("Day-1 recovery", _fmt(None if rhr_recovery.get("avg_pct_recovered_day1") is None else f"{rhr_recovery['avg_pct_recovered_day1']:.0f}%"))
    if rhr_trend.get("interpretation"):
        st.write(f"- {rhr_trend['interpretation']}")
    if rhr_stability_stats.get("interpretation"):
        st.write(f"- {rhr_stability_stats['interpretation']}")
    if rhr_recovery.get("interpretation"):
        st.write(f"- {rhr_recovery['interpretation']}")


def render_hrv_longevity_panel(
    *,
    hrv_age: Dict[str, Any],
    hrv_pattern: Dict[str, Any],
) -> None:
    st.markdown("### Longevity lens")
    if not hrv_age and hrv_pattern.get("error"):
        st.info(str(hrv_pattern.get("error")))
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("HRV age ref", _fmt(hrv_age.get("hrv_biological_age")))
    c2.metric("Age delta", _fmt(None if hrv_age.get("delta_years") is None else f"{hrv_age['delta_years']:+.1f} y"))
    c3.metric("Trend slope", _fmt(None if hrv_pattern.get("trend_slope_ms_per_month") is None else f"{hrv_pattern['trend_slope_ms_per_month']:+.2f} ms/mo"))
    c4.metric("7d CV", _fmt(None if hrv_pattern.get("cv_7d_current") is None else f"{hrv_pattern['cv_7d_current']:.1f}%"))
    c5.metric("Residual noise", _fmt(None if hrv_pattern.get("residual_volatility_ms") is None else f"{hrv_pattern['residual_volatility_ms']:.1f} ms"))
    if hrv_age.get("interpretation"):
        st.write(f"- {hrv_age['interpretation']}")
    if hrv_pattern.get("interpretation"):
        st.write(f"- {hrv_pattern['interpretation']}")
    weekly_pattern = hrv_pattern.get("weekly_pattern") or {}
    if weekly_pattern:
        wp = pd.DataFrame({"day": list(weekly_pattern.keys()), "hrv": list(weekly_pattern.values())})
        fig = px.bar(wp, x="day", y="hrv", title="Average HRV by day of week")
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, width="stretch")
        st.write(f"- Best recovery day: **{_fmt(hrv_pattern.get('best_recovery_day'))}**")
        st.write(f"- Worst recovery day: **{_fmt(hrv_pattern.get('worst_recovery_day'))}**")


def render_vo2_longevity_panel(
    *,
    vo2_analysis: Dict[str, Any],
    vo2_decay: Optional[str],
) -> None:
    st.markdown("### Longevity lens")
    if vo2_analysis.get("error"):
        st.info(str(vo2_analysis.get("error")))
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Risk category", _fmt(vo2_analysis.get("risk_category")))
    c2.metric("Next target", _fmt(None if vo2_analysis.get("next_target") is None else f"{vo2_analysis['next_target']:.0f}"))
    c3.metric("Target elite", _fmt(None if vo2_analysis.get("target_elite") is None else f"{vo2_analysis['target_elite']:.0f}"))
    c4.metric("Gap to next target", _fmt(None if vo2_analysis.get("gap_to_next_target") is None else f"{vo2_analysis['gap_to_next_target']:.1f}"))
    c5.metric("VO2 bio age", _fmt(vo2_analysis.get("vo2_biological_age")))
    if vo2_analysis.get("bio_age_delta") is not None:
        st.write(f"- VO2 biological age delta: **{vo2_analysis['bio_age_delta']:+.1f} years** vs chronological.")
    if vo2_analysis.get("rate_per_year") is not None:
        st.write(f"- Current improvement rate: **{vo2_analysis['rate_per_year']:+.1f} ml/kg/min per year**.")
    if vo2_analysis.get("months_to_target_estimate") is not None:
        st.write(f"- Estimated time to next target at your observed pace: **{vo2_analysis['months_to_target_estimate']:.0f} months**.")
    else:
        st.write("- Time-to-target is only shown when there are enough repeated VO₂ estimates to support a real trend.")
    if vo2_decay:
        st.warning(vo2_decay)
    for line in vo2_analysis.get("prescription", [])[:5]:
        st.write(f"- {line}")


def render_biomarker_card(
    title: str,
    *,
    acute: str,
    current: str,
    median28: str,
    slope90: str,
    stability: str,
    target: str,
    action: str,
    trust: Optional[Dict[str, Any]] = None,
) -> None:
    st.markdown(f"### {title}")
    st.caption("Acute / Trend / Stability / Prescription")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current", current)
    c2.metric("28d median", median28)
    c3.metric("90d slope", slope90)
    c4, c5, c6 = st.columns(3)
    c4.metric("Stability", stability)
    c5.metric("Target", target)
    c6.metric("Confidence", _fmt((trust or {}).get("confidence")))
    st.write(f"- **Acute state:** {acute}")
    st.write(f"- **Next best action:** {action}")
    if trust:
        st.write(
            f"- **Trust layer:** {trust.get('n_valid_28', trust.get('valid_points', 0))}/28 valid points, "
            f"{_fmt(trust.get('missing_pct_28', trust.get('missing_pct')))}% missing, source `{trust.get('source_used')}`."
        )
        if trust.get("pattern"):
            st.write(f"- **Pattern:** {str(trust['pattern']).replace('_', ' ').title()}")
        if trust.get("context_note"):
            st.write(f"- **Context:** {trust['context_note']}")


def render_biomarker_weekly_summary(
    *,
    rhr_trend: Dict[str, Any],
    hrv_pattern: Dict[str, Any],
    vo2_trend: Dict[str, Any],
    efficiency: Dict[str, Any],
    training_link: Dict[str, Any],
) -> None:
    improving: List[str] = []
    flat: List[str] = []
    deteriorating: List[str] = []

    rhr_slope = _safe_float(rhr_trend.get("slope_bpm_per_month"))
    if rhr_slope is not None:
        (improving if rhr_slope < -0.1 else deteriorating if rhr_slope > 0.1 else flat).append("Resting HR")
    hrv_slope = _safe_float(hrv_pattern.get("trend_slope_ms_per_month"))
    if hrv_slope is not None:
        (improving if hrv_slope > 0.3 else deteriorating if hrv_slope < -0.3 else flat).append("HRV")
    vo2_slope = _safe_float(vo2_trend.get("slope_90d"))
    if vo2_slope is not None:
        (improving if vo2_slope > 0.4 else deteriorating if vo2_slope < -0.4 else flat).append("VO₂max")
    eff_slope = _safe_float(efficiency.get("slope_pct_per_month"))
    if eff_slope is not None:
        (improving if eff_slope < -0.3 else deteriorating if eff_slope > 0.3 else flat).append("Aerobic efficiency")

    st.markdown("### This week")
    c1, c2, c3 = st.columns(3)
    c1.write(f"- **Improving:** {', '.join(improving) if improving else 'None clearly improving'}")
    c2.write(f"- **Flat:** {', '.join(flat) if flat else 'None clearly flat'}")
    c3.write(f"- **Deteriorating:** {', '.join(deteriorating) if deteriorating else 'None clearly deteriorating'}")
    if training_link.get("next_action"):
        st.write(f"- **What to do next week:** {training_link['next_action']}")


def render_integrated_biomarker_view(
    *,
    hrv_signal: MetricSignal,
    rhr_signal: MetricSignal,
    hrv_trust: Dict[str, Any],
    rhr_trust: Dict[str, Any],
    hrv_pattern: Dict[str, Any],
    rhr_trend: Dict[str, Any],
    rhr_stability_stats: Dict[str, Any],
    rhr_recovery: Dict[str, Any],
    vo2_analysis: Dict[str, Any],
    vo2_trend: Dict[str, Any],
    vo2_trust: Dict[str, Any],
    efficiency: Dict[str, Any],
    efficiency_trust: Dict[str, Any],
    training_link: Dict[str, Any],
    diagnostic_chips: Optional[List[str]] = None,
) -> None:
    st.subheader("Biomarker operating system")
    st.caption("This is the weekly center of gravity: acute state, long trend, stability, and the next best action for each biomarker pillar.")
    if diagnostic_chips:
        st.markdown("### Diagnostic context")
        st.write(" | ".join(diagnostic_chips))
    render_biomarker_weekly_summary(
        rhr_trend=rhr_trend,
        hrv_pattern=hrv_pattern,
        vo2_trend=vo2_trend,
        efficiency=efficiency,
        training_link=training_link,
    )

    row1 = st.columns(2)
    with row1[0]:
        recovery_text = _fmt(None if rhr_recovery.get("avg_pct_recovered_day1") is None else f"{rhr_recovery['avg_pct_recovered_day1']:.0f}% by day 1")
        render_biomarker_card(
            "Resting HR",
            acute=f"{format_metric_value('resting_hr', _safe_float(rhr_signal.get('value')))} vs {format_metric_value('resting_hr', _safe_float(rhr_signal.get('baseline28')))} baseline ({rhr_signal.get('signal', 'UNKNOWN')})",
            current=format_metric_value("resting_hr", _safe_float(rhr_signal.get("value"))),
            median28=format_metric_value("resting_hr", _safe_float(rhr_signal.get("baseline28"))),
            slope90=_fmt(None if rhr_trend.get("slope_bpm_per_month") is None else f"{rhr_trend['slope_bpm_per_month']:+.2f} bpm/mo"),
            stability=_fmt(None if rhr_stability_stats.get("cv_percent") is None else f"{rhr_stability_stats['cv_percent']:.1f}% CV"),
            target="Downward or stable trend, long-term toward a low personal resting-HR band",
            action=(
                "Add 1-2 strict Z2 sessions this week and lock bedtime within your personal window."
                if _safe_float(rhr_trend.get("slope_bpm_per_month")) is not None and _safe_float(rhr_trend.get("slope_bpm_per_month")) > 0
                else "Protect the low trend: keep hydration, sleep timing, and easy-day discipline."
            ),
            trust=rhr_trust,
        )
        st.write(f"- **Post-training recovery:** {recovery_text}")
    with row1[1]:
        residual_text = _fmt(None if hrv_pattern.get("residual_volatility_ms") is None else f"{hrv_pattern['residual_volatility_ms']:.1f} ms")
        render_biomarker_card(
            "HRV",
            acute=f"{format_metric_value('hrv_rmssd', _safe_float(hrv_signal.get('value')))} vs {format_metric_value('hrv_rmssd', _safe_float(hrv_signal.get('baseline28')))} baseline ({hrv_signal.get('signal', 'UNKNOWN')})",
            current=format_metric_value("hrv_rmssd", _safe_float(hrv_signal.get("value"))),
            median28=format_metric_value("hrv_rmssd", _safe_float(hrv_signal.get("baseline28"))),
            slope90=_fmt(None if hrv_pattern.get("trend_slope_ms_per_month") is None else f"{hrv_pattern['trend_slope_ms_per_month']:+.2f} ms/mo"),
            stability=_fmt(None if hrv_pattern.get("cv_7d_current") is None else f"{hrv_pattern['cv_7d_current']:.1f}% CV"),
            target="Stable-to-rising trend, 7d CV < 12%",
            action=(
                "Tighten bedtime drift, keep late meals/alcohol out, and hold intensity to Z1-Z2 until variability settles."
                if _safe_float(hrv_pattern.get("cv_7d_current")) is not None and _safe_float(hrv_pattern.get("cv_7d_current")) >= 12
                else "Keep the trend steady by protecting sleep timing and avoiding random medium-hard work."
            ),
            trust=hrv_trust,
        )
        st.write(f"- **Residual volatility:** {residual_text}")

    row2 = st.columns(2)
    with row2[0]:
        render_biomarker_card(
            "VO₂max",
            acute=f"{_fmt(vo2_analysis.get('risk_category'))} category, current {_fmt(vo2_analysis.get('vo2max'))}",
            current=_fmt(None if vo2_trend.get("current") is None else f"{vo2_trend['current']:.1f}"),
            median28=_fmt(None if vo2_trend.get("median28") is None else f"{vo2_trend['median28']:.1f}"),
            slope90=_fmt(None if vo2_trend.get("slope_90d") is None else f"{vo2_trend['slope_90d']:+.2f} /90d"),
            stability=_fmt(None if vo2_trend.get("stability_spread") is None else f"{vo2_trend['stability_spread']:.1f} spread"),
            target=_fmt(None if vo2_analysis.get("target_above_average") is None else f">= {vo2_analysis['target_above_average']:.0f} next"),
            action=(vo2_analysis.get("prescription") or ["Build Z2 volume and add one interval session when recovery allows."])[0],
            trust=vo2_trust,
        )
        st.write(f"- **Months to next target:** {_fmt(vo2_analysis.get('months_to_target_estimate'))}")
    with row2[1]:
        render_biomarker_card(
            "Aerobic efficiency",
            acute=_fmt(None if efficiency.get("current_drift") is None else f"Latest easy-session drift {efficiency['current_drift']:.1f}%"),
            current=_fmt(None if efficiency.get("current_drift") is None else f"{efficiency['current_drift']:.1f}% drift"),
            median28=_fmt(None if efficiency.get("median28_drift") is None else f"{efficiency['median28_drift']:.1f}% drift"),
            slope90=_fmt(None if efficiency.get("slope_pct_per_month") is None else f"{efficiency['slope_pct_per_month']:+.2f}%/mo"),
            stability=_fmt(None if efficiency.get("low_drift_rate") is None else f"{efficiency['low_drift_rate']:.0f}% <=5% drift"),
            target=_fmt(efficiency.get("target")),
            action=(
                "Start base sessions easier and fuel earlier until drift is back under 5%."
                if _safe_float(efficiency.get("current_drift")) is not None and _safe_float(efficiency.get("current_drift")) > 5
                else "Keep easy work truly easy and extend duration slowly."
            ),
            trust=efficiency_trust,
        )

    st.markdown("### Training -> biomarker link")
    if not training_link:
        st.info("Not enough aligned workout history yet to tell whether training is improving or degrading biomarkers.")
    else:
        st.write(f"- **Verdict:** {training_link.get('verdict')}")
        for bullet in training_link.get("bullets", [])[:3]:
            st.write(f"- {bullet}")
        st.write(f"- **Next action:** {training_link.get('next_action')}")


# ------------------------------
# Oura API client + pagination
# ------------------------------

@dataclass
class OuraClient:
    access_token: str

    def get(self, path: str, params: Optional[dict] = None) -> Tuple[int, dict]:
        url = f"{API_BASE}{path}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        retryable = {429, 500, 502, 503, 504}
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=40)
                try:
                    j = r.json() if r.text else {}
                except Exception:
                    j = {"_raw": r.text}

                if r.status_code in retryable and attempt < 2:
                    retry_after = r.headers.get("Retry-After")
                    try:
                        wait_s = max(0.75, float(retry_after)) if retry_after else 0.75 * (2 ** attempt)
                    except Exception:
                        wait_s = 0.75 * (2 ** attempt)
                    time.sleep(wait_s)
                    continue
                return r.status_code, j
            except Exception as exc:
                if attempt < 2:
                    time.sleep(0.75 * (2 ** attempt))
                    continue
                record_debug_event(f"Oura request failed for {path}", exc=exc)
                return 0, {"error": str(exc)}
        return 0, {"error": f"Failed to fetch {path}"}


@st.cache_data(ttl=300, show_spinner=False)
def get_all_pages(access_token: str, path: str, params: Dict[str, Any]) -> Tuple[int, dict]:
    """Fetch Oura usercollection endpoints that may paginate with next_token."""
    client = OuraClient(access_token=access_token)
    merged: List[dict] = []
    next_token: Optional[str] = None
    pages = 0
    last_code = 0
    while True:
        p = dict(params)
        if next_token:
            p["next_token"] = next_token
        code, doc = client.get(path, params=p)
        last_code = code
        pages += 1
        if code != 200:
            return code, {"error": doc, "_pages": pages}
        data = (doc or {}).get("data")
        if isinstance(data, list):
            merged.extend([x for x in data if isinstance(x, dict)])
        next_token = (doc or {}).get("next_token")
        if not next_token:
            break
        if pages >= 50:
            break
    return last_code, {"data": merged, "next_token": None, "_pages": pages}


@st.cache_data(ttl=300, show_spinner=False)
def get_single(access_token: str, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[int, dict]:
    return OuraClient(access_token=access_token).get(path, params=params)


# ------------------------------
# Endpoint registry (what to try)
# ------------------------------

ENDPOINTS: List[Tuple[str, str, str]] = [
    # key, path, kind
    ("personal", "/usercollection/personal_info", "single"),
    ("readiness", "/usercollection/daily_readiness", "paged"),
    ("sleep", "/usercollection/sleep", "paged"),
    ("daily_sleep", "/usercollection/daily_sleep", "paged"),
    ("activity", "/usercollection/daily_activity", "paged"),
    ("workout", "/usercollection/workout", "paged"),
    ("session", "/usercollection/session", "paged"),
    ("tag", "/usercollection/tag", "paged"),
    ("stress", "/usercollection/daily_stress", "paged"),
    ("spo2", "/usercollection/daily_spo2", "paged"),
    ("daily_hrv", "/usercollection/daily_hrv", "paged"),
    ("heartrate", "/usercollection/heartrate", "paged"),
    ("heart_health", "/usercollection/heart_health", "paged"),
    # VO2 has weird casing in docs; we try both.
    ("vo2_max", "/usercollection/vo2_max", "paged"),
    ("vO2_max", "/usercollection/vO2_max", "paged"),
]


def _fetch_endpoint_result(
    token: str,
    *,
    key: str,
    path: str,
    kind: str,
    params_range: Dict[str, Any],
    params_wide: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    if kind == "single":
        code, doc = get_single(token, path)
    else:
        code, doc = get_all_pages(token, path, params_range)
        if key in {"vo2_max", "vO2_max"} and code == 200 and isinstance(doc, dict) and not (doc.get("data") or []):
            code2, doc2 = get_all_pages(token, path, params_wide)
            if code2 == 200 and (doc2.get("data") or []):
                code, doc = code2, doc2

    df = df_from_doc(doc) if (kind != "single") else pd.DataFrame([doc]) if isinstance(doc, dict) and doc else pd.DataFrame()
    return key, {
        "key": key,
        "path": path,
        "kind": kind,
        "code": code,
        "doc": doc,
        "df": df,
        "rows": 0 if df is None else int(len(df)),
        "cols": [] if df is None else list(df.columns),
    }


def fetch_endpoints(
    token: str,
    *,
    start_d: date,
    end_d: date,
    wide_sparse_days: int = 365,
) -> Dict[str, Dict[str, Any]]:
    """Fetch what we can, and return a dict:

    {key: {code:int, doc:dict, df:DataFrame, path:str, rows:int}}

    Sparse metrics (VO2) are retried with a wider date window automatically.
    """
    params_range = {"start_date": _iso(start_d), "end_date": _iso(end_d)}
    params_wide = {"start_date": _iso(date.today() - timedelta(days=wide_sparse_days)), "end_date": _iso(date.today())}

    out: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(ENDPOINTS))) as pool:
        futures = {
            pool.submit(
                _fetch_endpoint_result,
                token,
                key=key,
                path=path,
                kind=kind,
                params_range=params_range,
                params_wide=params_wide,
            ): key
            for key, path, kind in ENDPOINTS
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                result_key, blob = future.result()
                out[result_key] = blob
            except Exception as exc:
                record_debug_event(f"Endpoint fetch failed: {key}", exc=exc)
                out[key] = {
                    "key": key,
                    "path": next(path for k, path, _ in ENDPOINTS if k == key),
                    "kind": next(kind for k, _, kind in ENDPOINTS if k == key),
                    "code": 0,
                    "doc": {"error": str(exc)},
                    "df": pd.DataFrame(),
                    "rows": 0,
                    "cols": [],
                }
    return out


# ------------------------------
# Insight layer (build from what we have)
# ------------------------------


def compute_daily_frame(data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Build a unified daily table keyed by day.

    We only include columns we can reliably map across accounts.
    """
    def base(df: pd.DataFrame, day_col: str = "day") -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["day"]).set_index("day")
        tmp = df.copy()
        if day_col not in df.columns:
            # try derive day
            for c in ["timestamp", "start_datetime", "end_datetime"]:
                if c in tmp.columns:
                    tmp["day"] = tmp[c].apply(_to_day)
                    break
        elif day_col != "day":
            tmp["day"] = tmp[day_col]
        if "day" not in tmp.columns:
            return pd.DataFrame(columns=["day"]).set_index("day")
        tmp = tmp.dropna(subset=["day"]).copy()
        tmp["day"] = tmp["day"].astype(str)
        tmp = tmp.sort_values("day")
        tmp = tmp.drop_duplicates(subset=["day"], keep="last")
        return tmp.set_index("day")

    frames: List[pd.DataFrame] = []
    sleep_priority_frame: Optional[pd.DataFrame] = None

    # Readiness
    df_r = base(data.get("readiness", {}).get("df", pd.DataFrame()))
    if not df_r.empty:
        keep = {}
        for src, dst in [
            ("score", "readiness"),
            ("resting_heart_rate", "resting_hr"),
            ("temperature_deviation", "temp_dev"),
            ("hrv_balance", "hrv_balance"),
        ]:
            if src in df_r.columns:
                keep[dst] = df_r[src]
        if keep:
            frames.append(pd.DataFrame(keep))

    # Daily sleep
    df_ds = base(data.get("daily_sleep", {}).get("df", pd.DataFrame()))
    if not df_ds.empty:
        keep = {}
        for src, dst in [
            ("score", "sleep_score"),
            ("total_sleep_duration", "sleep_total_s"),
            ("deep_sleep_duration", "sleep_deep_s"),
            ("rem_sleep_duration", "sleep_rem_s"),
            ("efficiency", "sleep_eff"),
            ("average_breath", "resp_rate"),
            ("breath_average", "resp_rate"),
            ("respiratory_rate", "resp_rate"),
        ]:
            if src in df_ds.columns:
                keep[dst] = df_ds[src]
        if keep:
            frames.append(pd.DataFrame(keep))

    # Daily activity
    df_a = base(data.get("activity", {}).get("df", pd.DataFrame()))
    if not df_a.empty:
        keep = {}
        for src, dst in [
            ("score", "activity_score"),
            ("steps", "steps"),
            ("active_calories", "active_cal"),
            ("total_calories", "total_cal"),
        ]:
            if src in df_a.columns:
                keep[dst] = df_a[src]
        if keep:
            frames.append(pd.DataFrame(keep))

    # Daily HRV
    df_hrv = base(data.get("daily_hrv", {}).get("df", pd.DataFrame()))
    if not df_hrv.empty:
        keep = {}
        # Oura commonly uses rmssd
        for src, dst in [
            ("rmssd", "hrv_rmssd"),
            ("hrv_rmssd", "hrv_rmssd"),
        ]:
            if src in df_hrv.columns:
                keep[dst] = df_hrv[src]
        if keep:
            frames.append(pd.DataFrame(keep))

    # Detailed sleep endpoint often contains the actual nightly HRV / nightly lowest HR
    # even when daily_hrv is unavailable for the account.
    df_sleep = base(data.get("sleep", {}).get("df", pd.DataFrame()))
    if not df_sleep.empty:
        keep = {}
        for src, dst in [
            ("average_hrv", "hrv_rmssd"),
            ("lowest_heart_rate", "resting_hr"),
            ("average_heart_rate", "sleep_avg_hr"),
            ("average_breath", "resp_rate"),
            ("breath_average", "resp_rate"),
            ("respiratory_rate", "resp_rate"),
            ("breathing_regularity", "breathing_regularity"),
        ]:
            if src in df_sleep.columns:
                keep[dst] = df_sleep[src]
        if keep:
            sleep_priority_frame = pd.DataFrame(keep)
            frames.append(sleep_priority_frame)

    # SpO2
    df_sp = base(data.get("spo2", {}).get("df", pd.DataFrame()))
    if not df_sp.empty:
        keep = {}
        for src, dst in [
            ("spo2_percentage", "spo2"),
            ("spo2", "spo2"),
        ]:
            if src in df_sp.columns:
                keep[dst] = df_sp[src]
        if keep:
            frames.append(pd.DataFrame(keep))

    # Stress
    df_st = base(data.get("stress", {}).get("df", pd.DataFrame()))
    if not df_st.empty:
        keep = {}
        for src, dst in [
            ("stress_high", "stress_high"),
            ("stress", "stress"),
        ]:
            if src in df_st.columns:
                keep[dst] = df_st[src]
        if keep:
            frames.append(pd.DataFrame(keep))

    if not frames:
        return pd.DataFrame(columns=["day"])

    daily = frames[0].copy()
    for frame in frames[1:]:
        daily = daily.combine_first(frame)
    if sleep_priority_frame is not None:
        daily = sleep_priority_frame.combine_first(daily)
    daily.index.name = "day"
    daily = daily.reset_index()
    daily = daily.drop_duplicates(subset=["day"], keep="last")
    daily = daily.sort_values("day")

    # numeric coercions
    for c in daily.columns:
        if c == "day":
            continue
        daily[c] = pd.to_numeric(daily[c], errors="coerce")

    return daily


def extract_tag_labels(row: pd.Series) -> List[str]:
    labels: List[str] = []
    raw_tags = row.get("tags")
    if isinstance(raw_tags, list):
        for item in raw_tags:
            if isinstance(item, str) and item.strip():
                labels.append(item.strip())
            elif isinstance(item, dict):
                for key in ["name", "label", "value", "tag"]:
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        labels.append(value.strip())
                        break
    elif isinstance(raw_tags, str) and raw_tags.strip():
        text = raw_tags.strip().strip("[]")
        parts = [part.strip(" '\"") for part in text.replace(";", ",").split(",")]
        labels.extend([part for part in parts if part])

    for key in ["text", "comment", "tag_type"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            labels.append(value.strip())

    deduped: List[str] = []
    seen = set()
    for label in labels:
        norm = label.lower()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(label)
    return deduped


def metric_delta_frame(daily: pd.DataFrame, metric: str, *, prior_window: int = 7) -> pd.DataFrame:
    if daily.empty or metric not in daily.columns:
        return pd.DataFrame(columns=["day", "value", "prior_avg", "delta"])

    metric_df = daily[["day", metric]].copy()
    metric_df["day"] = pd.to_datetime(metric_df["day"], errors="coerce").dt.normalize()
    metric_df["value"] = pd.to_numeric(metric_df[metric], errors="coerce")
    metric_df = metric_df.dropna(subset=["day", "value"]).sort_values("day").reset_index(drop=True)
    if metric_df.empty:
        return pd.DataFrame(columns=["day", "value", "prior_avg", "delta"])

    prior_avgs: List[Optional[float]] = []
    deltas: List[Optional[float]] = []
    for idx in range(len(metric_df)):
        prior = metric_df.iloc[max(0, idx - prior_window): idx]["value"].dropna()
        prior_avg = float(prior.mean()) if len(prior) >= min(3, prior_window) else None
        prior_avgs.append(prior_avg)
        value = _safe_float(metric_df.iloc[idx]["value"])
        deltas.append(None if prior_avg is None or value is None else value - prior_avg)

    metric_df["prior_avg"] = prior_avgs
    metric_df["delta"] = deltas
    return metric_df[["day", "value", "prior_avg", "delta"]]


def tag_effect_analysis(
    daily: pd.DataFrame,
    tags_df: pd.DataFrame,
    *,
    metric: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    if daily.empty or tags_df.empty:
        return pd.DataFrame()

    metric_df = metric_delta_frame(daily, metric)
    if metric_df.empty:
        return pd.DataFrame()
    lookup = metric_df.set_index("day")

    tags = tags_df.copy()
    if "day" not in tags.columns:
        for column in ["timestamp", "start_datetime"]:
            if column in tags.columns:
                tags["day"] = tags[column].apply(_to_day)
                break
    tags["day"] = pd.to_datetime(tags["day"], errors="coerce").dt.normalize()
    tags = tags.dropna(subset=["day"]).copy()
    if tags.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, row in tags.iterrows():
        next_day = row["day"] + pd.Timedelta(days=1)
        if next_day not in lookup.index:
            continue
        next_metric = lookup.loc[next_day]
        delta = _safe_float(next_metric.get("delta"))
        if delta is None:
            continue
        for label in extract_tag_labels(row):
            rows.append({
                "tag": label,
                "next_day": next_day.date().isoformat(),
                "next_day_value": _safe_float(next_metric.get("value")),
                "prior_avg": _safe_float(next_metric.get("prior_avg")),
                "delta": delta,
            })

    if not rows:
        return pd.DataFrame()

    effects = pd.DataFrame(rows)
    effects = effects.groupby("tag", as_index=False).agg(
        n=("delta", "size"),
        mean_delta=("delta", "mean"),
        median_delta=("delta", "median"),
        next_day_avg=("next_day_value", "mean"),
        prior_avg=("prior_avg", "mean"),
    )
    effects = effects[effects["n"] >= 2].copy()
    if effects.empty:
        return effects
    effects["favorable_delta"] = effects["mean_delta"] if higher_is_better else -effects["mean_delta"]
    effects = effects.sort_values(["favorable_delta", "n"], ascending=[False, False]).reset_index(drop=True)
    return effects


def _parse_local_dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(float(v))
    if isinstance(v, str):
        try:
            return dtparse.isoparse(v)
        except Exception:
            return None
    return None


def format_clock_from_anchor(minutes: Optional[float], *, anchor_hour: int = 18) -> str:
    if minutes is None:
        return "—"
    try:
        if math.isnan(float(minutes)):
            return "—"
    except Exception:
        return "—"
    total = int(round(float(minutes))) % (24 * 60)
    total = (total + anchor_hour * 60) % (24 * 60)
    hh = total // 60
    mm = total % 60
    return f"{hh:02d}:{mm:02d}"


def circular_minute_distance(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    diff = abs(float(a) - float(b))
    return min(diff, 1440.0 - diff)


def bool_from_value(v: Any) -> Optional[bool]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return bool(v)


def build_sleep_feature_frame(data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    df_sleep = data.get("sleep", {}).get("df", pd.DataFrame())
    if df_sleep.empty:
        return pd.DataFrame(columns=["day", "bedtime_min", "wake_min", "sleep_midpoint_hour", "median_bedtime_min", "bedtime_dev"])

    s = df_sleep.copy()
    start_col = "start_datetime" if "start_datetime" in s.columns else ("bedtime_start" if "bedtime_start" in s.columns else None)
    end_col = "end_datetime" if "end_datetime" in s.columns else ("bedtime_end" if "bedtime_end" in s.columns else None)
    if not start_col or not end_col:
        return pd.DataFrame(columns=["day", "bedtime_min", "wake_min", "sleep_midpoint_hour", "median_bedtime_min", "bedtime_dev"])

    if "day" not in s.columns:
        s["day"] = s[end_col].apply(_to_day)
    s["start_local"] = s[start_col].apply(_parse_local_dt)
    s["end_local"] = s[end_col].apply(_parse_local_dt)
    s = s.dropna(subset=["day", "start_local", "end_local"]).copy()
    if s.empty:
        return pd.DataFrame(columns=["day", "bedtime_min", "wake_min", "sleep_midpoint_hour", "median_bedtime_min", "bedtime_dev"])

    s["bedtime_min"] = s["start_local"].apply(clock_minutes_from_anchor)
    s["wake_min"] = s["end_local"].apply(clock_minutes_from_anchor)
    s["sleep_midpoint_hour"] = s.apply(
        lambda row: (((row["start_local"] + (row["end_local"] - row["start_local"]) / 2).hour * 60) + (row["start_local"] + (row["end_local"] - row["start_local"]) / 2).minute) / 60.0,
        axis=1,
    )
    s = s.sort_values("day").drop_duplicates(subset=["day"], keep="last").reset_index(drop=True)

    median_vals: List[Optional[float]] = []
    dev_vals: List[Optional[float]] = []
    for idx in range(len(s)):
        window = pd.to_numeric(s.iloc[max(0, idx - 28): idx]["bedtime_min"], errors="coerce").dropna()
        median_bedtime = float(window.median()) if len(window) >= 5 else None
        current_bedtime = _safe_float(s.iloc[idx]["bedtime_min"])
        median_vals.append(median_bedtime)
        dev_vals.append(circular_minute_distance(current_bedtime, median_bedtime))

    s["median_bedtime_min"] = median_vals
    s["bedtime_dev"] = dev_vals
    s["bedtime_clock"] = s["bedtime_min"].apply(format_clock_from_anchor)
    s["median_bedtime_clock"] = s["median_bedtime_min"].apply(format_clock_from_anchor)
    return s[["day", "bedtime_min", "wake_min", "sleep_midpoint_hour", "median_bedtime_min", "bedtime_dev", "bedtime_clock", "median_bedtime_clock"]]


def _matches_keywords(value: str, keywords: List[str]) -> bool:
    lower = value.lower()
    return any(keyword in lower for keyword in keywords)


def build_tag_feature_frame(data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    df_tag = data.get("tag", {}).get("df", pd.DataFrame())
    if df_tag.empty:
        return pd.DataFrame(columns=["day", "tags_text", "tags_available", "alcohol_tag", "late_meal_tag", "travel_tag", "supplement_tag"])

    t = df_tag.copy()
    if "day" not in t.columns:
        for c in ["timestamp", "start_datetime"]:
            if c in t.columns:
                t["day"] = t[c].apply(_to_day)
                break
    t = t.dropna(subset=["day"]).copy()
    if t.empty:
        return pd.DataFrame(columns=["day", "tags_text", "tags_available", "alcohol_tag", "late_meal_tag", "travel_tag", "supplement_tag"])

    alcohol_keys = ["alcohol", "wine", "beer", "cocktail", "drinks", "drinking"]
    late_meal_keys = ["late meal", "late dinner", "heavy meal", "big meal", "ate late", "food late"]
    travel_keys = ["travel", "flight", "jet lag", "hotel"]
    supplement_keys = ["supplement", "magnesium", "melatonin", "creatine", "electrolyte"]

    rows: List[Dict[str, Any]] = []
    for _, row in t.iterrows():
        labels = extract_tag_labels(row)
        joined = " | ".join(labels)
        rows.append(
            {
                "day": str(row.get("day")),
                "tags_text": joined,
                "alcohol_tag": any(_matches_keywords(label, alcohol_keys) for label in labels),
                "late_meal_tag": any(_matches_keywords(label, late_meal_keys) for label in labels),
                "travel_tag": any(_matches_keywords(label, travel_keys) for label in labels),
                "supplement_tag": any(_matches_keywords(label, supplement_keys) for label in labels),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["day", "tags_text", "tags_available", "alcohol_tag", "late_meal_tag", "travel_tag", "supplement_tag"])

    tag_df = pd.DataFrame(rows)
    tag_df = tag_df.groupby("day", as_index=False).agg(
        tags_text=("tags_text", lambda xs: " | ".join([str(x) for x in xs if str(x).strip()])),
        alcohol_tag=("alcohol_tag", "max"),
        late_meal_tag=("late_meal_tag", "max"),
        travel_tag=("travel_tag", "max"),
        supplement_tag=("supplement_tag", "max"),
    )
    tag_df["tags_available"] = True
    return tag_df


def build_behavior_event_frame(events_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "day",
        "behavior_events_text",
        "behavior_events_available",
        "behavior_alcohol_tag",
        "behavior_late_meal_tag",
        "behavior_travel_tag",
        "behavior_supplement_tag",
        "behavior_illness_tag",
        "behavior_sauna_tag",
        "behavior_cold_tag",
        "behavior_caffeine_timing_tag",
        "manual_wellness",
        "behavior_notes",
    ]
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for _, row in events_df.iterrows():
        day = str(row.get("day") or "")
        if not day:
            continue
        note_value = row.get("notes")
        note = "" if note_value is None or (isinstance(note_value, float) and math.isnan(note_value)) else str(note_value).strip()
        supplement_value = row.get("supplement")
        supplement = "" if supplement_value is None or (isinstance(supplement_value, float) and math.isnan(supplement_value)) else str(supplement_value).strip()
        event_parts: List[str] = []
        if bool_from_value(row.get("alcohol")):
            event_parts.append("Alcohol")
        if bool_from_value(row.get("late_meal")):
            event_parts.append("Late meal")
        if bool_from_value(row.get("travel")):
            event_parts.append("Travel")
        if bool_from_value(row.get("illness")):
            event_parts.append("Illness")
        if bool_from_value(row.get("sauna")):
            event_parts.append("Sauna")
        if bool_from_value(row.get("cold")):
            event_parts.append("Cold exposure")
        if bool_from_value(row.get("caffeine_late")):
            event_parts.append("Late caffeine")
        if supplement:
            event_parts.append(f"Supplement: {supplement}")
        wellness = _safe_float(row.get("manual_wellness"))
        if wellness is not None:
            event_parts.append(f"Wellness {int(round(wellness))}/10")
        payload: Dict[str, Any] = {
            "day": day,
            "behavior_events_text": " | ".join(event_parts + ([note] if note else [])),
            "behavior_events_available": True,
            "behavior_alcohol_tag": bool(bool_from_value(row.get("alcohol"))),
            "behavior_late_meal_tag": bool(bool_from_value(row.get("late_meal"))),
            "behavior_travel_tag": bool(bool_from_value(row.get("travel"))),
            "behavior_supplement_tag": bool(supplement),
            "behavior_illness_tag": bool(bool_from_value(row.get("illness"))),
            "behavior_sauna_tag": bool(bool_from_value(row.get("sauna"))),
            "behavior_cold_tag": bool(bool_from_value(row.get("cold"))),
            "behavior_caffeine_timing_tag": bool(bool_from_value(row.get("caffeine_late"))),
            "manual_wellness": _safe_float(row.get("manual_wellness")),
            "behavior_notes": note,
        }
        rows.append(payload)

    if not rows:
        return pd.DataFrame(columns=columns)

    events = pd.DataFrame(rows)
    return events[columns].sort_values("day").reset_index(drop=True)


def normalize_workout_rows(df_workout: pd.DataFrame) -> pd.DataFrame:
    if df_workout.empty:
        return pd.DataFrame()
    w = df_workout.copy()
    if "day" not in w.columns:
        for c in ["start_datetime", "timestamp"]:
            if c in w.columns:
                w["day"] = w[c].apply(_to_day)
                break
    w["day"] = w.get("day").astype(str)

    duration_col = None
    for c in ["duration_min", "duration", "duration_seconds"]:
        if c in w.columns:
            duration_col = c
            break
    if duration_col == "duration_min":
        w["duration_min"] = pd.to_numeric(w["duration_min"], errors="coerce")
    elif duration_col:
        w["duration_min"] = pd.to_numeric(w[duration_col], errors="coerce") / 60.0
    else:
        w["duration_min"] = None

    for c in ["type", "sport", "activity"]:
        if c in w.columns:
            w["type"] = w[c]
            break
    if "type" not in w.columns:
        w["type"] = "workout"
    return w


def normalize_hr_points(df_hr: pd.DataFrame) -> pd.DataFrame:
    if df_hr.empty:
        return pd.DataFrame()
    hp = df_hr.copy()
    ts_col = None
    for c in ["timestamp", "time", "datetime", "start_datetime"]:
        if c in hp.columns:
            ts_col = c
            break
    bpm_col = None
    for c in ["bpm", "heart_rate", "hr"]:
        if c in hp.columns:
            bpm_col = c
            break
    if not ts_col or not bpm_col:
        return pd.DataFrame()

    hp["ts"] = hp[ts_col].apply(_parse_dt)
    hp = hp.dropna(subset=["ts"]).copy()
    hp["ts_epoch"] = hp["ts"].apply(lambda x: x.timestamp())
    hp["bpm"] = pd.to_numeric(hp[bpm_col], errors="coerce")
    hp = hp.dropna(subset=["bpm"]).copy()
    return hp[["ts", "ts_epoch", "bpm"]].sort_values("ts_epoch")


def compute_analysis_range(
    *,
    start_d: date,
    end_d: date,
    analysis_lookback_days: int,
) -> date:
    min_start = end_d - timedelta(days=max(analysis_lookback_days - 1, 0))
    return min(start_d, min_start)


def compute_daily_training_load(
    data: Dict[str, Dict[str, Any]],
    *,
    max_hr: int,
    resting_hr: int,
    sex: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workouts = normalize_workout_rows(data.get("workout", {}).get("df", pd.DataFrame()))
    hr_points = normalize_hr_points(data.get("heartrate", {}).get("df", pd.DataFrame()))
    if workouts.empty:
        return pd.DataFrame(columns=["day", "trimp", "workout_minutes", "workout_count", "last_workout_type"]), workouts, hr_points

    rows: List[Dict[str, Any]] = []
    for _, row in workouts.iterrows():
        trimp = None
        start_dt, end_dt = workout_window(row)
        if not hr_points.empty and start_dt is not None and end_dt is not None:
            seg = hr_points[(hr_points["ts"] >= start_dt) & (hr_points["ts"] <= end_dt)].copy()
            if len(seg) >= 20:
                trimp = compute_banister_trimp(seg, max_hr=int(max_hr), resting_hr=int(resting_hr), sex=sex)
        rows.append(
            {
                "day": str(row.get("day")),
                "trimp": trimp,
                "workout_minutes": _safe_float(row.get("duration_min")),
                "workout_type": str(row.get("type") or "workout"),
            }
        )

    load_df = pd.DataFrame(rows)
    if load_df.empty:
        return pd.DataFrame(columns=["day", "trimp", "workout_minutes", "workout_count", "last_workout_type"]), workouts, hr_points

    load_df = load_df.sort_values("day")
    day_load = load_df.groupby("day", as_index=False).agg(
        trimp=("trimp", lambda s: s.sum(min_count=1)),
        workout_minutes=("workout_minutes", lambda s: s.sum(min_count=1)),
        workout_count=("workout_type", "size"),
        last_workout_type=("workout_type", "last"),
    )
    return day_load, workouts, hr_points


def enrich_daily_context(
    daily: pd.DataFrame,
    *,
    sleep_features: pd.DataFrame,
    tag_features: pd.DataFrame,
    training_load: pd.DataFrame,
    behavior_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if daily is not None and not daily.empty:
        frames.append(daily.copy())
    for frame in [sleep_features, tag_features, training_load, behavior_features]:
        if frame is not None and not frame.empty:
            frames.append(frame.copy())
    if not frames:
        return pd.DataFrame(columns=["day"])

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="day", how="outer")
    merged = merged.drop_duplicates(subset=["day"], keep="last").sort_values("day").reset_index(drop=True)

    if "sleep_total_s" in merged.columns:
        merged["sleep_hours"] = pd.to_numeric(merged["sleep_total_s"], errors="coerce") / 3600.0
    if "resp_rate" in merged.columns:
        merged["resp_rate"] = pd.to_numeric(merged["resp_rate"], errors="coerce")
        merged["resp_rate_baseline"] = merged["resp_rate"].rolling(28, min_periods=10).median().shift(1)
        merged["resp_rate_dev"] = merged["resp_rate"] - merged["resp_rate_baseline"]
    if "tags_text" not in merged.columns:
        merged["tags_text"] = None
    if "tags_available" not in merged.columns:
        merged["tags_available"] = False
    for col in ["alcohol_tag", "late_meal_tag", "travel_tag", "supplement_tag"]:
        if col not in merged.columns:
            merged[col] = None
    if "behavior_events_text" not in merged.columns:
        merged["behavior_events_text"] = None
    if "behavior_events_available" not in merged.columns:
        merged["behavior_events_available"] = False
    if "manual_wellness" not in merged.columns:
        merged["manual_wellness"] = None
    if "behavior_notes" not in merged.columns:
        merged["behavior_notes"] = None
    behavior_flag_map = {
        "alcohol_tag": "behavior_alcohol_tag",
        "late_meal_tag": "behavior_late_meal_tag",
        "travel_tag": "behavior_travel_tag",
        "supplement_tag": "behavior_supplement_tag",
    }
    for base_col, behavior_col in behavior_flag_map.items():
        if behavior_col not in merged.columns:
            merged[behavior_col] = False
        merged[base_col] = merged[base_col].apply(bool_from_value)
        merged[behavior_col] = merged[behavior_col].apply(bool_from_value).fillna(False)
        merged[base_col] = (merged[base_col].fillna(False) | merged[behavior_col]).astype(bool)
    for behavior_col in ["behavior_illness_tag", "behavior_sauna_tag", "behavior_cold_tag", "behavior_caffeine_timing_tag"]:
        if behavior_col not in merged.columns:
            merged[behavior_col] = False
        merged[behavior_col] = merged[behavior_col].apply(bool_from_value).fillna(False).astype(bool)
    merged["tags_available"] = merged["tags_available"].fillna(False) | merged["behavior_events_available"].fillna(False)
    merged["events_text"] = (
        merged[["tags_text", "behavior_events_text"]]
        .fillna("")
        .agg(lambda row: " | ".join([part for part in row if str(part).strip()]), axis=1)
    )
    return merged


def best_time_to_act(sleep_features: pd.DataFrame) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {
        "chronotype": None,
        "exercise_window": None,
        "parasympathetic_window": None,
        "light_window": "First hour after waking",
        "bedtime_target": None,
        "winddown_start": None,
        "note": None,
    }
    if sleep_features is None or sleep_features.empty:
        return out

    recent = sleep_features.dropna(subset=["sleep_midpoint_hour"]).sort_values("day").tail(14)
    midpoint = float(recent["sleep_midpoint_hour"].mean()) if len(recent) >= 5 else None
    chrono = chronotype_from_sleep_midpoint(midpoint)
    out["chronotype"] = chrono
    if chrono == "MORNING":
        out["exercise_window"] = "08:00-11:00"
        out["parasympathetic_window"] = "12:30-14:30"
    elif chrono == "EVENING":
        out["exercise_window"] = "17:00-20:00"
        out["parasympathetic_window"] = "14:00-16:00"
    else:
        out["exercise_window"] = "14:00-18:00"
        out["parasympathetic_window"] = "13:00-15:00"

    latest = sleep_features.sort_values("day").iloc[-1]
    median_bed = _safe_float(latest.get("median_bedtime_min"))
    if median_bed is not None:
        bedtime_target = (median_bed - 15.0) % (24 * 60)
        out["bedtime_target"] = format_clock_from_anchor(bedtime_target)
        out["winddown_start"] = format_clock_from_anchor((bedtime_target - 45.0) % (24 * 60))
    out["note"] = "Estimated from your sleep midpoint and bedtime history. Intraday HRV timing is not exposed reliably by this Oura account."
    return out


def compute_intervention_windows(
    hr_points: pd.DataFrame,
    chronotype: str,
) -> Dict[str, Dict[str, str]]:
    windows: Dict[str, Dict[str, str]] = {}
    if chronotype == "MORNING":
        windows = {
            "hard_training": {"window": "08:00-11:00", "why": "Morning types usually hit their best coordination and focus mid-morning."},
            "nsdr_reset": {"window": "13:00-14:30", "why": "Post-lunch is the cleanest parasympathetic window for a reset."},
            "easy_training": {"window": "15:00-17:00", "why": "Easy afternoon movement will not crowd bedtime."},
            "last_caffeine": {"window": "Before 12:00", "why": "Later caffeine tends to bleed into sleep pressure for morning chronotypes."},
            "cold_exposure": {"window": "07:00-09:00", "why": "Morning light and easy movement are the cleanest circadian anchors."},
        }
    elif chronotype == "EVENING":
        windows = {
            "hard_training": {"window": "17:00-20:00", "why": "Evening chronotypes usually peak later in the day."},
            "nsdr_reset": {"window": "14:00-16:00", "why": "Use a mid-afternoon reset before your best performance window."},
            "easy_training": {"window": "10:00-12:00", "why": "Easy morning work helps wake the system without burning your best hours."},
            "last_caffeine": {"window": "Before 14:00", "why": "A later cortisol rhythm means the caffeine cutoff also shifts later."},
            "cold_exposure": {"window": "08:00-10:00", "why": "Morning light and easy movement help wakefulness and anchor the clock earlier."},
        }
    else:
        windows = {
            "hard_training": {"window": "14:00-18:00", "why": "Intermediate chronotypes usually perform best in the afternoon."},
            "nsdr_reset": {"window": "13:00-15:00", "why": "This is the most reliable parasympathetic reset window for most people."},
            "easy_training": {"window": "08:00-10:00 or 18:00-19:00", "why": "Bookend the day with easy movement."},
            "last_caffeine": {"window": "Before 13:00", "why": "This leaves enough runway for nighttime sleep."},
            "cold_exposure": {"window": "07:00-09:00", "why": "Morning light plus easy movement is the most reliable circadian signal."},
        }

    if hr_points is not None and not hr_points.empty and len(hr_points) > 300:
        try:
            hp = hr_points.copy()
            hp["hour"] = hp["ts"].dt.hour
            hp = hp[(hp["hour"] >= 8) & (hp["hour"] <= 22)]
            if not hp.empty:
                hourly_hr = hp.groupby("hour")["bpm"].median()
                if not hourly_hr.empty:
                    min_hr_hour = int(hourly_hr.idxmin())
                    windows["nsdr_reset"] = {
                        "window": f"{min_hr_hour:02d}:00-{min(min_hr_hour + 1, 23):02d}:30",
                        "why": f"Your waking HR is naturally lowest around {min_hr_hour:02d}:00, which is a good time to stack NSDR or breathwork.",
                    }
        except Exception:
            pass
    return windows


def morning_protocol(
    hrv_signal: MetricSignal,
    rhr_signal: MetricSignal,
    sleep_signal: MetricSignal,
    daily: pd.DataFrame,
    zones: List[Tuple[str, float, float]],
    *,
    hrv_trust: Optional[Dict[str, Any]] = None,
    rhr_trust: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    actions: List[Dict[str, str]] = []
    latest = daily.sort_values("day").iloc[-1] if daily is not None and not daily.empty else pd.Series(dtype=object)
    timing = best_time_to_act(daily[["day", "sleep_midpoint_hour", "median_bedtime_min"]].dropna(how="all") if daily is not None and not daily.empty and "sleep_midpoint_hour" in daily.columns else pd.DataFrame())
    hrv_strength = str((hrv_trust or {}).get("recommendation_strength") or "MODERATE")
    rhr_strength = str((rhr_trust or {}).get("recommendation_strength") or "MODERATE")
    hrv_low_conf = str((hrv_trust or {}).get("confidence") or "LOW") == "LOW"
    rhr_low_conf = str((rhr_trust or {}).get("confidence") or "LOW") == "LOW"
    hrv_context_note = str((hrv_trust or {}).get("context_note") or "").strip()
    rhr_context_note = str((rhr_trust or {}).get("context_note") or "").strip()

    def _why(base: str, note: str) -> str:
        return f"{base} {note}".strip() if note else base

    if rhr_signal.get("signal") in {"RED", "AMBER"}:
        actions.extend(
            [
                {
                    "timing": "Now",
                    "action": "Drink water early and keep the first hour calm",
                    "why": _why("RHR is elevated. Start with low-risk recovery inputs before adding more stress." if not rhr_low_conf else "RHR may be elevated, but confidence is limited. Use calm, low-risk inputs first.", rhr_context_note),
                    "category": "hydration",
                    "strength": rhr_strength,
                },
                {
                    "timing": "Now",
                    "action": "Keep caffeine lighter than usual and avoid stacking it on an empty, stressed system",
                    "why": _why("If resting HR is already high, aggressive caffeine often pushes the day further in the wrong direction." if not rhr_low_conf else "Use a conservative caffeine plan because the RHR signal is sparse.", rhr_context_note),
                    "category": "stimulants",
                    "strength": rhr_strength,
                },
                {
                    "timing": "Before 10:00",
                    "action": "Get 10-15 minutes of outdoor light and an easy walk",
                    "why": _why("Light movement is a low-risk way to stabilize arousal and support circadian timing." if not rhr_low_conf else "Keep the intervention simple and low-risk because the RHR signal is low-confidence today.", rhr_context_note),
                    "category": "circadian",
                    "strength": rhr_strength,
                },
                {
                    "timing": "Before first meeting",
                    "action": "Do 5 minutes of box breathing (4-4-4-4) or physiological sighs",
                    "why": _why("This is the fastest parasympathetic lever for bringing HR down today." if not rhr_low_conf else "Breathwork is still the right low-cost move when the signal is uncertain.", rhr_context_note),
                    "category": "nervous_system",
                    "strength": rhr_strength,
                },
            ]
        )

    if hrv_signal.get("signal") == "RED":
        actions.append(
            {
                "timing": "Next 10 minutes",
                "action": "Do 10-20 minutes of NSDR or Yoga Nidra",
                "why": _why("HRV is suppressed. Downshift the nervous system before the day adds more load." if not hrv_low_conf else "HRV may be suppressed, but confidence is low. NSDR is still the safest leverage point.", hrv_context_note),
                "category": "recovery",
                "strength": hrv_strength,
            }
        )
    elif hrv_signal.get("signal") == "AMBER":
        actions.append(
            {
                "timing": "Next 30 minutes",
                "action": "Do 5 minutes of box breathing or a short NSDR session",
                "why": _why("HRV is mediocre today. A small nervous-system intervention can keep it from sliding further." if not hrv_low_conf else "Treat this as a gentle nudge, not a hard call, because HRV coverage is sparse.", hrv_context_note),
                "category": "recovery",
                "strength": hrv_strength,
            }
        )

    if hrv_signal.get("signal") in {"RED", "AMBER"}:
        z2 = zone_bounds_by_name(zones, "Z2")
        cap = f"<={z2[1]} bpm" if z2 else "Z1-Z2 only"
        actions.append(
            {
                "timing": "All day",
                "action": f"Cap exercise at {cap}. No intervals.",
                "why": _why("Hard training on suppressed HRV usually compounds the problem instead of solving it." if not hrv_low_conf else "Lean conservative because HRV is soft, but treat this as a moderate-confidence call.", hrv_context_note),
                "category": "training",
                "strength": hrv_strength,
            }
        )

    bedtime_target = timing.get("bedtime_target") or "15 minutes earlier than your normal bedtime"
    winddown = timing.get("winddown_start") or "45 minutes before bed"
    actions.append(
        {
            "timing": "Tonight",
            "action": f"In bed by {bedtime_target}. Start screen-off wind-down at {winddown}. Keep room at 18-19C.",
            "why": "Tonight is the highest-ROI opportunity to move tomorrow's HRV and resting HR.",
            "category": "sleep",
            "strength": "STRONG",
        }
    )

    alcohol_tag = bool_from_value(latest.get("alcohol_tag"))
    late_meal_tag = bool_from_value(latest.get("late_meal_tag"))
    if alcohol_tag or late_meal_tag:
        actions.append(
            {
                "timing": "Tonight",
                "action": "Fix alcohol and meal timing first: no alcohol and stop calories at least 3 hours before bed.",
                "why": "Your recent tags point to alcohol or a late meal as the simplest lever to fix before chasing supplements.",
                "category": "nutrition",
                "strength": "STRONG",
            }
        )
    else:
        actions.append(
            {
                "timing": "Tonight",
                "action": "Protect meal timing: finish dinner 3+ hours before bed and keep it lighter.",
                "why": "Late digestion is a common reason HRV stays low and resting HR stays elevated overnight.",
                "category": "nutrition",
                "strength": "MODERATE",
            }
        )

    if sleep_signal.get("signal") == "RED":
        actions.append(
            {
                "timing": "Today",
                "action": "Reduce cognitive load where possible and avoid stacking late-day stress.",
                "why": "Poor sleep plus low HRV is a fast route to another bad night.",
                "category": "schedule",
                "strength": "MODERATE",
            }
        )

    ordering = {"Now": 0, "Next 10 minutes": 1, "Next 30 minutes": 2, "Before first meeting": 3, "Before 10:00": 4, "All day": 5, "Today": 6, "Tonight": 7}
    return sorted(actions, key=lambda item: ordering.get(item["timing"], 99))


def what_moved_my_numbers(
    daily: pd.DataFrame,
    tags_df: pd.DataFrame,
    metric: str = "hrv_rmssd",
    lookback: int = 14,
) -> pd.DataFrame:
    if daily.empty or metric not in daily.columns:
        return pd.DataFrame()

    working = daily.copy()
    if tags_df is not None and not tags_df.empty and "alcohol_tag" not in working.columns:
        working = enrich_daily_context(
            working,
            sleep_features=pd.DataFrame(),
            tag_features=build_tag_feature_frame({"tag": {"df": tags_df}}),
            training_load=pd.DataFrame(),
        )

    metric_df = metric_delta_frame(working, metric).rename(
        columns={"day": "metric_day", "value": "metric_value", "prior_avg": "metric_prior_avg", "delta": "metric_delta"}
    )
    if metric_df.empty:
        return pd.DataFrame()

    working["day_dt"] = pd.to_datetime(working["day"], errors="coerce").dt.normalize()
    merged = metric_df.merge(working, left_on="metric_day", right_on="day_dt", how="left").sort_values("metric_day").tail(lookback)
    if merged.empty:
        return pd.DataFrame()

    higher_is_better = metric != "resting_hr"
    merged["worse_delta"] = -pd.to_numeric(merged["metric_delta"], errors="coerce") if higher_is_better else pd.to_numeric(merged["metric_delta"], errors="coerce")
    merged["sleep_hours"] = pd.to_numeric(merged.get("sleep_total_s"), errors="coerce") / 3600.0
    merged["prev_steps"] = pd.to_numeric(merged.get("steps"), errors="coerce").shift(1)
    merged["prev_trimp"] = pd.to_numeric(merged.get("trimp"), errors="coerce").shift(1)
    merged["prev_bedtime_dev"] = pd.to_numeric(merged.get("bedtime_dev"), errors="coerce")
    merged["prev_manual_wellness"] = pd.to_numeric(merged.get("manual_wellness"), errors="coerce").shift(1)
    merged["resp_rate_abs_dev"] = pd.to_numeric(merged.get("resp_rate_dev"), errors="coerce").abs()
    for col in ["alcohol_tag", "late_meal_tag", "travel_tag", "supplement_tag"]:
        merged[f"prev_{col}"] = merged[col].apply(bool_from_value).shift(1)
    for col in ["behavior_illness_tag", "behavior_sauna_tag", "behavior_cold_tag", "behavior_caffeine_timing_tag"]:
        if col in merged.columns:
            merged[f"prev_{col}"] = merged[col].apply(bool_from_value).shift(1)

    latest = merged.iloc[-1]
    rows: List[Dict[str, Any]] = []
    numeric_factors = [
        ("sleep_hours", "Sleep duration", "h"),
        ("prev_steps", "Previous-day steps", "steps"),
        ("prev_trimp", "Previous-day workout load", "TRIMP"),
        ("prev_bedtime_dev", "Bedtime deviation", "min"),
        ("prev_manual_wellness", "Manual wellness", "/10"),
        ("resp_rate_abs_dev", "Respiratory-rate deviation", "/min"),
    ]
    for col, label, unit in numeric_factors:
        subset = merged[[col, "worse_delta"]].dropna()
        if len(subset) < 10 or subset[col].nunique() < 4:
            continue
        corr = subset[col].corr(subset["worse_delta"])
        if corr is None or math.isnan(corr):
            continue
        median_value = float(subset[col].median())
        current_value = _safe_float(latest.get(col))
        risk_direction = "higher" if corr > 0 else "lower"
        current_triggered = None
        if current_value is not None:
            current_triggered = current_value > median_value if risk_direction == "higher" else current_value < median_value
        rows.append(
            {
                "factor": label,
                "kind": "numeric",
                "strength": abs(float(corr)),
                "impact": float(corr),
                "risk_direction": risk_direction,
                "current_value": current_value,
                "current_display": f"{current_value:.1f} {unit}".strip() if current_value is not None else "—",
                "current_triggered": current_triggered,
                "reference": median_value,
            }
        )

    binary_factors = [
        ("prev_alcohol_tag", "Alcohol tag"),
        ("prev_late_meal_tag", "Late meal tag"),
        ("prev_travel_tag", "Travel tag"),
        ("prev_supplement_tag", "Supplement tag"),
        ("prev_behavior_illness_tag", "Illness event"),
        ("prev_behavior_sauna_tag", "Sauna event"),
        ("prev_behavior_cold_tag", "Cold exposure event"),
        ("prev_behavior_caffeine_timing_tag", "Late caffeine event"),
    ]
    for col, label in binary_factors:
        subset = merged[[col, "worse_delta"]].dropna()
        if len(subset) < 8:
            continue
        subset = subset.copy()
        subset[col] = subset[col].astype(int)
        if subset[col].nunique() < 2:
            continue
        corr = subset[col].corr(subset["worse_delta"])
        if corr is None or math.isnan(corr):
            continue
        current_present = bool_from_value(latest.get(col))
        rows.append(
            {
                "factor": label,
                "kind": "binary",
                "strength": abs(float(corr)),
                "impact": float(corr),
                "risk_direction": "present" if corr > 0 else "absent",
                "current_value": current_present,
                "current_display": "present" if current_present else "not tagged",
                "current_triggered": current_present if corr > 0 else (not current_present if current_present is not None else None),
                "reference": None,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["strength", "current_triggered"], ascending=[False, False]).reset_index(drop=True)
    out["metric"] = metric
    out["latest_value"] = _safe_float(latest.get("metric_value"))
    out["latest_delta"] = _safe_float(latest.get("metric_delta"))
    out["latest_prior_avg"] = _safe_float(latest.get("metric_prior_avg"))
    out["latest_day"] = str(latest.get("metric_day"))[:10]
    return out


def dose_response(
    daily: pd.DataFrame,
    x_col: str,
    y_col: str,
    lag_days: int = 1,
    bins: int = 5,
    *,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    if daily.empty or x_col not in daily.columns or y_col not in daily.columns:
        return {"curve": pd.DataFrame(), "sweet_spot": None}

    df = daily[["day", x_col, y_col]].copy()
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=["day"]).sort_values("day")
    df[f"{y_col}_next"] = df[y_col].shift(-lag_days)
    df = df.dropna(subset=[x_col, f"{y_col}_next"]).copy()
    if len(df) < 8 or df[x_col].nunique() < 3:
        return {"curve": pd.DataFrame(), "sweet_spot": None}

    q = max(2, min(int(bins), int(df[x_col].nunique())))
    try:
        df["bin"] = pd.qcut(df[x_col], q=q, duplicates="drop")
    except Exception:
        return {"curve": pd.DataFrame(), "sweet_spot": None}

    result = (
        df.groupby("bin", observed=False)
        .agg(
            mean_y=(f"{y_col}_next", "mean"),
            n=(f"{y_col}_next", "size"),
            x_low=(x_col, "min"),
            x_high=(x_col, "max"),
        )
        .reset_index(drop=True)
    )
    if result.empty:
        return {"curve": result, "sweet_spot": None}

    result["bin_label"] = result.apply(lambda row: f"{row['x_low']:.1f}-{row['x_high']:.1f}", axis=1)
    pick_idx = int(result["mean_y"].idxmax()) if higher_is_better else int(result["mean_y"].idxmin())
    sweet_spot = result.iloc[pick_idx].to_dict()
    return {"curve": result, "sweet_spot": sweet_spot}


def dose_response_has_enough_support(
    curve_result: Dict[str, Any],
    *,
    min_pairs: int = 28,
    min_bins: int = 4,
    min_sweet_bin_n: int = 5,
) -> bool:
    curve_df = curve_result.get("curve")
    sweet = curve_result.get("sweet_spot")
    if not isinstance(curve_df, pd.DataFrame) or curve_df.empty or not isinstance(sweet, dict):
        return False
    n_series = pd.to_numeric(curve_df.get("n"), errors="coerce").fillna(0)
    if int(n_series.sum()) < int(min_pairs):
        return False
    if int(len(curve_df)) < int(min_bins):
        return False
    sweet_n = _safe_float(sweet.get("n"))
    if sweet_n is None or sweet_n < float(min_sweet_bin_n):
        return False
    return True


def project_next_metric(value: Optional[float], baseline: Optional[float], *, higher_is_better: bool) -> Optional[float]:
    if value is None:
        return None
    if baseline is None:
        return value
    gap = baseline - value if higher_is_better else value - baseline
    if gap <= 0:
        return value
    step = max(abs(gap) * 0.35, 1.5 if higher_is_better else 1.0)
    return value + step if higher_is_better else max(0.0, value - step)


def compute_personal_thresholds(analysis_daily: pd.DataFrame) -> Dict[str, Any]:
    thresholds: Dict[str, Any] = {
        "min_sleep_hours_for_hrv": 7.0,
        "target_sleep_hours_for_hrv": 8.0,
        "max_bedtime_dev_for_rhr": 30.0,
        "trimp_sweet_spot_low": 60.0,
        "trimp_sweet_spot_high": 120.0,
        "step_floor": 6000.0,
        "step_target": 9000.0,
        "threshold_source_flags": {
            "min_sleep_hours_for_hrv": "fallback",
            "target_sleep_hours_for_hrv": "fallback",
            "max_bedtime_dev_for_rhr": "fallback",
            "trimp_sweet_spot_low": "fallback",
            "trimp_sweet_spot_high": "fallback",
            "step_floor": "fallback",
            "step_target": "fallback",
        },
    }
    if analysis_daily is None or analysis_daily.empty:
        return thresholds

    sleep_curve = dose_response(analysis_daily, "sleep_hours", "hrv_rmssd", lag_days=1, bins=5, higher_is_better=True)
    sleep_sweet = sleep_curve.get("sweet_spot")
    if sleep_sweet and dose_response_has_enough_support(sleep_curve, min_pairs=28, min_bins=4, min_sweet_bin_n=5):
        thresholds["min_sleep_hours_for_hrv"] = max(6.0, float(sleep_sweet.get("x_low") or thresholds["min_sleep_hours_for_hrv"]))
        thresholds["target_sleep_hours_for_hrv"] = max(thresholds["min_sleep_hours_for_hrv"], float(sleep_sweet.get("x_high") or thresholds["target_sleep_hours_for_hrv"]))
        thresholds["threshold_source_flags"]["min_sleep_hours_for_hrv"] = "learned"
        thresholds["threshold_source_flags"]["target_sleep_hours_for_hrv"] = "learned"

    bedtime_curve = dose_response(analysis_daily, "bedtime_dev", "resting_hr", lag_days=1, bins=5, higher_is_better=False)
    bedtime_sweet = bedtime_curve.get("sweet_spot")
    if bedtime_sweet and dose_response_has_enough_support(bedtime_curve, min_pairs=28, min_bins=4, min_sweet_bin_n=5):
        thresholds["max_bedtime_dev_for_rhr"] = max(15.0, float(bedtime_sweet.get("x_high") or thresholds["max_bedtime_dev_for_rhr"]))
        thresholds["threshold_source_flags"]["max_bedtime_dev_for_rhr"] = "learned"

    trimp_curve = dose_response(analysis_daily, "trimp", "hrv_rmssd", lag_days=2, bins=5, higher_is_better=True)
    trimp_sweet = trimp_curve.get("sweet_spot")
    if trimp_sweet and dose_response_has_enough_support(trimp_curve, min_pairs=20, min_bins=4, min_sweet_bin_n=4):
        thresholds["trimp_sweet_spot_low"] = max(20.0, float(trimp_sweet.get("x_low") or thresholds["trimp_sweet_spot_low"]))
        thresholds["trimp_sweet_spot_high"] = max(thresholds["trimp_sweet_spot_low"], float(trimp_sweet.get("x_high") or thresholds["trimp_sweet_spot_high"]))
        thresholds["threshold_source_flags"]["trimp_sweet_spot_low"] = "learned"
        thresholds["threshold_source_flags"]["trimp_sweet_spot_high"] = "learned"

    steps_curve = dose_response(analysis_daily, "steps", "hrv_rmssd", lag_days=1, bins=5, higher_is_better=True)
    steps_sweet = steps_curve.get("sweet_spot")
    if steps_sweet and dose_response_has_enough_support(steps_curve, min_pairs=28, min_bins=4, min_sweet_bin_n=5):
        thresholds["step_floor"] = max(4000.0, float(steps_sweet.get("x_low") or thresholds["step_floor"]))
        thresholds["step_target"] = max(thresholds["step_floor"], float(steps_sweet.get("x_high") or thresholds["step_target"]))
        thresholds["threshold_source_flags"]["step_floor"] = "learned"
        thresholds["threshold_source_flags"]["step_target"] = "learned"

    return thresholds


def learn_personal_thresholds(daily: pd.DataFrame) -> Dict[str, Any]:
    return compute_personal_thresholds(daily)


def respiratory_rate_context(latest_row: pd.Series, daily: pd.DataFrame) -> Optional[str]:
    latest_dev = _safe_float(latest_row.get("resp_rate_dev"))
    if latest_dev is None:
        latest_rate = _safe_float(latest_row.get("resp_rate"))
        if latest_rate is None or daily is None or daily.empty or "resp_rate" not in daily.columns:
            return None
        recent = daily[["day", "resp_rate"]].copy()
        recent["day"] = pd.to_datetime(recent["day"], errors="coerce")
        recent["resp_rate"] = pd.to_numeric(recent["resp_rate"], errors="coerce")
        recent = recent.dropna(subset=["day", "resp_rate"]).sort_values("day")
        if recent.empty:
            return None
        latest_day = pd.to_datetime(latest_row.get("day"), errors="coerce")
        if pd.isna(latest_day):
            latest_day = recent["day"].max()
        baseline_pool = recent[recent["day"] < latest_day]["resp_rate"].tail(28)
        if len(baseline_pool) < 10:
            return None
        latest_dev = latest_rate - float(baseline_pool.median())

    if latest_dev is None or abs(latest_dev) < 0.5:
        return None
    direction = "elevated" if latest_dev > 0 else "suppressed"
    return f"Respiratory rate {direction} ({latest_dev:+.1f}/min)"


def diagnostic_context_chips(daily: pd.DataFrame, thresholds: Dict[str, Any]) -> List[str]:
    if daily is None or daily.empty:
        return []
    latest = daily.sort_values("day").iloc[-1]
    chips: List[str] = []
    bedtime_cap = _safe_float(thresholds.get("max_bedtime_dev_for_rhr")) or 30.0
    bedtime_dev = _safe_float(latest.get("bedtime_dev"))
    if bedtime_dev is not None and bedtime_dev > bedtime_cap:
        chips.append(f"Sleep regularity strained ({bedtime_dev:.0f} min drift)")
    temp_dev = _safe_float(latest.get("temp_dev"))
    if temp_dev is not None and abs(temp_dev) > 0.3:
        chips.append(f"Temp deviation {temp_dev:+.1f} C")
    resp_chip = respiratory_rate_context(latest, daily)
    if resp_chip:
        chips.append(resp_chip)
    spo2 = _safe_float(latest.get("spo2"))
    if spo2 is not None and spo2 < 95:
        chips.append(f"SpO₂ soft ({spo2:.0f}%)")
    stress_high = _safe_float(latest.get("stress_high"))
    if stress_high is not None and stress_high > 120:
        chips.append("High daytime stress")
    if bool_from_value(latest.get("behavior_illness_tag")):
        chips.append("Illness logged")
    if bool_from_value(latest.get("travel_tag")):
        chips.append("Travel / schedule disruption")
    return chips[:4]


def build_three_day_protocol(
    daily: pd.DataFrame,
    *,
    hrv_signal: MetricSignal,
    rhr_signal: MetricSignal,
    day_mode: DayMode,
    protocol_actions: List[Dict[str, str]],
    training_rx: TrainingPrescription,
) -> pd.DataFrame:
    latest = daily.sort_values("day").iloc[-1] if daily is not None and not daily.empty else pd.Series(dtype=object)
    projected_hrv = project_next_metric(_safe_float(hrv_signal.get("value")), _safe_float(hrv_signal.get("baseline7")), higher_is_better=True)
    projected_rhr = project_next_metric(_safe_float(rhr_signal.get("value")), _safe_float(rhr_signal.get("baseline7")), higher_is_better=False)
    sleep_target = "8h+ in bed"
    bedtime_target = latest.get("median_bedtime_clock")
    if isinstance(bedtime_target, str) and bedtime_target.strip():
        sleep_target = f"8h+ in bed by {format_clock_from_anchor((float(latest.get('median_bedtime_min')) - 15.0) % (24 * 60)) if _safe_float(latest.get('median_bedtime_min')) is not None else bedtime_target}"

    yesterday_training = str(latest.get("last_workout_type") or "—")
    today_action = protocol_actions[0]["action"] if protocol_actions else training_rx["instruction"]
    tomorrow_training = "Z2 base if AM signals improve" if str(day_mode.get("mode")) in {"RECOVER", "MAINTAIN"} else "Repeat planned build session only if HRV rises and RHR falls"
    return pd.DataFrame(
        [
            {
                "Metric": "HRV",
                "Yesterday (actual)": format_metric_value("hrv_rmssd", _safe_float(hrv_signal.get("value"))) + f" ({hrv_signal.get('signal')})" if hrv_signal.get("value") is not None else "—",
                "Today (plan)": f"Target: > {format_metric_value('hrv_rmssd', projected_hrv)}" if projected_hrv is not None else "Protect recovery",
                "Tomorrow (projected)": format_metric_value("hrv_rmssd", projected_hrv),
            },
            {
                "Metric": "Resting HR",
                "Yesterday (actual)": format_metric_value("resting_hr", _safe_float(rhr_signal.get("value"))) + f" ({rhr_signal.get('signal')})" if rhr_signal.get("value") is not None else "—",
                "Today (plan)": f"Target: < {format_metric_value('resting_hr', projected_rhr)}" if projected_rhr is not None else "Keep strain low",
                "Tomorrow (projected)": format_metric_value("resting_hr", projected_rhr),
            },
            {
                "Metric": "Training",
                "Yesterday (actual)": yesterday_training,
                "Today (plan)": training_rx["session"],
                "Tomorrow (projected)": tomorrow_training,
            },
            {
                "Metric": "Sleep target",
                "Yesterday (actual)": f"{_safe_float(latest.get('sleep_hours')):.1f}h" if _safe_float(latest.get("sleep_hours")) is not None else "—",
                "Today (plan)": sleep_target,
                "Tomorrow (projected)": "7.5h+ if protocol is followed",
            },
            {
                "Metric": "Key action",
                "Yesterday (actual)": "—",
                "Today (plan)": today_action,
                "Tomorrow (projected)": "Reassess AM signals and progress only if HRV/RHR move the right way",
            },
        ]
    )


def compute_weekly_habit_scorecard(
    daily: pd.DataFrame,
    *,
    thresholds: Optional[Dict[str, Any]] = None,
    goal: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    if daily.empty:
        return pd.DataFrame(), {}

    recent = daily.sort_values("day").tail(7).copy()
    tags_available = bool(recent.get("tags_available").fillna(False).any()) if "tags_available" in recent.columns else False
    thresholds = thresholds or compute_personal_thresholds(daily)
    goal_key = str(goal or "").lower()
    sleep_hours_min = _safe_float(thresholds.get("min_sleep_hours_for_hrv")) or 7.0
    sleep_hours_target = _safe_float(thresholds.get("target_sleep_hours_for_hrv")) or 8.0
    bedtime_dev_max = _safe_float(thresholds.get("max_bedtime_dev_for_rhr")) or 30.0
    steps_floor = _safe_float(thresholds.get("step_floor")) or 6000.0
    steps_target = _safe_float(thresholds.get("step_target")) or 9000.0
    trimp_low = _safe_float(thresholds.get("trimp_sweet_spot_low")) or 60.0
    trimp_high = _safe_float(thresholds.get("trimp_sweet_spot_high")) or 120.0

    if "stress" in goal_key or "recovery" in goal_key:
        habits = [
            (f"Sleep >= {sleep_hours_target:.1f}h", lambda row: None if _safe_float(row.get("sleep_total_s")) is None else (_safe_float(row.get("sleep_total_s")) / 3600.0) >= sleep_hours_target),
            (f"Bedtime drift <= {bedtime_dev_max:.0f}m", lambda row: None if _safe_float(row.get("bedtime_dev")) is None else _safe_float(row.get("bedtime_dev")) <= bedtime_dev_max),
            ("No alcohol / no late meal", lambda row: None if not tags_available else (not bool_from_value(row.get("alcohol_tag")) and not bool_from_value(row.get("late_meal_tag")))),
            (f"Training load <= {trimp_high:.0f} TRIMP", lambda row: None if _safe_float(row.get("trimp")) is None else (_safe_float(row.get("trimp")) <= trimp_high)),
            ("No late caffeine", lambda row: None if "behavior_caffeine_timing_tag" not in row else not bool_from_value(row.get("behavior_caffeine_timing_tag"))),
        ]
    elif "vo2" in goal_key or "performance" in goal_key:
        habits = [
            (f"Sleep >= {sleep_hours_min:.1f}h", lambda row: None if _safe_float(row.get("sleep_total_s")) is None else (_safe_float(row.get("sleep_total_s")) / 3600.0) >= sleep_hours_min),
            (f"Steps >= {steps_target:.0f}", lambda row: None if _safe_float(row.get("steps")) is None else _safe_float(row.get("steps")) >= steps_target),
            (f"Training load {trimp_low:.0f}-{trimp_high:.0f} TRIMP", lambda row: None if _safe_float(row.get("trimp")) is None else trimp_low <= _safe_float(row.get("trimp")) <= trimp_high),
            ("No alcohol / no late meal", lambda row: None if not tags_available else (not bool_from_value(row.get("alcohol_tag")) and not bool_from_value(row.get("late_meal_tag")))),
            (f"Bedtime drift <= {bedtime_dev_max:.0f}m", lambda row: None if _safe_float(row.get("bedtime_dev")) is None else _safe_float(row.get("bedtime_dev")) <= bedtime_dev_max),
        ]
    elif "fat" in goal_key or "metabolic" in goal_key or "body composition" in goal_key:
        habits = [
            (f"Sleep >= {sleep_hours_min:.1f}h", lambda row: None if _safe_float(row.get("sleep_total_s")) is None else (_safe_float(row.get("sleep_total_s")) / 3600.0) >= sleep_hours_min),
            (f"Steps >= {steps_target:.0f}", lambda row: None if _safe_float(row.get("steps")) is None else _safe_float(row.get("steps")) >= steps_target),
            ("No alcohol / no late meal", lambda row: None if not tags_available else (not bool_from_value(row.get("alcohol_tag")) and not bool_from_value(row.get("late_meal_tag")))),
            (f"Training load <= {trimp_high:.0f} TRIMP", lambda row: None if _safe_float(row.get("trimp")) is None else (_safe_float(row.get("trimp")) <= trimp_high)),
            (f"Bedtime drift <= {bedtime_dev_max:.0f}m", lambda row: None if _safe_float(row.get("bedtime_dev")) is None else _safe_float(row.get("bedtime_dev")) <= bedtime_dev_max),
        ]
    else:
        habits = [
            (f"Sleep >= {sleep_hours_min:.1f}h", lambda row: None if _safe_float(row.get("sleep_total_s")) is None else (_safe_float(row.get("sleep_total_s")) / 3600.0) >= sleep_hours_min),
            (f"Bedtime drift <= {bedtime_dev_max:.0f}m", lambda row: None if _safe_float(row.get("bedtime_dev")) is None else _safe_float(row.get("bedtime_dev")) <= bedtime_dev_max),
            (f"Steps >= {steps_floor:.0f}", lambda row: None if _safe_float(row.get("steps")) is None else _safe_float(row.get("steps")) >= steps_floor),
            ("No alcohol / no late meal", lambda row: None if not tags_available else (not bool_from_value(row.get("alcohol_tag")) and not bool_from_value(row.get("late_meal_tag")))),
            (f"Training load {trimp_low:.0f}-{trimp_high:.0f} TRIMP", lambda row: None if _safe_float(row.get("trimp")) is None else trimp_low <= _safe_float(row.get("trimp")) <= trimp_high),
        ]

    rows: List[Dict[str, Any]] = []
    for _, row in recent.iterrows():
        entry: Dict[str, Any] = {"day": str(row.get("day"))}
        hit_count = 0
        for name, check in habits:
            result = check(row)
            entry[name] = result
            if result is True:
                hit_count += 1
        entry["habit_hits"] = hit_count
        entry["hrv_rmssd"] = _safe_float(row.get("hrv_rmssd"))
        rows.append(entry)

    score_df = pd.DataFrame(rows)
    hrv_high = score_df[score_df["habit_hits"] >= 4]["hrv_rmssd"].dropna()
    hrv_low = score_df[score_df["habit_hits"] < 3]["hrv_rmssd"].dropna()
    summary = {
        "high_hit_hrv": float(hrv_high.mean()) if len(hrv_high) else None,
        "low_hit_hrv": float(hrv_low.mean()) if len(hrv_low) else None,
    }
    if summary["high_hit_hrv"] is not None and summary["low_hit_hrv"] is not None:
        summary["hrv_delta"] = float(summary["high_hit_hrv"]) - float(summary["low_hit_hrv"])
    else:
        summary["hrv_delta"] = None
    summary["thresholds"] = thresholds
    summary["goal"] = goal
    return score_df, summary


def describe_driver_row(row: pd.Series) -> str:
    factor = str(row.get("factor") or "Factor")
    kind = str(row.get("kind") or "")
    if kind == "binary":
        present = bool_from_value(row.get("current_value"))
        if present:
            return f"{factor}: present"
        return f"{factor}: not tagged"

    display = str(row.get("current_display") or "—")
    risk_direction = str(row.get("risk_direction") or "")
    if risk_direction == "lower":
        return f"{factor}: low ({display})"
    if risk_direction == "higher":
        return f"{factor}: high ({display})"
    return f"{factor}: {display}"


def summarize_driver_analysis(drivers: pd.DataFrame, *, metric: str) -> Dict[str, Any]:
    if drivers is None or drivers.empty:
        return {"headline": None, "top_rows": pd.DataFrame(), "primary_lever": None}

    latest = drivers.iloc[0]
    latest_delta = _safe_float(latest.get("latest_delta"))
    latest_value = _safe_float(latest.get("latest_value"))
    prior_avg = _safe_float(latest.get("latest_prior_avg"))
    if metric == "hrv_rmssd":
        headline = (
            f"HRV changed {latest_delta:+.1f} ms vs your prior 7-night baseline."
            if latest_delta is not None
            else f"Latest HRV: {format_metric_value(metric, latest_value)}."
        )
    else:
        headline = (
            f"Resting HR changed {latest_delta:+.1f} bpm vs your prior 7-night baseline."
            if latest_delta is not None
            else f"Latest Resting HR: {format_metric_value(metric, latest_value)}."
        )
    if prior_avg is not None:
        headline += f" Baseline: {format_metric_value(metric, prior_avg)}."

    triggered = drivers[drivers["current_triggered"] == True]
    ordered = pd.concat([triggered, drivers], ignore_index=True).drop_duplicates(subset=["factor"]).head(3)
    lever_row = triggered.head(1)
    if lever_row.empty:
        lever_row = drivers.head(1)
    primary_lever = describe_driver_row(lever_row.iloc[0]) if not lever_row.empty else None
    return {"headline": headline, "top_rows": ordered, "primary_lever": primary_lever}


def compute_vo2(data: Dict[str, Dict[str, Any]]) -> Tuple[Optional[float], pd.DataFrame]:
    # Prefer the doc that actually has rows
    df1 = data.get("vO2_max", {}).get("df", pd.DataFrame())
    df2 = data.get("vo2_max", {}).get("df", pd.DataFrame())
    df = df1 if len(df1) else df2

    vo2 = latest(df, ["vo2_max", "value", "vo2max"])
    vo2f = _safe_float(vo2)

    if not df.empty and "day" not in df.columns:
        if "timestamp" in df.columns:
            df = df.copy()
            df["day"] = df["timestamp"].apply(_to_day)

    if not df.empty:
        cols = [c for c in ["day", "timestamp", "vo2_max", "value", "vo2max"] if c in df.columns]
        df = df[cols].copy() if cols else df
        if "day" in df.columns:
            df = df.dropna(subset=["day"]).sort_values("day")
    return vo2f, df


def compute_cva(data: Dict[str, Dict[str, Any]]) -> Tuple[Optional[float], pd.DataFrame]:
    df = data.get("heart_health", {}).get("df", pd.DataFrame())
    cva = latest(df, ["cardiovascular_age", "cva"])
    cvaf = _safe_float(cva)

    if not df.empty and "day" not in df.columns:
        if "timestamp" in df.columns:
            df = df.copy()
            df["day"] = df["timestamp"].apply(_to_day)

    if not df.empty:
        cols = [c for c in ["day", "timestamp", "cardiovascular_age", "cva"] if c in df.columns]
        df = df[cols].copy() if cols else df
        if "day" in df.columns:
            df = df.dropna(subset=["day"]).sort_values("day")
    return cvaf, df


# ------------------------------
# UI
# ------------------------------


def inject_brand_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

        :root {
          --pw-bg: #f8f9fb;
          --pw-bg-soft: #f2f5f7;
          --pw-surface: rgba(255, 255, 255, 0.82);
          --pw-surface-strong: rgba(255, 255, 255, 0.95);
          --pw-ink: #0f1115;
          --pw-ink-soft: #111827;
          --pw-muted: #98a3af;
          --pw-muted-strong: #66727f;
          --pw-line: rgba(17, 24, 39, 0.07);
          --pw-line-strong: rgba(17, 24, 39, 0.12);
          --pw-glacier: rgba(206, 218, 232, 0.42);
          --pw-mist: rgba(190, 203, 217, 0.25);
          --pw-accent: #334155;
          --pw-shadow: 0 20px 60px rgba(15, 17, 21, 0.045);
          --pw-shadow-soft: 0 10px 28px rgba(15, 17, 21, 0.03);
          --pw-radius: 22px;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
          background:
            radial-gradient(circle at top right, var(--pw-glacier) 0%, rgba(206, 218, 232, 0) 36%),
            radial-gradient(circle at 15% 24%, rgba(232, 238, 244, 0.9) 0%, rgba(232, 238, 244, 0) 24%),
            radial-gradient(circle at bottom left, var(--pw-mist) 0%, rgba(190, 203, 217, 0) 32%),
            linear-gradient(180deg, #fbfcfd 0%, var(--pw-bg) 50%, var(--pw-bg-soft) 100%);
          color: var(--pw-ink);
          font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stHeader"] {
          background: rgba(245, 245, 247, 0.65);
          backdrop-filter: blur(18px);
          border-bottom: 1px solid rgba(17, 24, 39, 0.05);
        }

        [data-testid="stToolbar"] {
          right: 1rem;
        }

        [data-testid="stSidebar"] {
          background:
            linear-gradient(180deg, rgba(248, 249, 251, 0.96) 0%, rgba(242, 245, 247, 0.94) 100%);
          border-right: 1px solid var(--pw-line);
        }

        section[data-testid="stSidebar"] > div {
          padding-top: 1.2rem;
        }

        .block-container {
          max-width: 1440px;
          padding-top: 2rem;
          padding-bottom: 5rem;
          padding-left: 1.35rem;
          padding-right: 1.35rem;
        }

        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
          font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: var(--pw-ink-soft);
          letter-spacing: -0.03em;
          line-height: 0.96;
          font-weight: 700;
        }

        p, li, div, span, label {
          font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
          color: #2e3945;
          line-height: 1.68;
          letter-spacing: -0.01em;
        }

        [data-testid="stMarkdownContainer"] ul,
        [data-testid="stMarkdownContainer"] ol {
          margin-top: 0.35rem;
          margin-bottom: 0.8rem;
          padding-left: 1.2rem;
        }

        [data-testid="stMarkdownContainer"] li {
          margin-bottom: 0.48rem;
        }

        [data-testid="stCaptionContainer"] p {
          font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          font-size: 0.88rem;
          line-height: 1.62;
          letter-spacing: -0.01em;
          color: #7a8794;
          text-transform: none;
        }

        [data-testid="stWidgetLabel"] p {
          font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          font-size: 0.74rem;
          font-weight: 600;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: #7f8895;
        }

        .pw-ref,
        .pw-overline,
        .pw-chipline {
          font-family: "Space Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: var(--pw-muted-strong);
        }

        .pw-hero {
          position: relative;
          overflow: hidden;
          margin: 0 0 2.7rem 0;
          padding: 2.8rem 2.7rem 2.35rem;
          border: 1px solid var(--pw-line);
          border-radius: 28px;
          background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(249, 251, 252, 0.92) 44%, rgba(241, 245, 248, 0.96) 100%);
          box-shadow: var(--pw-shadow);
          backdrop-filter: blur(16px);
        }

        .pw-hero::after {
          content: "";
          position: absolute;
          inset: auto -10% -35% auto;
          width: 26rem;
          height: 26rem;
          background: radial-gradient(circle, rgba(173, 189, 204, 0.22) 0%, rgba(173, 189, 204, 0) 68%);
          pointer-events: none;
        }

        .pw-hero h1 {
          margin: 0;
          font-size: clamp(3rem, 7vw, 5.7rem);
          line-height: 0.9;
          letter-spacing: -0.055em;
          color: var(--pw-ink);
          max-width: 11ch;
        }

        .pw-hero .pw-subtitle {
          margin-top: 0.28rem;
          font-size: clamp(2.5rem, 6vw, 5.2rem);
          line-height: 0.94;
          letter-spacing: -0.055em;
          color: var(--pw-muted);
          font-weight: 500;
          max-width: 12ch;
        }

        .pw-hero .pw-body {
          margin-top: 1.2rem;
          max-width: 50rem;
          color: #6f7c88;
          font-size: 1rem;
          line-height: 1.72;
        }

        .pw-hero .pw-ref {
          margin-bottom: 1.1rem;
          font-size: 0.82rem;
        }

        .stTabs [data-baseweb="tab-list"] {
          gap: 1.5rem;
          border-bottom: 1px solid var(--pw-line);
          margin-bottom: 0.8rem;
          overflow-x: auto;
          overflow-y: hidden;
          scrollbar-width: none;
          flex-wrap: nowrap;
        }

        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
          display: none;
        }

        .stTabs [data-baseweb="tab"] {
          height: auto;
          padding: 0.15rem 0 1.05rem;
          background: transparent !important;
          color: var(--pw-muted-strong) !important;
          font-weight: 500;
          letter-spacing: -0.02em;
          flex: 0 0 auto;
        }

        .stTabs [aria-selected="true"] {
          color: var(--pw-ink-soft) !important;
          box-shadow: inset 0 -2px 0 0 var(--pw-ink-soft);
        }

        [data-testid="stMetric"] {
          background: linear-gradient(180deg, var(--pw-surface-strong) 0%, var(--pw-surface) 100%);
          border: 1px solid var(--pw-line);
          border-radius: var(--pw-radius);
          padding: 1.1rem 1.12rem;
          box-shadow: var(--pw-shadow-soft);
        }

        [data-testid="stMetricLabel"] {
          color: var(--pw-muted-strong);
          font-family: "Space Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: 0.72rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
          color: var(--pw-ink-soft);
          font-weight: 700;
          letter-spacing: -0.04em;
        }

        [data-testid="stMetricDelta"] {
          color: var(--pw-accent);
        }

        [data-testid="stAlert"],
        [data-testid="stDataFrame"],
        .stExpander,
        .stForm,
        [data-testid="stFileUploader"] {
          border-radius: var(--pw-radius);
        }

        [data-testid="stAlert"] {
          border: 1px solid var(--pw-line);
          background: rgba(255, 255, 255, 0.78);
          box-shadow: var(--pw-shadow-soft);
        }

        [data-testid="stDataFrame"] {
          border: 1px solid var(--pw-line);
          overflow: hidden;
          background: rgba(255, 255, 255, 0.78);
        }

        [data-testid="stDataFrame"] > div {
          overflow-x: auto;
        }

        .stExpander {
          border: 1px solid var(--pw-line) !important;
          background: rgba(255, 255, 255, 0.72);
          box-shadow: var(--pw-shadow-soft);
        }

        .stButton > button,
        .stDownloadButton > button,
        button[kind="primary"],
        button[kind="secondary"] {
          border-radius: 999px;
          border: 1px solid var(--pw-line-strong);
          background: rgba(255, 255, 255, 0.86);
          color: var(--pw-ink-soft);
          box-shadow: 0 8px 18px rgba(15, 17, 21, 0.04);
          font-weight: 600;
          letter-spacing: -0.01em;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        button[kind="primary"]:hover,
        button[kind="secondary"]:hover {
          border-color: rgba(17, 24, 39, 0.2);
          background: rgba(255, 255, 255, 0.98);
          color: var(--pw-ink);
        }

        .stTextInput input,
        .stDateInput input,
        .stTextArea textarea,
        .stSelectbox [data-baseweb="select"] > div,
        .stNumberInput input {
          border-radius: 16px !important;
          border-color: var(--pw-line-strong) !important;
          background: rgba(255, 255, 255, 0.78) !important;
          color: var(--pw-ink-soft) !important;
        }

        .stSlider [data-baseweb="slider"] [role="slider"] {
          background: var(--pw-ink-soft);
          box-shadow: 0 0 0 6px rgba(173, 189, 204, 0.18);
        }

        .stSlider [data-baseweb="slider"] > div > div {
          background: rgba(17, 24, 39, 0.12);
        }

        .stCheckbox label,
        .stRadio label {
          color: var(--pw-ink-soft);
        }

        .stMarkdown a {
          color: var(--pw-accent);
          text-decoration-color: rgba(51, 65, 85, 0.35);
        }

        hr, .stDivider {
          border-color: var(--pw-line);
        }

        .pw-page-intro {
          position: relative;
          overflow: hidden;
          margin: 0 0 1.7rem;
          padding: 1.8rem 1.85rem 1.55rem;
          border: 1px solid var(--pw-line);
          border-radius: 26px;
          background: linear-gradient(180deg, rgba(255,255,255,0.84) 0%, rgba(249,251,252,0.78) 100%);
          box-shadow: var(--pw-shadow-soft);
        }

        .pw-page-intro::after {
          content: "";
          position: absolute;
          inset: auto -8% -55% auto;
          width: 24rem;
          height: 24rem;
          background: radial-gradient(circle, rgba(200, 210, 230, 0.18) 0%, rgba(200, 210, 230, 0) 70%);
          pointer-events: none;
        }

        .pw-page-title {
          margin: 0.15rem 0 0;
          font-size: clamp(2.15rem, 4vw, 3.55rem);
          line-height: 0.98;
          letter-spacing: -0.05em;
          color: var(--pw-ink);
          max-width: 14ch;
        }

        .pw-page-body {
          margin-top: 0.8rem;
          max-width: 46rem;
          color: #6d7884;
          font-size: 1rem;
          line-height: 1.72;
        }

        .pw-page-subtitle {
          margin-top: 0.4rem;
          max-width: 18ch;
          color: rgba(255,255,255,0.96);
          font-size: clamp(1.65rem, 3vw, 2.5rem);
          font-weight: 600;
          letter-spacing: -0.045em;
          line-height: 1.02;
          text-wrap: balance;
        }

        .pw-page-hero {
          min-height: 340px;
          display: flex;
          align-items: flex-end;
          padding: 2.1rem 2rem;
          background:
            linear-gradient(90deg, rgba(14, 21, 28, 0.74) 0%, rgba(14, 21, 28, 0.46) 30%, rgba(14, 21, 28, 0.08) 62%),
            linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.18) 100%),
            url("https://cdn.shopify.com/s/files/1/0645/5030/5960/files/background.webp") center center / cover no-repeat;
          border-color: rgba(255,255,255,0.14);
        }

        .pw-page-hero-content {
          position: relative;
          z-index: 2;
          max-width: 34rem;
        }

        .pw-page-hero .pw-overline {
          color: rgba(255,255,255,0.72);
        }

        .pw-page-hero .pw-page-title {
          color: rgba(255,255,255,0.98);
          font-size: clamp(3.1rem, 7vw, 5.4rem);
          max-width: none;
        }

        .pw-page-hero .pw-page-body {
          margin-top: 0.95rem;
          max-width: 31rem;
          color: rgba(255,255,255,0.82);
        }

        .pw-action-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 0.92rem;
          margin: 0.45rem 0 0.9rem;
        }

        .pw-action-card,
        .pw-rail-card {
          border: 1px solid var(--pw-line);
          border-radius: 22px;
          background: rgba(255, 255, 255, 0.8);
          box-shadow: var(--pw-shadow-soft);
        }

        .pw-action-card {
          padding: 1rem 1.05rem 0.95rem;
        }

        .pw-action-topline {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 0.75rem;
          margin-bottom: 0.35rem;
        }

        .pw-action-head {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 0.7rem;
        }

        .pw-action-title {
          color: var(--pw-ink-soft);
          font-size: 1rem;
          line-height: 1.38;
          font-weight: 600;
          letter-spacing: -0.02em;
        }

        .pw-action-why {
          margin-top: 0.65rem;
          color: #75818d;
          font-size: 0.92rem;
          line-height: 1.58;
        }

        .pw-pill {
          flex: 0 0 auto;
          padding: 0.28rem 0.56rem;
          border-radius: 999px;
          background: rgba(206, 218, 232, 0.28);
          color: #5f6d79;
          font-size: 0.67rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .pw-chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 0.25rem;
        }

        .pw-chip {
          padding: 0.48rem 0.72rem;
          border-radius: 999px;
          border: 1px solid var(--pw-line);
          background: rgba(255,255,255,0.9);
          color: #64717d;
          font-size: 0.84rem;
          line-height: 1.2;
        }

        .pw-rail-card {
          padding: 1.15rem 1.1rem;
          margin-bottom: 0.95rem;
        }

        .pw-rail-title {
          margin-bottom: 0.52rem;
        }

        .pw-rail-value {
          color: var(--pw-ink);
          font-size: 1.34rem;
          font-weight: 700;
          letter-spacing: -0.03em;
          line-height: 1.08;
        }

        .pw-rail-copy {
          margin-top: 0.5rem;
          color: #73808c;
          font-size: 0.94rem;
          line-height: 1.62;
        }

        .pw-spacer-sm {
          height: 0.45rem;
        }

        .pw-spacer-md {
          height: 1rem;
        }

        @media (max-width: 1180px) {
          .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
          }

          section[data-testid="stSidebar"] {
            min-width: min(86vw, 340px) !important;
            max-width: min(86vw, 340px) !important;
          }

          div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: 0.9rem !important;
            align-items: stretch;
          }

          div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: min(300px, 100%) !important;
            flex: 1 1 300px !important;
            width: auto !important;
          }

          .pw-page-hero {
            min-height: 300px;
            padding: 1.75rem 1.35rem;
            background-position: 62% center;
          }

          .pw-page-hero-content {
            max-width: 28rem;
          }

          .pw-page-hero .pw-page-title {
            font-size: clamp(2.75rem, 8vw, 4.5rem);
          }

          .pw-page-subtitle {
            max-width: 15ch;
          }
        }

        @media (max-width: 900px) {
          .block-container {
            padding-top: 1.25rem;
            padding-left: 0.95rem;
            padding-right: 0.95rem;
            padding-bottom: 3.6rem;
          }

          .pw-hero {
            padding: 2rem 1.35rem 1.65rem;
            border-radius: 24px;
          }

          .pw-page-intro {
            padding: 1.35rem 1.2rem 1.15rem;
            border-radius: 22px;
          }
        }

        @media (max-width: 760px) {
          [data-testid="stSidebar"],
          [data-testid="collapsedControl"] {
            display: none !important;
          }

          .block-container {
            padding-top: 0.8rem;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-bottom: 2.8rem;
          }

          .pw-hero {
            margin-bottom: 1.35rem;
            padding: 1.45rem 1rem 1.1rem;
            border-radius: 20px;
          }

          .pw-hero::after {
            width: 18rem;
            height: 18rem;
            inset: auto -22% -44% auto;
          }

          .pw-hero .pw-ref {
            font-size: 0.72rem;
            margin-bottom: 0.7rem;
          }

          .pw-hero h1 {
            font-size: clamp(2.1rem, 14vw, 3.25rem);
            max-width: 8ch;
          }

          .pw-hero .pw-subtitle {
            margin-top: 0.18rem;
            font-size: clamp(1.9rem, 11vw, 3rem);
            max-width: 9ch;
          }

          .pw-hero .pw-body {
            margin-top: 0.85rem;
            max-width: none;
            font-size: 0.92rem;
            line-height: 1.52;
          }

          .pw-page-intro {
            margin-bottom: 1rem;
            padding: 1rem 0.92rem 0.95rem;
            border-radius: 18px;
          }

          .pw-page-hero {
            min-height: 248px;
            align-items: flex-end;
            padding: 1.15rem 0.95rem 1rem;
            background-position: 68% center;
          }

          .pw-page-hero-content {
            max-width: none;
          }

          .pw-page-title {
            font-size: clamp(1.8rem, 11vw, 2.7rem);
            max-width: none;
          }

          .pw-page-hero .pw-page-title {
            font-size: clamp(2.2rem, 13vw, 3.35rem);
            line-height: 0.92;
          }

          .pw-page-subtitle {
            margin-top: 0.28rem;
            max-width: 14ch;
            font-size: clamp(1.15rem, 6.6vw, 1.72rem);
            line-height: 1.04;
          }

          .pw-page-body,
          .pw-page-hero .pw-page-body {
            margin-top: 0.7rem;
            max-width: none;
            font-size: 0.91rem;
            line-height: 1.5;
          }

          .pw-action-card,
          .pw-rail-card {
            border-radius: 18px;
          }

          .pw-action-card {
            padding: 0.92rem 0.86rem 0.88rem;
          }

          .pw-action-topline,
          .pw-action-head {
            gap: 0.55rem;
          }

          .pw-action-title {
            font-size: 0.95rem;
            line-height: 1.32;
          }

          .pw-action-why,
          .pw-rail-copy {
            font-size: 0.88rem;
            line-height: 1.48;
          }

          .pw-pill {
            padding: 0.24rem 0.46rem;
            font-size: 0.61rem;
          }

          .pw-chip-row {
            gap: 0.45rem;
          }

          .pw-chip {
            padding: 0.4rem 0.56rem;
            font-size: 0.78rem;
          }

          .pw-rail-card {
            padding: 1rem 0.92rem;
            margin-bottom: 0.8rem;
          }

          .pw-rail-value {
            font-size: 1.15rem;
          }

          [data-testid="stMetric"] {
            padding: 0.95rem 0.92rem;
          }

          [data-testid="stMetricLabel"] {
            font-size: 0.66rem;
          }

          [data-testid="stMetricValue"] {
            font-size: clamp(1.55rem, 8vw, 2.15rem);
          }

          [data-testid="stCaptionContainer"] p {
            font-size: 0.82rem;
            line-height: 1.5;
          }

          .stButton > button,
          .stDownloadButton > button,
          button[kind="primary"],
          button[kind="secondary"] {
            width: 100%;
            min-height: 2.8rem;
            justify-content: center;
          }

          .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            margin-bottom: 0.55rem;
          }

          .stTabs [data-baseweb="tab"] {
            padding-bottom: 0.82rem;
            font-size: 0.94rem;
          }

          div[data-testid="stHorizontalBlock"] {
            flex-direction: column;
            gap: 0.8rem !important;
          }

          div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: 100% !important;
            width: 100% !important;
            flex: 1 1 100% !important;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_plotly_theme() -> None:
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color="#111827", size=13),
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0.58)",
            colorway=["#111827", "#555e68", "#78818a", "#adbdcc", "#3b4d80", "#a5cc4f"],
            margin=dict(l=24, r=24, t=44, b=24),
            title=dict(font=dict(size=18, family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color="#111827")),
            legend=dict(
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(17,24,39,0.08)",
                borderwidth=1,
                font=dict(size=12, color="#555e68"),
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="rgba(17,24,39,0.08)",
                tickfont=dict(color="#6b7280"),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(17,24,39,0.08)",
                zeroline=False,
                showline=False,
                tickfont=dict(color="#6b7280"),
            ),
        )
    )
    pio.templates["pythonwater"] = template
    pio.templates.default = "pythonwater"


def render_brand_hero() -> None:
    st.markdown(
        """
        <section class="pw-hero">
          <div class="pw-ref">PYTHON WATER // REF: BIO-OPT-V14</div>
          <h1>Biomarker Operating System</h1>
          <div class="pw-subtitle">for the Modern Physiology.</div>
          <p class="pw-body">
            Oura Deep Insights, reframed as a clean clinical cockpit for HRV, resting HR,
            VO₂max, exercise efficiency, and the next best action to improve cardiovascular aging.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_today_page_intro(goal_profile: GoalProfile, day_mode: DayMode) -> None:
    title = "Today"
    subtitle = str(day_mode.get("summary") or "Protect recovery and keep the next move clear.").strip()
    st.markdown(
        f"""
        <div class="pw-page-intro pw-page-hero">
          <div class="pw-page-hero-overlay"></div>
          <div class="pw-page-hero-content">
            <div class="pw-overline">{html.escape(goal_profile.label)}</div>
            <div class="pw-page-title">{html.escape(title)}</div>
            <div class="pw-page-subtitle">{html.escape(subtitle)}</div>
            <div class="pw-page-body">Do the first actions now. Set up tonight so tomorrow is better. Use Biomarkers for the longer trend.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_protocol_action_cards(actions: List[Dict[str, str]], *, limit: int = 4) -> None:
    if not actions:
        st.info("No immediate interventions triggered. Stay on plan and protect tonight's sleep.")
        return
    shown = actions[:limit]
    columns = st.columns(2, gap="large")
    for idx, action in enumerate(shown):
        with columns[idx % 2]:
            with st.container(border=True):
                strength = str(action.get("strength") or "").strip()
                overline = html.escape(str(action.get("timing") or "Now"))
                title = str(action.get("action") or "").strip()
                why = str(action.get("why") or "").strip()
                if strength:
                    st.markdown(
                        f"""
                        <div class="pw-action-topline">
                          <div class="pw-overline">{overline}</div>
                          <span class="pw-pill">{html.escape(strength)}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"<div class='pw-overline'>{overline}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='pw-action-title'>{html.escape(title)}</div>", unsafe_allow_html=True)
                st.caption(why)


def render_chip_cloud(chips: List[str]) -> None:
    if not chips:
        st.caption("No secondary strain markers are clearly elevated today.")
        return
    rendered = "".join(f"<span class='pw-chip'>{html.escape(str(chip))}</span>" for chip in chips)
    st.markdown(f"<div class='pw-chip-row'>{rendered}</div>", unsafe_allow_html=True)


def render_today_support_card(
    *,
    title: str,
    overline: str,
    body_lines: List[str],
    value: Optional[str] = None,
) -> None:
    body_html = "".join(f"<div class='pw-rail-copy'>{html.escape(str(line))}</div>" for line in body_lines if str(line).strip())
    value_html = f"<div class='pw-rail-value'>{html.escape(value)}</div>" if value else ""
    st.markdown(
        f"""
        <div class="pw-rail-card">
          <div class="pw-rail-title pw-overline">{html.escape(overline)}</div>
          {value_html}
          <div class="pw-action-title">{html.escape(title)}</div>
          {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_peer_comparison_tab(
    *,
    active_label: str,
    snapshots: List[Dict[str, Any]],
    community_name: Optional[str] = None,
) -> None:
    st.subheader("Community comparison")
    if community_name:
        st.caption(community_name)
    st.caption("Only members who opted in to community sharing are shown here. Compare raw biometrics for context, but use each person's own baseline as the primary read.")

    if len(snapshots) < 2:
        st.info("Add at least one more connected account in the sidebar to compare friends.")
        return

    valid = [snapshot for snapshot in snapshots if not snapshot.get("error")]
    invalid = [snapshot for snapshot in snapshots if snapshot.get("error")]
    for snapshot in invalid:
        st.warning(f"{snapshot.get('label')}: {snapshot.get('error')}")
    if len(valid) < 2:
        st.info("Not enough accounts with usable daily data to compare yet.")
        return

    card_columns = st.columns(min(3, len(valid)), gap="large")
    for idx, snapshot in enumerate(valid[:3]):
        compare_index = snapshot.get("compare_index")
        index_text = "—" if compare_index is None else f"{compare_index:.0f}/100"
        title = str(snapshot.get("display_name") or snapshot.get("label") or "Friend")
        if str(snapshot.get("label")) == str(active_label):
            title = f"{title} (you)"
        body_lines = [
            str(snapshot.get("baseline_summary") or ""),
            f"HRV: {format_metric_value('hrv_rmssd', _safe_float(snapshot.get('hrv_signal', {}).get('value')))} ({snapshot.get('hrv_signal', {}).get('signal', 'UNKNOWN')})",
            f"RHR: {format_metric_value('resting_hr', _safe_float(snapshot.get('rhr_signal', {}).get('value')))} ({snapshot.get('rhr_signal', {}).get('signal', 'UNKNOWN')})",
        ]
        with card_columns[idx % len(card_columns)]:
            render_today_support_card(
                title=title,
                overline="Relative to self",
                value=index_text,
                body_lines=body_lines,
            )

    normalized_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    trend_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    for snapshot in valid:
        label = str(snapshot.get("display_name") or snapshot.get("label") or "Friend")
        if str(snapshot.get("label")) == str(active_label):
            label = f"{label} (you)"
        hrv_signal = snapshot.get("hrv_signal", {})
        rhr_signal = snapshot.get("rhr_signal", {})
        sleep_signal = snapshot.get("sleep_signal", {})
        readiness_signal = snapshot.get("readiness_signal", {})
        activity_signal = snapshot.get("activity_signal", {})
        hrv_trust = snapshot.get("hrv_trust", {})
        rhr_trust = snapshot.get("rhr_trust", {})
        normalized_rows.append(
            {
                "Person": label,
                "Recovery index": "—" if snapshot.get("compare_index") is None else f"{snapshot['compare_index']:.0f}",
                "Status": snapshot.get("baseline_summary"),
                "HRV vs 28d": format_delta(_safe_float(hrv_signal.get("delta28")), " ms", 1),
                "RHR vs 28d": format_delta(_safe_float(rhr_signal.get("delta28")), " bpm", 1),
                "Sleep vs 28d": format_delta(_safe_float(sleep_signal.get("delta28")), "", 0),
                "Readiness vs 28d": format_delta(_safe_float(readiness_signal.get("delta28")), "", 0),
                "HRV trust": _fmt(hrv_trust.get("confidence")),
                "RHR trust": _fmt(rhr_trust.get("confidence")),
            }
        )
        raw_rows.append(
            {
                "Person": label,
                "HRV": format_metric_value("hrv_rmssd", _safe_float(hrv_signal.get("value"))),
                "Resting HR": format_metric_value("resting_hr", _safe_float(rhr_signal.get("value"))),
                "Sleep score": format_metric_value("sleep_score", _safe_float(sleep_signal.get("value"))),
                "Readiness": format_metric_value("readiness", _safe_float(readiness_signal.get("value"))),
                "Activity": format_metric_value("activity_score", _safe_float(activity_signal.get("value"))),
                "VO₂max": _fmt(None if snapshot.get("vo2_value") is None else f"{snapshot['vo2_value']:.1f}"),
            }
        )
        trend_rows.append(
            {
                "Person": label,
                "HRV trend": _fmt(None if snapshot.get("hrv_pattern", {}).get("trend_slope_ms_per_month") is None else f"{snapshot['hrv_pattern']['trend_slope_ms_per_month']:+.2f} ms/mo"),
                "RHR trend": _fmt(None if snapshot.get("rhr_trend", {}).get("slope_bpm_per_month") is None else f"{snapshot['rhr_trend']['slope_bpm_per_month']:+.2f} bpm/mo"),
                "VO₂ trend": _fmt(None if snapshot.get("vo2_trend", {}).get("slope_90d") is None else f"{snapshot['vo2_trend']['slope_90d']:+.2f} /90d"),
                "Interpretation": snapshot.get("baseline_summary"),
            }
        )
        stability_rows.append(
            {
                "Person": label,
                "HRV 7d CV": _fmt(None if snapshot.get("hrv_pattern", {}).get("cv_7d_current") is None else f"{snapshot['hrv_pattern']['cv_7d_current']:.1f}%"),
                "HRV residual noise": _fmt(None if snapshot.get("hrv_pattern", {}).get("residual_volatility_ms") is None else f"{snapshot['hrv_pattern']['residual_volatility_ms']:.1f} ms"),
                "RHR 28d CV": _fmt(None if snapshot.get("rhr_stability", {}).get("cv_percent") is None else f"{snapshot['rhr_stability']['cv_percent']:.1f}%"),
                "HRV trust": _fmt(snapshot.get("hrv_trust", {}).get("confidence")),
                "RHR trust": _fmt(snapshot.get("rhr_trust", {}).get("confidence")),
            }
        )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Relative to each person's own baseline")
        st.dataframe(pd.DataFrame(normalized_rows), use_container_width=True, hide_index=True)
        st.caption("Start here. This avoids the misleading trap of comparing raw HRV or raw resting HR across different bodies.")
    with right:
        st.markdown("### Raw biomarker snapshot")
        st.dataframe(pd.DataFrame(raw_rows), use_container_width=True, hide_index=True)
        st.caption("Use raw values for context, not for declaring a winner.")

    st.markdown("### Trend direction")
    st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)
    st.markdown("### Stability and confidence")
    st.dataframe(pd.DataFrame(stability_rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Oura Deep Insights", layout="wide")
    inject_brand_theme()
    configure_plotly_theme()
    restore_device_session_from_query(ACCOUNT_STORE_PATH)
    handle_oura_oauth_callback(ACCOUNT_STORE_PATH)
    # Personal-first mode: ignore community state until that feature is reintroduced.
    st.session_state["community_id"] = ""
    st.session_state["compare_account_ids"] = []
    oauth_config = oura_oauth_config()
    oauth_browser_enabled = browser_oura_oauth_enabled()
    oauth_flash_message = str(st.session_state.pop("oauth_flash_message", "") or "").strip()
    oauth_flash_error = str(st.session_state.pop("oauth_flash_error", "") or "").strip()
    if oauth_flash_message:
        st.success(oauth_flash_message)
    if oauth_flash_error:
        st.error(oauth_flash_error)

    public_invite_only = str(os.environ.get("OURA_PUBLIC_INVITE_ONLY", "")).strip().lower() in {"1", "true", "yes", "on"}
    bootstrap_owner_code = str(os.environ.get("OURA_BOOTSTRAP_CODE") or "").strip().upper()
    saved_accounts = load_connected_accounts(ACCOUNT_STORE_PATH)
    saved_communities = load_communities(ACCOUNT_STORE_PATH)
    session_token = os.environ.get("OURA_ACCESS_TOKEN", "").strip()
    session_account: Optional[Dict[str, Any]] = None
    active_account_id: Optional[str] = None
    compare_account_ids: List[str] = []
    current_device_session_token = str(st.session_state.get("device_session_token") or "").strip()
    current_community_id = str(st.session_state.get("community_id") or "")
    current_member_id = str(st.session_state.get("community_member_id") or "")
    if current_member_id and not any(str(account.get("id") or "") == current_member_id for account in saved_accounts):
        if current_device_session_token:
            delete_device_session(current_device_session_token)
        for state_key in ["community_id", "community_member_id", "active_account_id", "compare_account_ids", "device_session_token"]:
            st.session_state.pop(state_key, None)
        sync_device_session_query_param(None)
        current_community_id = ""
        current_member_id = ""
        current_device_session_token = ""

    refresh_value = _safe_float(st.session_state.get("auto_refresh_minutes"))
    if refresh_value is None:
        refresh_value = 6.0 if current_member_id else 0.0
    refresh_minutes = int(refresh_value)
    if refresh_minutes not in {0, 3, 6}:
        refresh_minutes = 6 if current_member_id else 0
    st.session_state["auto_refresh_minutes"] = refresh_minutes
    maybe_enable_auto_refresh(refresh_minutes)

    def personal_saved_accounts() -> List[Dict[str, Any]]:
        return [
            account
            for account in saved_accounts
            if not str(account.get("community_id") or "").strip()
        ]

    def render_community_controls(*, key_prefix: str, show_title: bool = True) -> None:
        nonlocal saved_accounts, saved_communities, session_account
        local_community_id = str(st.session_state.get("community_id") or "")
        local_member_id = str(st.session_state.get("community_member_id") or "")
        local_community = get_community_by_id(ACCOUNT_STORE_PATH, local_community_id) if local_community_id else None
        local_member_account = next((account for account in saved_accounts if str(account.get("id")) == local_member_id), None)

        if show_title:
            st.header("Community")

        refresh_options = {"Off": 0, "Every 3 min": 3, "Every 6 min": 6}
        current_refresh = int(_safe_float(st.session_state.get("auto_refresh_minutes")) or 0)
        if current_refresh not in refresh_options.values():
            current_refresh = 0
        refresh_labels = list(refresh_options.keys())
        selected_refresh = next((label for label, value in refresh_options.items() if value == current_refresh), "Off")
        refresh_label = st.selectbox(
            "Auto-refresh",
            options=refresh_labels,
            index=refresh_labels.index(selected_refresh),
            key=f"{key_prefix}_auto_refresh",
        )
        st.session_state["auto_refresh_minutes"] = refresh_options[refresh_label]

        if local_community is not None:
            community_accounts = accounts_for_community(ACCOUNT_STORE_PATH, str(local_community.get("id") or ""))
            shared_accounts = [account for account in community_accounts if _coerce_bool(account.get("share_enabled"))]
            st.caption(f"**{local_community.get('name')}**")
            if local_member_account is not None:
                is_owner = str(local_community.get("owner_account_id") or "") == str(local_member_account.get("id") or "")
                if is_owner:
                    st.caption("You created this community.")
                share_key = f"{key_prefix}_community_share_enabled_{local_member_id}"
                share_enabled = st.checkbox(
                    "Share my biomarker summary with this community",
                    value=_coerce_bool(local_member_account.get("share_enabled")),
                    key=share_key,
                )
                if share_enabled != _coerce_bool(local_member_account.get("share_enabled")):
                    update_connected_account_share_setting(ACCOUNT_STORE_PATH, str(local_member_account.get("id") or ""), share_enabled)
                    saved_accounts = load_connected_accounts(ACCOUNT_STORE_PATH)
                    local_member_account = next((account for account in saved_accounts if str(account.get("id")) == local_member_id), local_member_account)
                with st.expander("Invite a friend", expanded=False):
                    st.caption("Generate a one-time invitation code. Only existing members can create new invites.")
                    if st.button("Generate personal invitation code", key=f"{key_prefix}_generate_member_invite"):
                        try:
                            invite = generate_member_invitation(
                                ACCOUNT_STORE_PATH,
                                community_id=str(local_community.get("id") or ""),
                                inviter_account_id=str(local_member_account.get("id") or ""),
                            )
                            st.session_state["latest_invite_code"] = str(invite.get("invite_code") or "")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                    latest_invite_code = str(st.session_state.get("latest_invite_code") or "").strip().upper()
                    if latest_invite_code:
                        st.text_input("Latest invite code", value=latest_invite_code, disabled=True, key=f"{key_prefix}_latest_invite_code_display")
                st.caption(f"Members connected: {len(community_accounts)} | Sharing enabled: {len(shared_accounts)}")
            if st.button("Leave this community", key=f"{key_prefix}_leave_current_community"):
                if str(st.session_state.get("device_session_token") or "").strip():
                    delete_device_session(str(st.session_state.get("device_session_token") or "").strip())
                sync_device_session_query_param(None)
                for state_key in ["community_id", "community_member_id", "active_account_id", "compare_account_ids", "device_session_token"]:
                    st.session_state.pop(state_key, None)
                st.session_state["auto_refresh_minutes"] = 0
                st.rerun()

            with st.expander("Reconnect or update my Oura account", expanded=local_member_account is None):
                reconnect_label = st.text_input(
                    "Your display name",
                    value=str((local_member_account or {}).get("label") or ""),
                    key=f"{key_prefix}_community_reconnect_label",
                )
                reconnect_share_enabled = _coerce_bool((local_member_account or {}).get("share_enabled"))
                if oauth_browser_enabled:
                    if st.button("Connect Oura in browser", key=f"{key_prefix}_community_reconnect_browser"):
                        begin_oura_oauth_flow(
                            action="reconnect_account",
                            payload={
                                "label": reconnect_label,
                                "account_id": str((local_member_account or {}).get("id") or ""),
                                "community_id": str(local_community.get("id") or ""),
                                "share_enabled": reconnect_share_enabled,
                            },
                        )
                else:
                    st.caption("Browser Oura connect is disabled on this web host. Paste your token manually below.")
                with st.expander("Advanced: paste token manually", expanded=True):
                    reconnect_file = st.file_uploader(
                        "Upload oura_tokens.json",
                        type=["json"],
                        key=f"{key_prefix}_community_reconnect_file",
                    )
                    reconnect_input = st.text_area(
                        "Or paste token JSON / access token",
                        height=120,
                        key=f"{key_prefix}_community_reconnect_token_input",
                    )
                    if st.button("Save my Oura connection", key=f"{key_prefix}_community_reconnect_account"):
                        raw_token_input = reconnect_input
                        if reconnect_file is not None:
                            raw_token_input = reconnect_file.getvalue().decode("utf-8")
                        try:
                            saved = upsert_connected_account(
                                ACCOUNT_STORE_PATH,
                                label=reconnect_label,
                                token_input=raw_token_input,
                                account_id=str((local_member_account or {}).get("id") or ""),
                                community_id=str(local_community.get("id") or ""),
                                share_enabled=reconnect_share_enabled,
                            )
                            st.session_state["community_member_id"] = str(saved["id"])
                            st.session_state["active_account_id"] = str(saved["id"])
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
        else:
            personal_accounts = [
                account
                for account in saved_accounts
                if not str(account.get("community_id") or "").strip()
            ]
            personal_account_options = {str(account.get("id") or ""): str(account.get("label") or account.get("profile_name") or "You") for account in personal_accounts}
            current_personal_account = next(
                (
                    account
                    for account in personal_accounts
                    if str(account.get("id") or "") == str(st.session_state.get("active_account_id") or st.session_state.get("community_member_id") or "")
                ),
                None,
            )
            if current_personal_account is None and personal_accounts:
                current_personal_account = personal_accounts[0]

            with st.expander("Personal dashboard", expanded=current_personal_account is None):
                if current_personal_account is not None:
                    st.caption(f"Connected as **{current_personal_account.get('label') or current_personal_account.get('profile_name') or 'You'}**")
                    selected_personal_id = st.selectbox(
                        "Saved personal profile",
                        options=list(personal_account_options.keys()),
                        index=list(personal_account_options.keys()).index(str(current_personal_account.get("id") or "")),
                        format_func=lambda account_id: personal_account_options.get(str(account_id), str(account_id)),
                        key=f"{key_prefix}_personal_account_id",
                    )
                    if str(selected_personal_id) != str(current_personal_account.get("id") or ""):
                        current_personal_account = next(
                            (account for account in personal_accounts if str(account.get("id") or "") == str(selected_personal_id)),
                            current_personal_account,
                        )
                    if st.button("Open personal dashboard", key=f"{key_prefix}_open_personal_dashboard"):
                        st.session_state["community_id"] = ""
                        st.session_state["community_member_id"] = str(current_personal_account.get("id") or "")
                        st.session_state["active_account_id"] = str(current_personal_account.get("id") or "")
                        st.session_state["compare_account_ids"] = []
                        st.rerun()
                    st.divider()

                personal_label_default = str((current_personal_account or {}).get("label") or "")
                personal_label = st.text_input("Your name", value=personal_label_default, key=f"{key_prefix}_personal_label")
                if oauth_browser_enabled:
                    st.caption("Use browser connect first. The token uploader below is only an advanced fallback.")
                    if st.button("Connect Oura and open personal dashboard", key=f"{key_prefix}_personal_oauth_button"):
                        begin_oura_oauth_flow(
                            action="connect_personal",
                            payload={
                                "label": personal_label,
                                "account_id": str((current_personal_account or {}).get("id") or ""),
                            },
                        )
                else:
                    missing_items = [str(item) for item in oauth_config.get("missing", []) if str(item).strip()]
                    if missing_items:
                        st.caption(
                            "Browser Oura connect is not configured on this host yet. "
                            f"Missing: {', '.join(missing_items)}. Use the manual token fallback below."
                        )
                    else:
                        st.caption("Browser Oura connect is disabled on this host. Use the manual token fallback below.")
                with st.expander("Advanced fallback: add or update Oura token", expanded=current_personal_account is None and not oauth_browser_enabled):
                    personal_file = st.file_uploader("Upload oura_tokens.json", type=["json"], key=f"{key_prefix}_personal_file")
                    personal_token_input = st.text_area("Or paste token JSON / access token", height=120, key=f"{key_prefix}_personal_token_input")
                    if st.button("Save personal dashboard access", key=f"{key_prefix}_save_personal_dashboard"):
                        raw_token_input = personal_token_input
                        if personal_file is not None:
                            raw_token_input = personal_file.getvalue().decode("utf-8")
                        try:
                            saved = upsert_connected_account(
                                ACCOUNT_STORE_PATH,
                                label=personal_label,
                                token_input=raw_token_input,
                                account_id=str((current_personal_account or {}).get("id") or "") or None,
                                community_id=None,
                                share_enabled=False,
                            )
                            saved_accounts = load_connected_accounts(ACCOUNT_STORE_PATH)
                            st.session_state["community_id"] = ""
                            st.session_state["community_member_id"] = str(saved.get("id") or "")
                            st.session_state["active_account_id"] = str(saved.get("id") or "")
                            st.session_state["compare_account_ids"] = []
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

            bootstrap_unlocked = not public_invite_only
            if public_invite_only:
                st.caption("Community is invite-only. Connect your personal dashboard first, then join with a personal code generated by an existing member.")
                if bootstrap_owner_code:
                    owner_code = st.text_input(
                        "Owner bootstrap code",
                        type="password",
                        key=f"{key_prefix}_owner_bootstrap_code",
                        help="Only the owner code can create a new community on the public site.",
                    ).strip().upper()
                    bootstrap_unlocked = owner_code == bootstrap_owner_code
                    if owner_code and not bootstrap_unlocked:
                        st.warning("Owner code not recognized.")
                else:
                    st.caption("Owner creation is disabled until an owner bootstrap code is configured on the host.")
            else:
                st.caption("Connect your personal dashboard first. Community is optional and only used for comparison.")

            if bootstrap_unlocked:
                with st.expander("Create a community", expanded=not public_invite_only):
                    if current_personal_account is None:
                        st.info("Connect your personal dashboard first.")
                    else:
                        community_name = st.text_input("Community name", key=f"{key_prefix}_create_community_name")
                        owner_share_enabled = st.checkbox("Share my biomarker summary with this community", value=True, key=f"{key_prefix}_create_community_share_enabled")
                        if st.button("Create community", key=f"{key_prefix}_create_community_button"):
                            try:
                                community, account, invite = create_community_for_existing_account(
                                    ACCOUNT_STORE_PATH,
                                    community_name=community_name,
                                    account_id=str(current_personal_account.get("id") or ""),
                                    share_enabled=owner_share_enabled,
                                )
                                saved_accounts = load_connected_accounts(ACCOUNT_STORE_PATH)
                                st.session_state["community_id"] = str(community["id"])
                                st.session_state["community_member_id"] = str(account["id"])
                                st.session_state["active_account_id"] = str(account["id"])
                                st.session_state["latest_invite_code"] = str(invite.get("invite_code") or "")
                                st.rerun()
                            except Exception as exc:
                                st.error(str(exc))

            with st.expander("Join with personal invitation code", expanded=False):
                if current_personal_account is None:
                    st.info("Connect your personal dashboard first.")
                else:
                    invite_code = st.text_input("Personal invitation code", key=f"{key_prefix}_join_community_invite_code").strip().upper()
                    member_share_enabled = st.checkbox("Share my biomarker summary with this community", value=True, key=f"{key_prefix}_join_community_share_enabled")
                    if st.button("Join community", key=f"{key_prefix}_join_community_button"):
                        try:
                            community, account, invite = join_community_for_existing_account(
                                ACCOUNT_STORE_PATH,
                                invite_code=invite_code,
                                account_id=str(current_personal_account.get("id") or ""),
                                share_enabled=member_share_enabled,
                            )
                            saved_accounts = load_connected_accounts(ACCOUNT_STORE_PATH)
                            st.session_state["community_id"] = str(community["id"])
                            st.session_state["community_member_id"] = str(account["id"])
                            st.session_state["active_account_id"] = str(account["id"])
                            st.session_state["latest_invite_code"] = str(invite.get("invite_code") or "")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

            session_token_value = ""
            if session_token:
                st.caption("Using `OURA_ACCESS_TOKEN` from the environment for a private single-user session.")
            else:
                session_token_value = st.text_input("Private single-user access token", type="password", key=f"{key_prefix}_session_access_token").strip()
            if session_token or session_token_value:
                private_token = session_token or session_token_value
                session_account = {
                    "id": "__session__",
                    "label": "Current session",
                    "profile_name": "Current session",
                    "email": "",
                    "profile": {},
                    "token_bundle": {"access_token": private_token, "_fetched_at": _utcnow().isoformat()},
                    "community_id": "",
                    "share_enabled": False,
                    "created_at": "",
                    "updated_at": "",
                    "last_refreshed_at": "",
                }

    def render_personal_controls(*, key_prefix: str, show_title: bool = True) -> None:
        nonlocal saved_accounts, session_account

        if show_title:
            st.header("Oura access")

        refresh_options = {"Off": 0, "Every 3 min": 3, "Every 6 min": 6}
        current_refresh = int(_safe_float(st.session_state.get("auto_refresh_minutes")) or 0)
        if current_refresh not in refresh_options.values():
            current_refresh = 0
        refresh_labels = list(refresh_options.keys())
        selected_refresh = next((label for label, value in refresh_options.items() if value == current_refresh), "Off")
        refresh_label = st.selectbox(
            "Auto-refresh",
            options=refresh_labels,
            index=refresh_labels.index(selected_refresh),
            key=f"{key_prefix}_auto_refresh_personal",
        )
        st.session_state["auto_refresh_minutes"] = refresh_options[refresh_label]

        personal_accounts = personal_saved_accounts()
        personal_account_options = {
            str(account.get("id") or ""): str(account.get("label") or account.get("profile_name") or "You")
            for account in personal_accounts
        }
        current_personal_account = next(
            (
                account
                for account in personal_accounts
                if str(account.get("id") or "") == str(st.session_state.get("active_account_id") or st.session_state.get("community_member_id") or "")
            ),
            None,
        )
        if current_personal_account is None and personal_accounts:
            current_personal_account = personal_accounts[0]

        with st.expander("Personal dashboard", expanded=current_personal_account is None):
            if current_personal_account is not None:
                st.caption(f"Connected as **{current_personal_account.get('label') or current_personal_account.get('profile_name') or 'You'}**")
                selected_personal_id = st.selectbox(
                    "Saved personal profile",
                    options=list(personal_account_options.keys()),
                    index=list(personal_account_options.keys()).index(str(current_personal_account.get("id") or "")),
                    format_func=lambda account_id: personal_account_options.get(str(account_id), str(account_id)),
                    key=f"{key_prefix}_personal_account_id",
                )
                if str(selected_personal_id) != str(current_personal_account.get("id") or ""):
                    current_personal_account = next(
                        (account for account in personal_accounts if str(account.get("id") or "") == str(selected_personal_id)),
                        current_personal_account,
                    )
                if st.button("Open personal dashboard", key=f"{key_prefix}_open_personal_dashboard"):
                    st.session_state["community_id"] = ""
                    st.session_state["community_member_id"] = str(current_personal_account.get("id") or "")
                    st.session_state["active_account_id"] = str(current_personal_account.get("id") or "")
                    st.session_state["compare_account_ids"] = []
                    st.rerun()
                st.divider()

            personal_label_default = str((current_personal_account or {}).get("label") or "")
            personal_label = st.text_input("Your name", value=personal_label_default, key=f"{key_prefix}_personal_label")
            if oauth_browser_enabled:
                st.caption("Authorize Oura in the browser to load the full personal dashboard.")
                if st.button("Connect Oura", key=f"{key_prefix}_personal_oauth_button"):
                    begin_oura_oauth_flow(
                        action="connect_personal",
                        payload={
                            "label": personal_label,
                            "account_id": str((current_personal_account or {}).get("id") or ""),
                        },
                    )
            else:
                missing_items = [str(item) for item in oauth_config.get("missing", []) if str(item).strip()]
                if missing_items:
                    st.warning(f"Browser Oura connect is not configured on this host yet. Missing: {', '.join(missing_items)}.")
                else:
                    st.warning("Browser Oura connect is disabled on this host.")
            with st.expander("Advanced fallback: add or update Oura token", expanded=current_personal_account is None and not oauth_browser_enabled):
                personal_file = st.file_uploader("Upload oura_tokens.json", type=["json"], key=f"{key_prefix}_personal_file")
                personal_token_input = st.text_area("Or paste token JSON / access token", height=120, key=f"{key_prefix}_personal_token_input")
                if st.button("Save personal dashboard access", key=f"{key_prefix}_save_personal_dashboard"):
                    raw_token_input = personal_token_input
                    if personal_file is not None:
                        raw_token_input = personal_file.getvalue().decode("utf-8")
                    try:
                        saved = upsert_connected_account(
                            ACCOUNT_STORE_PATH,
                            label=personal_label,
                            token_input=raw_token_input,
                            account_id=str((current_personal_account or {}).get("id") or "") or None,
                            community_id=None,
                            share_enabled=False,
                        )
                        saved_accounts = load_connected_accounts(ACCOUNT_STORE_PATH)
                        st.session_state["community_id"] = ""
                        st.session_state["community_member_id"] = str(saved.get("id") or "")
                        st.session_state["active_account_id"] = str(saved.get("id") or "")
                        st.session_state["compare_account_ids"] = []
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

        with st.expander("Advanced: temporary session token", expanded=False):
            session_token_value = ""
            if session_token:
                st.caption("Using `OURA_ACCESS_TOKEN` from the environment for a private single-user session.")
            else:
                session_token_value = st.text_input("Private single-user access token", type="password", key=f"{key_prefix}_session_access_token").strip()
            if session_token or session_token_value:
                private_token = session_token or session_token_value
                session_account = {
                    "id": "__session__",
                    "label": "Current session",
                    "profile_name": "Current session",
                    "email": "",
                    "profile": {},
                    "token_bundle": {"access_token": private_token, "_fetched_at": _utcnow().isoformat()},
                    "community_id": "",
                    "share_enabled": False,
                    "created_at": "",
                    "updated_at": "",
                    "last_refreshed_at": "",
                }

    def render_mobile_view_controls() -> None:
        account_options: List[Dict[str, Any]] = []
        if session_account is not None:
            account_options.append(session_account)
        account_options.extend(personal_saved_accounts())

        if account_options:
            account_labels = {
                str(account["id"]): str(account.get("label") or account.get("profile_name") or "Friend")
                for account in account_options
            }
            active_ids = [str(account["id"]) for account in account_options]
            active_default = str(st.session_state.get("active_account_id") or active_ids[0])
            if active_default not in active_ids:
                active_default = active_ids[0]
            active_mobile = st.selectbox(
                "Dashboard account",
                options=active_ids,
                index=active_ids.index(active_default),
                format_func=lambda account_id: account_labels.get(str(account_id), str(account_id)),
                key="mobile_active_account_id",
            )
            st.session_state["active_account_id"] = active_mobile
        st.session_state["compare_account_ids"] = []

        goal_options = [
            "Performance (endurance)",
            "Performance (strength / hybrid)",
            "Longevity / healthspan",
            "Body composition",
            "Stress resilience",
        ]
        current_goal = str(st.session_state.get("goal") or goal_options[0])
        if current_goal not in goal_options:
            current_goal = goal_options[0]
        st.session_state["goal"] = st.selectbox(
            "Optimize for",
            options=goal_options,
            index=goal_options.index(current_goal),
            key="mobile_goal",
        )

        today_local = date.today()
        st.session_state["start_date"] = st.date_input(
            "Start date",
            value=st.session_state.get("start_date", today_local - timedelta(days=60)),
            key="mobile_start_date",
        )
        st.session_state["end_date"] = st.date_input(
            "End date",
            value=st.session_state.get("end_date", today_local),
            key="mobile_end_date",
        )
        st.session_state["wide_sparse_days"] = st.slider(
            "Sparse-metric lookback",
            min_value=90,
            max_value=730,
            value=int(_safe_float(st.session_state.get("wide_sparse_days")) or 365),
            step=30,
            key="mobile_wide_sparse_days",
        )
        st.session_state["analysis_lookback_days"] = st.slider(
            "Recovery-analysis lookback",
            min_value=90,
            max_value=180,
            value=int(_safe_float(st.session_state.get("analysis_lookback_days")) or 120),
            step=15,
            key="mobile_analysis_lookback_days",
        )
        st.session_state["max_hr"] = st.number_input(
            "Max HR",
            min_value=120,
            max_value=240,
            value=int(st.session_state.get("max_hr", int(os.environ.get("OURA_MAX_HR", "190")))),
            key="mobile_max_hr",
        )
        st.session_state["resting_hr"] = st.number_input(
            "Resting HR",
            min_value=30,
            max_value=120,
            value=int(st.session_state.get("resting_hr", int(os.environ.get("OURA_RESTING_HR", "50")))),
            key="mobile_resting_hr",
        )
        st.session_state["show_raw"] = st.checkbox(
            "Show raw JSON samples",
            value=bool(st.session_state.get("show_raw", False)),
            key="mobile_show_raw",
        )

    toolbar_left, toolbar_right = st.columns([12, 1])
    with toolbar_right:
        with st.popover("⚙️", use_container_width=True):
            render_personal_controls(key_prefix="mobile", show_title=False)
            st.divider()
            render_mobile_view_controls()

    render_brand_hero()

    with st.sidebar:
        render_personal_controls(key_prefix="sidebar", show_title=True)

    current_community_id = ""
    current_member_id = str(st.session_state.get("community_member_id") or "")
    current_community = None

    with st.sidebar:
        account_options: List[Dict[str, Any]] = []
        if session_account is not None:
            account_options.append(session_account)
        account_options.extend(personal_saved_accounts())

        account_labels = {str(account["id"]): str(account.get("label") or account.get("profile_name") or "Friend") for account in account_options}
        if account_options:
            default_active_id = current_member_id or str(account_options[0].get("id"))
            active_account_id = st.selectbox(
                "Dashboard account",
                options=[str(account["id"]) for account in account_options],
                index=max(0, [str(account["id"]) for account in account_options].index(default_active_id)) if default_active_id in [str(account["id"]) for account in account_options] else 0,
                format_func=lambda account_id: account_labels.get(str(account_id), str(account_id)),
                key="active_account_id",
            )
            compare_account_ids = []
        else:
            st.warning("Connect your Oura account or use a temporary access token to continue.")
            st.stop()

        st.caption(f"Account store: `{pathlib.Path(ACCOUNT_STORE_PATH).expanduser()}`")
        st.divider()
        st.header("Goal")
        goal = st.selectbox(
            "Optimize for",
            options=[
                "Performance (endurance)",
                "Performance (strength / hybrid)",
                "Longevity / healthspan",
                "Body composition",
                "Stress resilience",
            ],
            index=0,
            key="goal",
        )
        goal_profile = get_goal_profile(goal)
        st.caption(goal_profile.focus)
        st.divider()
        st.subheader("Date range")
        today = date.today()
        default_days = 60
        start_d = st.date_input("Start date", value=today - timedelta(days=default_days), key="start_date")
        end_d = st.date_input("End date", value=today, key="end_date")
        wide_sparse_days = st.slider("Sparse-metric lookback (VO₂ etc.)", 90, 730, 365, step=30, key="wide_sparse_days")
        analysis_lookback_days = st.slider("Recovery-analysis lookback", 90, 180, 120, step=15, key="analysis_lookback_days")
        st.caption("Retrospectives and dose-response curves can use up to this much history even if the visible dashboard range is shorter.")
        st.divider()
        st.subheader("Athlete settings")
        max_hr = st.number_input("Max HR", min_value=120, max_value=240, value=int(os.environ.get("OURA_MAX_HR", "190")), key="max_hr")
        resting_hr = st.number_input("Resting HR", min_value=30, max_value=120, value=int(os.environ.get("OURA_RESTING_HR", "50")), key="resting_hr")
        st.divider()
        st.subheader("Diagnostics")
        show_raw = st.checkbox("Show raw JSON samples", value=False, key="show_raw")

    all_saved_accounts = load_connected_accounts(ACCOUNT_STORE_PATH)
    saved_accounts = all_saved_accounts
    visible_saved_accounts = [
        account
        for account in saved_accounts
        if not str(account.get("community_id") or "").strip()
    ]

    account_options = ([session_account] if session_account is not None else []) + visible_saved_accounts
    if not account_options:
        st.error("Missing Oura account connection. Connect your Oura account in the sidebar, or use a temporary access token.")
        st.stop()

    account_refresh_warnings: List[str] = []
    registry_changed = False

    def resolve_account_for_use(account: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        nonlocal all_saved_accounts, saved_accounts, visible_saved_accounts, registry_changed
        if str(account.get("id")) == "__session__":
            token_value = str(account.get("token_bundle", {}).get("access_token") or "").strip()
            if not token_value:
                raise ValueError("Current session token is empty.")
            return token_value, account
        token_value, updated_account, warning, changed = resolve_connected_account(account)
        if warning:
            account_refresh_warnings.append(warning)
        if changed:
            all_saved_accounts = [
                updated_account if str(existing.get("id")) == str(updated_account.get("id")) else existing
                for existing in all_saved_accounts
            ]
            saved_accounts = all_saved_accounts
            visible_saved_accounts = [
                existing
                for existing in saved_accounts
                if not str(existing.get("community_id") or "").strip()
            ]
            registry_changed = True
        return token_value, updated_account

    active_account = next((account for account in account_options if str(account.get("id")) == str(active_account_id)), None)
    if active_account is None:
        active_account = account_options[0]
    current_device_session_token = str(st.session_state.get("device_session_token") or "").strip()
    if _coerce_bool(active_account.get("pending_connection")):
        active_account_id_value = str(active_account.get("id") or "")
        st.info("This dashboard profile exists, but its Oura connection is still incomplete. Reconnect Oura in the Personal dashboard panel when you want biomarker data to load.")
        st.stop()
    try:
        active_token, active_account = resolve_account_for_use(active_account)
    except Exception as exc:
        st.error(f"Could not use the selected account: {exc}")
        st.stop()
    active_account_id_value = str(active_account.get("id") or "")
    can_edit_active_account = True
    behavior_event_path = behavior_event_path_for_account(active_account_id_value)
    workout_intent_path = workout_intent_path_for_account(active_account_id_value)

    compare_accounts: List[Dict[str, Any]] = []
    for account_id in compare_account_ids:
        selected = next((account for account in saved_accounts if str(account.get("id")) == str(account_id)), None)
        if selected is not None:
            compare_accounts.append(selected)

    resolved_compare_accounts: List[Tuple[Dict[str, Any], str]] = []
    for account in compare_accounts:
        try:
            token_value, updated_account = resolve_account_for_use(account)
            resolved_compare_accounts.append((updated_account, token_value))
        except Exception as exc:
            account_refresh_warnings.append(f"{account.get('label')}: {exc}")

    if active_account_id_value and active_account_id_value != "__session__":
        session_record = persist_device_session(
            community_id="",
            member_id=active_account_id_value,
            active_account_id=active_account_id_value,
            compare_account_ids=[],
            refresh_minutes=int(st.session_state.get("auto_refresh_minutes") or 6),
            token=current_device_session_token or None,
        )
        st.session_state["device_session_token"] = str(session_record.get("token") or "")
        sync_device_session_query_param(str(session_record.get("token") or ""))
    else:
        sync_device_session_query_param(None)

    if registry_changed:
        store = load_account_store(ACCOUNT_STORE_PATH)
        save_account_store(
            accounts=all_saved_accounts,
            communities=store.get("communities", []),
            invitations=store.get("invitations", []),
            path=ACCOUNT_STORE_PATH,
        )

    if start_d > end_d:
        st.warning("Start date is after end date")
        st.stop()

    analysis_start_d = compute_analysis_range(
        start_d=start_d,
        end_d=end_d,
        analysis_lookback_days=int(analysis_lookback_days),
    )

    with st.spinner(f"Fetching Oura data for {active_account.get('label') or 'selected account'}…"):
        data = fetch_endpoints(active_token, start_d=start_d, end_d=end_d, wide_sparse_days=wide_sparse_days)
        analysis_data = data
        if analysis_start_d < start_d:
            analysis_data = fetch_endpoints(active_token, start_d=analysis_start_d, end_d=end_d, wide_sparse_days=wide_sparse_days)

    # Build unified daily frame
    daily = compute_daily_frame(data)
    analysis_daily = compute_daily_frame(analysis_data)
    vo2, vo2_df = compute_vo2(data)
    analysis_vo2, analysis_vo2_df = compute_vo2(analysis_data)
    cva, cva_df = compute_cva(data)

    # Personal info
    personal = data.get("personal", {}).get("doc", {}) if isinstance(data.get("personal", {}).get("doc", {}), dict) else {}
    birthdate = personal.get("birthdate")

    # Oura personal payload varies; allow manual override
    sex_guess = None
    for k in ["sex", "gender"]:
        if k in personal and personal.get(k):
            sex_guess = str(personal.get(k)).strip().upper()
            break

    sex = st.sidebar.selectbox(
        "Sex (for benchmarks)",
        options=["M", "F"],
        index=0 if (sex_guess or "M").startswith("M") else 1,
        help="Used only for percentile-style benchmarks (VO₂max/HRV/RHR).",
        key="sex",
    )

    age_years = None
    try:
        if birthdate:
            bd = dtparse.isoparse(str(birthdate))
            today_d = date.today()
            age_years = today_d.year - bd.date().year - (
                (today_d.month, today_d.day) < (bd.date().month, bd.date().day)
            )
    except Exception:
        age_years = None

    sleep_features = build_sleep_feature_frame(data)
    tag_features = build_tag_feature_frame(data)
    behavior_events = load_behavior_events(behavior_event_path)
    behavior_features = build_behavior_event_frame(behavior_events)
    training_load, _workout_overview, _hr_points_overview = compute_daily_training_load(
        data,
        max_hr=int(max_hr),
        resting_hr=int(resting_hr),
        sex=sex,
    )
    analysis_sleep_features = build_sleep_feature_frame(analysis_data)
    analysis_tag_features = build_tag_feature_frame(analysis_data)
    analysis_behavior_features = behavior_features
    analysis_training_load, analysis_workout_overview, _analysis_hr_points = compute_daily_training_load(
        analysis_data,
        max_hr=int(max_hr),
        resting_hr=int(resting_hr),
        sex=sex,
    )
    daily = enrich_daily_context(
        daily,
        sleep_features=sleep_features,
        tag_features=tag_features,
        training_load=training_load,
        behavior_features=behavior_features,
    )
    analysis_daily = enrich_daily_context(
        analysis_daily,
        sleep_features=analysis_sleep_features,
        tag_features=analysis_tag_features,
        training_load=analysis_training_load,
        behavior_features=analysis_behavior_features,
    )
    analysis_window_label = f"{analysis_start_d.isoformat()} to {end_d.isoformat()}"
    has_tag_history = not analysis_tag_features.empty or not behavior_events.empty

    zones = hr_zones_karvonen(max_hr=int(max_hr), resting_hr=int(resting_hr))
    hrv_signal = compute_metric_signal(daily, "hrv_rmssd", higher_is_better=True)
    rhr_signal = compute_metric_signal(daily, "resting_hr", higher_is_better=False)
    sleep_signal = compute_metric_signal(daily, "sleep_score", higher_is_better=True)
    readiness_signal = compute_metric_signal(daily, "readiness", higher_is_better=True)
    hrv_trust = compute_biomarker_trust(analysis_daily, "hrv_rmssd", higher_is_better=True)
    rhr_trust = compute_biomarker_trust(analysis_daily, "resting_hr", higher_is_better=False)
    day_mode = determine_day_mode(hrv_signal, rhr_signal, readiness_signal, sleep_signal)
    training_rx = build_training_prescription(goal, str(day_mode.get("mode") or "MAINTAIN"), zones)
    timing_guide = best_time_to_act(analysis_sleep_features if not analysis_sleep_features.empty else sleep_features)
    protocol_actions = morning_protocol(
        hrv_signal,
        rhr_signal,
        sleep_signal,
        daily,
        zones,
        hrv_trust=hrv_trust,
        rhr_trust=rhr_trust,
    )
    tags_df = analysis_data.get("tag", {}).get("df", pd.DataFrame())
    experiment_rows = pd.concat([tags_df, behavior_events_to_tag_rows(behavior_events)], ignore_index=True, sort=False)
    hrv_drivers = what_moved_my_numbers(analysis_daily, experiment_rows, metric="hrv_rmssd", lookback=max(60, int(analysis_lookback_days)))
    rhr_drivers = what_moved_my_numbers(analysis_daily, experiment_rows, metric="resting_hr", lookback=max(60, int(analysis_lookback_days)))
    rolling_protocol = build_three_day_protocol(
        daily,
        hrv_signal=hrv_signal,
        rhr_signal=rhr_signal,
        day_mode=day_mode,
        protocol_actions=protocol_actions,
        training_rx=training_rx,
    )
    personal_thresholds = compute_personal_thresholds(analysis_daily)
    weekly_habits, weekly_habit_summary = compute_weekly_habit_scorecard(analysis_daily, thresholds=personal_thresholds, goal=goal)
    diagnostic_chips = diagnostic_context_chips(daily, personal_thresholds)
    rhr_trend = rhr_trend_slope(analysis_daily, window_days=min(int(analysis_lookback_days), 120))
    rhr_stability_stats = rhr_stability(analysis_daily, window=28)
    rhr_recovery = rhr_recovery_rate(analysis_daily, analysis_workout_overview)
    hrv_age = hrv_biological_age_estimate(analysis_daily, age_years, sex)
    hrv_pattern = hrv_pattern_analysis(analysis_daily, window=min(int(analysis_lookback_days), 120))
    vo2_for_analysis = analysis_vo2 if analysis_vo2 is not None else vo2
    vo2_history_for_analysis = analysis_vo2_df if not analysis_vo2_df.empty else vo2_df
    vo2_longevity = vo2_longevity_analysis(vo2_for_analysis, vo2_history_for_analysis, age_years, sex)
    vo2_trend = vo2_trend_summary(vo2_history_for_analysis, window_days=max(int(analysis_lookback_days), 90))
    vo2_decay = vo2_decay_alert(vo2_history_for_analysis)
    vo2_trust = {
        "valid_points": int(vo2_trend.get("valid_points") or 0),
        "expected_points": max(1, min(int(analysis_lookback_days), 365) // 30),
        "missing_pct": None,
        "source_used": biomarker_source_label("vo2_max"),
        "confidence": "HIGH" if int(vo2_trend.get("valid_points") or 0) >= 4 else "MEDIUM" if int(vo2_trend.get("valid_points") or 0) >= 2 else "LOW",
        "pattern": "Sparse metric. Trust the direction only when you have repeated qualifying recordings.",
    }
    aerobic_efficiency = aerobic_efficiency_summary(
        analysis_workout_overview,
        _analysis_hr_points,
        zones=zones,
        max_hr=int(max_hr),
        resting_hr=int(resting_hr),
        sex=sex,
    )
    training_effects = compute_training_biomarker_effects(
        analysis_daily,
        analysis_workout_overview,
        _analysis_hr_points,
        max_hr=int(max_hr),
        resting_hr=int(resting_hr),
        sex=sex,
        intents_path=workout_intent_path,
    )
    efficiency_trust = {
        "valid_points": int(aerobic_efficiency.get("valid_sessions") or 0),
        "expected_points": 8,
        "missing_pct": None,
        "source_used": biomarker_source_label("aerobic_efficiency"),
        "confidence": "HIGH" if int(aerobic_efficiency.get("valid_sessions") or 0) >= 8 else "MEDIUM" if int(aerobic_efficiency.get("valid_sessions") or 0) >= 4 else "LOW",
        "pattern": "Based only on HR-aligned easy/base sessions with enough duration to estimate drift.",
    }
    training_link = training_biomarker_link(
        analysis_daily,
        efficiency=aerobic_efficiency,
        rhr_trend=rhr_trend,
        hrv_pattern=hrv_pattern,
        effects=training_effects,
    )
    longevity_score = longevity_composite_score(rhr_trend, hrv_pattern, vo2_longevity, weekly_habit_summary)
    prior_cutoff = end_d - timedelta(days=30)
    prior_analysis_daily = analysis_daily[pd.to_datetime(analysis_daily["day"], errors="coerce").dt.date <= prior_cutoff].copy() if not analysis_daily.empty else pd.DataFrame()
    prior_weekly_habits, prior_habit_summary = compute_weekly_habit_scorecard(prior_analysis_daily, thresholds=personal_thresholds, goal=goal) if not prior_analysis_daily.empty else (pd.DataFrame(), {})
    prior_rhr_trend = rhr_trend_slope(prior_analysis_daily, window_days=min(int(analysis_lookback_days), 120)) if not prior_analysis_daily.empty else {}
    prior_hrv_pattern = hrv_pattern_analysis(prior_analysis_daily, window=min(int(analysis_lookback_days), 120)) if not prior_analysis_daily.empty else {}
    prior_vo2_df = vo2_history_for_analysis[pd.to_datetime(vo2_history_for_analysis.get("day"), errors="coerce").dt.date <= prior_cutoff].copy() if not vo2_history_for_analysis.empty and "day" in vo2_history_for_analysis.columns else pd.DataFrame()
    prior_vo2_val = None
    if not prior_vo2_df.empty:
        val_col = next((c for c in ["vo2_max", "value", "vo2max"] if c in prior_vo2_df.columns), None)
        if val_col:
            vals = pd.to_numeric(prior_vo2_df[val_col], errors="coerce").dropna()
            prior_vo2_val = float(vals.iloc[-1]) if not vals.empty else None
    prior_vo2_analysis = vo2_longevity_analysis(prior_vo2_val, prior_vo2_df, age_years, sex) if prior_vo2_val is not None else {}
    prior_longevity_score = longevity_composite_score(prior_rhr_trend, prior_hrv_pattern, prior_vo2_analysis, prior_habit_summary) if prior_rhr_trend or prior_hrv_pattern or prior_vo2_analysis else {}
    decision_score = longevity_decision_score(
        rhr_trend,
        hrv_pattern,
        vo2_longevity,
        aerobic_efficiency,
        weekly_habit_summary,
        score_30d_ago=_safe_float(prior_longevity_score.get("score")) if prior_longevity_score else None,
    )

    active_comparison_snapshot = {
        "label": str(active_account.get("label") or "Current session"),
        "display_name": str((personal.get("first_name") if isinstance(personal, dict) else "") or active_account.get("label") or "You"),
        "hrv_signal": hrv_signal,
        "rhr_signal": rhr_signal,
        "sleep_signal": sleep_signal,
        "readiness_signal": readiness_signal,
        "activity_signal": compute_metric_signal(daily, "activity_score", higher_is_better=True),
        "hrv_trust": hrv_trust,
        "rhr_trust": rhr_trust,
        "rhr_trend": rhr_trend,
        "rhr_stability": rhr_stability_stats,
        "hrv_pattern": hrv_pattern,
        "vo2_value": vo2,
        "vo2_trend": vo2_trend,
        "compare_index": compute_peer_recovery_index(
            hrv_signal=hrv_signal,
            rhr_signal=rhr_signal,
            sleep_signal=sleep_signal,
            readiness_signal=readiness_signal,
        ),
        "baseline_summary": "",
    }
    compare_index_value = _safe_float(active_comparison_snapshot.get("compare_index"))
    if compare_index_value is None:
        active_comparison_snapshot["baseline_summary"] = "Not enough baseline history yet."
    elif compare_index_value >= 58:
        active_comparison_snapshot["baseline_summary"] = "Above personal normal."
    elif compare_index_value <= 42:
        active_comparison_snapshot["baseline_summary"] = "Below personal normal."
    else:
        active_comparison_snapshot["baseline_summary"] = "Near personal normal."

    comparison_snapshots: List[Dict[str, Any]] = [active_comparison_snapshot]
    if resolved_compare_accounts:
        with st.spinner("Fetching comparison accounts…"):
            for compare_account, compare_token in resolved_compare_accounts:
                try:
                    comparison_snapshots.append(
                        build_peer_comparison_snapshot(
                            account_label=str(compare_account.get("label") or "Friend"),
                            token=compare_token,
                            start_d=start_d,
                            end_d=end_d,
                            analysis_start_d=analysis_start_d,
                            wide_sparse_days=int(wide_sparse_days),
                        )
                    )
                except Exception as exc:
                    record_debug_event(f"Comparison fetch failed: {compare_account.get('label')}", exc=exc)
                    comparison_snapshots.append(
                        {
                            "label": str(compare_account.get("label") or "Friend"),
                            "display_name": str(compare_account.get("label") or "Friend"),
                            "error": str(exc),
                        }
                    )

    viewing_bits = [f"Viewing account: **{active_account.get('label') or 'Current session'}**"]
    st.caption(" | ".join(viewing_bits))
    for warning in account_refresh_warnings[:3]:
        st.warning(warning)

    # Tabs
    tabs = st.tabs([
        "Today",
        "Biomarkers",
        "Training",
        "Experiments",
        "Legacy: Recovery trends",
        "Legacy: VO₂ / CVA",
        "Legacy: Resting HR",
        "Legacy: HRV",
        "Data Access",
    ])

    # ------------------
    # Today
    # ------------------
    with guarded_tab(tabs[0], "Today"):
        disclaimer()
        render_today_page_intro(goal_profile, day_mode)

        if daily.empty:
            st.warning("No daily data available in selected range. Check Data Access.")
        else:
            hrv_delta = format_delta(hrv_signal.get("delta_prev"), " ms", 1) if hrv_signal.get("delta_prev") is not None else None
            rhr_delta = format_delta(rhr_signal.get("delta_prev"), " bpm", 1) if rhr_signal.get("delta_prev") is not None else None
            sleep_delta = format_delta(sleep_signal.get("delta_prev"), "", 0) if sleep_signal.get("delta_prev") is not None else None
            today_action = protocol_actions[0]["action"] if protocol_actions else training_rx["instruction"]
            tomorrow_plan = "Reassess morning signals, then progress only if HRV stays up and resting HR stays controlled."
            tomorrow_rows = rolling_protocol[rolling_protocol["Metric"].isin(["Training", "Sleep target", "Key action"])].copy() if not rolling_protocol.empty else pd.DataFrame()
            tomorrow_bits = [str(value) for value in tomorrow_rows["Tomorrow (projected)"].tolist() if str(value).strip() and str(value).strip() != "—"] if not tomorrow_rows.empty else []
            if tomorrow_bits:
                tomorrow_plan = " ".join(tomorrow_bits[:2])
            next_week_plan = str(training_link.get("next_action") or "Keep training load steady and protect sleep timing.") if isinstance(training_link, dict) else "Keep training load steady and protect sleep timing."

            c1, c2, c3, c4 = st.columns(4, gap="large")
            c1.metric("Day mode", str(day_mode.get("mode") or "—"))
            c2.metric("HRV rmSSD", format_metric_value("hrv_rmssd", _safe_float(hrv_signal.get("value"))), hrv_delta)
            c3.metric("Resting HR", format_metric_value("resting_hr", _safe_float(rhr_signal.get("value"))), rhr_delta, delta_color="inverse")
            c4.metric("Sleep score", format_metric_value("sleep_score", _safe_float(sleep_signal.get("value"))), sleep_delta)
            st.markdown("<div class='pw-spacer-sm'></div>", unsafe_allow_html=True)

            horizon_cols = st.columns(3, gap="large")
            with horizon_cols[0]:
                render_today_support_card(
                    title="Do today",
                    overline="Today",
                    value=str(day_mode.get("mode") or "—"),
                    body_lines=[
                        today_action,
                        f"Training: {training_rx['session']}",
                        f"Keep HR near {training_rx['hr_target']}",
                    ],
                )
            with horizon_cols[1]:
                render_today_support_card(
                    title="Set up tomorrow",
                    overline="Tomorrow",
                    body_lines=[
                        tomorrow_plan,
                        f"Bedtime target: {timing_guide.get('bedtime_target') or 'Protect an early bedtime.'}",
                    ],
                )
            with horizon_cols[2]:
                render_today_support_card(
                    title="Move next week",
                    overline="Next week",
                    body_lines=[
                        next_week_plan,
                        str(decision_score.get("fastest_lever") or "Fastest lever not clear yet."),
                    ],
                )
            st.markdown("<div class='pw-spacer-md'></div>", unsafe_allow_html=True)

            protocol_left, protocol_right = st.columns([1.28, 0.92], gap="large")
            with protocol_left:
                st.markdown("### Next best actions")
                st.caption("Start here. These are the highest-ROI moves for the next 2-12 hours.")
                render_protocol_action_cards(protocol_actions, limit=4)
                if len(protocol_actions) > 4:
                    with st.expander("See the full protocol", expanded=False):
                        for action in protocol_actions[4:]:
                            strength = str(action.get("strength") or "")
                            strength_text = f" [{strength.lower()}]" if strength else ""
                            st.write(f"- **{action['timing']}**: {action['action']}{strength_text}")
                            st.caption(action["why"])

                st.markdown("### What likely moved the numbers")
                st.caption(f"Uses up to {analysis_window_label} of recovery history when available.")
                if not has_tag_history:
                    st.caption("Tag alcohol, late meals, supplements, and travel in Oura for 2-3 weeks to unlock behavior-level diagnosis.")
                retro_cols = st.columns(2, gap="large")
                for col, metric_key, label in [
                    (retro_cols[0], "hrv_rmssd", "HRV"),
                    (retro_cols[1], "resting_hr", "Resting HR"),
                ]:
                    drivers = hrv_drivers if metric_key == "hrv_rmssd" else rhr_drivers
                    summary = summarize_driver_analysis(drivers, metric=metric_key)
                    trust = hrv_trust if metric_key == "hrv_rmssd" else rhr_trust
                    with col:
                        st.markdown(f"#### {label}")
                        if not summary["headline"]:
                            st.info("Not enough history yet to explain what moved this number.")
                        else:
                            st.write(summary["headline"])
                            st.caption(describe_signal_trust(trust))
                            top_rows = summary["top_rows"]
                            if top_rows is not None and not top_rows.empty:
                                for _, row in top_rows.head(2).iterrows():
                                    st.write(f"- {describe_driver_row(row)}")
                            if summary["primary_lever"]:
                                st.write(f"**Primary lever:** {summary['primary_lever']}")

            with protocol_right:
                render_today_support_card(
                    title=training_rx["session"],
                    overline="Today allows",
                    value=training_rx["hr_target"],
                    body_lines=[
                        day_mode["summary"],
                        f"Do: {training_rx['instruction']}",
                        f"Avoid: {training_rx['avoid']}",
                    ],
                )
                if day_mode.get("reasons"):
                    st.caption("Signals: " + " | ".join(str(reason) for reason in day_mode.get("reasons", [])[:3]))

                st.markdown("### Context")
                render_chip_cloud(diagnostic_chips)

                st.markdown("### Signal trust")
                render_today_support_card(
                    title="HRV",
                    overline="Confidence",
                    body_lines=[describe_signal_trust(hrv_trust)],
                )
                render_today_support_card(
                    title="Resting HR",
                    overline="Confidence",
                    body_lines=[describe_signal_trust(rhr_trust)],
                )
                if hrv_trust.get("confidence") == "LOW" or rhr_trust.get("confidence") == "LOW":
                    st.info("Recent RH/HRV coverage is sparse. Use today’s plan conservatively and lean harder on sleep timing, hydration, and easy movement.")

                st.markdown("### Best time to act")
                t1, t2 = st.columns(2)
                t1.metric("Chronotype", _fmt(timing_guide.get("chronotype")))
                t2.metric("Hard-session window", _fmt(timing_guide.get("exercise_window")))
                t3, t4 = st.columns(2)
                t3.metric("NSDR / breathing window", _fmt(timing_guide.get("parasympathetic_window")))
                t4.metric("Light / walk window", _fmt(timing_guide.get("light_window")))
                if timing_guide.get("bedtime_target"):
                    st.write(f"- **Tonight:** in bed by **{timing_guide['bedtime_target']}**")
                if timing_guide.get("winddown_start"):
                    st.write(f"- **Wind-down starts:** **{timing_guide['winddown_start']}**")
                if timing_guide.get("note"):
                    st.caption(str(timing_guide["note"]))

            signal_rows: List[dict] = []
            for key, signal in [
                ("hrv_rmssd", hrv_signal),
                ("resting_hr", rhr_signal),
                ("sleep_score", sleep_signal),
                ("readiness", readiness_signal),
            ]:
                if signal.get("value") is None:
                    continue
                unit = metric_unit(key)
                decimals = 1 if key in {"hrv_rmssd", "resting_hr"} else 0
                trust = hrv_trust if key == "hrv_rmssd" else rhr_trust if key == "resting_hr" else {}
                signal_rows.append(
                    {
                        "Metric": metric_label(key),
                        "Today": format_metric_value(key, _safe_float(signal.get("value"))),
                        "vs yesterday": format_delta(signal.get("delta_prev"), unit, decimals),
                        "vs 7d baseline": format_delta(signal.get("delta7"), unit, decimals),
                        "vs 28d baseline": format_delta(signal.get("delta28"), unit, decimals),
                        "Signal": signal.get("signal"),
                        "Valid nights": _fmt(trust.get("n_valid_28")) if trust else "—",
                        "Confidence": _fmt(trust.get("confidence")) if trust else "—",
                        "Pattern": _fmt(str(trust.get("pattern") or "").replace("_", " ").title()) if trust else "—",
                        "Source": _fmt(trust.get("source_used")) if trust else "—",
                        "What it means": interpret_metric_signal(key, signal),
                    }
                )

            lower_left, lower_right = st.columns([1.02, 0.98], gap="large")
            with lower_left:
                with st.expander("Immediate feedback and baselines", expanded=False):
                    st.caption("For HRV, higher is better. For resting HR, lower is better.")
                    if signal_rows:
                        st.dataframe(pd.DataFrame(signal_rows), width="stretch")
                with st.expander("3-day rolling protocol", expanded=False):
                    st.dataframe(rolling_protocol, width="stretch")
                    st.caption("Weekly trend and decision scoring now live in the Biomarkers tab so Today stays operational.")

            with lower_right:
                with st.expander("Goal playbook and metric levers", expanded=False):
                    st.markdown("### Goal playbook")
                    st.write(f"- **HRV:** {goal_profile.hrv_play}")
                    st.write(f"- **Resting HR:** {goal_profile.rhr_play}")
                    st.write(f"- **VO₂ max:** {goal_profile.vo2_play}")

                    st.markdown("### Improve the metrics that matter")
                    hrv_col, rhr_col, vo2_col = st.columns(3, gap="large")
                    with hrv_col:
                        st.markdown(f"#### HRV ({hrv_signal.get('signal', 'UNKNOWN')})")
                        for line in hrv_action_lines(goal, str(day_mode.get("mode") or "MAINTAIN"), zones, hrv_signal):
                            st.write(f"- {line}")
                    with rhr_col:
                        st.markdown(f"#### Resting HR ({rhr_signal.get('signal', 'UNKNOWN')})")
                        for line in rhr_action_lines(str(day_mode.get("mode") or "MAINTAIN"), zones, rhr_signal):
                            st.write(f"- {line}")
                    with vo2_col:
                        st.markdown("#### VO₂ max path")
                        for line in vo2_action_lines(goal, str(day_mode.get("mode") or "MAINTAIN"), zones):
                            st.write(f"- {line}")


    # ------------------
    # Biomarkers
    # ------------------
    with guarded_tab(tabs[1], "Biomarkers"):
        render_integrated_biomarker_view(
            hrv_signal=hrv_signal,
            rhr_signal=rhr_signal,
            hrv_trust=hrv_trust,
            rhr_trust=rhr_trust,
            hrv_pattern=hrv_pattern,
            rhr_trend=rhr_trend,
            rhr_stability_stats=rhr_stability_stats,
            rhr_recovery=rhr_recovery,
            vo2_analysis=vo2_longevity,
            vo2_trend=vo2_trend,
            vo2_trust=vo2_trust,
            efficiency=aerobic_efficiency,
            efficiency_trust=efficiency_trust,
            training_link=training_link,
            diagnostic_chips=diagnostic_chips,
        )
        st.divider()
        render_longevity_score_panel(decision_score, compact=False)

    # ------------------
    # Timeline
    # ------------------
    with guarded_tab(tabs[4], "Legacy: Recovery trends"):
        st.subheader("Recovery trend analysis")
        st.caption("Legacy deep-dive view retained for compatibility.")
        st.caption("One metric at a time, centered on what changes your next decision.")

        if daily.empty:
            st.info("No daily rows to plot.")
        else:
            metric_labels = [
                ("readiness", "Readiness"),
                ("sleep_score", "Sleep score"),
                ("sleep_total_s", "Sleep duration (hours)"),
                ("activity_score", "Activity score"),
                ("steps", "Steps"),
                ("hrv_rmssd", "HRV rmSSD"),
                ("resting_hr", "Resting HR"),
                ("temp_dev", "Temp deviation"),
                ("spo2", "SpO₂"),
                ("stress_high", "Stress high"),
            ]
            available = [(k, lbl) for k, lbl in metric_labels if k in daily.columns and daily[k].notna().any()]
            if not available:
                st.info("No common daily metrics available to plot.")
            else:
                preferred_idx = 0
                found_preferred = False
                for wanted in goal_profile.priority_metrics:
                    for idx, (key, _) in enumerate(available):
                        if key == wanted:
                            preferred_idx = idx
                            found_preferred = True
                            break
                    if found_preferred:
                        break
                pick = st.selectbox(
                    "Metric",
                    options=[k for k, _ in available],
                    format_func=lambda x: dict(available).get(x, x),
                    index=preferred_idx,
                )

                m = daily[["day", pick]].copy()
                m = m.dropna(subset=["day"]).copy()
                m["day"] = pd.to_datetime(m["day"], errors="coerce")
                m[pick] = pd.to_numeric(m[pick], errors="coerce")
                m = m.dropna(subset=["day"]).sort_values("day")

                # Unit normalization for sleep duration
                if pick == "sleep_total_s":
                    m[pick] = m[pick] / 3600.0

                s = m[pick]
                last_v = _safe_float(s.iloc[-1]) if len(s) else None
                avg7 = _safe_float(s.tail(7).mean()) if len(s.dropna()) >= 3 else None
                avg28 = _safe_float(s.tail(28).mean()) if len(s.dropna()) >= 10 else None
                delta7 = (last_v - avg7) if (last_v is not None and avg7 is not None) else None

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latest", _fmt(last_v))
                c2.metric("7‑day avg", _fmt(None if avg7 is None else round(avg7, 2)))
                c3.metric("28‑day avg", _fmt(None if avg28 is None else round(avg28, 2)))
                c4.metric("Δ vs 7‑day", _fmt(None if delta7 is None else round(delta7, 2)))

                # Plot: metric + 28d rolling average
                m["roll28"] = s.rolling(28, min_periods=7).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=m["day"], y=m[pick], mode="lines+markers", name="Daily"))
                if m["roll28"].notna().any():
                    fig.add_trace(go.Scatter(x=m["day"], y=m["roll28"], mode="lines", name="28‑day trend"))
                fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, width="stretch")

                # Short, actionable takeaway (simple rules; we can refine later)
                st.markdown("### Takeaway")
                if last_v is None:
                    st.write("No data for this metric in the selected range.")
                else:
                    if pick in {"readiness", "sleep_score", "activity_score"}:
                        if last_v >= 80:
                            st.write("Good today. Keep the routine consistent.")
                        elif last_v >= 65:
                            st.write("OK but not perfect. Consider reducing intensity and protecting sleep.")
                        else:
                            st.write("Low. Make today a recovery-focused day (easy movement + earlier bedtime).")
                    elif pick == "sleep_total_s":
                        if last_v >= 7.0:
                            st.write("Sleep duration is solid. Keep bedtime consistent.")
                        elif last_v >= 6.0:
                            st.write("Borderline sleep. Priority tonight: +60–90 min in bed.")
                        else:
                            st.write("Short sleep. Treat today as recovery: lower training load + earlier bedtime.")
                    elif pick == "resting_hr":
                        if avg28 is not None and last_v > avg28 + 3:
                            st.write("Resting HR is elevated vs baseline. Reduce intensity + hydrate + prioritize sleep.")
                        else:
                            st.write("Resting HR looks normal for you.")
                    elif pick == "hrv_rmssd":
                        if avg28 is not None and last_v < avg28 * 0.85:
                            st.write("HRV is below baseline. Keep training easy and prioritize recovery inputs.")
                        else:
                            st.write("HRV looks normal for you.")
                    else:
                        st.write("Use this to spot trends; actionable rules for this metric can be added next.")

                st.divider()
                st.markdown("### Personal dose-response curves")
                st.caption(f"These curves use up to {analysis_window_label} of history so stable routines still produce useful bins.")
                curve_specs = [
                    {
                        "title": "Sleep duration -> next-day HRV",
                        "x_col": "sleep_hours",
                        "y_col": "hrv_rmssd",
                        "lag_days": 1,
                        "higher_is_better": True,
                        "x_unit": "h",
                        "y_unit": "ms",
                    },
                    {
                        "title": "Training load -> 2-day-later HRV",
                        "x_col": "trimp",
                        "y_col": "hrv_rmssd",
                        "lag_days": 2,
                        "higher_is_better": True,
                        "x_unit": "TRIMP",
                        "y_unit": "ms",
                    },
                    {
                        "title": "Bedtime deviation -> next-day Resting HR",
                        "x_col": "bedtime_dev",
                        "y_col": "resting_hr",
                        "lag_days": 1,
                        "higher_is_better": False,
                        "x_unit": "min",
                        "y_unit": "bpm",
                    },
                ]
                dose_cols = st.columns(3)
                for col, spec in zip(dose_cols, curve_specs):
                    with col:
                        st.markdown(f"#### {spec['title']}")
                        curve = dose_response(
                            analysis_daily,
                            x_col=spec["x_col"],
                            y_col=spec["y_col"],
                            lag_days=spec["lag_days"],
                            bins=5,
                            higher_is_better=spec["higher_is_better"],
                        )
                        curve_df = curve.get("curve", pd.DataFrame())
                        sweet = curve.get("sweet_spot")
                        if curve_df is None or curve_df.empty or sweet is None:
                            st.info("Not enough clean history yet. If your values cluster tightly, that is a real finding: another lever may matter more than this one.")
                        else:
                            fig_curve = px.bar(curve_df, x="bin_label", y="mean_y", title="", labels={"bin_label": spec["x_unit"], "mean_y": spec["y_unit"]})
                            fig_curve.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                            st.plotly_chart(fig_curve, width="stretch", config={"displaylogo": False})
                            st.write(
                                f"- Sweet spot: **{sweet['x_low']:.1f}-{sweet['x_high']:.1f} {spec['x_unit']}** -> **{sweet['mean_y']:.1f} {spec['y_unit']}**"
                            )
                            if spec["x_col"] == "sleep_hours":
                                st.write(
                                    f"- Threshold: below **{curve_df.iloc[0]['x_high']:.1f} h** your next-day HRV is materially lower than your best bin."
                                )
                            elif spec["x_col"] == "bedtime_dev":
                                st.write(
                                    f"- Cost: when bedtime drift reaches **{curve_df.iloc[-1]['x_low']:.0f}+ min**, next-day resting HR is worse than your sweet spot."
                                )
                            else:
                                st.write(
                                    f"- Room to push: the best 2-day-later HRV shows up around **{sweet['x_low']:.0f}-{sweet['x_high']:.0f} TRIMP** in your own data."
                                )

                st.divider()
                st.markdown("### Weekly habit scorecard")
                if weekly_habits is None or weekly_habits.empty:
                    st.info("Not enough daily context yet to score habits.")
                else:
                    habit_cols = [c for c in weekly_habits.columns if c not in {"day", "habit_hits", "hrv_rmssd"}]
                    heat = weekly_habits[["day"] + habit_cols].copy()
                    heat["day"] = pd.to_datetime(heat["day"], errors="coerce").dt.strftime("%m-%d")
                    z_vals: List[List[int]] = []
                    text_vals: List[List[str]] = []
                    for habit in habit_cols:
                        row_z: List[int] = []
                        row_text: List[str] = []
                        for value in heat[habit].tolist():
                            truthy = bool_from_value(value)
                            if truthy is None:
                                row_z.append(-1)
                                row_text.append("--")
                            elif truthy:
                                row_z.append(1)
                                row_text.append("OK")
                            else:
                                row_z.append(0)
                                row_text.append("MISS")
                        z_vals.append(row_z)
                        text_vals.append(row_text)

                    fig_heat = go.Figure(
                        data=go.Heatmap(
                            z=z_vals,
                            x=heat["day"].tolist(),
                            y=habit_cols,
                            zmin=-1,
                            zmax=1,
                            text=text_vals,
                            texttemplate="%{text}",
                            colorscale=[
                                [0.0, "#d9d9d9"],
                                [0.499, "#d9d9d9"],
                                [0.5, "#f4a259"],
                                [0.749, "#f4a259"],
                                [0.75, "#4c956c"],
                                [1.0, "#4c956c"],
                            ],
                            showscale=False,
                        )
                    )
                    fig_heat.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_heat, width="stretch", config={"displaylogo": False})

                    hit_days = int((weekly_habits["habit_hits"] >= 4).sum())
                    st.write(f"- You hit **4+/5 habits on {hit_days} of {len(weekly_habits)} days**.")
                    thresholds_used = weekly_habit_summary.get("thresholds") or {}
                    if thresholds_used:
                        st.write(
                            f"- Personal thresholds in use: sleep >= **{(_safe_float(thresholds_used.get('min_sleep_hours_for_hrv')) or 7.0):.1f}h**, "
                            f"bedtime drift <= **{(_safe_float(thresholds_used.get('max_bedtime_dev_for_rhr')) or 30.0):.0f} min**, "
                            f"steps floor **{(_safe_float(thresholds_used.get('step_floor')) or 6000.0):.0f}** / target **{(_safe_float(thresholds_used.get('step_target')) or 9000.0):.0f}**, "
                            f"training sweet spot **{(_safe_float(thresholds_used.get('trimp_sweet_spot_low')) or 60.0):.0f}-{(_safe_float(thresholds_used.get('trimp_sweet_spot_high')) or 120.0):.0f} TRIMP**."
                        )
                        source_flags = thresholds_used.get("threshold_source_flags") or {}
                        learned = [key for key, source in source_flags.items() if source == "learned"]
                        fallback = [key for key, source in source_flags.items() if source != "learned"]
                        if learned or fallback:
                            st.caption(
                                f"Threshold sources: {len(learned)} learned from your history, {len(fallback)} fallback defaults."
                            )
                    high_hrv = _safe_float(weekly_habit_summary.get("high_hit_hrv"))
                    low_hrv = _safe_float(weekly_habit_summary.get("low_hit_hrv"))
                    if high_hrv is not None and low_hrv is not None:
                        st.write(f"- HRV averages **{high_hrv:.1f} ms** on 4+/5 days vs **{low_hrv:.1f} ms** on <3/5 days.")
                    else:
                        st.write("- Need a few more days of complete habit data to quantify the HRV spread.")

    # ------------------
    # Training
    # ------------------
    with guarded_tab(tabs[2], "Training"):
        st.subheader("Training")
        st.caption("First decide what today's recovery allows. Then score how a completed session actually executed.")

        st.markdown("### Today's exercise prescription")
        rx1, rx2, rx3 = st.columns(3)
        rx1.metric("Recovery state", str(day_mode.get("mode") or "—"))
        rx2.metric("Primary session", training_rx["session"])
        rx3.metric("Primary HR target", training_rx["hr_target"])
        st.write(f"- **Why:** {day_mode['summary']}")
        st.write(f"- **Do:** {training_rx['instruction']}")
        st.write(f"- **Avoid:** {training_rx['avoid']}")
        if day_mode.get("reasons"):
            st.write(f"- **Signals:** {', '.join(day_mode['reasons'])}")
        if training_link:
            st.markdown("### Is training helping biomarkers?")
            st.write(f"- **Verdict:** {training_link.get('verdict')}")
            for bullet in training_link.get("bullets", [])[:2]:
                st.write(f"- {bullet}")
            st.write(f"- **Next move:** {training_link.get('next_action')}")
        if training_effects is not None and not training_effects.empty:
            st.markdown("### Training impact summary")
            impact = summarize_training_biomarker_effects(training_effects)
            c1, c2, c3 = st.columns(3)
            c1.metric("Training impact", _fmt(impact.get("verdict")).replace("_", " ").title())
            hrr60_series = pd.to_numeric(training_effects.get("hrr60"), errors="coerce").dropna() if "hrr60" in training_effects.columns else pd.Series(dtype=float)
            c2.metric("HRR60", _fmt(None if hrr60_series.empty else f"{hrr60_series.mean():.1f} bpm"))
            c3.metric("HRR120", _fmt(None if pd.to_numeric(training_effects.get('hrr120'), errors='coerce').dropna().empty else f"{pd.to_numeric(training_effects.get('hrr120'), errors='coerce').dropna().mean():.1f} bpm"))
            for reason in impact.get("reasons", []):
                st.write(f"- {reason}")
            st.write(f"- **What to do next week:** {impact.get('next_week')}")
        st.divider()

        df_workout = data.get("workout", {}).get("df", pd.DataFrame())
        df_hr = data.get("heartrate", {}).get("df", pd.DataFrame())

        if df_workout.empty:
            st.info("No workout rows returned (or endpoint unavailable).")
        else:
            if not can_edit_active_account:
                st.info("Viewing a shared account read-only. Only the connected account owner can save workout intent labels or manual annotations.")
            w = df_workout.copy()

            # Persisted workout intents (at-home analysis; survives restarts)
            intents_path = workout_intent_path
            intent_labels = load_intent_labels(intents_path)

            # Standardize day
            if "day" not in w.columns:
                for c in ["start_datetime", "timestamp"]:
                    if c in w.columns:
                        w["day"] = w[c].apply(_to_day)
                        break
            w["day"] = w.get("day").astype(str)

            # Duration minutes
            dur = None
            for c in ["duration", "duration_seconds"]:
                if c in w.columns:
                    dur = c
                    break
            if dur:
                w["duration_min"] = pd.to_numeric(w[dur], errors="coerce") / 60.0
            else:
                w["duration_min"] = None

            # Type
            for c in ["type", "sport", "activity"]:
                if c in w.columns:
                    w["type"] = w[c]
                    break
            if "type" not in w.columns:
                w["type"] = "workout"

            # HR points normalization
            hr_points = pd.DataFrame()
            if not df_hr.empty:
                hp = df_hr.copy()
                # guess columns
                ts_col = None
                for c in ["timestamp", "time", "datetime", "start_datetime"]:
                    if c in hp.columns:
                        ts_col = c
                        break
                bpm_col = None
                for c in ["bpm", "heart_rate", "hr"]:
                    if c in hp.columns:
                        bpm_col = c
                        break

                if ts_col and bpm_col:
                    hp["ts"] = hp[ts_col].apply(_parse_dt)
                    hp = hp.dropna(subset=["ts"]).copy()
                    hp["ts_epoch"] = hp["ts"].apply(lambda x: x.timestamp())
                    hp["bpm"] = pd.to_numeric(hp[bpm_col], errors="coerce")
                    hp = hp.dropna(subset=["bpm"]).copy()
                    hr_points = hp[["ts", "ts_epoch", "bpm"]].sort_values("ts_epoch")

            # Choose a workout
            w_sorted = w.copy()

            # Personalization models (computed from your own history)
            pmodel = compute_personalization_models(w_sorted, hr_points, int(max_hr), int(resting_hr))
            z2_cap_bpm = _safe_float(pmodel.get("z2_cap_bpm"))
            intent_baselines = pmodel.get("intent_baselines", {}) if isinstance(pmodel, dict) else {}
            if "start_datetime" in w_sorted.columns:
                w_sorted["start_dt"] = w_sorted["start_datetime"].apply(_parse_dt)
            else:
                w_sorted["start_dt"] = pd.NaT
            w_sorted = w_sorted.sort_values(["day"], ascending=False)

            def workout_key(row: pd.Series) -> str:
                # Prefer precise start_datetime; fall back to day+type+duration
                sdt = row.get("start_datetime") if "start_datetime" in row else None
                if sdt is not None and pd.notna(sdt):
                    return str(sdt)
                d = str(row.get("day") or "")
                t = str(row.get("type") or "workout")
                dm = row.get("duration_min")
                dm_s = f"{float(dm):.0f}" if dm is not None and pd.notna(dm) else ""
                return f"{d}|{t}|{dm_s}"

            # Apply persisted labels onto workout rows (if any)
            try:
                if not intent_labels.empty:
                    label_map = dict(zip(intent_labels["workout_key"].astype(str), intent_labels["workout_intent"].astype(str)))
                    w_sorted["workout_key"] = w_sorted.apply(workout_key, axis=1)
                    w_sorted["workout_intent"] = w_sorted["workout_key"].astype(str).map(label_map).fillna("")
            except Exception:
                w_sorted["workout_key"] = w_sorted.apply(workout_key, axis=1)
                if "workout_intent" not in w_sorted.columns:
                    w_sorted["workout_intent"] = ""

            def workout_label(row: pd.Series) -> str:
                d = str(row.get("day") or "")
                t = str(row.get("type") or "workout")
                dm = row.get("duration_min")
                dm_s = f"{float(dm):.0f}m" if dm is not None and pd.notna(dm) else ""
                it = str(row.get("workout_intent") or "").strip()
                it_s = f" · {it}" if it else ""
                return f"{d} — {t} {dm_s}{it_s}".strip()

            options = list(w_sorted.index[:200])
            if options:
                st.markdown("### Analyze a completed workout")
                pick_idx = st.selectbox(
                    "Pick a workout to analyze",
                    options=options,
                    format_func=lambda i: workout_label(w_sorted.loc[i]),
                    index=0,
                )
                wr = w_sorted.loc[pick_idx]

                st.markdown("### Last workout effectiveness")

                start_dt, end_dt = workout_window(wr)
                start_dt = start_dt or _parse_dt(str(wr.get("day")) + "T00:00:00Z")
                duration_min = _safe_float(wr.get("duration_min"))

                # Filter HR samples for this workout
                seg = pd.DataFrame()
                if not hr_points.empty and start_dt is not None and end_dt is not None:
                    seg = hr_points[(hr_points["ts"] >= start_dt) & (hr_points["ts"] <= end_dt)].copy()

                # Defaults to avoid UnboundLocalError when HR data is missing
                drift = None
                trimp = None

                # KPIs
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Type", _fmt(wr.get("type")))
                k2.metric("Duration", f"{duration_min:.0f} min" if duration_min is not None else "—")
                k3.metric("Calories", _fmt(wr.get("calories")))
                k4.metric("Avg HR", _fmt(round(float(seg["bpm"].mean()), 0)) if not seg.empty else "—")
                k5.metric("Max HR", _fmt(round(float(seg["bpm"].max()), 0)) if not seg.empty else "—")

                # Context (how recovered were you that day?)
                try:
                    day_key = str(wr.get("day"))
                    if not daily.empty and "day" in daily.columns and (daily["day"].astype(str) == day_key).any():
                        drow = daily[daily["day"].astype(str) == day_key].iloc[-1]
                    elif not daily.empty and daily.index.name == "day" and day_key in list(daily.index.astype(str)):
                        drow = daily.loc[day_key]
                    else:
                        drow = None

                    if drow is not None:
                        st.markdown("#### Day context (baseline vs you)")
                        metrics = [
                            ("readiness", "Readiness", True),
                            ("sleep_score", "Sleep", True),
                            ("resting_hr", "Resting HR", False),
                            ("hrv_rmssd", "HRV rmSSD", True),
                            ("temp_dev", "Temp dev", False),
                            ("resp_rate_dev", "Resp rate dev", False),
                            ("spo2", "SpO₂", True),
                            ("stress_high", "Stress high", False),
                        ]
                        cards = st.columns(4)
                        idx = 0
                        for key, label, higher_better in metrics:
                            if key in daily.columns and pd.notna(drow.get(key)):
                                v = _safe_float(drow.get(key))
                                mean7, sd7 = baseline_mean_sd(daily[key], 7)
                                z7 = z_score(v, mean7, sd7)
                                status = traffic_light_from_z(z7, higher_is_better=higher_better)
                                cards[idx % 4].metric(label, _fmt(v), help=f"7d z={z7:.2f} ({status})" if z7 is not None else None)
                                idx += 1
                except Exception:
                    pass

                # HR chart
                if seg.empty:
                    st.info("No heart rate samples found for this workout window (heartrate stream may not align with workout timing for this account).")
                else:
                    fig = px.line(seg, x="ts", y="bpm", title="Heart rate during workout")
                    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, width="stretch")

                    # Zones
                    tz = time_in_zones(seg, zones)
                    # normalize to minutes
                    tz_min = (tz / 60.0).sort_index()

                    # Intent options depend on goal
                    # Core intent taxonomy (athlete-friendly). Some intents are not HR-scored.
                    # Use goal to order/prioritize, but keep a rich list.
                    core_intents: List[Tuple[str, List[str]]] = [
                        ("Recovery (easy)", ["Z1"]),
                        ("Aerobic base (Z2)", ["Z2"]),
                        ("Tempo / threshold (Z3)", ["Z3"]),
                        ("Intervals (Z4–Z5)", ["Z4", "Z5"]),
                        ("Mobility / rehab (not HR scored)", []),
                        ("Strength — hypertrophy (not HR scored)", []),
                        ("Strength — maximal (not HR scored)", []),
                        ("Power / sprint (not HR scored)", []),
                        ("Skill / technique (not HR scored)", []),
                        ("Longevity / healthspan (Z1–Z2)", ["Z1", "Z2"]),
                        ("Conditioning (Z2–Z3)", ["Z2", "Z3"]),
                        ("HIIT (Z4–Z5)", ["Z4", "Z5"]),
                        ("Downregulation / recovery", ["Z1"]),
                    ]

                    # Build intent list order based on goal
                    if goal.startswith("Performance (endurance)"):
                        order = [
                            "Recovery (easy)",
                            "Aerobic base (Z2)",
                            "Tempo / threshold (Z3)",
                            "Intervals (Z4–Z5)",
                            "Mobility / rehab (not HR scored)",
                            "Skill / technique (not HR scored)",
                            "Longevity / healthspan (Z1–Z2)",
                        ]
                    elif goal.startswith("Performance (strength"):
                        order = [
                            "Strength — hypertrophy (not HR scored)",
                            "Strength — maximal (not HR scored)",
                            "Power / sprint (not HR scored)",
                            "Conditioning (Z2–Z3)",
                            "HIIT (Z4–Z5)",
                            "Recovery (easy)",
                            "Mobility / rehab (not HR scored)",
                        ]
                    elif goal.startswith("Stress resilience"):
                        order = [
                            "Downregulation / recovery",
                            "Longevity / healthspan (Z1–Z2)",
                            "Recovery (easy)",
                            "Mobility / rehab (not HR scored)",
                            "Skill / technique (not HR scored)",
                        ]
                    else:
                        order = [
                            "Longevity / healthspan (Z1–Z2)",
                            "Recovery (easy)",
                            "Aerobic base (Z2)",
                            "Conditioning (Z2–Z3)",
                            "Tempo / threshold (Z3)",
                            "HIIT (Z4–Z5)",
                            "Intervals (Z4–Z5)",
                            "Mobility / rehab (not HR scored)",
                            "Strength — hypertrophy (not HR scored)",
                            "Strength — maximal (not HR scored)",
                            "Power / sprint (not HR scored)",
                            "Skill / technique (not HR scored)",
                        ]

                    intents_dict = {k: v for k, v in core_intents}
                    # Keep only known keys and maintain unique order
                    ordered_keys = [k for k in order if k in intents_dict]
                    for k, _ in core_intents:
                        if k not in ordered_keys:
                            ordered_keys.append(k)
                    intents = {k: intents_dict.get(k, []) for k in ordered_keys}

                    # ------------------------------
                    # Plan + intent + compliance
                    # ------------------------------
                    with st.expander("Plan this session (targets + caps)", expanded=False):
                        planned_intent = st.selectbox(
                            "Planned session type",
                            options=list(intents.keys()),
                            index=0,
                            help="Pick the intended stimulus; we’ll show HR targets and caps.",
                            key=f"planned_intent_{pick_idx}",
                        )
                        planned_duration = st.slider(
                            "Planned duration (min)",
                            min_value=10,
                            max_value=180,
                            value=int(round(duration_min)) if duration_min is not None and pd.notna(duration_min) else 45,
                            step=5,
                            key=f"planned_dur_{pick_idx}",
                        )
                        pzones = intents.get(planned_intent, [])
                        pbounds = target_bounds_for_zones(zones, pzones) if pzones else None

                        # Personalized caps / floors (simple and actionable)
                        hr_cap = None
                        hr_floor = None
                        if pzones:
                            # Cap = top of easiest included zone (recovery/aerobic)
                            if planned_intent.startswith("Recovery") or "Aerobic" in planned_intent or "Easy" in planned_intent or "Downregulation" in planned_intent:
                                b = zone_bounds_by_name(zones, pzones[0])
                                if b:
                                    hr_cap = b[1]
                            # For intervals: floor = bottom of Z4; recovery cap = top of Z2
                            if "Intervals" in planned_intent or "HIIT" in planned_intent:
                                b4 = zone_bounds_by_name(zones, "Z4")
                                b2 = zone_bounds_by_name(zones, "Z2")
                                if b4:
                                    hr_floor = b4[0]
                                if b2:
                                    hr_cap = b2[1]

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Planned target zones", ", ".join(pzones) if pzones else "—")
                        c2.metric("Planned target HR", f"{pbounds[0]}–{pbounds[1]} bpm" if pbounds else "—")
                        c3.metric("Planned duration", f"{planned_duration} min")
                        if hr_cap is not None:
                            st.write(f"- **HR cap**: stay ≤ **{hr_cap} bpm**")
                        if hr_floor is not None:
                            st.write(f"- **Work interval HR floor**: aim ≥ **{hr_floor} bpm** on hard reps")
                        if "Intervals" in planned_intent or "HIIT" in planned_intent:
                            st.write("- **Between reps**: recover until HR drops to ~Z2 before starting the next rep.")

                    intent = st.selectbox(
                        "Workout intent (used to score HR compliance when applicable)",
                        options=list(intents.keys()),
                        index=0,
                        help="Label the workout intent. For intents marked 'not HR scored', we’ll skip HR compliance and focus on load/consistency instead.",
                        key=f"intent_{pick_idx}",
                    )

                    # Persist label into the workout table for at-home analysis
                    try:
                        if can_edit_active_account:
                            wk_key = None
                            try:
                                wk_key = str(w_sorted.loc[pick_idx].get("workout_key"))
                            except Exception:
                                wk_key = str(workout_key(w_sorted.loc[pick_idx]))
                            save_intent_label(intents_path, wk_key, intent)

                            if "workout_intent" not in w_sorted.columns:
                                w_sorted["workout_intent"] = ""
                            w_sorted.loc[pick_idx, "workout_intent"] = intent
                    except Exception:
                        pass
                    target_zones = intents.get(intent, [])

                    target_bounds = target_bounds_for_zones(zones, target_zones) if target_zones else None
                    compliance = compliance_pct(tz, target_zones) if target_zones else None

                    # Compute caps/floors for the scored intent
                    score_hr_cap = None
                    score_hr_floor = None
                    if target_zones:
                        if intent.startswith("Recovery") or "Aerobic" in intent or "Easy" in intent or "Downregulation" in intent:
                            b = zone_bounds_by_name(zones, target_zones[0])
                            if b:
                                score_hr_cap = b[1]
                        if "Intervals" in intent or "HIIT" in intent:
                            b4 = zone_bounds_by_name(zones, "Z4")
                            b2 = zone_bounds_by_name(zones, "Z2")
                            if b4:
                                score_hr_floor = b4[0]
                            if b2:
                                score_hr_cap = b2[1]

                    # Personal targets overlay (learned)
                    learned_cap = None
                    if z2_cap_bpm is not None and ("Aerobic" in intent or "Z2" in intent or "Longevity" in intent or "Easy aerobic" in intent):
                        learned_cap = int(round(float(z2_cap_bpm)))

                    t1, t2, t3, t4, t5, t6 = st.columns(6)
                    t1.metric("Target zones", ", ".join(target_zones) if target_zones else "—")
                    t2.metric("Target HR", (f"{target_bounds[0]}–{target_bounds[1]} bpm" if target_bounds else "—"))
                    t3.metric("HR cap", f"≤{score_hr_cap} bpm" if score_hr_cap is not None else "—")
                    t4.metric("Interval floor", f"≥{score_hr_floor} bpm" if score_hr_floor is not None else "—")
                    t5.metric("Personal aerobic cap", f"≤{learned_cap} bpm" if learned_cap is not None else "—")
                    if compliance is None:
                        t6.metric("HR compliance", "—")
                    else:
                        t6.metric("HR compliance", f"{compliance:.0f}%")

                    if compliance is not None:
                        if compliance >= 85:
                            st.caption("Compliance: ✅ on target.")
                        elif compliance >= 60:
                            st.caption("Compliance: ⚠️ some drift — tighten pacing early or follow the HR cap/floor cues.")
                        else:
                            st.caption("Compliance: ❌ missed stimulus — either too easy or too hard for the intended goal.")

                    drift = compute_hr_drift(seg)
                    trimp = compute_banister_trimp(
                        seg,
                        max_hr=int(max_hr),
                        resting_hr=int(resting_hr),
                        sex=sex,
                    )

                    # ------------------------------
                    # Workout grade + takeaway
                    # ------------------------------
                    grade_score = None
                    if target_zones and compliance is not None:
                        grade_score = float(compliance)

                        # personalized drift penalty vs your own baseline for this intent
                        try:
                            ib = intent_baselines.get(str(intent)) if isinstance(intent_baselines, dict) else None
                            drift_med = _safe_float((ib or {}).get("drift_median"))
                            drift_iqr = _safe_float((ib or {}).get("drift_iqr"))
                        except Exception:
                            drift_med, drift_iqr = None, None

                        if drift is not None and ("Aerobic" in intent or "Z2" in intent):
                            if drift_med is not None and drift_iqr is not None and drift_iqr > 0:
                                # penalize if drift is > median + 1*IQR (moderate) or +2*IQR (high)
                                if drift > drift_med + 2 * drift_iqr:
                                    grade_score -= 20
                                elif drift > drift_med + 1 * drift_iqr:
                                    grade_score -= 10
                            else:
                                # fallback generic
                                if drift > 8:
                                    grade_score -= 20
                                elif drift > 5:
                                    grade_score -= 10
                        # penalize if day context is poor
                        try:
                            day_key = str(wr.get("day"))
                            drow = None
                            if not daily.empty and "day" in daily.columns and (daily["day"].astype(str) == day_key).any():
                                drow = daily[daily["day"].astype(str) == day_key].iloc[-1]
                            elif not daily.empty and daily.index.name == "day" and day_key in list(daily.index.astype(str)):
                                drow = daily.loc[day_key]
                            if drow is not None:
                                rctx = _safe_float(drow.get("readiness"))
                                sctx = _safe_float(drow.get("sleep_score"))
                                if (rctx is not None and rctx < 65) or (sctx is not None and sctx < 70):
                                    grade_score -= 10
                        except Exception:
                            pass
                        grade_score = max(0.0, min(100.0, grade_score))

                    st.markdown("### Workout grade + takeaway")
                    g1, g2, g3 = st.columns(3)
                    g1.metric("Grade", letter_grade(grade_score))
                    g2.metric("Execution score", _fmt(None if grade_score is None else round(grade_score, 0)))
                    g3.metric("Main signal", "HR compliance" if target_zones else "Non‑HR session")

                    takeaway: List[str] = []
                    if target_zones:
                        if compliance is not None and compliance >= 85:
                            takeaway.append("Execution: on-target.")
                        elif compliance is not None and compliance >= 60:
                            takeaway.append("Execution: some drift.")
                        elif compliance is not None:
                            takeaway.append("Execution: missed stimulus.")

                        # drift feedback personalized when we have a baseline
                        if drift is not None and ("Aerobic" in intent or "Z2" in intent):
                            try:
                                ib = intent_baselines.get(str(intent)) if isinstance(intent_baselines, dict) else None
                                drift_med = _safe_float((ib or {}).get("drift_median"))
                                drift_iqr = _safe_float((ib or {}).get("drift_iqr"))
                            except Exception:
                                drift_med, drift_iqr = None, None

                            if drift_med is not None and drift_iqr is not None and drift_iqr > 0:
                                if drift > drift_med + 2 * drift_iqr:
                                    takeaway.append("Aerobic efficiency: drift is high vs YOUR baseline → start easier / fuel earlier / shorten.")
                                elif drift > drift_med + 1 * drift_iqr:
                                    takeaway.append("Aerobic efficiency: drift elevated vs YOUR baseline → tighten pacing early.")
                                else:
                                    takeaway.append("Aerobic efficiency: within your normal range.")
                            else:
                                if drift > 8:
                                    takeaway.append("Aerobic efficiency: excessive drift → start easier / fuel earlier / shorten.")
                                elif drift > 5:
                                    takeaway.append("Aerobic efficiency: moderate drift → tighten pacing early.")
                    else:
                        takeaway.append("This intent is not HR-scored. Track quality via consistency + volume + soreness/performance notes.")

                    # Drivers (simple root-cause)
                    drivers: List[str] = []
                    try:
                        day_key = str(wr.get("day"))
                        drow = None
                        if not daily.empty and "day" in daily.columns and (daily["day"].astype(str) == day_key).any():
                            drow = daily[daily["day"].astype(str) == day_key].iloc[-1]
                        elif not daily.empty and daily.index.name == "day" and day_key in list(daily.index.astype(str)):
                            drow = daily.loc[day_key]
                        if drow is not None:
                            rctx = _safe_float(drow.get("readiness"))
                            sctx = _safe_float(drow.get("sleep_score"))
                            if sctx is not None and sctx < 70:
                                drivers.append("Low sleep score likely reduced execution.")
                            if rctx is not None and rctx < 65:
                                drivers.append("Low readiness suggests fatigue/illness/stress.")
                            td = _safe_float(drow.get("temp_dev"))
                            if td is not None and abs(td) > 0.3:
                                drivers.append("Temp deviation elevated → consider illness/alcohol/late meal.")
                            rd = _safe_float(drow.get("resp_rate_dev"))
                            if rd is not None and abs(rd) >= 0.5:
                                drivers.append("Respiratory rate shifted → recovery strain may be physiological, not just pacing.")
                            sp = _safe_float(drow.get("spo2"))
                            if sp is not None and sp < 97:
                                drivers.append("SpO₂ lower than typical → airway/altitude/recovery check.")
                    except Exception:
                        pass

                    if takeaway:
                        st.write("**Takeaway:**")
                        for t in takeaway[:3]:
                            st.write(f"- {t}")
                    if drivers:
                        st.write("**Likely drivers today:**")
                        for d in drivers[:3]:
                            st.write(f"- {d}")

                    selected_effect = pd.DataFrame()
                    if training_effects is not None and not training_effects.empty:
                        current_workout_key = str(wr.get("workout_key") or workout_key(wr))
                        selected_effect = training_effects[training_effects["workout_key"].astype(str) == current_workout_key].tail(1)
                    if not selected_effect.empty:
                        effect_row = selected_effect.iloc[-1]
                        sleep_debt_h = _safe_float(effect_row.get("sleep_debt_h"))
                        temp_dev_today = _safe_float(effect_row.get("temp_dev"))
                        prior_day_load = _safe_float(effect_row.get("prior_day_load"))
                        st.markdown("#### Context at workout start")
                        cx1, cx2, cx3 = st.columns(3)
                        cx1.metric("Sleep debt", _fmt(None if sleep_debt_h is None else f"{sleep_debt_h:.1f}h"))
                        cx2.metric("Temp deviation", _fmt(None if temp_dev_today is None else f"{temp_dev_today:+.1f} C"))
                        cx3.metric("Prior-day load", _fmt(None if prior_day_load is None else f"{prior_day_load:.0f} TRIMP"))
                        context_reasons: List[str] = []
                        if sleep_debt_h is not None and sleep_debt_h > 0.75:
                            context_reasons.append(f"Started with meaningful sleep debt ({sleep_debt_h:.1f}h).")
                        if temp_dev_today is not None and abs(temp_dev_today) > 0.3:
                            context_reasons.append(f"Temperature deviation was elevated ({temp_dev_today:+.1f} C).")
                        sweet_high = _safe_float(personal_thresholds.get("trimp_sweet_spot_high")) if isinstance(personal_thresholds, dict) else None
                        if prior_day_load is not None and sweet_high is not None and prior_day_load > sweet_high:
                            context_reasons.append(f"Prior-day load ({prior_day_load:.0f} TRIMP) was above your sweet spot.")
                        elif prior_day_load is not None and sweet_high is None and prior_day_load > 120:
                            context_reasons.append(f"Prior-day load ({prior_day_load:.0f} TRIMP) was high.")
                        if context_reasons:
                            st.write("- **Context verdict:** This session may have looked worse because context was poor, not because fitness regressed.")
                            for line in context_reasons[:3]:
                                st.write(f"- {line}")

                    cA, cB, cC = st.columns(3)
                    cA.metric("Est. load (TRIMP)", _fmt(None if trimp is None else round(trimp, 1)))
                    cB.metric("HR drift", _fmt(None if drift is None else f"{drift:.1f}%"))
                    # "Effectiveness" heuristic: did it match an intended stimulus?
                    stimulus = "Aerobic" if tz_min.get("Z2", 0) >= max(10, (duration_min or 0) * 0.35) else "Mixed/High"
                    cC.metric("Stimulus", stimulus)

                    # Time in zones chart
                    zdf = pd.DataFrame({"zone": list(tz_min.index), "minutes": list(tz_min.values)})
                    zdf = zdf[zdf["minutes"].notna()]
                    if not zdf.empty:
                        fig2 = px.bar(zdf, x="zone", y="minutes", title="Time in HR zones (min)")
                        fig2.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig2, width="stretch")

                    # Similar workout comparison (at-home)
                    st.markdown("### Compare to similar workouts")
                    try:
                        candidates = w_sorted.copy()
                        # show baseline for this intent (personal)
                        ib = intent_baselines.get(str(intent)) if isinstance(intent_baselines, dict) else None
                        if ib:
                            st.caption(f"Personal baseline for this intent (n={ib.get('n')}): avg HR ~{ib.get('avg_hr_median'):.0f} bpm; drift ~{ib.get('drift_median'):.1f}%" if ib.get('avg_hr_median') is not None and ib.get('drift_median') is not None else f"Personal baseline for this intent (n={ib.get('n')})")
                        candidates = candidates[candidates.index != pick_idx]
                        # same intent
                        if "workout_intent" in candidates.columns:
                            candidates = candidates[candidates["workout_intent"].astype(str) == str(intent)]
                        # duration within ±20%
                        if duration_min is not None and "duration_min" in candidates.columns:
                            candidates = candidates[pd.to_numeric(candidates["duration_min"], errors="coerce").between(0.8*float(duration_min), 1.2*float(duration_min))]
                        candidates = candidates.head(5)

                        comp_rows: List[dict] = []
                        for idx2, r2 in candidates.iterrows():
                            s2, e2 = workout_window(r2)
                            s2 = s2 or _parse_dt(str(r2.get("day")) + "T00:00:00Z")
                            if s2 is None or e2 is None:
                                continue
                            seg2 = hr_points[(hr_points["ts"] >= s2) & (hr_points["ts"] <= e2)].copy() if not hr_points.empty else pd.DataFrame()
                            if seg2.empty:
                                continue
                            avg2 = float(seg2["bpm"].mean())
                            drift2 = compute_hr_drift(seg2)
                            comp_rows.append({
                                "day": str(r2.get("day")),
                                "avg_hr": round(avg2, 0),
                                "drift_%": None if drift2 is None else round(drift2, 1),
                                "sleep_h": _safe_float(daily.loc[daily["day"].astype(str) == str(r2.get("day")), "sleep_total_s"].iloc[-1]) / 3600.0 if not daily.loc[daily["day"].astype(str) == str(r2.get("day")), "sleep_total_s"].empty and _safe_float(daily.loc[daily["day"].astype(str) == str(r2.get("day")), "sleep_total_s"].iloc[-1]) is not None else None,
                                "temp_dev": _safe_float(daily.loc[daily["day"].astype(str) == str(r2.get("day")), "temp_dev"].iloc[-1]) if "temp_dev" in daily.columns and not daily.loc[daily["day"].astype(str) == str(r2.get("day")), "temp_dev"].empty else None,
                                "prior_day_load": _safe_float(daily.loc[daily["day"].astype(str) < str(r2.get("day")), "trimp"].tail(1).iloc[-1]) if "trimp" in daily.columns and len(daily.loc[daily["day"].astype(str) < str(r2.get("day")), "trimp"].tail(1)) else None,
                            })

                        if comp_rows:
                            comp_df = pd.DataFrame(comp_rows).sort_values("day", ascending=False)
                            st.dataframe(comp_df, width="stretch")
                            # quick interpretation
                            cur_avg = float(seg["bpm"].mean())
                            prev_avgs = [r["avg_hr"] for r in comp_rows if r.get("avg_hr") is not None]
                            if prev_avgs:
                                base = float(pd.Series(prev_avgs).median())
                                delta = cur_avg - base
                                if abs(delta) >= 5:
                                    st.caption(f"Your avg HR is {delta:+.0f} bpm vs similar sessions — fatigue/heat/fueling can explain this.")
                            if not selected_effect.empty:
                                effect_row = selected_effect.iloc[-1]
                                session_context: List[str] = []
                                if _safe_float(effect_row.get("sleep_debt_h")) is not None and _safe_float(effect_row.get("sleep_debt_h")) > 0.75:
                                    session_context.append(f"sleep debt {_safe_float(effect_row.get('sleep_debt_h')):.1f}h")
                                if _safe_float(effect_row.get("temp_dev")) is not None and abs(_safe_float(effect_row.get("temp_dev"))) > 0.3:
                                    session_context.append(f"temp dev {_safe_float(effect_row.get('temp_dev')):+.1f} C")
                                if _safe_float(effect_row.get("prior_day_load")) is not None:
                                    session_context.append(f"prior-day load {_safe_float(effect_row.get('prior_day_load')):.0f} TRIMP")
                                if session_context:
                                    st.caption("Current-session context: " + ", ".join(session_context) + ". Read poor comparisons as context-limited first, fitness-limited second.")
                            if not comp_df.empty:
                                context_bits: List[str] = []
                                if comp_df["sleep_h"].notna().any():
                                    context_bits.append("sleep duration")
                                if comp_df["temp_dev"].notna().any():
                                    context_bits.append("temp deviation")
                                if comp_df["prior_day_load"].notna().any():
                                    context_bits.append("prior-day load")
                                if context_bits:
                                    st.caption(f"Context included for these comparisons: {', '.join(context_bits)}.")
                    except Exception:
                        st.info("Similar workout comparison unavailable.")

                    # Recommendations (goal-specific)
                    st.markdown("### How to make this training more effective")
                    recs: List[str] = []
                    if goal.startswith("Performance (endurance)"):
                        # If too much Z3, warn grey zone
                        z2 = float(tz_min.get("Z2", 0) or 0)
                        z3 = float(tz_min.get("Z3", 0) or 0)
                        z4p = float(tz_min.get("Z4", 0) or 0) + float(tz_min.get("Z5", 0) or 0)
                        if duration_min and z3 / max(1.0, duration_min) > 0.25 and z4p < 3:
                            recs.append("Too much Z3 (grey zone). Next aerobic day: keep HR capped to stay in Z2.")
                        if duration_min and z2 / max(1.0, duration_min) < 0.35 and z4p < 3:
                            recs.append("Low true Z2 time. Start easier in the first 10–15 min to prevent drift into Z3.")
                        if drift is not None and drift > 6:
                            recs.append("High HR drift. Improve aerobic efficiency: hydrate, fuel earlier, slow pace, or shorten duration.")
                    elif goal.startswith("Performance (strength"):
                        recs.append("If this was a strength session: track sets/reps/RPE manually (Oura API doesn’t include lifting details).")
                        recs.append("Use conditioning sessions as separate workouts so we can score them by zones + drift.")
                    elif goal.startswith("Stress resilience"):
                        recs.append("Keep intensity mostly Z1–Z2. Avoid late-day Z4+ if sleep is impacted.")
                    else:
                        recs.append("Set a target for each session (Z2 base / intervals / recovery). The dashboard will score compliance.")

                    if not recs:
                        recs = ["Solid session. Keep consistency and progress volume gradually."]
                    for rline in recs[:5]:
                        st.write(f"- {rline}")

            st.divider()
            st.markdown("## Weekly coaching insights")

            # ------------------------------
            # Weekly intensity distribution (from workout-linked HR when available)
            # ------------------------------
            st.markdown("### 1) Intensity distribution (Z1–Z5)")
            st.caption("Built from heart-rate samples inside workout windows when available. If HR doesn’t align for some sessions, those sessions are skipped.")

            # Aggregate last 28 days
            horizon_days = 28
            cutoff = date.today() - timedelta(days=horizon_days)
            w28 = w_sorted.copy()
            try:
                w28["day_dt"] = pd.to_datetime(w28["day"], errors="coerce").dt.date
                w28 = w28[w28["day_dt"].notna() & (w28["day_dt"] >= cutoff)]
            except Exception:
                w28 = w28

            zone_rows: List[dict] = []
            load_rows: List[dict] = []

            if hr_points.empty:
                st.info("Heart-rate stream unavailable for zone distribution. We can still show volume and workout frequency.")
            else:
                for _, row in w28.iterrows():
                    sdt, edt = workout_window(row)
                    if sdt is None or edt is None:
                        continue
                    seg = hr_points[(hr_points["ts"] >= sdt) & (hr_points["ts"] <= edt)].copy()
                    if len(seg) < 20:
                        continue
                    tz = time_in_zones(seg, zones) / 60.0
                    # weekly bucket
                    wk = pd.to_datetime(str(row.get("day")), errors="coerce").to_period("W").start_time.date().isoformat()
                    for zname, mins in tz.items():
                        zone_rows.append({"week": wk, "zone": str(zname), "minutes": float(mins)})

                    # TRIMP
                    try:
                        tr = compute_banister_trimp(
                            seg,
                            max_hr=int(max_hr),
                            resting_hr=int(resting_hr),
                            sex=sex,
                        )
                        day_iso = None
                        try:
                            day_iso = pd.to_datetime(str(row.get("day")), errors="coerce").date().isoformat()
                        except Exception:
                            day_iso = None
                        if tr is not None:
                            load_rows.append({"week": wk, "day": day_iso, "trimp": tr})
                    except Exception:
                        pass

                if zone_rows:
                    zdf = pd.DataFrame(zone_rows)
                    zagg = zdf.groupby(["week", "zone"], as_index=False)["minutes"].sum()

                    # stacked bar in zone order
                    order = ["<Z1", "Z1", "Z2", "Z3", "Z4", "Z5"]
                    zagg["zone"] = pd.Categorical(zagg["zone"], categories=order, ordered=True)
                    zagg = zagg.sort_values(["week", "zone"])

                    fig = px.bar(zagg, x="week", y="minutes", color="zone", title="Weekly minutes by HR zone", category_orders={"zone": order})
                    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), barmode="stack")
                    st.plotly_chart(fig, width="stretch")

                    # Grey-zone + polarization flags (last complete week)
                    last_week = sorted(zagg["week"].unique())[-1]
                    wk_df = zagg[zagg["week"] == last_week].copy()
                    total = float(wk_df["minutes"].sum()) if not wk_df.empty else 0.0
                    z2 = float(wk_df[wk_df["zone"] == "Z2"]["minutes"].sum())
                    z3 = float(wk_df[wk_df["zone"] == "Z3"]["minutes"].sum())
                    z4p = float(wk_df[wk_df["zone"].isin(["Z4", "Z5"])]["minutes"].sum())

                    st.markdown("**Coach flags (last week):**")
                    flags: List[str] = []
                    if total > 0:
                        if (z3 / total) > 0.25 and z4p < 10:
                            flags.append("Too much Z3 (grey zone). Make easy days truly easy (stay Z2) or commit to Z4 intervals.")
                        if (z2 / total) < 0.35 and total > 60:
                            flags.append("Low Z2 base volume. Add 1–2 strict Z2 sessions/week.")
                        if z4p == 0 and total > 90 and goal.startswith("Performance (endurance)"):
                            flags.append("No high-intensity stimulus recorded. Consider 1 interval session/week if appropriate.")
                    if not flags:
                        flags = ["Structure looks reasonable. Keep consistency and progress gradually."]
                    for f in flags:
                        st.write(f"- {f}")
                else:
                    st.info("Not enough HR-aligned workouts in the last 28 days to compute zone distribution.")

            # ------------------------------
            # Load ramp rate + deload suggestion
            # ------------------------------
            st.markdown("### 2) Load ramp rate (don’t spike)")
            st.caption("Uses Banister-style TRIMP when HR aligns; otherwise falls back to weekly minutes.")

            ramp_msg = None
            if load_rows:
                ldf = pd.DataFrame(load_rows)
                lweek = ldf.groupby("week", as_index=False)["trimp"].sum().sort_values("week")
                fig = px.line(lweek, x="week", y="trimp", markers=True, title="Weekly load (TRIMP)")
                fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

                if len(lweek) >= 3:
                    last = float(lweek.iloc[-1]["trimp"])
                    prev4 = lweek.iloc[max(0, len(lweek)-5):len(lweek)-1]["trimp"]
                    base = float(prev4.mean()) if len(prev4) else None
                    if base and base > 0:
                        ramp = (last - base) / base * 100.0
                        if ramp > 20:
                            ramp_msg = f"Load spike: +{ramp:.0f}% vs recent baseline. Consider a deload (−20–30% volume) this week."
                        elif ramp < -20:
                            ramp_msg = f"Load drop: {ramp:.0f}% vs baseline. If unplanned, check illness/travel/sleep."
                        else:
                            ramp_msg = f"Ramp looks controlled ({ramp:+.0f}% vs baseline)."
            else:
                # fallback to weekly minutes
                if "duration_min" in w.columns and w["duration_min"].notna().any():
                    ww = w.dropna(subset=["day"]).copy()
                    ww["day"] = pd.to_datetime(ww["day"], errors="coerce")
                    ww = ww.dropna(subset=["day"])
                    ww["week"] = ww["day"].dt.to_period("W").astype(str)
                    mins = ww.groupby("week")["duration_min"].sum().reset_index().sort_values("week")
                    fig = px.line(mins, x="week", y="duration_min", markers=True, title="Weekly training minutes")
                    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, width="stretch")

                    if len(mins) >= 3:
                        last = float(mins.iloc[-1]["duration_min"])
                        prev4 = mins.iloc[max(0, len(mins)-5):len(mins)-1]["duration_min"]
                        base = float(prev4.mean()) if len(prev4) else None
                        if base and base > 0:
                            ramp = (last - base) / base * 100.0
                            if ramp > 25:
                                ramp_msg = f"Volume spike: +{ramp:.0f}% vs baseline. Consider deloading to protect adaptation." 
                            else:
                                ramp_msg = f"Volume change vs baseline: {ramp:+.0f}%."

            st.write(f"- {_fmt(ramp_msg)}" if ramp_msg else "- Not enough history yet to compute ramp rate.")

            # ------------------------------
            # Fitness/Fatigue model (CTL/ATL/TSB)
            # ------------------------------
            st.markdown("### 3) Fitness/Fatigue (CTL/ATL/TSB)")
            st.caption("At-home analysis: are you building fitness (CTL) without drifting into fatigue (ATL)?")

            try:
                if load_rows:
                    ldf = pd.DataFrame(load_rows)
                    if "day" in ldf.columns:
                        lday = ldf.dropna(subset=["day"]).groupby("day", as_index=False)["trimp"].sum().sort_values("day")
                        if not lday.empty and len(lday) >= 10:
                            lday["day"] = pd.to_datetime(lday["day"], errors="coerce")
                            lday = lday.dropna(subset=["day"]).sort_values("day")
                            lday = lday.set_index("day")

                            ctl = ewma(lday["trimp"], 42)
                            atl = ewma(lday["trimp"], 7)
                            tsb = ctl - atl

                            # Latest snapshot
                            last_ctl = _safe_float(ctl.iloc[-1])
                            last_atl = _safe_float(atl.iloc[-1])
                            last_tsb = _safe_float(tsb.iloc[-1])

                            s1, s2, s3, s4 = st.columns(4)
                            s1.metric("CTL (42d)", _fmt(None if last_ctl is None else round(last_ctl, 1)))
                            s2.metric("ATL (7d)", _fmt(None if last_atl is None else round(last_atl, 1)))
                            s3.metric("TSB (CTL−ATL)", _fmt(None if last_tsb is None else round(last_tsb, 1)))

                            status = "—"
                            if last_tsb is not None:
                                if last_tsb >= 5:
                                    status = "Fresh"
                                elif last_tsb >= -5:
                                    status = "Productive"
                                elif last_tsb >= -15:
                                    status = "Fatigued"
                                else:
                                    status = "Overreaching risk"
                            s4.metric("Status", status)

                            plot_df = pd.DataFrame({"Daily load": lday["trimp"], "CTL": ctl, "ATL": atl, "TSB": tsb}).reset_index().rename(columns={"index": "day"})
                            fig = px.line(plot_df, x="day", y=["Daily load", "CTL", "ATL", "TSB"], title="Training load model")
                            fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.info("Need more HR-aligned workouts (≈10+ days) to compute CTL/ATL/TSB.")
                    else:
                        st.info("No daily load series available yet.")
                else:
                    st.info("No TRIMP load series available yet (needs HR-aligned workouts).")
            except Exception:
                st.info("Load model unavailable due to missing history.")

            # ------------------------------
            # Sleep consistency + training effectiveness linkage
            # ------------------------------
            st.markdown("### 4) Sleep consistency (athlete lever)")
            df_sleep = data.get("sleep", {}).get("df", pd.DataFrame())
            if df_sleep.empty:
                st.info("Sleep sessions not available.")
            else:
                s = df_sleep.copy()
                # pick timestamps
                start_col = "start_datetime" if "start_datetime" in s.columns else ("bedtime_start" if "bedtime_start" in s.columns else None)
                end_col = "end_datetime" if "end_datetime" in s.columns else ("bedtime_end" if "bedtime_end" in s.columns else None)
                if start_col and end_col:
                    s["start"] = s[start_col].apply(_parse_dt)
                    s["end"] = s[end_col].apply(_parse_dt)
                    s = s.dropna(subset=["start", "end"]).copy()
                    # last 14
                    s = s.sort_values("start").tail(14)

                    s["bed_min"] = s["start"].apply(clock_minutes_from_anchor)
                    s["wake_min"] = s["end"].apply(clock_minutes_from_anchor)

                    bed_sd = float(pd.Series(s["bed_min"]).std()) if len(s) >= 5 else None
                    wake_sd = float(pd.Series(s["wake_min"]).std()) if len(s) >= 5 else None

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Bedtime variability", f"{bed_sd:.0f} min" if bed_sd is not None else "—")
                    c2.metric("Wake variability", f"{wake_sd:.0f} min" if wake_sd is not None else "—")
                    c3.metric("Nights analyzed", str(len(s)))

                    insight: List[str] = []
                    if bed_sd is not None and bed_sd > 45:
                        insight.append("Bedtime is inconsistent. Tightening to ±30 min is one of the highest-ROI performance levers.")
                    if wake_sd is not None and wake_sd > 45:
                        insight.append("Wake time is inconsistent. Keep wake time stable to anchor circadian rhythm.")
                    if not insight:
                        insight.append("Sleep timing is fairly consistent — good foundation for adaptation.")

                    for it in insight:
                        st.write(f"- {it}")

                    # Chronotype + timing (simple, useful at-home)
                    try:
                        midpoints: List[datetime] = []
                        for _, rr in s.iterrows():
                            stt = rr.get("start")
                            enn = rr.get("end")
                            if isinstance(stt, datetime) and isinstance(enn, datetime):
                                dur_s = (enn - stt).total_seconds()
                                if dur_s > 0:
                                    midpoints.append(stt + timedelta(seconds=dur_s / 2.0))
                        avg_mid = None
                        if len(midpoints) >= 7:
                            avg_mid = sum([m.hour + m.minute / 60.0 for m in midpoints]) / len(midpoints)
                        chrono = chronotype_from_sleep_midpoint(avg_mid)

                        win = "—"
                        avoid = "—"
                        if chrono == "MORNING":
                            win = "08:00–11:00"
                            avoid = "Late evening hard sessions"
                        elif chrono == "EVENING":
                            win = "17:00–20:00"
                            avoid = "Before 10:00 high intensity"
                        elif chrono == "INTERMEDIATE":
                            win = "14:00–18:00"
                            avoid = "—"

                        st.markdown("#### Chronotype + best training window")
                        a1, a2, a3 = st.columns(3)
                        a1.metric("Chronotype", chrono)
                        a2.metric("Best window (hard sessions)", win)
                        a3.metric("Avoid", avoid)

                        st.markdown("#### Optimal intervention windows")
                        windows = compute_intervention_windows(hr_points, chrono)
                        labels = {
                            "hard_training": "Hard training",
                            "easy_training": "Easy training / walk",
                            "nsdr_reset": "NSDR / breathwork",
                            "cold_exposure": "Morning light / walk",
                            "last_caffeine": "Last caffeine",
                        }
                        for key in ["hard_training", "easy_training", "nsdr_reset", "cold_exposure", "last_caffeine"]:
                            window = windows.get(key)
                            if not window:
                                continue
                            st.write(f"- **{labels.get(key, key)}:** {window['window']}")
                            st.caption(window["why"])
                    except Exception:
                        pass
                else:
                    st.info("Sleep timing fields missing in your sleep payload; can’t compute consistency.")

            st.divider()
            st.markdown("### Workout log")
            cols = [c for c in ["day", "type", "workout_intent", "duration_min", "calories", "distance"] if c in w.columns]
            st.dataframe(w[cols].sort_values("day", ascending=False).head(200), width="stretch")

    # ------------------
    # Fitness
    # ------------------
    with guarded_tab(tabs[5], "Legacy: VO₂ / CVA"):
        st.subheader("VO₂ / CVA")
        st.caption("Legacy deep-dive view retained for compatibility.")
        st.caption(f"Longevity analysis uses up to {analysis_window_label} of history.")

        c1, c2, c3 = st.columns(3)
        c1.metric("VO₂max", _fmt(vo2))
        c2.metric("CVA", _fmt(cva))
        delta = None
        if age_years is not None and cva is not None:
            try:
                delta = float(cva) - float(age_years)
            except Exception:
                delta = None
        c3.metric("CVA Δ vs age", f"{delta:.1f}" if delta is not None else "—")

        if vo2_df.empty:
            st.info(
                "VO₂max not available from the API for this account/date range. "
                "Check Data Access → vO2_max rows. If it's 200 + 0 rows, Oura isn't providing VO₂max."
            )
        else:
            st.markdown("### VO₂max")
            dfp = vo2_df.copy()
            # Prefer day; else derive from timestamp
            if "day" not in dfp.columns and "timestamp" in dfp.columns:
                dfp = dfp.copy()
                dfp["day"] = dfp["timestamp"].apply(_to_day)
            if "day" in dfp.columns:
                dfp["day"] = pd.to_datetime(dfp["day"], errors="coerce")

            val_col = "vo2_max" if "vo2_max" in dfp.columns else ("value" if "value" in dfp.columns else ("vo2max" if "vo2max" in dfp.columns else None))
            if val_col:
                dfp[val_col] = pd.to_numeric(dfp[val_col], errors="coerce")

            # Show last recorded date + guidance on next recording
            last_day = None
            try:
                if "day" in dfp.columns and dfp["day"].notna().any():
                    last_day = dfp.dropna(subset=["day"]).sort_values("day").iloc[-1]["day"]
            except Exception:
                last_day = None

            cA, cB = st.columns(2)
            cA.metric("Last recorded", last_day.date().isoformat() if last_day is not None else "—")
            cB.metric("Next recording", "Not scheduled (Oura updates when it has enough qualifying data)")
            st.caption(
                "Oura doesn’t expose a schedule for VO₂max updates. It typically updates only when you do qualifying activity (often walking/running outdoors with enough steady data)."
            )

            if "day" in dfp.columns and val_col:
                st.markdown("#### History")
                fig = px.line(dfp.dropna(subset=["day"]), x="day", y=val_col, markers=True)
                fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, width="stretch")

        render_vo2_longevity_panel(
            vo2_analysis=vo2_longevity,
            vo2_decay=vo2_decay,
        )
        render_longevity_score_panel(decision_score, compact=False)

        if not cva_df.empty:
            st.markdown("### CVA history")
            dfp = cva_df.copy()
            if "day" in dfp.columns:
                dfp["day"] = pd.to_datetime(dfp["day"], errors="coerce")
                val_col = "cardiovascular_age" if "cardiovascular_age" in dfp.columns else ("cva" if "cva" in dfp.columns else None)
                if val_col:
                    dfp[val_col] = pd.to_numeric(dfp[val_col], errors="coerce")
                    fig = px.line(dfp.dropna(subset=["day"]), x="day", y=val_col, markers=True)
                    fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, width="stretch")

    # ------------------
    # Resting HR
    # ------------------
    with guarded_tab(tabs[6], "Legacy: Resting HR"):
        st.caption("Legacy deep-dive view retained for compatibility.")
        render_nightly_metric_tab(
            daily=daily,
            key="resting_hr",
            label="Resting HR",
            higher_is_better=False,
            goal=goal,
            mode=str(day_mode.get("mode") or "MAINTAIN"),
            zones=zones,
            trust=rhr_trust,
        )
        st.caption(f"Trend, stability, and recovery-rate analysis use up to {analysis_window_label} of history.")
        render_rhr_longevity_panel(
            rhr_trend=rhr_trend,
            rhr_stability_stats=rhr_stability_stats,
            rhr_recovery=rhr_recovery,
        )

    # ------------------
    # HRV
    # ------------------
    with guarded_tab(tabs[7], "Legacy: HRV"):
        st.caption("Legacy deep-dive view retained for compatibility.")
        render_nightly_metric_tab(
            daily=daily,
            key="hrv_rmssd",
            label="HRV rmSSD",
            higher_is_better=True,
            goal=goal,
            mode=str(day_mode.get("mode") or "MAINTAIN"),
            zones=zones,
            trust=hrv_trust,
        )
        st.caption(f"Biological-age and pattern analysis use up to {analysis_window_label} of history.")
        render_hrv_longevity_panel(
            hrv_age=hrv_age,
            hrv_pattern=hrv_pattern,
        )

    # ------------------
    # Experiments (Tags)
    # ------------------
    with guarded_tab(tabs[3], "Experiments"):
        st.subheader("Experiments")
        st.caption("This is the N-of-1 lab. Oura tags still work, but experiments no longer depend on them.")
        st.caption(f"Using experiment history from {analysis_window_label}.")
        if not can_edit_active_account:
            st.info("Viewing a shared account read-only. Only the connected account owner can log or clear manual experiment events.")

        st.markdown("### Log a daily event record")
        selected_day = st.date_input("Day", value=end_d, key="event_record_day")
        selected_day_iso = selected_day.isoformat()
        existing_record = behavior_events[behavior_events["day"].astype(str) == selected_day_iso].tail(1)
        current_record = existing_record.iloc[-1] if not existing_record.empty else pd.Series(dtype=object)
        with st.form("behavior_event_form", clear_on_submit=False):
            f1, f2, f3, f4 = st.columns(4)
            alcohol = f1.checkbox("Alcohol", value=bool(bool_from_value(current_record.get("alcohol"))), key="event_alcohol")
            late_meal = f2.checkbox("Late meal", value=bool(bool_from_value(current_record.get("late_meal"))), key="event_late_meal")
            travel = f3.checkbox("Travel", value=bool(bool_from_value(current_record.get("travel"))), key="event_travel")
            illness = f4.checkbox("Illness", value=bool(bool_from_value(current_record.get("illness"))), key="event_illness")
            f5, f6, f7, f8 = st.columns(4)
            sauna = f5.checkbox("Sauna", value=bool(bool_from_value(current_record.get("sauna"))), key="event_sauna")
            cold = f6.checkbox("Cold", value=bool(bool_from_value(current_record.get("cold"))), key="event_cold")
            caffeine_late = f7.checkbox("Late caffeine", value=bool(bool_from_value(current_record.get("caffeine_late"))), key="event_caffeine_late")
            wellness_options = [None] + list(range(1, 11))
            current_wellness = int(_safe_float(current_record.get("manual_wellness"))) if _safe_float(current_record.get("manual_wellness")) is not None else None
            manual_wellness = f8.selectbox("Wellness (1-10)", options=wellness_options, index=wellness_options.index(current_wellness) if current_wellness in wellness_options else 0, key="event_manual_wellness")
            supplement_value = current_record.get("supplement")
            supplement = st.text_input("Supplement", value="" if supplement_value is None or (isinstance(supplement_value, float) and math.isnan(supplement_value)) else str(supplement_value), key="event_supplement")
            notes_value = current_record.get("notes")
            notes = st.text_area("Notes", value="" if notes_value is None or (isinstance(notes_value, float) and math.isnan(notes_value)) else str(notes_value), key="event_notes")
            save_cols = st.columns(2)
            save_pressed = save_cols[0].form_submit_button("Save daily record", disabled=not can_edit_active_account)
            clear_pressed = save_cols[1].form_submit_button("Clear this day", disabled=not can_edit_active_account)
            if save_pressed:
                save_event_record(
                    behavior_event_path,
                    EventRecord(
                        day=selected_day_iso,
                        alcohol=alcohol,
                        late_meal=late_meal,
                        travel=travel,
                        illness=illness,
                        supplement=supplement,
                        sauna=sauna,
                        cold=cold,
                        caffeine_late=caffeine_late,
                        manual_wellness=int(manual_wellness) if manual_wellness is not None else None,
                        notes=notes,
                    ),
                )
                st.success("Daily event record saved. Refreshing analysis.")
                st.rerun()
            if clear_pressed:
                delete_event_record(behavior_event_path, selected_day_iso)
                st.success("Daily event record cleared. Refreshing analysis.")
                st.rerun()

        df_tag = analysis_data.get("tag", {}).get("df", pd.DataFrame())
        manual_events = behavior_events.copy()
        manual_event_rows = behavior_events_to_tag_rows(manual_events)
        combined_events = pd.concat([df_tag, manual_event_rows], ignore_index=True, sort=False)

        st.markdown("### Logged experiment events")
        if manual_events.empty:
            st.info("No manual events logged yet.")
            st.write("- Log **alcohol**, **late meal**, **travel**, **illness**, **supplement**, **sauna**, **cold exposure**, and **late caffeine** here.")
            st.write("- 2-3 weeks is enough to start seeing behavior-to-next-day effects.")
        else:
            st.dataframe(
                manual_events.sort_values("day", ascending=False).head(20)[
                    ["day", "alcohol", "late_meal", "travel", "illness", "supplement", "sauna", "cold", "caffeine_late", "manual_wellness", "notes", "updated_at"]
                ].rename(
                    columns={
                        "day": "Day",
                        "alcohol": "Alcohol",
                        "late_meal": "Late meal",
                        "travel": "Travel",
                        "illness": "Illness",
                        "supplement": "Supplement",
                        "sauna": "Sauna",
                        "cold": "Cold",
                        "caffeine_late": "Late caffeine",
                        "manual_wellness": "Wellness",
                        "notes": "Notes",
                        "updated_at": "Updated at",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        st.markdown("### Experiment history")
        if combined_events.empty:
            st.info("No Oura tags or manual events available yet.")
            st.write("- Start tagging or logging **alcohol**, **late meals**, **supplements**, and **travel**.")
            st.write("- Without experiment inputs, diagnosis falls back to sleep, training load, steps, and temperature deviation.")
        else:
            t = combined_events.copy()
            if "day" not in t.columns:
                for c in ["timestamp", "start_datetime"]:
                    if c in t.columns:
                        t["day"] = t[c].apply(_to_day)
                        break
            show_cols = [c for c in ["day", "source", "tag_type", "tags", "text", "comment"] if c in t.columns]
            st.dataframe(t[show_cols].sort_values("day", ascending=False).head(200), width="stretch")
            metric_choice = st.selectbox(
                "Analyze next-day effect on",
                options=[
                    ("hrv_rmssd", "HRV rmSSD"),
                    ("resting_hr", "Resting HR"),
                    ("sleep_score", "Sleep score"),
                    ("readiness", "Readiness"),
                ],
                format_func=lambda item: item[1],
                key="tag_effect_metric",
            )
            metric_key, metric_label_choice = metric_choice
            effects = tag_effect_analysis(
                analysis_daily,
                t,
                metric=metric_key,
                higher_is_better=metric_key != "resting_hr",
            )
            if effects.empty:
                st.info("Not enough repeated experiment days with next-day data to estimate effects yet. You need at least two occurrences of an event plus next-day metric coverage.")
            else:
                show = effects.copy()
                show["mean_delta"] = pd.to_numeric(show["mean_delta"], errors="coerce").round(1)
                show["median_delta"] = pd.to_numeric(show["median_delta"], errors="coerce").round(1)
                show["next_day_avg"] = pd.to_numeric(show["next_day_avg"], errors="coerce").round(1)
                show["prior_avg"] = pd.to_numeric(show["prior_avg"], errors="coerce").round(1)
                st.markdown(f"#### Next-day {metric_label_choice} effect vs prior baseline")
                st.dataframe(
                    show.rename(
                        columns={
                            "tag": "Event",
                            "n": "N",
                            "mean_delta": "Mean Δ",
                            "median_delta": "Median Δ",
                            "next_day_avg": "Next-day avg",
                            "prior_avg": "Prior avg",
                        }
                    )[["Event", "N", "Mean Δ", "Median Δ", "Next-day avg", "Prior avg"]],
                    width="stretch",
                )
                st.caption("Mean and median deltas compare the day after each event against the prior 7 available nights or days for that metric.")

    # ------------------
    # Data Access
    # ------------------
    with guarded_tab(tabs[8], "Data Access"):
        st.subheader("Data Access (what the API is *actually* returning)")
        st.caption(f"Experiment storage for this account: `{pathlib.Path(behavior_event_path).expanduser()}`")
        rows = []
        for key, blob in data.items():
            rows.append({
                "key": key,
                "path": blob.get("path"),
                "code": blob.get("code"),
                "rows": blob.get("rows"),
                "pages": (blob.get("doc") or {}).get("_pages") if isinstance(blob.get("doc"), dict) else None,
            })
        st.dataframe(pd.DataFrame(rows).sort_values(["code", "rows"], ascending=[True, False]), width="stretch")

        st.markdown("### Endpoint details")
        pick = st.selectbox("Inspect", options=sorted(list(data.keys())))
        b = data[pick]
        st.write(f"**{pick}** → `{b['path']}` (HTTP {b['code']}, rows={b['rows']})")
        st.write("Columns:")
        st.code(", ".join(b.get("cols", [])[:200]) or "—")

        if show_raw:
            st.markdown("Raw JSON (truncated)")
            st.json(b.get("doc"), expanded=False)

        st.markdown("### Debug log")
        debug_events = get_debug_events()
        if debug_events:
            st.code("\n".join(debug_events[-50:]))
        else:
            st.caption("No debug events captured in this session.")


if __name__ == "__main__":
    main()
