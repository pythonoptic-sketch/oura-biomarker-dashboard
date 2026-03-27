FROM python:3.12-slim

WORKDIR /app

COPY tools/oura_streamlit_dashboard.requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY tools/oura_streamlit_dashboard.py.gz.b64.part00 /tmp/oura_streamlit_dashboard.py.gz.b64.part00
COPY tools/oura_streamlit_dashboard.py.gz.b64.part01 /tmp/oura_streamlit_dashboard.py.gz.b64.part01
COPY tools/oura_streamlit_dashboard.py.gz.b64.part02 /tmp/oura_streamlit_dashboard.py.gz.b64.part02
COPY tools/oura_streamlit_dashboard.py.gz.b64.part03 /tmp/oura_streamlit_dashboard.py.gz.b64.part03

RUN python - <<'PY'
import base64
import gzip
from pathlib import Path

encoded = b''.join(
    Path(f'/tmp/oura_streamlit_dashboard.py.gz.b64.part0{i}').read_bytes()
    for i in range(4)
)
decoded = gzip.decompress(base64.b64decode(encoded))
Path('/app/oura_streamlit_dashboard.py').write_bytes(decoded)
PY

RUN mkdir -p /data

ENV OURA_ACCOUNT_STORE_PATH=/data/accounts.json
ENV OURA_OAUTH_STATE_STORE_PATH=/data/oauth_states.json
ENV OURA_DEVICE_SESSION_STORE_PATH=/data/device_sessions.json
ENV OURA_BEHAVIOR_EVENT_PATH=/data/events.csv
ENV OURA_WORKOUT_INTENT_PATH=/data/workout_intents.csv
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1

VOLUME ["/data"]
EXPOSE 8501

CMD ["/bin/sh", "-c", "streamlit run /app/oura_streamlit_dashboard.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
