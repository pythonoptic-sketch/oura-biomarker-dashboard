# Oura Streamlit Dashboard Deployment

This dashboard is packaged for a stable web deployment via Render.

## Why this path

- stable public URL instead of a rotating `trycloudflare.com` tunnel
- persistent disk for accounts, event logs, invite state, device sessions, and OAuth state
- Docker-based deploy matches the current local app with minimal changes

## Files

- Docker image: `tools/oura_streamlit_dashboard.Dockerfile`
- Render blueprint: `render.yaml`

## Render deployment

1. Push the repo to GitHub.
2. In Render, create a new Blueprint or Web Service from the repo.
3. Use the checked-in `render.yaml`.
4. Set the secret environment variables in Render:
   - `OURA_BOOTSTRAP_CODE`
   - `OURA_CLIENT_ID`
   - `OURA_CLIENT_SECRET`
   - `OURA_OAUTH_REDIRECT_URI`
5. After the first deploy, note the stable Render URL, for example:
   - `https://oura-biomarker-dashboard.onrender.com`
6. Set `OURA_OAUTH_REDIRECT_URI` to that exact stable URL.
7. In the Oura developer app, add that exact same URL as an allowed redirect URI.

## Persistence

The service expects a mounted disk at `/data` and stores:

- `accounts.json`
- `oauth_states.json`
- `device_sessions.json`
- `events.csv`
- `workout_intents.csv`

Without a persistent disk, users, invites, saved sessions, and logs will reset on restart.

## Oura OAuth requirement

Oura requires the redirect URI in the app settings to exactly match the live public URL. Do not use a rotating quick-tunnel URL for production OAuth.
