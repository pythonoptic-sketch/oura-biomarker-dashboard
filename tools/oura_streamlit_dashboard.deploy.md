# Oura Streamlit Dashboard Deployment

This dashboard is packaged for a stable web deployment via Render.

## Why this path

- stable public URL instead of a rotating `trycloudflare.com` tunnel
- free web-service deploy for initial testing
- Docker-based deploy matches the current local app with minimal changes
- Python 3.12 base image to stay on the stable wheel path for pandas/Streamlit dependencies

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

The service stores local state under `/data`:

- `accounts.json`
- `oauth_states.json`
- `device_sessions.json`
- `events.csv`
- `workout_intents.csv`

On Render's free plan this storage is ephemeral. Users, invites, saved sessions, and logs can reset on restart or redeploy.

For free initial testing this is acceptable.

For durable multi-user use, move these stores into a real database or switch to a paid persistent disk.

## Oura OAuth requirement

Oura requires the redirect URI in the app settings to exactly match the live public URL. Do not use a rotating quick-tunnel URL for production OAuth.
