# Deployment

Frontend on **Vercel**, backend on **Railway**, study assets bundled with the
frontend so Phase 1 + Phase 2 work even if the backend is asleep or down.

```
┌──────────────────────────────┐         ┌────────────────────────────────────┐
│ Vercel (frontend)            │         │ Railway (backend)                  │
│  - Vite SPA                  │  HTTPS  │  - FastAPI + EfficientNet-B4       │
│  - bundled study-analyses    │  ────►  │  - LayerCAM, BiSeNet, forensics    │
│    .json + quiz-heatmaps/    │         │  - OpenAI VLM                      │
│  - reads VITE_API_BASE_URL   │         │  - reads CORS_ORIGINS              │
└──────────────────────────────┘         └────────────────────────────────────┘
                                                  ▲
                                                  │ Supabase (anon key from
                                                  │ frontend writes results)
```

The frontend never blocks on the backend during the study. Phase 3 (the
post-study product trial) is the only path that needs the backend reachable.

ML weights (EfficientNet-B4, BiSeNet, MediaPipe Face Landmarker) are
downloaded at startup, not baked into the image — keeps the Docker image
small but adds ~30–60 s to the first cold boot.

---

## First-time setup

### Backend → Railway

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy
   from GitHub repo** → select the XADE repo.
2. **Service settings → Root directory:** `backend`
   Railway picks up [`backend/railway.json`](../backend/railway.json) and
   builds [`backend/Dockerfile`](../backend/Dockerfile) with `backend/` as
   the build context.
3. **Variables** (Railway secret manager — never commit any of these):

   | Key | Source |
   |---|---|
   | `OPENAI_API_KEY` | OpenAI dashboard |
   | `SUPABASE_URL` | Supabase project settings |
   | `SUPABASE_KEY` | Supabase Settings → API → **service_role** key (not anon) |
   | `CORS_ORIGINS` | comma-separated, see below |
   | `HF_TOKEN` | optional, only if EfficientNet-B4 weights become gated |
   | `ANTHROPIC_API_KEY` | optional — currently unused by `/study` |
   | `GOOGLE_GEMINI_API_KEY` | optional — currently unused by `/study` |

   `CORS_ORIGINS` should at minimum cover the production Vercel URL. The
   regex `https://xade.*\.vercel\.app` in [`app/main.py`](../backend/app/main.py)
   already covers `*.vercel.app` preview URLs, so add custom domains here:

   ```
   https://xade.vercel.app,https://study.xade.example.com
   ```

4. Railway auto-injects `PORT`. The Dockerfile already binds to
   `0.0.0.0:$PORT`. The healthcheck path `/health` and `ON_FAILURE` restart
   policy come from [`backend/railway.json`](../backend/railway.json).
5. Deploy. First build takes ~3–5 min (CPU torch wheel is large).
6. Health check: `https://<railway-url>/health` should return 200 with
   `database: healthy`, `detection_model: loaded`, `vlm_service:
   initialized`.
7. **Record the Railway URL** — you need it for the Vercel
   `VITE_API_BASE_URL` variable.

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project** → import the
   same GitHub repo.
2. **Build & Output Settings** are picked up from
   [`vercel.json`](../vercel.json):
   - Build: `npm --prefix desktop run build:web`
   - Output: `desktop/dist`
   - SPA rewrites: every path → `/index.html`
3. **Environment Variables** (Vercel project → Settings → Environment
   Variables — these are inlined at build time, so changes need a redeploy):

   | Key | Value |
   |---|---|
   | `VITE_SUPABASE_URL` | `https://<project>.supabase.co` |
   | `VITE_SUPABASE_ANON_KEY` | Supabase anon key |
   | `VITE_API_BASE_URL` | the Railway URL from Backend step 7 |

   See [`desktop/.env.example`](../desktop/.env.example) for the same
   list with example values.
4. Deploy. PRs against `main` get preview URLs automatically; production
   deploys from `main`.
5. Smoke-test the deployed URL: Phase 1 should run with no backend
   reachable (the bundled `study-analyses.json` and `quiz-heatmaps/` cover
   Phase 2). Submitting Phase 3 results should hit the Railway backend.

### Closing the loop

After both are live, copy the production Vercel URL back into Railway's
`CORS_ORIGINS` so Phase 3 (uploading to the live backend) works without
CORS errors. Preview URLs are already covered by the regex in
[`app/main.py`](../backend/app/main.py).

---

## Verifying a deploy

```bash
# Should return 200 within 2 s on a warm container
curl https://<your-railway-url>/health
```

Expected response:

```json
{
  "status": "healthy",
  "database": "healthy",
  "detection_model": "loaded",
  "vlm_service": "initialized"
}
```

---

## Redeploy

| Trigger | What happens |
|---|---|
| Push to `main` | Both Vercel (production) and Railway (production) auto-deploy. |
| Open a PR against `main` | Vercel builds a preview URL. Railway does not preview by default. |
| Bumped study assets (regenerated `study-analyses.json` or `quiz-heatmaps/*`) | Commit them to git and push. Vercel rebuilds and serves the new bundle. **Don't** rely on `/api/v1/study/precompute` from Vercel — Vercel can't write back to the repo. Run `/precompute` against your local backend, commit the output, push. |
| Backend code change only | Push to `main`. Railway redeploys. |
| Manual redeploy (after rotating a secret) | Railway dashboard → **Deployments** → **Redeploy**, or Vercel dashboard → **Deployments** → **⋯ Redeploy**. |

### Re-running `/study/precompute`

```bash
# Run locally — needs OPENAI_API_KEY in backend/.env
cd backend
uvicorn app.main:app --reload &
curl -X POST http://localhost:8000/api/v1/study/precompute --max-time 1800
```

The endpoint writes
[`desktop/public/study-analyses.json`](../desktop/public/study-analyses.json)
and the per-image assets under
[`desktop/public/quiz-heatmaps/`](../desktop/public/quiz-heatmaps/).
Commit and push to trigger a Vercel redeploy.

---

## Secret rotation

1. **OpenAI key compromised:** rotate in the OpenAI dashboard, paste the
   new key into Railway's Variables panel, redeploy.
2. **Supabase service role key:** rotate in Supabase → Settings → API,
   update Railway, redeploy. The frontend uses the anon key only — it
   doesn't need rotating unless you also rotate that one (in which case
   update Vercel and redeploy too).
3. **Anything committed to git by accident:** rotate the key first
   (immediate effect), then optionally purge from history with
   `git filter-repo`.

---

## Cost notes

- **Vercel:** Hobby tier is free for the study window. ~40 MB of bundled
  study assets is well inside the limit.
- **Railway:** ~$5–10/month for shared CPU + 2 GB. The Dockerfile uses
  the CPU-only torch wheel. Shut the service down once participant
  collection ends to stop the meter.
- **OpenAI:** each `/api/v1/study/precompute` run is 12 images × one
  explanation each, dominated by image tokens — budget ~$0.50 per full
  precompute. Don't precompute on every push; use the cached
  `study-analyses.json` and only re-run when the study image set or the
  prompt changes.

---

## Switching backend to Hugging Face Spaces (fallback)

If Railway cost is a problem after the study:

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
   → **Docker** SDK.
2. Push the `backend/` directory as the Space repo.
3. Set the same environment variables in the Space **Settings →
   Repository secrets**.
4. Update `VITE_API_BASE_URL` in Vercel to the new Space URL and redeploy.

Note: HF Spaces free tier has a ~30 s cold start after idle. For the
thesis defence demo this is acceptable; for an active study window
consider the persistent paid tier (~$9/mo).

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Phase 2 shows blank Facial Regions / "Unavailable" ELA tile | `study-analyses.json` is from before the ranker / ELA wiring. Re-run `/precompute` locally, commit, push. |
| Phase 3 upload returns CORS error | Production Vercel URL is not in Railway's `CORS_ORIGINS` and doesn't match the `xade.*\.vercel\.app` regex. Add the URL. |
| Railway health check times out | Cold start is exceeding 120 s (the value in `backend/railway.json`). Bump it, or bake the EfficientNet-B4 weights into the image at build time instead of lazy-loading from HF Hub. |
| OpenAI rate limit during precompute | Already retried with backoff in [`study.py`](../backend/app/routers/study.py). If it still fails, lower precompute concurrency or wait a minute. |
| Vercel build OOM | Unlikely with this bundle, but if it happens, drop `study-analyses.json` from the bundle and load it on demand from a CDN. |
| Frontend hits `localhost:8000` in production | `VITE_API_BASE_URL` not set in Vercel, or set on the wrong environment (Production vs Preview). Vite needs the var set on the environment that built the deploy. |
