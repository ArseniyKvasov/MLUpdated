# AI request proxy (Cloudflare Worker)

Routes the ML service's Groq and Gemini API calls through Cloudflare instead
of hitting the providers directly. The real provider keys live only as
Worker secrets; the ML service only holds a shared secret to authenticate to
the Worker (see `CLOUDFLARE_AI_WORKER_URL` / `CLOUDFLARE_AI_WORKER_SECRET` in
`.env.example`).

## Deploy

```bash
cd cloudflare
npx wrangler deploy
```

This publishes to `https://ai-proxy.<your-subdomain>.workers.dev` (rename the
`name` in `wrangler.toml` if you want a different subdomain, or attach a
custom domain/route in the Cloudflare dashboard).

## Secrets

Set these on the Worker (never in the ML service's `.env`):

```bash
npx wrangler secret put WORKER_SHARED_SECRET
npx wrangler secret put GROQ_API_KEY
npx wrangler secret put GEMINI_API_KEY
```

- `WORKER_SHARED_SECRET` - any random string you generate yourself (e.g.
  `openssl rand -hex 32`). Must match `CLOUDFLARE_AI_WORKER_SECRET` in the ML
  service's `.env`.
- `GROQ_API_KEY` - your real Groq API key.
- `GEMINI_API_KEY` - your real Gemini API key.

## ML service configuration

In the ML service's `.env`:

```env
CLOUDFLARE_AI_WORKER_URL=https://ai-proxy.your-subdomain.workers.dev
CLOUDFLARE_AI_WORKER_SECRET=<same value as WORKER_SHARED_SECRET>
```

`GROQ_API_KEY` / `GEMINI_API_KEY` in the ML service's own `.env` become
optional once these two are set.
