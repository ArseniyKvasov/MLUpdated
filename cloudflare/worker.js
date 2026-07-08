/**
 * AI request proxy for the ML service.
 *
 * The ML service (GroqService in app/groq_service.py) sends its Groq and
 * Gemini traffic here instead of directly to api.groq.com /
 * generativelanguage.googleapis.com. This Worker holds the real provider
 * API keys as secrets and injects them before forwarding, so the ML
 * service's own environment never needs to know them - only the shared
 * secret below.
 *
 * Routes:
 *   /groq/*   -> https://api.groq.com/openai/v1/*
 *   /google/* -> https://generativelanguage.googleapis.com/*
 *
 * Required secrets (wrangler secret put <NAME>):
 *   WORKER_SHARED_SECRET  - must match CLOUDFLARE_AI_WORKER_SECRET in the ML service .env
 *   GROQ_API_KEY           - real Groq API key
 *   GEMINI_API_KEY         - real Gemini API key
 */

const GROQ_UPSTREAM = "https://api.groq.com/openai/v1";
const GOOGLE_UPSTREAM = "https://generativelanguage.googleapis.com";

export default {
  async fetch(request, env) {
    const suppliedSecret = request.headers.get("X-Worker-Secret");
    if (!env.WORKER_SHARED_SECRET || suppliedSecret !== env.WORKER_SHARED_SECRET) {
      return new Response("Unauthorized", { status: 401 });
    }

    const url = new URL(request.url);

    let upstreamBase;
    let prefix;
    if (url.pathname.startsWith("/groq/")) {
      upstreamBase = GROQ_UPSTREAM;
      prefix = "/groq";
    } else if (url.pathname.startsWith("/google/")) {
      upstreamBase = GOOGLE_UPSTREAM;
      prefix = "/google";
    } else {
      return new Response("Not found", { status: 404 });
    }

    const upstreamUrl = upstreamBase + url.pathname.slice(prefix.length) + url.search;

    const headers = new Headers(request.headers);
    headers.delete("X-Worker-Secret");
    headers.delete("host");

    if (prefix === "/groq") {
      if (!env.GROQ_API_KEY) {
        return new Response("Worker missing GROQ_API_KEY secret", { status: 500 });
      }
      headers.set("Authorization", `Bearer ${env.GROQ_API_KEY}`);
    } else {
      if (!env.GEMINI_API_KEY) {
        return new Response("Worker missing GEMINI_API_KEY secret", { status: 500 });
      }
      headers.set("x-goog-api-key", env.GEMINI_API_KEY);
    }

    const upstreamRequest = new Request(upstreamUrl, {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      // Required by the Workers runtime when streaming a request body through.
      duplex: ["GET", "HEAD"].includes(request.method) ? undefined : "half",
    });

    const upstreamResponse = await fetch(upstreamRequest);

    // Stream the response straight back; strip hop-by-hop headers that don't
    // make sense to relay.
    const responseHeaders = new Headers(upstreamResponse.headers);
    responseHeaders.delete("content-encoding");
    responseHeaders.delete("transfer-encoding");

    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: responseHeaders,
    });
  },
};
