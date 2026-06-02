"""Global runtime knobs for GPA, with environment-variable overrides.

These are process-wide scalars (concurrency ceiling, OpenAI client retry/timeout,
rate-limiter tuning) that don't fit the per-row CSV config convention used in
Configuration_Files/. Each constant can be overridden per deployment via an env
var without editing tracked files — which suits the way the toad worker spawns a
fresh GPA subprocess (with its own env) per MCP call.

Per-model rate limits themselves are NOT hardcoded here: the rate limiter reads
them live from OpenAI's ``x-ratelimit-limit-*`` response headers. TIER5_FALLBACK
below is only a seed used for the very first request of a run, before any header
has been seen, and as a safety net for models the headers don't report.
"""

import os


def _int(name, default):
    """Env override -> int, falling back to default on missing/garbage values."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    """Env override -> float, falling back to default on missing/garbage values."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# Hard ceiling on concurrent in-flight requests, applied on top of whatever
# max_parallel_requests a caller passes — and the size of the HTTP connection
# pool. On a single machine the binding constraint is local sockets/DNS, not
# OpenAI's per-minute limits: a few hundred simultaneous connections overwhelm the
# resolver and cause APIConnectionError (getaddrinfo) failures. 50 is verified to
# run cleanly here; raise via GPA_MAX_CONCURRENCY only if the host tolerates more.
MAX_CONCURRENCY_CEILING = _int("GPA_MAX_CONCURRENCY", 50)

# openai-python client retry/timeout. The SDK retries 408/409/429/5xx with
# exponential backoff + jitter and honors Retry-After — the final net for any
# residual 429 the proactive limiter doesn't prevent.
OPENAI_MAX_RETRIES = _int("GPA_OPENAI_MAX_RETRIES", 5)
OPENAI_TIMEOUT_SECONDS = _float("GPA_OPENAI_TIMEOUT", 120.0)

# Fraction of the model's published RPM/TPM the limiter actually targets, leaving
# headroom for token-estimate error, clock skew, and other org traffic.
RATE_LIMIT_TARGET_UTILIZATION = _float("GPA_RATE_LIMIT_UTILIZATION", 0.9)

# Token-bucket burst capacity expressed in seconds of budget. 1.0 means the bucket
# holds ~one second's worth of the per-minute limit, so at most ~RPM/60 requests
# can be admitted in any single second (e.g. ~500/s for a 30,000 RPM model) — this
# is the "parcel them out so we never spike" control.
BURST_WINDOW_SECONDS = _float("GPA_BURST_WINDOW_SECONDS", 1.0)

# Seed limits (requests/min, tokens/min) used before the first response header is
# seen, and for models the headers don't report. Keyed by model name; "default"
# applies to anything not listed. Values reflect this account's Tier-5 limits but
# are deliberately on the safe side — the live headers override them within the
# first request of each run.
TIER5_FALLBACK = {
    "gpt-5.5": (15_000, 40_000_000),
    "gpt-5-mini": (30_000, 180_000_000),
    "gpt-4o-mini": (30_000, 150_000_000),
    "gpt-4o": (10_000, 30_000_000),
    # Conservative catch-all for any unlisted model until headers refine it.
    "default": (10_000, 10_000_000),
}


def fallback_limits(model):
    """(rpm, tpm) seed for ``model`` from TIER5_FALLBACK, or the generic default."""
    return TIER5_FALLBACK.get(model, TIER5_FALLBACK["default"])
