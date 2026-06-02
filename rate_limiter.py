"""Proactive, self-pacing rate limiter for OpenAI calls.

The point is to *parcel requests out* at a rate that never exceeds the model's
per-minute limit, rather than dumping a whole minute's budget at once. OpenAI's
limits are per-minute (rolling 60s), but firing ~1,500 requests in one instant
saturates their edge and returns HTTP 429 ``insufficient_quota`` even with a
healthy credit balance. A token bucket refilling at ``limit / 60`` per second,
with a burst capacity of roughly one second of budget, means at most ~``RPM/60``
requests can ever be admitted in a single second (e.g. ~500/s for a 30,000 RPM
model) — the spike simply cannot happen.

Two buckets are enforced together: requests/min (RPM) and tokens/min (TPM). A
call is admitted only when both allow it.

The real, account-specific limits come from OpenAI's ``x-ratelimit-limit-*``
response headers (``configure_from_headers``); a seed from settings.TIER5_FALLBACK
is used only for the very first request, before any header has been seen.
``reconcile`` additionally backs off hard if the server reports the per-minute
budget is exhausted (e.g. other traffic on the org), honoring the reset headers.

NOTE: this limiter is per-process. That is correct today because the toad worker
runs jobs serially and each GPA MCP call is a fresh subprocess, so only one run's
traffic exists at a time. If GPA ever runs multiple concurrent runs in one process
(or multiple worker processes against one org), a shared/cross-process limiter
would be needed; the SDK's own retry/backoff remains a backstop regardless.
"""

import asyncio
import re
import time

from unified_logger import LogLevel, get_logger

# OpenAI reset headers look like "2ms", "0s", "1s", "6m0s", "1h2m3s".
_RESET_RE = re.compile(
    r"(?:(?P<h>\d+)h)?(?:(?P<m>\d+)m(?!s))?(?:(?P<s>\d+)s)?(?:(?P<ms>\d+)ms)?"
)


def _parse_reset_seconds(value):
    """Parse an OpenAI ratelimit reset header (e.g. '6m0s', '250ms') to seconds.

    Returns 0.0 on anything unrecognized so callers degrade to "retry now"."""
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    m = _RESET_RE.fullmatch(s)
    if not m:
        return 0.0
    h = int(m.group("h") or 0)
    mins = int(m.group("m") or 0)
    secs = int(m.group("s") or 0)
    ms = int(m.group("ms") or 0)
    return h * 3600 + mins * 60 + secs + ms / 1000.0


def _header_int(headers, name):
    """Read an int header from a mapping or an httpx.Headers-like object."""
    if headers is None:
        return None
    try:
        raw = headers.get(name)
    except AttributeError:
        raw = None
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


class AsyncRateLimiter:
    """Dual (RPM + TPM) token bucket that paces dispatch to ``limit / 60`` per sec.

    Args:
        rpm: requests-per-minute limit (<=0 disables the request bucket).
        tpm: tokens-per-minute limit (<=0 disables the token bucket).
        utilization: fraction of the published limit to actually target.
        burst_seconds: bucket capacity expressed in seconds of budget.
        logger: optional unified logger; defaults to the shared one.
        source: where the limits came from ("fallback" or "headers"), for logging.
    """

    def __init__(self, rpm, tpm, *, utilization=0.9, burst_seconds=1.0,
                 logger=None, source="fallback"):
        self._lock = asyncio.Lock()
        self._log = logger or get_logger()
        self._util = max(0.0, float(utilization))
        self._burst = max(0.001, float(burst_seconds))
        self.source = source

        # Per-bucket state, set by _set_limits.
        self._req_rate = 0.0   # tokens/sec refilled
        self._req_cap = 0.0    # bucket capacity
        self._req_avail = 0.0
        self._tok_rate = 0.0
        self._tok_cap = 0.0
        self._tok_avail = 0.0
        # Hard backoff windows derived from server "remaining<=0" + reset headers.
        self._req_blocked_until = 0.0
        self._tok_blocked_until = 0.0

        self._now = time.monotonic
        self._last = self._now()
        self._oversized_warned = False

        # Stats surfaced in the run summary.
        self.throttle_count = 0
        self.throttle_wait_seconds = 0.0

        # Start the buckets EMPTY (fill=False) rather than full. A full bucket would
        # let the initial burst dump cap tokens at t=0 on top of the first second's
        # refill (~2x rate in the opening second). Empty means the opening second is
        # capped at the refill rate itself (~RPM/60), i.e. "no more than ~500 in the
        # first second" for a 30k-RPM model. The cost is a sub-second wait before the
        # very first request, which is negligible at real limits.
        self._set_limits(rpm, tpm, fill=False)

    # ---- configuration -------------------------------------------------

    def _set_limits(self, rpm, tpm, *, fill=False):
        """(Re)compute bucket rate/capacity from per-minute limits.

        rate = limit * utilization / 60 (per second); capacity = rate * burst.
        A non-positive limit disables that bucket (treated as unlimited).
        """
        if rpm and rpm > 0:
            self._req_rate = rpm * self._util / 60.0
            self._req_cap = max(1.0, self._req_rate * self._burst)
        else:
            self._req_rate = 0.0
            self._req_cap = 0.0
        if tpm and tpm > 0:
            self._tok_rate = tpm * self._util / 60.0
            self._tok_cap = max(1.0, self._tok_rate * self._burst)
        else:
            self._tok_rate = 0.0
            self._tok_cap = 0.0
        if fill:
            self._req_avail = self._req_cap
            self._tok_avail = self._tok_cap
        else:
            # Never hand out more than the new capacity allows.
            self._req_avail = min(self._req_avail, self._req_cap)
            self._tok_avail = min(self._tok_avail, self._tok_cap)
        self.rpm = rpm
        self.tpm = tpm

    def configure_from_headers(self, headers):
        """Set the limits from ``x-ratelimit-limit-{requests,tokens}`` headers.

        Called once from the cache-warm response (before the big fan-out) and
        cheaply on later responses; limits rarely change mid-run. No-op if the
        headers don't carry the limit fields."""
        rpm = _header_int(headers, "x-ratelimit-limit-requests")
        tpm = _header_int(headers, "x-ratelimit-limit-tokens")
        if rpm is None and tpm is None:
            return
        new_rpm = rpm if rpm is not None else self.rpm
        new_tpm = tpm if tpm is not None else self.tpm
        changed = (new_rpm != self.rpm) or (new_tpm != self.tpm) or self.source != "headers"
        self._set_limits(new_rpm, new_tpm, fill=False)
        if changed:
            self.source = "headers"
            self._log.log(
                LogLevel.INFO,
                f"Rate limiter configured from headers: {new_rpm} RPM / {new_tpm} TPM "
                f"(targeting {self._util:.0%} -> {self._req_rate:.1f} req/s, "
                f"burst {self._req_cap:.0f}).",
                source_file="rate_limiter.py")

    def reconcile(self, headers):
        """Refresh limits and hard-back-off if the server says budget is exhausted.

        When ``x-ratelimit-remaining-*`` hits 0 we pause new admissions until the
        matching ``x-ratelimit-reset-*`` window elapses — this catches the genuine
        per-minute-cap case (including budget consumed by other org traffic) that
        the local rate model wouldn't otherwise see."""
        self.configure_from_headers(headers)
        rem_req = _header_int(headers, "x-ratelimit-remaining-requests")
        rem_tok = _header_int(headers, "x-ratelimit-remaining-tokens")
        now = self._now()
        if rem_req is not None and rem_req <= 0:
            wait = _parse_reset_seconds(
                headers.get("x-ratelimit-reset-requests") if headers else None)
            self._req_blocked_until = max(self._req_blocked_until, now + wait)
        if rem_tok is not None and rem_tok <= 0:
            wait = _parse_reset_seconds(
                headers.get("x-ratelimit-reset-tokens") if headers else None)
            self._tok_blocked_until = max(self._tok_blocked_until, now + wait)

    # ---- admission -----------------------------------------------------

    def _refill(self, now):
        elapsed = max(0.0, now - self._last)
        self._last = now
        if self._req_rate > 0:
            self._req_avail = min(self._req_cap, self._req_avail + elapsed * self._req_rate)
        if self._tok_rate > 0:
            self._tok_avail = min(self._tok_cap, self._tok_avail + elapsed * self._tok_rate)

    async def acquire(self, estimated_tokens):
        """Block until both buckets admit one request of ~``estimated_tokens``.

        Deducts the cost on admission. Bookkeeping happens under the lock; the
        wait sleeps lock-free so callers don't serialize behind each other."""
        est = max(0, int(estimated_tokens or 0))
        while True:
            async with self._lock:
                now = self._now()
                self._refill(now)

                waits = []

                # Hard server-imposed backoff windows take precedence.
                if now < self._req_blocked_until:
                    waits.append(self._req_blocked_until - now)
                if now < self._tok_blocked_until:
                    waits.append(self._tok_blocked_until - now)

                # Request bucket (skipped when disabled).
                req_ok = True
                if self._req_rate > 0:
                    if self._req_avail >= 1.0:
                        req_ok = True
                    else:
                        req_ok = False
                        waits.append((1.0 - self._req_avail) / self._req_rate)

                # Token bucket (skipped when disabled).
                tok_ok = True
                tok_cost = 0.0
                if self._tok_rate > 0 and est > 0:
                    # A single request larger than the whole bucket can never fully
                    # fit; clamp its cost to capacity so it drains the bucket (and
                    # paces the next call) instead of deadlocking. Practically
                    # impossible at real limits (cap ~ millions of tokens), but it
                    # matters under tiny test limits.
                    tok_cost = min(float(est), self._tok_cap)
                    if est > self._tok_cap and not self._oversized_warned:
                        self._oversized_warned = True
                        self._log.log(
                            LogLevel.WARNING,
                            f"A request (~{est} tokens) exceeds the TPM burst "
                            f"capacity ({self._tok_cap:.0f}); admitting it alone "
                            f"after the bucket drains.",
                            source_file="rate_limiter.py")
                    if self._tok_avail >= tok_cost:
                        tok_ok = True
                    else:
                        tok_ok = False
                        waits.append((tok_cost - self._tok_avail) / self._tok_rate)

                if req_ok and tok_ok and not waits:
                    if self._req_rate > 0:
                        self._req_avail -= 1.0
                    if tok_cost:
                        self._tok_avail -= tok_cost
                    return

                wait = max(waits) if waits else 0.0

            # Sleep outside the lock so other coroutines can re-check meanwhile.
            wait = max(wait, 0.001)
            self.throttle_count += 1
            self.throttle_wait_seconds += wait
            await asyncio.sleep(wait)

    def stats(self):
        """Throttle stats for the run summary."""
        return {
            "limiter_source": self.source,
            "rpm": self.rpm,
            "tpm": self.tpm,
            "throttle_count": self.throttle_count,
            "throttle_wait_seconds": round(self.throttle_wait_seconds, 2),
        }
