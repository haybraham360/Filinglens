"""Concurrent /ask load test (specs/06 acceptance: 20 concurrent requests, no errors).

    python -m loadtest --url http://localhost:8000 --concurrency 20 --requests 60

Sends distinct questions (so the answer cache doesn't hide the work) from N concurrent
clients and reports status codes and latency percentiles. The API's per-IP rate limit
applies to the load test too: raise API_RATE_LIMIT_PER_MINUTE on the target, or the run
measures the limiter (429s) rather than the service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter

import httpx

TEMPLATES = [
    "What does {t} say about supply chain risk? (load test {i})",
    "What legal proceedings does {t} describe? (load test {i})",
    "What are {t}'s main risk factors? (load test {i})",
    "How does {t} describe competition? (load test {i})",
]


async def run(
    url: str, concurrency: int, total: int, tickers: list[str], mode: str
) -> dict[str, object]:
    """Fire ``total`` requests with ``concurrency`` in flight; return a summary."""
    sem = asyncio.Semaphore(concurrency)
    statuses: Counter[str] = Counter()
    latencies: list[float] = []
    errors: list[str] = []

    async with httpx.AsyncClient(base_url=url, timeout=120) as client:

        async def one(i: int) -> None:
            q = TEMPLATES[i % len(TEMPLATES)].format(t=tickers[i % len(tickers)], i=i)
            async with sem:
                t0 = time.perf_counter()
                try:
                    resp = await client.post("/ask", json={"question": q, "mode": mode})
                    statuses[str(resp.status_code)] += 1
                    if resp.status_code != 200:
                        errors.append(f"{resp.status_code}: {resp.text[:120]}")
                    else:
                        latencies.append(time.perf_counter() - t0)
                except httpx.HTTPError as exc:
                    statuses["transport_error"] += 1
                    errors.append(repr(exc))

        start = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(total)))
        wall = time.perf_counter() - start

    lat = sorted(latencies)
    pct = lambda q: lat[min(len(lat) - 1, max(0, round(q / 100 * len(lat)) - 1))] if lat else None  # noqa: E731
    return {
        "requests": total,
        "concurrency": concurrency,
        "statuses": dict(statuses),
        "ok": statuses.get("200", 0) == total,
        "wall_s": round(wall, 2),
        "throughput_rps": round(total / wall, 2),
        "latency_s": {
            "p50": pct(50),
            "p95": pct(95),
            "max": lat[-1] if lat else None,
            "mean": statistics.fmean(lat) if lat else None,
        },
        "first_errors": errors[:5],
    }


def main() -> int:
    """CLI entry point; exit code 1 if any request failed."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--requests", type=int, default=60)
    p.add_argument("--tickers", default="AAPL,MSFT,JPM")
    p.add_argument("--mode", default="agent", choices=["agent", "rag"])
    args = p.parse_args()
    summary = asyncio.run(
        run(args.url, args.concurrency, args.requests, args.tickers.split(","), args.mode)
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
