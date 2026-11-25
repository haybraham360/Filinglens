import math
from pathlib import Path

import httpx
import pytest
import respx
from httpx import Response

from filinglens.ingestion.client import (
    EdgarClient,
    EdgarError,
    EdgarForbiddenError,
    EdgarNotFoundError,
)
from filinglens.ingestion.ratelimit import RateLimiter
from tests.edgar_fakes import IDENTITY, make_client

URL = "https://data.sec.gov/submissions/CIK0000320193.json"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.parametrize("rate", [8.0, 10.0, 2.5])
def test_rate_limiter_never_exceeds_rate_in_any_one_second_window(rate: float) -> None:
    clock = FakeClock()
    limiter = RateLimiter(rate, clock=clock.time, sleep=clock.sleep)
    stamps = []
    for _ in range(200):
        limiter.acquire()
        stamps.append(clock.now)

    # Spacing of 1/rate puts at most ceil(rate) requests in any half-open 1s window
    # (exactly `rate` for integer rates); 1e-9 absorbs float accumulation.
    for i, start in enumerate(stamps):
        in_window = sum(1 for t in stamps[i:] if t < start + 1.0 - 1e-9)
        assert in_window <= math.ceil(rate)
    assert stamps[-1] == pytest.approx(199 / rate)


def test_rate_limiter_does_not_sleep_when_requests_are_already_spaced() -> None:
    clock = FakeClock()
    sleeps: list[float] = []
    limiter = RateLimiter(8, clock=clock.time, sleep=sleeps.append)
    for _ in range(3):
        limiter.acquire()
        clock.now += 1.0
    assert sleeps == []


@respx.mock
def test_every_request_carries_declared_user_agent(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=Response(200, json={"ok": True}))
    with make_client(tmp_path) as client:
        client.get_json(URL)
        client.get_json(URL, refresh=True)
        client.download(URL, tmp_path / "out.json")
    assert route.call_count == 3
    for call in route.calls:
        assert call.request.headers["User-Agent"] == IDENTITY


@pytest.mark.parametrize("identity", ["", "   ", "FilingLens no-email"])
def test_identity_without_email_is_rejected(tmp_path: Path, identity: str) -> None:
    with pytest.raises(ValueError, match="EDGAR_IDENTITY"):
        EdgarClient(identity, tmp_path)


@respx.mock
def test_responses_are_cached_on_disk(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=Response(200, json={"n": 1}))
    with make_client(tmp_path) as client:
        assert client.get_json(URL) == {"n": 1}
    with make_client(tmp_path) as fresh:  # new process, same cache dir
        assert fresh.get_json(URL) == {"n": 1}
        assert fresh.stats.network_requests == 0
        assert fresh.stats.cache_hits == 1
    assert route.call_count == 1


@respx.mock
def test_404_is_cached_and_raised(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=Response(404))
    with make_client(tmp_path) as client:
        for _ in range(2):
            with pytest.raises(EdgarNotFoundError):
                client.get_json(URL)
    assert route.call_count == 1


@respx.mock
def test_retries_429_and_5xx_with_backoff_then_succeeds(tmp_path: Path) -> None:
    respx.get(URL).mock(
        side_effect=[
            Response(429, headers={"Retry-After": "3"}),
            Response(503),
            httpx.ConnectError("boom"),
            Response(200, json={"ok": True}),
        ]
    )
    sleeps: list[float] = []
    with make_client(tmp_path, sleep=sleeps.append, backoff_base=1.0) as client:
        assert client.get_json(URL) == {"ok": True}
        assert client.stats.network_requests == 4
    assert sleeps[0] == 3.0  # honours Retry-After
    assert 2.0 <= sleeps[1] < 3.0  # base * 2**1 + jitter
    assert 4.0 <= sleeps[2] < 5.0


@respx.mock
def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=Response(500))
    with make_client(tmp_path, max_retries=2) as client, pytest.raises(EdgarError, match="500"):
        client.get_json(URL)
    assert route.call_count == 3


@respx.mock
def test_403_is_not_retried(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=Response(403))
    with make_client(tmp_path) as client, pytest.raises(EdgarForbiddenError):
        client.get_json(URL)
    assert route.call_count == 1


@respx.mock
def test_download_skips_existing_file(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=Response(200, content=b"doc"))
    dest = tmp_path / "raw" / "doc.htm"
    with make_client(tmp_path) as client:
        assert client.download(URL, dest) is True
        assert client.download(URL, dest) is False
    assert dest.read_bytes() == b"doc"
    assert route.call_count == 1
    assert not list(dest.parent.glob("*.tmp"))
