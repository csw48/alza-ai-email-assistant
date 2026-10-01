import threading
import time

import pytest

from app.idempotency import IdempotencyCache


def test_concurrent_duplicate_waits_for_first_result():
    cache = IdempotencyCache()
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.3)
        return "odpoveď"

    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.run("<m1>", slow))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1
    assert sorted(dup for _, dup in results) == [False, True]
    assert all(r == "odpoveď" for r, _ in results)


def test_waiter_retries_when_first_attempt_fails():
    cache = IdempotencyCache()
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            time.sleep(0.2)
            raise RuntimeError("Gemini down")
        return "ok"

    errors, results = [], []

    def call():
        try:
            results.append(cache.run("<m2>", flaky))
        except RuntimeError as e:
            errors.append(e)

    t1 = threading.Thread(target=call)
    t1.start()
    time.sleep(0.05)
    t2 = threading.Thread(target=call)
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 1
    assert results == [("ok", False)]


def test_expired_entries_are_evicted():
    cache = IdempotencyCache(ttl_seconds=0)
    cache.run("<old>", lambda: "a")
    time.sleep(0.01)
    result, duplicate = cache.run("<old>", lambda: "b")
    assert (result, duplicate) == ("b", False)


def test_max_items_bounds_memory():
    cache = IdempotencyCache(max_items=2)
    for i in range(5):
        cache.run(f"<m{i}>", lambda: "x")
    assert len(cache._items) <= 3
