# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import asyncio
import sqlite3
import time

import pytest

from tools.monitor.app import create_app
from tools.monitor.collector import MetricsCollector
from tools.monitor.main import default_database
from tools.monitor.store import MonitorStore


def sample(model="A", target="http://a", count=3):
    return {
        "timestamp": time.time(), "model_name": model, "target": target,
        "gauges": {"running": 0, "waiting": 0, "kv_cache_usage": 0},
        "counter_deltas": {"requests_finished": count}, "histogram_deltas": {}, "elapsed": 1,
    }


@pytest.mark.parametrize("changed", [{"model": "B"}, {"target": "http://b"}])
def test_database_rejects_different_identity_across_restarts(tmp_path, changed):
    path = tmp_path / "monitor.db"
    store = MonitorStore(path)
    store.record(sample())
    store.close()
    store = MonitorStore(path)
    try:
        with pytest.raises(ValueError, match="different target or model"):
            store.record(sample(**changed))
        assert store.summary()["today"]["request_count"] == 3
        assert store.summary()["model_name"] == "A"
    finally:
        store.close()


def test_default_database_is_target_specific():
    assert default_database("http://a") == default_database("http://a/")
    assert default_database("http://a") != default_database("http://b")


def test_preview_database_is_rejected(tmp_path):
    path = tmp_path / "monitor.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE samples (timestamp REAL)")
    with pytest.raises(ValueError, match="preview database"):
        MonitorStore(path)


def test_collector_recovers_after_real_database_write_lock(tmp_path):
    async def check():
        path = tmp_path / "monitor.db"
        store = MonitorStore(path)
        store._connection.execute("PRAGMA busy_timeout=20")
        collector = MetricsCollector("http://a", store)
        count = 10
        collector.fetch = lambda: {
            "server_id": "server", "model_name": "A",
            "replicas": [{"counters": {"requests_finished": count}}],
        }
        blocker = sqlite3.connect(path)
        try:
            await collector.collect_once()
            assert collector.status.connected
            blocker.execute("BEGIN IMMEDIATE")
            count = 13
            await collector.collect_once()
            assert not collector.status.connected
            assert "locked" in collector.status.last_error
            assert collector._previous["counters"]["requests_finished"] == 10
            blocker.rollback()
            count = 15
            await collector.collect_once()
            assert collector.status.connected
            assert collector.status.last_error == ""
            assert store.summary()["today"]["request_count"] == 5
        finally:
            blocker.close()
            store.close()
    asyncio.run(check())


def test_failed_background_task_is_visible_without_database(tmp_path):
    async def check():
        store = MonitorStore(tmp_path / "monitor.db")
        collector = MetricsCollector("http://a", store)
        collector.status.connected = True
        async def fail():
            raise RuntimeError("unexpected failure")
        collector.collect_once = fail
        app = create_app(collector, store)
        async with app.router.lifespan_context(app):
            task = app.state.collector_task
            with pytest.raises(RuntimeError, match="unexpected failure"):
                await task
            endpoint = next(route.endpoint for route in app.routes if route.path == "/api/status")
            status = await endpoint()
            assert not status["connected"]
            assert "unexpected failure" in status["last_error"]
    asyncio.run(check())


def test_target_credentials_are_not_persisted(tmp_path):
    path = tmp_path / "monitor.db"
    store = MonitorStore(path)
    store.record(sample(target="http://user:secret@example.com"))
    identity = store._connection.execute("SELECT target FROM identity").fetchone()[0]
    assert "secret" not in identity
    assert len(identity) == 64
    store.close()
