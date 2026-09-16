# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

try:
    from fastapi import FastAPI, Query
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:
    raise ImportError(
        "PyPTO Monitor requires fastapi and uvicorn. Install the serving runtime dependencies."
    ) from exc

from .collector import MetricsCollector
from .store import MonitorStore


def create_app(collector: MetricsCollector, store: MonitorStore) -> FastAPI:
    app = FastAPI(title="PyPTO Monitor", docs_url=None, redoc_url=None)
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

    def collector_done(task: asyncio.Task) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            collector.fail(error)
            logging.getLogger(__name__).error("Metrics collector stopped: %s", error)

    @app.on_event("startup")
    async def startup() -> None:
        app.state.collector_task = asyncio.create_task(collector.run())
        app.state.collector_task.add_done_callback(collector_done)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        collector.stop()
        task = getattr(app.state, "collector_task", None)
        try:
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        finally:
            store.close()

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/status")
    async def status() -> dict:
        return collector.status_snapshot()

    @app.get("/api/summary")
    async def summary(window: int = Query(default=300, ge=10, le=86400)) -> dict:
        data = await asyncio.to_thread(store.summary, window)
        data["collector"] = collector.status_snapshot()
        return data

    @app.get("/api/history")
    async def history(range: int = Query(default=3600, ge=60, le=86400)) -> dict:  # noqa: A002
        return {
            "range_seconds": range,
            "points": await asyncio.to_thread(store.history, range),
        }

    @app.get("/api/daily")
    async def daily(days: int = Query(default=30, ge=1, le=365)) -> dict:
        return {"days": await asyncio.to_thread(store.daily, days)}

    return app
