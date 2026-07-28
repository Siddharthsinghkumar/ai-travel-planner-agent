"""Runtime loop ownership for background price tracking."""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import FastAPI

from core.env_config import get_env_int


async def run_price_tracker_loop(app: FastAPI, *, logger) -> None:
    from tools import price_tracker

    interval = max(60, get_env_int("PRICE_TRACKER_INTERVAL_SECONDS", 1800))
    try:
        cleanup_summary = await asyncio.to_thread(price_tracker.cleanup_invalid_held_tracking_rows)
        app.state.price_tracker_status["startup_cleanup"] = cleanup_summary
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        app.state.price_tracker_status["startup_cleanup"] = {
            "status": "failed",
            "error": str(exc),
        }
        logger.warning(
            "price_tracker_startup_cleanup_failed",
            extra={"exception_type": type(exc).__name__},
        )
    while getattr(app.state, "price_tracker_enabled", False):
        try:
            app.state.price_tracker_status["last_started_at"] = datetime.utcnow().isoformat() + "Z"
            alerts = await price_tracker.check_held_booking_prices()
            app.state.price_tracker_status["last_alert_count"] = len(alerts)
            app.state.price_tracker_status["last_error"] = None
            # Session memory hygiene: prune expired sessions on tracker cadence
            try:
                from agents.planner_agent import _session_memory
                removed = await asyncio.to_thread(_session_memory.cleanup_expired)
                if removed:
                    logger.debug("session_memory_cleanup", extra={"removed_sessions": removed})
            except Exception:
                pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            app.state.price_tracker_status["last_error"] = str(exc)
            logger.exception("price_tracker_loop_error")
        finally:
            app.state.price_tracker_status["last_completed_at"] = datetime.utcnow().isoformat() + "Z"
        await asyncio.sleep(interval)

