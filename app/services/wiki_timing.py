"""Request-scoped backend timing for the PDF-to-Wiki pipeline."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger("uvicorn.error")


@dataclass(slots=True)
class _WikiTimingState:
    operation: str
    context: str
    durations: dict[str, float] = field(default_factory=dict)


_current_timing: ContextVar[_WikiTimingState | None] = ContextVar(
    "wiki_generate_timing", default=None
)


def _log_duration(state: _WikiTimingState, stage: str, duration: float) -> None:
    logger.info("%s %s %s=%.3fs", state.operation, state.context, stage, duration)


@contextmanager
def _request_timing(operation: str, context: str) -> Iterator[None]:
    """Collect pipeline timings and always log the total request duration."""

    state = _WikiTimingState(operation=operation, context=context)
    token = _current_timing.set(state)
    started_at = time.perf_counter()
    try:
        yield
    finally:
        total = time.perf_counter() - started_at
        for stage, duration in state.durations.items():
            _log_duration(state, stage, duration)
        _log_duration(state, "total", total)
        _current_timing.reset(token)


@contextmanager
def upload_request_timing(filename: str) -> Iterator[None]:
    """Time PDF upload and extraction in the upload request only."""

    with _request_timing("wiki.upload", f"filename={filename}"):
        yield


@contextmanager
def wiki_request_timing(document_id: int) -> Iterator[None]:
    """Time focused generation using an already-persisted Document."""

    with _request_timing("wiki.generate", f"document_id={document_id}"):
        yield


@contextmanager
def time_wiki_stage(stage: str) -> Iterator[None]:
    """Measure one stage only while handling a Wiki generation request."""

    state = _current_timing.get()
    if state is None:
        yield
        return

    started_at = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started_at
        state.durations[stage] = state.durations.get(stage, 0.0) + elapsed
