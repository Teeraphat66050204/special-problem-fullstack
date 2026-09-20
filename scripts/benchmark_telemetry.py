"""Best-effort host telemetry for Wiki benchmark generation runs."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - exercised by installations without the optional runtime
    psutil = None  # type: ignore[assignment]

MEBIBYTE = 1024 * 1024
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

GpuReader = Callable[[], tuple[float | None, float | None]]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        match = _NUMBER.search(value.replace(",", ""))
        if not match:
            return None
        number = float(match.group())
    else:
        return None
    return number if math.isfinite(number) else None


def summarize_samples(
    samples: Sequence[Mapping[str, float | None]],
    *,
    gpu_provider: str | None = None,
) -> dict[str, Any]:
    """Summarize available samples without manufacturing unavailable measurements."""

    field_map = {
        "system_cpu_percent": "system_cpu",
        "process_cpu_percent": "process_cpu",
        "ram_mb": "ram",
        "process_rss_mb": "process_rss",
        "gpu_percent": "gpu",
        "vram_mb": "vram",
    }
    summary: dict[str, Any] = {
        "sample_count": len(samples),
        "gpu_provider": gpu_provider,
    }
    for sample_field, output_prefix in field_map.items():
        values = [
            float(value)
            for sample in samples
            if (value := sample.get(sample_field)) is not None and math.isfinite(float(value))
        ]
        if output_prefix.endswith("cpu"):
            mean_field = f"{output_prefix}_mean_percent"
            peak_field = f"{output_prefix}_peak_percent"
        elif output_prefix in {"ram", "process_rss", "vram"}:
            mean_field = f"{output_prefix}_mean_mb"
            peak_field = f"{output_prefix}_peak_mb"
        else:
            mean_field = "gpu_mean_percent"
            peak_field = "gpu_peak_percent"
        summary[mean_field] = _mean(values)
        summary[peak_field] = max(values) if values else None

    cpu_ram_available = all(
        summary[field] is not None
        for field in (
            "system_cpu_mean_percent",
            "process_cpu_mean_percent",
            "ram_mean_mb",
            "process_rss_mean_mb",
        )
    )
    gpu_available = summary["gpu_mean_percent"] is not None and summary["vram_mean_mb"] is not None
    if cpu_ram_available and gpu_available:
        summary["status"] = "cpu_ram_available_gpu_available"
    elif cpu_ram_available:
        summary["status"] = "cpu_ram_available_gpu_unavailable"
    else:
        summary["status"] = "unavailable"
    return summary


def _run_json_command(command: Sequence[str]) -> Any:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3,
        creationflags=creation_flags,
    )
    return json.loads(completed.stdout)


def _walk_json(value: Any, prefix: str = "") -> list[tuple[str, object]]:
    found: list[tuple[str, object]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_walk_json(child, child_key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_json(child, prefix))
    else:
        found.append((prefix.casefold().replace(" ", "_"), value))
    return found


def _first_matching_number(
    fields: Sequence[tuple[str, object]], predicates: Sequence[str]
) -> float | None:
    for key, value in fields:
        if any(predicate in key for predicate in predicates):
            number = _finite_number(value)
            if number is not None:
                return number
    return None


def _nvidia_reader(executable: str) -> tuple[float | None, float | None]:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3,
        creationflags=creation_flags,
    )
    gpu_values: list[float] = []
    vram_values: list[float] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        gpu = _finite_number(parts[0])
        vram = _finite_number(parts[1])
        if gpu is not None:
            gpu_values.append(gpu)
        if vram is not None:
            vram_values.append(vram)
    return _mean(gpu_values), sum(vram_values) if vram_values else None


def _amd_smi_reader(executable: str) -> tuple[float | None, float | None]:
    fields = _walk_json(_run_json_command([executable, "metric", "-u", "-v", "--json"]))
    gpu = _first_matching_number(fields, ("gfx_activity", "gfx_util", "gpu_util"))
    vram = _first_matching_number(fields, ("vram_used", "used_vram"))
    return gpu, vram


def _rocm_smi_reader(executable: str) -> tuple[float | None, float | None]:
    fields = _walk_json(
        _run_json_command([executable, "--showuse", "--showmeminfo", "vram", "--json"])
    )
    gpu = _first_matching_number(fields, ("gpu_use", "gpu_busy", "gpu_use_(%)"))
    vram_bytes = _first_matching_number(fields, ("vram_total_used_memory", "vram_used"))
    return gpu, vram_bytes / MEBIBYTE if vram_bytes is not None else None


def detect_gpu_reader() -> tuple[str | None, GpuReader | None]:
    """Select a supported vendor CLI when one is available on PATH."""

    if executable := shutil.which("nvidia-smi"):
        return "nvidia-smi", lambda: _nvidia_reader(executable)
    if executable := shutil.which("amd-smi"):
        return "amd-smi", lambda: _amd_smi_reader(executable)
    if executable := shutil.which("rocm-smi"):
        return "rocm-smi", lambda: _rocm_smi_reader(executable)
    return None, None


class ResourceTelemetrySampler:
    """Sample process, system, and optional GPU metrics without raising to callers."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self._samples: list[dict[str, float | None]] = []
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = None
        try:
            self._gpu_provider, self._gpu_reader = detect_gpu_reader()
        except Exception:
            self._gpu_provider, self._gpu_reader = None, None

    def start(self) -> None:
        if psutil is not None:
            with suppress(Exception):
                self._process = psutil.Process()
            with suppress(Exception):
                psutil.cpu_percent(interval=None)
            if self._process is not None:
                with suppress(Exception):
                    self._process.cpu_percent(interval=None)
        try:
            self._thread = threading.Thread(
                target=self._sample_until_stopped,
                name="benchmark-resource-telemetry",
                daemon=True,
            )
            self._thread.start()
        except Exception:
            self._thread = None

    def _sample_until_stopped(self) -> None:
        while not self._stopped.is_set():
            self._sample()
            if self._stopped.wait(self.interval_seconds):
                break

    def _sample(self) -> None:
        sample: dict[str, float | None] = {
            "system_cpu_percent": None,
            "process_cpu_percent": None,
            "ram_mb": None,
            "process_rss_mb": None,
            "gpu_percent": None,
            "vram_mb": None,
        }
        if psutil is not None:
            with suppress(Exception):
                sample["system_cpu_percent"] = float(psutil.cpu_percent(interval=None))
            with suppress(Exception):
                sample["ram_mb"] = float(psutil.virtual_memory().used) / MEBIBYTE
            if self._process is not None:
                with suppress(Exception):
                    sample["process_cpu_percent"] = float(self._process.cpu_percent(interval=None))
                with suppress(Exception):
                    sample["process_rss_mb"] = float(self._process.memory_info().rss) / MEBIBYTE
        if self._gpu_reader is not None:
            with suppress(Exception):
                sample["gpu_percent"], sample["vram_mb"] = self._gpu_reader()
        self._samples.append(sample)

    def stop(self) -> dict[str, Any]:
        self._stopped.set()
        if self._thread is not None:
            with suppress(Exception):
                self._thread.join(timeout=max(4.0, self.interval_seconds * 2))
        try:
            return summarize_samples(self._samples, gpu_provider=self._gpu_provider)
        except Exception:
            return summarize_samples([], gpu_provider=self._gpu_provider)
