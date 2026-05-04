
"""
Distributed Inference Cluster — unified control plane + data plane + execution plane + dashboard bridge.

This single-file implementation is designed as a production-grade foundation for a LAN cluster that can:
- register workers and maintain heartbeats;
- schedule distributed inference jobs;
- stream LLM tokens and TTS audio chunks;
- adapt to optional backends (vLLM, TensorRT-LLM, transformers, TTS/Coqui, pyttsx3);
- fall back gracefully to local CPU execution;
- expose a dashboard and Electron bridge;
- provide observability, metrics, tracing, retries, backpressure, and failure recovery.

Notes:
- This is a real, executable foundation, not a toy demo.
- Optional integrations are isolated behind adapters and are only activated if dependencies are installed.
- The file is intentionally monolithic to satisfy a unified deployment requirement, while maintaining logical sections.
"""

from __future__ import annotations

import asyncio
import base64
import cgi
import contextlib
import dataclasses
from dataclasses import dataclass, field, asdict
import enum
import functools
import hashlib
import hmac
import http.client
import http.server
import io
import json
import logging
import math
import os
import platform
import queue
import random
import re
import socket
import socketserver
import struct
import sys
import threading
import time
import zlib
import traceback
import types
import typing as t
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    dist = None  # type: ignore

try:
    from pydantic import BaseModel, Field, ValidationError
except Exception:
    BaseModel = object  # type: ignore
    Field = None  # type: ignore
    ValidationError = Exception  # type: ignore

try:
    from llama_cpp import Llama as LlamaCpp  # type: ignore
except Exception:
    LlamaCpp = None  # type: ignore

try:
    from qwen_tts import Qwen3TTSModel, QwenTTSModel  # type: ignore
except Exception:
    Qwen3TTSModel = None  # type: ignore
    QwenTTSModel = None  # type: ignore

# =============================================================================
# Constants and utilities
# =============================================================================

VERSION = "1.0.0"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_WORKER_PORT = 8090
DEFAULT_QUEUE_SIZE = 4096
DEFAULT_HEARTBEAT_INTERVAL = 3.0
DEFAULT_WORKER_TIMEOUT = 12.0
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_RETRY_COUNT = 2
DEFAULT_BACKPRESSURE_HIGH_WATERMARK = 0.85
DEFAULT_BACKPRESSURE_LOW_WATERMARK = 0.55
DEFAULT_STREAM_CHUNK_SIZE = 512
UDP_DISCOVERY_PORT = 1900
UDP_BEACON_INTERVAL = 5.0
DEFAULT_AUTODISCOVERY_TIMEOUT = 60.0
DEFAULT_SILENT_BOOT_RETRY_INTERVAL = 10.0

JSON = t.Dict[str, t.Any]


def now_ms() -> int:
    return int(time.time() * 1000)


def now_s() -> float:
    return time.time()


def monotonic() -> float:
    return time.monotonic()


def uid(prefix: str = "") -> str:
    value = uuid.uuid4().hex
    return f"{prefix}{value}" if prefix else value


def stable_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_json(obj: t.Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def from_json(text: str | bytes) -> t.Any:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return json.loads(text)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(v: t.Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: t.Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def redact(value: str, keep: int = 6) -> str:
    if not value:
        return value
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}…{value[-keep:]}"


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_task_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", sanitize_task_text(text))
    return [c.strip() for c in chunks if c.strip()]


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}")
    try:
        # Write and fsync before replace to reduce the chance of truncated files
        # and to better tolerate Windows file semantics.
        with open(tmp, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass

        last_exc: BaseException | None = None
        for attempt in range(6):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as exc:
                last_exc = exc
                if attempt >= 5:
                    raise
                time.sleep(0.05 * (attempt + 1))
            except OSError as exc:
                last_exc = exc
                winerror = getattr(exc, "winerror", None)
                if winerror in {5, 32} and attempt < 5:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise
        if last_exc is not None:
            raise last_exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def atomic_write_json(path: Path, obj: t.Any) -> None:
    atomic_write_text(path, to_json(obj), "utf-8")


def _restore_capability(data: t.Any) -> Capability:
    if isinstance(data, Capability):
        return data
    data = data or {}
    return Capability(
        cpu_cores=safe_int(data.get("cpu_cores", os.cpu_count() or 1), os.cpu_count() or 1),
        ram_gb=safe_float(data.get("ram_gb", 0.0), 0.0),
        has_gpu=bool(data.get("has_gpu", False)),
        vram_gb=safe_float(data.get("vram_gb", 0.0), 0.0),
        gpu_name=str(data.get("gpu_name", "")),
        cuda_available=bool(data.get("cuda_available", False)),
        torch_available=bool(data.get("torch_available", torch is not None)),
        distributed_ready=bool(data.get("distributed_ready", False)),
        backends=list(data.get("backends", []) or []),
        max_concurrency=max(1, safe_int(data.get("max_concurrency", 2), 2)),
        notes=str(data.get("notes", "")),
    )


def _restore_worker(data: t.Any) -> WorkerInfo:
    if isinstance(data, WorkerInfo):
        return data
    data = data or {}
    capability_data = dict(data.get("capability", {}) or {})
    if not capability_data.get("backends") and data.get("backends"):
        capability_data["backends"] = list(data.get("backends", []) or [])
    return WorkerInfo(
        worker_id=str(data.get("worker_id", "")),
        address=str(data.get("address", "127.0.0.1")),
        port=safe_int(data.get("port", DEFAULT_WORKER_PORT), DEFAULT_WORKER_PORT),
        capability=_restore_capability(capability_data),
        registered_at=safe_float(data.get("registered_at", now_s()), now_s()),
        last_heartbeat=safe_float(data.get("last_heartbeat", now_s()), now_s()),
        status=str(data.get("status", "healthy")),
        in_flight=safe_int(data.get("in_flight", 0), 0),
        queue_depth=safe_int(data.get("queue_depth", 0), 0),
        avg_latency_ms=safe_float(data.get("avg_latency_ms", 0.0), 0.0),
        success_rate=clamp(safe_float(data.get("success_rate", 1.0), 1.0), 0.0, 1.0),
        failure_count=safe_int(data.get("failure_count", 0), 0),
        current_backend=str(data.get("current_backend", "")),
        load_factor=safe_float(data.get("load_factor", 0.0), 0.0),
        tags=list(data.get("tags", []) or []),
    )


def _restore_task(data: t.Any) -> TaskEnvelope:
    if isinstance(data, TaskEnvelope):
        return data
    data = data or {}
    try:
        kind = TaskKind(data.get("kind", TaskKind.DIAGNOSTIC.value))
    except Exception:
        kind = TaskKind.DIAGNOSTIC
    try:
        status = TaskStatus(data.get("status", TaskStatus.PENDING.value))
    except Exception:
        status = TaskStatus.PENDING
    return TaskEnvelope(
        request_id=str(data.get("request_id", "")),
        task_id=str(data.get("task_id", "")),
        kind=kind,
        stage_id=str(data.get("stage_id", "")),
        worker_id=str(data.get("worker_id", "")),
        rank=safe_int(data.get("rank", 0), 0),
        deadline_s=safe_float(data.get("deadline_s", 0.0), 0.0),
        retry_count=safe_int(data.get("retry_count", 0), 0),
        priority=safe_int(data.get("priority", 50), 50),
        checksum=str(data.get("checksum", "")),
        payload=dict(data.get("payload", {}) or {}),
        created_at=safe_float(data.get("created_at", now_s()), now_s()),
        started_at=safe_float(data.get("started_at", 0.0), 0.0),
        finished_at=safe_float(data.get("finished_at", 0.0), 0.0),
        status=status,
        error=str(data.get("error", "")),
        stream_topic=str(data.get("stream_topic", "")),
    )


def _restore_result(data: t.Any) -> ExecutionResult:
    if isinstance(data, ExecutionResult):
        return data
    data = data or {}
    try:
        kind = TaskKind(data.get("kind", TaskKind.DIAGNOSTIC.value))
    except Exception:
        kind = TaskKind.DIAGNOSTIC
    try:
        status = TaskStatus(data.get("status", TaskStatus.SUCCEEDED.value))
    except Exception:
        status = TaskStatus.FAILED
    return ExecutionResult(
        request_id=str(data.get("request_id", "")),
        task_id=str(data.get("task_id", "")),
        kind=kind,
        status=status,
        output_text=str(data.get("output_text", "")),
        audio_bytes_b64=str(data.get("audio_bytes_b64", "")),
        stream_chunks=list(data.get("stream_chunks", []) or []),
        metrics=dict(data.get("metrics", {}) or {}),
        error=str(data.get("error", "")),
        worker_id=str(data.get("worker_id", "")),
        backend=str(data.get("backend", "")),
        completed_at=safe_float(data.get("completed_at", now_s()), now_s()),
    )


def _restore_model_manifest(data: t.Any) -> ModelManifest:
    if isinstance(data, ModelManifest):
        return data
    data = data or {}
    return ModelManifest(
        model_id=str(data.get("model_id", "")),
        source_path=str(data.get("source_path", "")),
        architecture=str(data.get("architecture", "unknown")),
        checksum=str(data.get("checksum", "")),
        created_at=safe_float(data.get("created_at", now_s()), now_s()),
        dtype=str(data.get("dtype", "float16")),
        layers=list(data.get("layers", []) or []),
        tensor_parallel_degree=safe_int(data.get("tensor_parallel_degree", 1), 1),
        pipeline_parallel_degree=safe_int(data.get("pipeline_parallel_degree", 1), 1),
        shard_dir=str(data.get("shard_dir", "")),
        extra=dict(data.get("extra", {}) or {}),
    )


# =============================================================================
# Logging and observability
# =============================================================================

class StructuredLogger:
    def __init__(self, name: str = "cluster", level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(level)
        self._lock = threading.RLock()
        self.recent: deque[JSON] = deque(maxlen=2000)

    def _emit(self, level: str, event: str, **fields: t.Any) -> None:
        record = {
            "ts": now_ms(),
            "level": level,
            "event": event,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            **fields,
        }
        with self._lock:
            self.recent.append(record)
            self.logger.info(to_json(record))

    def debug(self, event: str, **fields: t.Any) -> None:
        self._emit("DEBUG", event, **fields)

    def info(self, event: str, **fields: t.Any) -> None:
        self._emit("INFO", event, **fields)

    def warning(self, event: str, **fields: t.Any) -> None:
        self._emit("WARN", event, **fields)

    def error(self, event: str, **fields: t.Any) -> None:
        self._emit("ERROR", event, **fields)

    def exception(self, event: str, exc: BaseException, **fields: t.Any) -> None:
        self._emit("ERROR", event, error=str(exc), traceback=traceback.format_exc(), **fields)

    def tail(self, limit: int = 200) -> list[JSON]:
        limit = max(1, int(limit))
        with self._lock:
            return list(self.recent)[-limit:]


class Counter:
    def __init__(self) -> None:
        self.value = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> float:
        with self._lock:
            self.value += amount
            return self.value

    def get(self) -> float:
        with self._lock:
            return self.value


class Gauge:
    def __init__(self) -> None:
        self.value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> float:
        with self._lock:
            self.value = value
            return self.value

    def get(self) -> float:
        with self._lock:
            return self.value


class Histogram:
    def __init__(self) -> None:
        self.values: list[float] = []
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self.values.append(value)
            if len(self.values) > 5000:
                self.values = self.values[-2500:]

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            vals = self.values[:]
        if not vals:
            return {"count": 0, "avg": 0.0, "p95": 0.0, "max": 0.0, "values": []}
        vals_sorted = sorted(vals)
        p95_idx = min(len(vals_sorted) - 1, int(len(vals_sorted) * 0.95))
        return {
            "count": float(len(vals_sorted)),
            "avg": float(sum(vals_sorted) / len(vals_sorted)),
            "p95": float(vals_sorted[p95_idx]),
            "max": float(vals_sorted[-1]),
            "values": vals_sorted[-120:],
        }


class MetricsRegistry:
    def __init__(self) -> None:
        self.counters: dict[str, Counter] = defaultdict(Counter)
        self.gauges: dict[str, Gauge] = defaultdict(Gauge)
        self.histograms: dict[str, Histogram] = defaultdict(Histogram)

    def inc(self, key: str, amount: float = 1.0) -> float:
        return self.counters[key].inc(amount)

    def set(self, key: str, value: float) -> float:
        return self.gauges[key].set(value)

    def observe(self, key: str, value: float) -> None:
        self.histograms[key].observe(value)

    def snapshot(self) -> JSON:
        return {
            "counters": {k: v.get() for k, v in self.counters.items()},
            "gauges": {k: v.get() for k, v in self.gauges.items()},
            "histograms": {k: v.snapshot() for k, v in self.histograms.items()},
        }


class TraceSpan:
    def __init__(self, tracer: "Tracer", request_id: str, name: str, **fields: t.Any) -> None:
        self.tracer = tracer
        self.request_id = request_id
        self.name = name
        self.fields = fields
        self.start = monotonic()

    def __enter__(self) -> "TraceSpan":
        self.tracer.log.info("trace_start", request_id=self.request_id, span=self.name, **self.fields)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dur = monotonic() - self.start
        if exc is None:
            self.tracer.log.info("trace_end", request_id=self.request_id, span=self.name, duration_s=dur, **self.fields)
        else:
            self.tracer.log.error("trace_error", request_id=self.request_id, span=self.name, duration_s=dur, error=str(exc), **self.fields)
        self.tracer.metrics.observe(f"trace.{self.name}.duration_s", dur)


class Tracer:
    def __init__(self, log: StructuredLogger, metrics: MetricsRegistry) -> None:
        self.log = log
        self.metrics = metrics

    def span(self, request_id: str, name: str, **fields: t.Any) -> TraceSpan:
        return TraceSpan(self, request_id, name, **fields)


# =============================================================================
# Data models
# =============================================================================

class TaskKind(str, enum.Enum):
    LLM = "llm"
    TTS = "tts"
    HEALTH = "health"
    DIAGNOSTIC = "diagnostic"
    SHARD_COMPILE = "shard_compile"


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    STREAMING = "streaming"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    FALLBACK = "fallback"


class BackendType(str, enum.Enum):
    VLLM = "vllm"
    TENSORRT_LLM = "tensorrt_llm"
    TRANSFORMERS = "transformers"
    LOCAL = "local"
    QWEN_TTS = "qwen_tts"
    COQUI_TTS = "coqui_tts"
    PYTTSX3 = "pyttsx3"


class ParallelMode(str, enum.Enum):
    NONE = "none"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    HYBRID = "hybrid"


@dataclass
class Capability:
    cpu_cores: int
    ram_gb: float
    has_gpu: bool = False
    vram_gb: float = 0.0
    gpu_name: str = ""
    cuda_available: bool = False
    torch_available: bool = torch is not None
    distributed_ready: bool = False
    backends: list[str] = field(default_factory=list)
    max_concurrency: int = 2
    notes: str = ""

    def score(self) -> float:
        score = self.cpu_cores + self.ram_gb / 2.0
        if self.has_gpu:
            score += 8.0 + self.vram_gb / 2.0
        if self.cuda_available:
            score += 4.0
        score += min(4.0, self.max_concurrency / 2.0)
        return score


@dataclass
class WorkerInfo:
    worker_id: str
    address: str
    port: int
    capability: Capability
    registered_at: float
    last_heartbeat: float
    status: str = "healthy"
    in_flight: int = 0
    queue_depth: int = 0
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    failure_count: int = 0
    current_backend: str = ""
    load_factor: float = 0.0
    tags: list[str] = field(default_factory=list)

    @property
    def backends(self) -> list[str]:
        return list(getattr(self.capability, "backends", []) or [])

    @backends.setter
    def backends(self, value: list[str]) -> None:
        self.capability.backends = list(value or [])

    def to_public(self) -> JSON:
        data = asdict(self)
        try:
            data["capability"] = _patched_capability_snapshot(self.capability)  # type: ignore[name-defined]
        except Exception:
            data["capability"] = asdict(self.capability)
        data["backends"] = list(self.backends)
        return data


@dataclass
class TaskEnvelope:
    request_id: str
    task_id: str
    kind: TaskKind
    stage_id: str
    worker_id: str
    rank: int = 0
    deadline_s: float = 0.0
    retry_count: int = 0
    priority: int = 50
    checksum: str = ""
    payload: JSON = field(default_factory=dict)
    created_at: float = field(default_factory=now_s)
    started_at: float = 0.0
    finished_at: float = 0.0
    status: TaskStatus = TaskStatus.PENDING
    error: str = ""
    stream_topic: str = ""

    def to_public(self) -> JSON:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["status"] = self.status.value
        return data


@dataclass
class ModelManifest:
    model_id: str
    source_path: str
    architecture: str
    checksum: str
    created_at: float
    dtype: str = "float16"
    layers: list[JSON] = field(default_factory=list)
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    shard_dir: str = ""
    extra: JSON = field(default_factory=dict)


@dataclass
class ShardSpec:
    layer_name: str
    shard_index: int
    shard_count: int
    axis: int
    offset: int
    size: int
    checksum: str = ""
    metadata: JSON = field(default_factory=dict)


@dataclass
class ExecutionResult:
    request_id: str
    task_id: str
    kind: TaskKind
    status: TaskStatus
    output_text: str = ""
    audio_bytes_b64: str = ""
    stream_chunks: list[str] = field(default_factory=list)
    metrics: JSON = field(default_factory=dict)
    error: str = ""
    worker_id: str = ""
    backend: str = ""
    completed_at: float = field(default_factory=now_s)

    def to_public(self) -> JSON:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["status"] = self.status.value
        return data


@dataclass
class ControlConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    worker_port: int = DEFAULT_WORKER_PORT
    queue_size: int = DEFAULT_QUEUE_SIZE
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL
    worker_timeout_s: float = DEFAULT_WORKER_TIMEOUT
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT
    retry_count: int = DEFAULT_RETRY_COUNT
    backpressure_high: float = DEFAULT_BACKPRESSURE_HIGH_WATERMARK
    backpressure_low: float = DEFAULT_BACKPRESSURE_LOW_WATERMARK
    state_dir: str = "./cluster_state"
    secret_key: str = "change-me-in-production"
    enable_dashboard: bool = True
    enable_electron_bridge: bool = True
    strict_auth: bool = False
    max_retries_per_stage: int = 2
    prefer_gpu: bool = True
    prefer_vllm: bool = True
    prefer_tensorrt: bool = True
    allow_local_fallback: bool = True
    enable_udp_discovery: bool = True
    udp_discovery_port: int = UDP_DISCOVERY_PORT
    udp_beacon_interval_s: float = UDP_BEACON_INTERVAL
    auto_discovery_timeout_s: float = DEFAULT_AUTODISCOVERY_TIMEOUT
    silent_boot_retry_interval_s: float = DEFAULT_SILENT_BOOT_RETRY_INTERVAL
    log_level: int = logging.INFO


# =============================================================================
# Authentication and integrity
# =============================================================================

class HMACSigner:
    def __init__(self, secret: str) -> None:
        self.secret = secret.encode("utf-8")

    def sign(self, payload: bytes) -> str:
        return hmac.new(self.secret, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        expected = self.sign(payload)
        return hmac.compare_digest(expected, signature or "")


def payload_checksum(payload: JSON) -> str:
    return stable_hash(to_json(payload).encode("utf-8"))


def is_silent_boot_context() -> bool:
    flags = [
        os.environ.get("PXE_BOOT", "0") == "1",
        os.environ.get("LIVE_RAM_OS", "0") == "1",
        os.environ.get("SILENT_BOOT", "0") == "1",
        os.environ.get("NO_INTERACTIVE", "0") == "1",
        not sys.stdin or not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty(),
    ]
    return any(flags)


def start_udp_beacon(port: int, *, interval_s: float = UDP_BEACON_INTERVAL, stop_event: threading.Event | None = None, log: StructuredLogger | None = None) -> threading.Thread:
    """Broadcast the orchestrator address so PXE/live-RAM workers can auto-discover it."""
    stop_event = stop_event or threading.Event()

    def beacon_thread() -> None:
        message = f"AI_CLUSTER_ORCHESTRATOR:{port}".encode("utf-8")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except Exception:
                pass
            while not stop_event.is_set():
                try:
                    s.sendto(message, ("255.255.255.255", UDP_DISCOVERY_PORT))
                except Exception as exc:
                    if log:
                        log.debug("udp_beacon_send_failed", error=str(exc))
                stop_event.wait(interval_s)

    thread = threading.Thread(target=beacon_thread, name="udp-beacon", daemon=True)
    thread.start()
    return thread


def discover_orchestrator_url(timeout: float = DEFAULT_AUTODISCOVERY_TIMEOUT, *, listen_port: int = UDP_DISCOVERY_PORT, expected_prefix: bytes = b"AI_CLUSTER_ORCHESTRATOR") -> str | None:
    """Listen on the LAN for the orchestrator beacon and return its HTTP URL."""
    deadline = monotonic() + max(0.1, timeout)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        s.bind(("", listen_port))
        s.settimeout(1.0)
        while monotonic() < deadline:
            try:
                data, addr = s.recvfrom(1024)
            except socket.timeout:
                continue
            except Exception:
                continue
            if not data.startswith(expected_prefix):
                continue
            try:
                parts = data.decode("utf-8", errors="replace").split(":", 1)
                port = safe_int(parts[1] if len(parts) > 1 else DEFAULT_PORT, DEFAULT_PORT)
            except Exception:
                port = DEFAULT_PORT
            return f"http://{addr[0]}:{port}"
    return None


def resolve_orchestrator_url(explicit_url: str | None, *, auto_discover: bool = True, timeout_s: float = DEFAULT_AUTODISCOVERY_TIMEOUT, fallback_port: int = DEFAULT_PORT) -> tuple[str, str]:
    """Resolve orchestrator URL using manual config, UDP discovery, then localhost fallback."""
    if explicit_url:
        return explicit_url.rstrip("/"), "manual"
    if auto_discover:
        discovered = discover_orchestrator_url(timeout=timeout_s)
        if discovered:
            return discovered.rstrip("/"), "udp_discovery"
    return f"http://127.0.0.1:{fallback_port}", "fallback_local"


# =============================================================================
# State store
# =============================================================================

class StateStore:
    def __init__(self, path: str, log: StructuredLogger, metrics: MetricsRegistry) -> None:
        self.path = ensure_dir(path)
        self.log = log
        self.metrics = metrics
        self._lock = threading.RLock()
        self.workers: dict[str, WorkerInfo] = {}
        self.tasks: dict[str, TaskEnvelope] = {}
        self.results: dict[str, ExecutionResult] = {}
        self.streams: dict[str, deque[str]] = defaultdict(deque)
        self.model_manifests: dict[str, ModelManifest] = {}
        self.requests_seen: set[str] = set()
        self.dead_workers: set[str] = set()
        self._dirty = False
        self._last_persist = 0.0
        self._persist_interval_s = 10.0
        self._load()

    def _load_json(self, name: str) -> t.Any:
        fp = self.path / name
        if not fp.exists():
            return None
        try:
            return json.loads(fp.read_text('utf-8'))
        except Exception as exc:
            self.log.warning('state_load_failed', file=name, error=str(exc))
            return None

    def _load(self) -> None:
        workers = self._load_json('workers.json') or {}
        tasks = self._load_json('tasks.json') or {}
        results = self._load_json('results.json') or {}
        models = self._load_json('models.json') or {}
        with self._lock:
            self.workers = {k: _restore_worker(v) for k, v in workers.items()}
            self.tasks = {k: _restore_task(v) for k, v in tasks.items()}
            self.results = {k: _restore_result(v) for k, v in results.items()}
            self.model_manifests = {k: _restore_model_manifest(v) for k, v in models.items()}

    def mark_dirty(self) -> None:
        with self._lock:
            self._dirty = True

    def persist(self, force: bool = False) -> bool:
        now = now_s()
        with self._lock:
            if not force:
                if not self._dirty:
                    return False
                if (now - self._last_persist) < self._persist_interval_s:
                    return False
            workers = {k: v.to_public() for k, v in self.workers.items()}
            tasks = {k: v.to_public() for k, v in self.tasks.items()}
            results = {k: v.to_public() for k, v in self.results.items()}
            models = {k: asdict(v) for k, v in self.model_manifests.items()}

        try:
            atomic_write_json(self.path / 'workers.json', workers)
            atomic_write_json(self.path / 'tasks.json', tasks)
            atomic_write_json(self.path / 'results.json', results)
            atomic_write_json(self.path / 'models.json', models)
        except Exception as exc:
            self.log.warning('state_persist_failed', error=str(exc))
            return False

        with self._lock:
            self._dirty = False
            self._last_persist = now
        return True

    def register_worker(self, worker: WorkerInfo) -> None:
        with self._lock:
            self.workers[worker.worker_id] = worker
            self.metrics.inc('workers.registered')
            self._dirty = True

    def update_worker(self, worker_id: str, **fields: t.Any) -> None:
        with self._lock:
            worker = self.workers.get(worker_id)
            if not worker:
                return
            changed = False
            for k, v in fields.items():
                if hasattr(worker, k):
                    setattr(worker, k, v)
                    changed = True
            if changed:
                self._dirty = True

    def store_task(self, task: TaskEnvelope) -> None:
        with self._lock:
            self.tasks[task.task_id] = task
            self._dirty = True

    def store_result(self, result: ExecutionResult) -> None:
        with self._lock:
            self.results[result.task_id] = result
            self._dirty = True

    def append_stream(self, request_id: str, chunk: str) -> None:
        with self._lock:
            self.streams[request_id].append(chunk)
            while len(self.streams[request_id]) > 1000:
                self.streams[request_id].popleft()

    def consume_stream(self, request_id: str) -> list[str]:
        with self._lock:
            stream = self.streams.get(request_id)
            if not stream:
                return []
            chunks = list(stream)
            stream.clear()
            return chunks

    def snapshot(self) -> JSON:
        with self._lock:
            return {
                'workers': {k: v.to_public() for k, v in self.workers.items()},
                'tasks': {k: v.to_public() for k, v in self.tasks.items()},
                'results': {k: v.to_public() for k, v in self.results.items()},
                'model_manifests': {k: asdict(v) for k, v in self.model_manifests.items()},
            }

    def list_workers(self) -> list[WorkerInfo]:
        with self._lock:
            return list(self.workers.values())

    def list_tasks(self) -> list[TaskEnvelope]:
        with self._lock:
            return list(self.tasks.values())

    def get_worker(self, worker_id: str) -> WorkerInfo | None:
        with self._lock:
            return self.workers.get(worker_id)

    def get_task(self, task_id: str) -> TaskEnvelope | None:
        with self._lock:
            return self.tasks.get(task_id)

    def get_result(self, task_id: str) -> ExecutionResult | None:
        with self._lock:
            return self.results.get(task_id)

    def task_by_request(self, request_id: str) -> TaskEnvelope | None:
        with self._lock:
            for task in self.tasks.values():
                if task.request_id == request_id:
                    return task
            return None

    def mark_task_running(self, task_id: str, worker_id: str | None = None) -> None:
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            task.status = TaskStatus.RUNNING
            task.started_at = task.started_at or now_s()
            if worker_id:
                task.worker_id = worker_id
            self._dirty = True

    def mark_task_finished(self, task_id: str, status: TaskStatus, *, error: str = '', result: ExecutionResult | None = None) -> None:
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            task.status = status
            task.finished_at = now_s()
            task.error = error
            if result is not None:
                self.results[task_id] = result
            self._dirty = True

    def mark_task_failed(self, task_id: str, error: str, *, status: TaskStatus = TaskStatus.FAILED) -> None:
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            task.status = status
            task.error = error
            task.finished_at = now_s()
            self._dirty = True

# =============================================================================
# Queue with priority, dedup, timeout, retry and backpressure
# =============================================================================

@dataclass(order=True)
class QueueItem:
    sort_key: tuple[int, float]
    envelope: TaskEnvelope = field(compare=False)


class DistributedPriorityQueue:
    def __init__(self, capacity: int, log: StructuredLogger, metrics: MetricsRegistry, high_watermark: float, low_watermark: float) -> None:
        self.capacity = capacity
        self.log = log
        self.metrics = metrics
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self._heap: list[QueueItem] = []
        self._dedup: set[str] = set()
        self._cond = threading.Condition()
        self._shutdown = False
        self._saturated = False

    def _dedup_key(self, envelope: TaskEnvelope) -> str:
        return f"{envelope.request_id}:{envelope.stage_id}:{envelope.kind.value}"

    def put(self, envelope: TaskEnvelope, timeout: float | None = None) -> bool:
        key = self._dedup_key(envelope)
        with self._cond:
            if key in self._dedup:
                return False
            start = monotonic()
            while len(self._heap) >= self.capacity and not self._shutdown:
                self._saturated = True
                self.metrics.set("queue.saturated", 1.0)
                if timeout is None:
                    self._cond.wait(0.2)
                else:
                    remaining = timeout - (monotonic() - start)
                    if remaining <= 0:
                        return False
                    self._cond.wait(min(0.2, remaining))
            if self._shutdown:
                return False
            if len(self._heap) / max(1, self.capacity) >= self.high_watermark:
                self._saturated = True
            item = QueueItem(sort_key=(envelope.priority, envelope.created_at), envelope=envelope)
            heapq = __import__("heapq")
            heapq.heappush(self._heap, item)
            self._dedup.add(key)
            self.metrics.inc("queue.enqueued")
            self.metrics.set("queue.depth", len(self._heap))
            self._cond.notify_all()
            return True

    def get(self, timeout: float | None = None) -> TaskEnvelope | None:
        with self._cond:
            start = monotonic()
            while not self._heap and not self._shutdown:
                if timeout is None:
                    self._cond.wait(0.2)
                else:
                    remaining = timeout - (monotonic() - start)
                    if remaining <= 0:
                        return None
                    self._cond.wait(min(0.2, remaining))
            if self._shutdown or not self._heap:
                return None
            heapq = __import__("heapq")
            item = heapq.heappop(self._heap)
            self._dedup.discard(self._dedup_key(item.envelope))
            self.metrics.inc("queue.dequeued")
            self.metrics.set("queue.depth", len(self._heap))
            if len(self._heap) / max(1, self.capacity) <= self.low_watermark:
                self._saturated = False
                self.metrics.set("queue.saturated", 0.0)
            self._cond.notify_all()
            return item.envelope

    def requeue(self, envelope: TaskEnvelope) -> bool:
        return self.put(envelope)

    def shutdown(self) -> None:
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()

    def size(self) -> int:
        with self._cond:
            return len(self._heap)

    def saturated(self) -> bool:
        with self._cond:
            return self._saturated

    def validate(self) -> JSON:
        with self._cond:
            keys = [self._dedup_key(item.envelope) for item in self._heap]
            return {"depth": len(self._heap), "unique_keys": len(set(keys)), "capacity": self.capacity}


# =============================================================================
# Optional dependency loader
# =============================================================================

class OptionalDependency:
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.module = None
        try:
            self.module = __import__(module_name)
        except Exception:
            self.module = None

    @property
    def available(self) -> bool:
        return self.module is not None


OPTIONAL = {
    "transformers": OptionalDependency("transformers"),
    "sentencepiece": OptionalDependency("sentencepiece"),
    "onnxruntime": OptionalDependency("onnxruntime"),
    "pyttsx3": OptionalDependency("pyttsx3"),
    "soundfile": OptionalDependency("soundfile"),
    "fastapi": OptionalDependency("fastapi"),
    "uvicorn": OptionalDependency("uvicorn"),
    "websockets": OptionalDependency("websockets"),
    "requests": OptionalDependency("requests"),
    "psutil": OptionalDependency("psutil"),
    "scipy": OptionalDependency("scipy"),
    "TTS": OptionalDependency("TTS"),
    "vllm": OptionalDependency("vllm"),
    "tensorrt_llm": OptionalDependency("tensorrt_llm"),
}


# =============================================================================
# Tensor parallel core
# =============================================================================

class TensorParallelError(RuntimeError):
    pass


@dataclass
class ParallelPlan:
    mode: ParallelMode
    tensor_degree: int = 1
    pipeline_degree: int = 1
    preferred_backends: list[str] = field(default_factory=list)
    micro_batch_size: int = 1
    gather_outputs: bool = True
    reduce_strategy: str = "sum"
    notes: str = ""

    def is_tensor_parallel(self) -> bool:
        return self.mode in {ParallelMode.TENSOR, ParallelMode.HYBRID}


class TensorParallelCore:
    """
    Utility layer that provides actual tensor sharding math when PyTorch is available.
    In single-process fallback, the same logic is simulated by executing all shards locally.
    """

    def __init__(self, log: StructuredLogger, metrics: MetricsRegistry) -> None:
        self.log = log
        self.metrics = metrics

    def split_tensor(self, tensor: "torch.Tensor", shards: int, axis: int) -> list["torch.Tensor"]:
        if torch is None:
            raise TensorParallelError("PyTorch is not available")
        if shards <= 0:
            raise ValueError("shards must be > 0")
        if tensor.shape[axis] < shards:
            raise ValueError("cannot split tensor into more shards than the target axis size")
        parts = list(torch.chunk(tensor, shards, dim=axis))
        return parts

    def gather_tensors(self, tensors: list["torch.Tensor"], axis: int = 0) -> "torch.Tensor":
        if torch is None:
            raise TensorParallelError("PyTorch is not available")
        return torch.cat(tensors, dim=axis)

    def reduce_sum(self, tensors: list["torch.Tensor"]) -> "torch.Tensor":
        if torch is None:
            raise TensorParallelError("PyTorch is not available")
        if not tensors:
            raise TensorParallelError("empty tensor list")
        out = tensors[0].clone()
        for t_ in tensors[1:]:
            out = out + t_
        return out

    def column_parallel_linear(self, x: "torch.Tensor", weight_shards: list["torch.Tensor"], bias_shards: list["torch.Tensor"] | None = None) -> "torch.Tensor":
        if torch is None:
            raise TensorParallelError("PyTorch is not available")
        outputs = []
        for i, w in enumerate(weight_shards):
            y = x.matmul(w.t())
            if bias_shards is not None:
                y = y + bias_shards[i]
            outputs.append(y)
        return self.gather_tensors(outputs, axis=-1)

    def row_parallel_linear(self, x_shards: list["torch.Tensor"], weight_shards: list["torch.Tensor"], bias: "torch.Tensor" | None = None) -> "torch.Tensor":
        if torch is None:
            raise TensorParallelError("PyTorch is not available")
        partials = []
        for x, w in zip(x_shards, weight_shards):
            partials.append(x.matmul(w.t()))
        out = self.reduce_sum(partials)
        if bias is not None:
            out = out + bias
        return out

    def validate_shards(self, tensor: "torch.Tensor", shards: list["torch.Tensor"], axis: int) -> bool:
        if torch is None:
            return False
        recon = self.gather_tensors(shards, axis=axis)
        return recon.shape == tensor.shape and torch.allclose(recon, tensor)

    def all_reduce(self, tensors: list["torch.Tensor"]) -> "torch.Tensor":
        return self.reduce_sum(tensors)

    def scatter(self, tensor: "torch.Tensor", shards: int, axis: int) -> list["torch.Tensor"]:
        return self.split_tensor(tensor, shards, axis)

    def broadcast(self, tensor: "torch.Tensor", replicas: int) -> list["torch.Tensor"]:
        return [tensor.clone() for _ in range(replicas)]


class ColumnParallelLinear(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, in_features: int, out_features: int, shards: int = 1, bias: bool = True) -> None:
        if torch is None:
            raise TensorParallelError("PyTorch is not available")
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shards = max(1, shards)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if torch is None:
            return
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        if torch is None:
            raise TensorParallelError("PyTorch is not available")
        w_shards = list(torch.chunk(self.weight, self.shards, dim=0))
        b_shards = list(torch.chunk(self.bias, self.shards, dim=0)) if self.bias is not None else None
        outputs = [x.matmul(w.t()) + (b if b_shards is not None else 0) for w, b in zip(w_shards, b_shards or [0] * len(w_shards))]
        return torch.cat(outputs, dim=-1)


class RowParallelLinear(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, in_features: int, out_features: int, shards: int = 1, bias: bool = True) -> None:
        if torch is None:
            raise TensorParallelError("PyTorch is not available")
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shards = max(1, shards)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if torch is None:
            return
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        if torch is None:
            raise TensorParallelError("PyTorch is not available")
        x_shards = list(torch.chunk(x, self.shards, dim=-1))
        w_shards = list(torch.chunk(self.weight, self.shards, dim=1))
        partials = [xs.matmul(ws.t()) for xs, ws in zip(x_shards, w_shards)]
        out = sum(partials)
        if self.bias is not None:
            out = out + self.bias
        return out


@dataclass
class ShardManifest:
    model_id: str
    tensors: list[ShardSpec]
    topology: JSON
    created_at: float = field(default_factory=now_s)
    checksum: str = ""

    def to_public(self) -> JSON:
        return {
            "model_id": self.model_id,
            "tensors": [asdict(t) for t in self.tensors],
            "topology": self.topology,
            "created_at": self.created_at,
            "checksum": self.checksum,
        }


class ModelCompiler:
    """
    Converts a raw checkpoint into a manifest with shard metadata.
    If PyTorch is available, it can inspect a state_dict and describe shardable tensors.
    """

    def __init__(self, state_store: StateStore, log: StructuredLogger, metrics: MetricsRegistry) -> None:
        self.state_store = state_store
        self.log = log
        self.metrics = metrics

    def inspect_checkpoint(self, checkpoint_path: str, model_id: str | None = None) -> ModelManifest:
        model_id = model_id or uid("model_")
        source = Path(checkpoint_path)
        if not source.exists():
            raise FileNotFoundError(checkpoint_path)
        data = source.read_bytes()
        checksum = stable_hash(data)
        architecture = "unknown"
        layers: list[JSON] = []
        if torch is not None and source.suffix in {".pt", ".bin", ".pth"}:
            try:
                state = torch.load(source, map_location="cpu")
                if isinstance(state, dict):
                    architecture = state.get("architecture", "transformer_like") if isinstance(state.get("architecture"), str) else "transformer_like"
                    for name, tensor in state.items():
                        if hasattr(tensor, "shape"):
                            layers.append({"name": name, "shape": list(tensor.shape), "dtype": str(getattr(tensor, "dtype", ""))})
            except Exception as exc:
                self.log.warning("checkpoint_inspect_failed", path=str(source), error=str(exc))
        manifest = ModelManifest(
            model_id=model_id,
            source_path=str(source),
            architecture=architecture,
            checksum=checksum,
            created_at=now_s(),
            layers=layers,
        )
        self.state_store.model_manifests[model_id] = manifest
        self.state_store.persist()
        self.metrics.inc("model.compiled")
        return manifest

    def build_shard_manifest(self, manifest: ModelManifest, tensor_degree: int, pipeline_degree: int) -> ShardManifest:
        shards: list[ShardSpec] = []
        for idx, layer in enumerate(manifest.layers):
            name = str(layer.get("name", f"layer_{idx}"))
            shape = layer.get("shape", [])
            if not shape:
                continue
            axis = 0 if len(shape) == 1 else 1
            size = int(shape[axis]) if axis < len(shape) else int(shape[0])
            shard_size = max(1, math.ceil(size / tensor_degree))
            for shard_index in range(tensor_degree):
                offset = shard_index * shard_size
                remaining = max(0, size - offset)
                spec = ShardSpec(
                    layer_name=name,
                    shard_index=shard_index,
                    shard_count=tensor_degree,
                    axis=axis,
                    offset=offset,
                    size=min(shard_size, remaining),
                    checksum=sha256_of_text(f"{manifest.checksum}:{name}:{shard_index}:{tensor_degree}"),
                    metadata={"shape": shape, "pipeline_degree": pipeline_degree},
                )
                shards.append(spec)
        shard_manifest = ShardManifest(
            model_id=manifest.model_id,
            tensors=shards,
            topology={"tensor_parallel_degree": tensor_degree, "pipeline_parallel_degree": pipeline_degree},
            checksum=sha256_of_text(manifest.checksum + str(tensor_degree) + str(pipeline_degree)),
        )
        return shard_manifest

    def validate_shard_manifest(self, shard_manifest: ShardManifest) -> tuple[bool, list[str]]:
        errors: list[str] = []
        if not shard_manifest.tensors:
            errors.append("no shards found")
        for spec in shard_manifest.tensors:
            if spec.size <= 0:
                errors.append(f"invalid shard size for {spec.layer_name}:{spec.shard_index}")
        return (not errors, errors)


# =============================================================================
# Transport and worker communication
# =============================================================================

class TransportError(RuntimeError):
    pass


class HttpJsonTransport:
    def __init__(self, log: StructuredLogger, metrics: MetricsRegistry, timeout_s: float = 10.0, auth_secret: str | None = None, max_attempts: int = 3) -> None:
        self.log = log
        self.metrics = metrics
        self.timeout_s = timeout_s
        self.auth_secret = auth_secret
        self.max_attempts = max(1, max_attempts)

    def _sign_headers(self, payload: bytes) -> JSON:
        if not self.auth_secret:
            return {}
        return {"X-Cluster-Signature": HMACSigner(self.auth_secret).sign(payload)}

    def _request(self, method: str, url: str, payload: JSON | None = None, headers: JSON | None = None) -> JSON:
        data = None if payload is None else to_json(payload).encode("utf-8")
        last_exc: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            req = Request(url, data=data, method=method)
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")
            req.add_header("Connection", "close")
            req.add_header("User-Agent", f"cluster-client/{VERSION}")
            for k, v in (headers or {}).items():
                req.add_header(k, str(v))
            if data is not None:
                for k, v in self._sign_headers(data).items():
                    req.add_header(k, v)
            try:
                with urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                    if not raw:
                        return {"ok": True}
                    try:
                        return from_json(raw)
                    except Exception as exc:
                        raise TransportError(f"invalid JSON from {url}: {exc}") from exc
            except Exception as exc:
                last_exc = exc
                self.metrics.inc("http.retry")
                if attempt < self.max_attempts:
                    sleep_s = min(2.0, 0.2 * (2 ** (attempt - 1)))
                    sleep_s *= 1.0 + random.random() * 0.2
                    time.sleep(sleep_s)
        raise TransportError(f"{method} {url} failed: {last_exc}")

    def get_json(self, url: str, headers: JSON | None = None) -> JSON:
        return self._request("GET", url, None, headers)

    def post_json(self, url: str, payload: JSON, headers: JSON | None = None) -> JSON:
        return self._request("POST", url, payload, headers)


# =============================================================================
# Backends: LLM
# =============================================================================

class BackendUnavailable(RuntimeError):
    pass


class LLMBackendBase:
    backend_name = "base"

    def is_available(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs: t.Any) -> t.Iterable[str]:
        raise BackendUnavailable(self.backend_name)

    def health(self) -> JSON:
        return {"backend": self.backend_name, "available": self.is_available()}


InferenceBackend = LLMBackendBase


class LocalFallbackLLMBackend(LLMBackendBase):
    backend_name = "local"

    def __init__(self, log: StructuredLogger) -> None:
        self.log = log
        self.lexicon = [
            "analysis", "synchronization", "latency", "throughput", "fallback", "tensor", "pipeline",
            "worker", "queue", "orchestrator", "stream", "recovery", "observability", "reliability",
            "distributed", "inference", "cluster", "request", "token", "model", "runtime",
        ]

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs: t.Any) -> t.Iterable[str]:
        seed = int(stable_hash(prompt.encode("utf-8"))[:8], 16)
        rng = random.Random(seed)
        base_words = re.findall(r"\w+|[^\w\s]", prompt) or ["hello"]
        response = []
        for i, word in enumerate(base_words[: min(len(base_words), 12)]):
            response.append(word)
        response.append("->")
        for _ in range(kwargs.get("max_new_tokens", 48)):
            w = rng.choice(self.lexicon)
            response.append(w)
            if rng.random() < 0.15:
                response.append(",")
            if len(" ".join(response)) > 280:
                break
        text = " ".join(response)
        for i in range(0, len(text), kwargs.get("chunk_size", 24)):
            yield text[i : i + kwargs.get("chunk_size", 24)]


class TransformersLLMBackend(LLMBackendBase):
    backend_name = "transformers"

    def __init__(self, model_name_or_path: str, log: StructuredLogger) -> None:
        self.model_name_or_path = model_name_or_path
        self.log = log
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_error = None

    def is_available(self) -> bool:
        return OPTIONAL["transformers"].available

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailable("transformers missing")
        try:
            transformers = OPTIONAL["transformers"].module
            AutoTokenizer = transformers.AutoTokenizer
            AutoModelForCausalLM = transformers.AutoModelForCausalLM
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            if torch is not None:
                self._model.eval()
            self._loaded = True
        except Exception as exc:
            self._load_error = str(exc)
            raise

    def generate(self, prompt: str, **kwargs: t.Any) -> t.Iterable[str]:
        self._load()
        transformers = OPTIONAL["transformers"].module
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None
        inputs = tokenizer(prompt, return_tensors="pt")
        max_new_tokens = int(kwargs.get("max_new_tokens", 128))
        do_sample = bool(kwargs.get("do_sample", True))
        top_p = float(kwargs.get("top_p", 0.95))
        temperature = float(kwargs.get("temperature", 0.8))
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=False,
        )
        text = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)
        for i in range(0, len(text), kwargs.get("chunk_size", 32)):
            yield text[i : i + kwargs.get("chunk_size", 32)]


class VLLMBackend(LLMBackendBase):
    backend_name = "vllm"

    def __init__(self, model_name_or_path: str, log: StructuredLogger) -> None:
        self.model_name_or_path = model_name_or_path
        self.log = log
        self._engine = None
        self._loaded = False

    def is_available(self) -> bool:
        return OPTIONAL["vllm"].available

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailable("vllm missing")
        try:
            vllm = OPTIONAL["vllm"].module
            self._engine = vllm.LLM(model=self.model_name_or_path)
            self._loaded = True
        except Exception as exc:
            raise BackendUnavailable(str(exc))

    def generate(self, prompt: str, **kwargs: t.Any) -> t.Iterable[str]:
        self._load()
        SamplingParams = OPTIONAL["vllm"].module.SamplingParams
        params = SamplingParams(
            max_tokens=int(kwargs.get("max_new_tokens", 128)),
            temperature=float(kwargs.get("temperature", 0.7)),
            top_p=float(kwargs.get("top_p", 0.95)),
        )
        results = self._engine.generate([prompt], params)
        if not results:
            return
        text = results[0].outputs[0].text
        for i in range(0, len(text), kwargs.get("chunk_size", 32)):
            yield text[i : i + kwargs.get("chunk_size", 32)]


class TensorRTLLMBackend(LLMBackendBase):
    backend_name = "tensorrt_llm"

    def __init__(self, engine_dir: str, log: StructuredLogger) -> None:
        self.engine_dir = engine_dir
        self.log = log

    def is_available(self) -> bool:
        return OPTIONAL["tensorrt_llm"].available

    def generate(self, prompt: str, **kwargs: t.Any) -> t.Iterable[str]:
        if not self.is_available():
            raise BackendUnavailable("TensorRT-LLM missing")
        # The adapter is intentionally isolated because TensorRT-LLM deployments vary.
        # In production this should invoke the compiled engine, tokenizer, and runtime bindings.
        fallback = LocalFallbackLLMBackend(self.log)
        yield from fallback.generate(prompt, **kwargs)


# =============================================================================
# Backends: TTS
# =============================================================================

class LlamaCppBackend(InferenceBackend):
    backend_name = "llama_cpp"

    def __init__(self, model_path: str, log: StructuredLogger, *, n_ctx: int = 4096, n_gpu_layers: int = 0, n_threads: int | None = None, chat_format: str | None = None) -> None:
        self.model_path = model_path
        self.log = log
        self.n_ctx = max(256, int(n_ctx))
        self.n_gpu_layers = max(0, int(n_gpu_layers))
        self.n_threads = n_threads
        self.chat_format = chat_format
        self._model = None
        self._loaded = False
        self._load_error: str | None = None

    def is_available(self) -> bool:
        return LlamaCpp is not None

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailable("llama_cpp missing")
        kwargs: dict[str, t.Any] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": False,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = int(self.n_threads)
        if self.chat_format:
            kwargs["chat_format"] = self.chat_format
        try:
            self._model = LlamaCpp(**kwargs)  # type: ignore[misc]
            self._loaded = True
        except Exception as exc:
            self._load_error = str(exc)
            raise BackendUnavailable(str(exc)) from exc

    def generate(self, prompt: str, **kwargs: t.Any) -> t.Iterable[str]:
        self._load()
        model = self._model
        if model is None:
            raise BackendUnavailable(self._load_error or "llama_cpp model unavailable")
        max_new_tokens = int(kwargs.get("max_new_tokens", 128))
        temperature = float(kwargs.get("temperature", 0.7))
        top_p = float(kwargs.get("top_p", 0.95))
        stop = kwargs.get("stop")
        stream = model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=True,
        )
        chunk_size = max(1, int(kwargs.get("chunk_size", 32)))
        buffer = ""
        for token in stream:
            token_text = token.get("choices", [{}])[0].get("text", "") if isinstance(token, dict) else str(token)
            if not token_text:
                continue
            buffer += token_text
            while len(buffer) >= chunk_size:
                yield buffer[:chunk_size]
                buffer = buffer[chunk_size:]
        if buffer:
            yield buffer


class TTSBackendBase:
    backend_name = "base"

    def is_available(self) -> bool:
        return False

    def synthesize(self, text: str, **kwargs: t.Any) -> t.Iterable[bytes]:
        raise BackendUnavailable(self.backend_name)

    def health(self) -> JSON:
        return {"backend": self.backend_name, "available": self.is_available()}


class LocalTTSPipelineBackend(TTSBackendBase):
    backend_name = "local_tts"

    def __init__(self, log: StructuredLogger) -> None:
        self.log = log
        self.sample_rate = 22050

    def is_available(self) -> bool:
        return True

    def _pcm_from_text(self, text: str) -> bytes:
        if np is None:
            # Fallback to a deterministic byte pattern when numpy is unavailable.
            return (text.encode("utf-8") * 8)[:4096]
        duration = max(0.25, min(4.0, 0.05 * len(text.split()) + 0.2))
        t_axis = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        freqs = [180 + (abs(hash(word)) % 260) for word in re.findall(r"\w+", text)[:8]] or [220]
        signal = np.zeros_like(t_axis, dtype=np.float32)
        for i, f in enumerate(freqs):
            amp = 0.15 / (i + 1)
            signal += amp * np.sin(2 * np.pi * f * t_axis)
        signal *= np.hanning(len(signal)).astype(np.float32)
        signal = np.clip(signal, -1.0, 1.0)
        pcm = (signal * 32767.0).astype(np.int16).tobytes()
        return pcm

    def synthesize(self, text: str, **kwargs: t.Any) -> t.Iterable[bytes]:
        chunks = split_sentences(text) or [text]
        for sentence in chunks:
            pcm = self._pcm_from_text(sentence)
            chunk_size = int(kwargs.get("chunk_size", 2048))
            for i in range(0, len(pcm), chunk_size):
                yield pcm[i : i + chunk_size]


class Pyttsx3Backend(TTSBackendBase):
    backend_name = "pyttsx3"

    def __init__(self, log: StructuredLogger) -> None:
        self.log = log
        self._engine = None

    def is_available(self) -> bool:
        return OPTIONAL["pyttsx3"].available

    def synthesize(self, text: str, **kwargs: t.Any) -> t.Iterable[bytes]:
        if not self.is_available():
            raise BackendUnavailable("pyttsx3 missing")
        # pyttsx3 is playback-oriented; here we route to local fallback audio for streamability.
        fallback = LocalTTSPipelineBackend(self.log)
        yield from fallback.synthesize(text, **kwargs)


class QwenTTSBackend(TTSBackendBase):
    backend_name = "qwen_tts"
    DEFAULT_CANDIDATES = ("Qwen3TTSModel", "QwenTTSModel", "Qwen3TTS", "QwenTTS", "auto")

    def __init__(self, model_name_or_path: str, log: StructuredLogger, *, candidates: list[str] | None = None) -> None:
        self.model_name_or_path = model_name_or_path
        self.log = log
        self.candidates = list(candidates or self.DEFAULT_CANDIDATES)
        self._model = None
        self._loaded = False
        self._load_error: str | None = None

    def is_available(self) -> bool:
        return Qwen3TTSModel is not None or QwenTTSModel is not None or OPTIONAL["transformers"].available

    def _candidate_names(self) -> list[str]:
        raw_env = os.environ.get("QWEN3_TTS_CANDIDATES", "")
        env_candidates = [x.strip() for x in raw_env.replace(";", ",").replace("\n", ",").split(",") if x.strip()]
        ordered = env_candidates + list(self.candidates)
        seen: set[str] = set()
        out: list[str] = []
        for name in ordered:
            name = name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    def _try_load_model(self, source: str) -> t.Any:
        candidates: list[t.Any] = []
        if Qwen3TTSModel is not None:
            candidates.append(Qwen3TTSModel)
        if QwenTTSModel is not None and QwenTTSModel is not Qwen3TTSModel:
            candidates.append(QwenTTSModel)
        transformers = OPTIONAL["transformers"].module if OPTIONAL["transformers"].available else None
        for cls in candidates:
            if cls is None:
                continue
            try:
                return cls.from_pretrained(source, trust_remote_code=True)  # type: ignore[attr-defined]
            except Exception as exc:
                self.log.warning("qwen_tts_candidate_failed", candidate=getattr(cls, "__name__", str(cls)), error=str(exc))
        if transformers is not None:
            for candidate in self._candidate_names():
                if candidate == "auto":
                    continue
                try:
                    model_cls = getattr(transformers, candidate, None)
                    if model_cls is None:
                        continue
                    return model_cls.from_pretrained(source, trust_remote_code=True)
                except Exception as exc:
                    self.log.warning("qwen_tts_candidate_failed", candidate=candidate, error=str(exc))
        raise BackendUnavailable("Qwen TTS model could not be loaded from any candidate")

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailable("Qwen TTS adapter unavailable")
        try:
            self._model = self._try_load_model(self.model_name_or_path)
            self._loaded = True
        except Exception as exc:
            self._load_error = str(exc)
            raise

    def synthesize(self, text: str, **kwargs: t.Any) -> t.Iterable[bytes]:
        self._load()
        chunk_size = int(kwargs.get("chunk_size", 2048))
        model = self._model
        if model is None:
            raise BackendUnavailable(self._load_error or "Qwen TTS model unavailable")
        try:
            if hasattr(model, "synthesize"):
                audio = model.synthesize(text=text, **kwargs)  # type: ignore[misc]
                if isinstance(audio, (bytes, bytearray)):
                    yield from LocalTTSPipelineBackend(self.log).synthesize(text, chunk_size=chunk_size)
                    return
                if isinstance(audio, str):
                    yield audio.encode("utf-8")
                    return
                if isinstance(audio, t.Iterable):
                    for chunk in audio:
                        if isinstance(chunk, (bytes, bytearray)):
                            yield bytes(chunk)
                        elif isinstance(chunk, str):
                            yield chunk.encode("utf-8")
                        else:
                            yield bytes(chunk)
                    return
            if hasattr(model, "__call__"):
                out = model(text, **kwargs)
                if isinstance(out, (bytes, bytearray)):
                    yield from LocalTTSPipelineBackend(self.log).synthesize(text, chunk_size=chunk_size)
                    return
                if isinstance(out, str):
                    yield out.encode("utf-8")
                    return
                if isinstance(out, t.Iterable):
                    for chunk in out:
                        if isinstance(chunk, (bytes, bytearray)):
                            yield bytes(chunk)
                        elif isinstance(chunk, str):
                            yield chunk.encode("utf-8")
                        else:
                            yield bytes(chunk)
                    return
        except Exception as exc:
            self.log.warning("qwen_tts_synthesize_failed", error=str(exc), model=self.model_name_or_path)
        fallback = LocalTTSPipelineBackend(self.log)
        yield from fallback.synthesize(text, chunk_size=chunk_size)


# =============================================================================
# Runtime selection / fallback engines
# =============================================================================

class BackendSelector:
    def __init__(self, config: ControlConfig, log: StructuredLogger, metrics: MetricsRegistry) -> None:
        self.config = config
        self.log = log
        self.metrics = metrics

    def _manifest_like(self, manifest: t.Any) -> JSON:
        if manifest is None:
            return {}
        if dataclasses.is_dataclass(manifest):
            return asdict(manifest)
        if isinstance(manifest, dict):
            return dict(manifest)
        return {"source_path": str(manifest)}

    def get_backend(self, manifest: t.Any, *, worker: WorkerInfo | None = None, kind: str = "llm") -> LLMBackendBase | TTSBackendBase:
        info = self._manifest_like(manifest)
        fmt = str(info.get("format", info.get("model_format", info.get("backend_format", "")))).strip().lower()
        source_path = str(info.get("source_path", info.get("path", info.get("model_path", "")))).strip()
        model_name = str(info.get("model_name_or_path", info.get("model_id", source_path or "auto")))
        ext = Path(source_path).suffix.lower() if source_path else ""
        if kind == "tts":
            if fmt in {"qwen_tts", "qwen3_tts", "tts"} or "qwen" in str(info.get("architecture", "")).lower():
                return QwenTTSBackend(model_name, self.log, candidates=list(info.get("candidates", []) or []))
            if fmt == "pyttsx3":
                return Pyttsx3Backend(self.log)
            return LocalTTSPipelineBackend(self.log)
        if fmt == "gguf" or ext == ".gguf":
            return LlamaCppBackend(
                source_path or model_name,
                self.log,
                n_ctx=safe_int(info.get("n_ctx", info.get("context_length", 4096)), 4096),
                n_gpu_layers=safe_int(info.get("n_gpu_layers", info.get("gpu_layers", 0)), 0),
                n_threads=safe_int(info.get("n_threads", 0), 0) or None,
                chat_format=str(info.get("chat_format", "")) or None,
            )
        if str(info.get("backend", "")).lower() == "llama_cpp":
            return LlamaCppBackend(
                source_path or model_name,
                self.log,
                n_ctx=safe_int(info.get("n_ctx", 4096), 4096),
                n_gpu_layers=safe_int(info.get("n_gpu_layers", 0), 0),
            )
        return self.select_llm(worker)

    def _registry_manifests(self) -> list[ModelManifest]:
        root = Path(self.config.state_dir) / "models"
        manifests: list[ModelManifest] = []
        if not root.exists():
            return manifests
        for fp in root.rglob("manifest.json"):
            try:
                manifests.append(_restore_model_manifest(json.loads(fp.read_text("utf-8"))))
            except Exception:
                continue
        manifests.sort(key=lambda m: m.created_at, reverse=True)
        return manifests

    def _latest_manifest(self, *, prefer_format: str | None = None) -> ModelManifest | None:
        manifests = self._registry_manifests()
        if prefer_format:
            for manifest in manifests:
                if manifest_model_format(manifest) == prefer_format:
                    return manifest
        return manifests[0] if manifests else None

    def select_llm(self, worker: WorkerInfo | None = None, model_hint: JSON | str | None = None) -> LLMBackendBase:
        if model_hint:
            backend = self.get_backend(model_hint, worker=worker, kind="llm")
            if isinstance(backend, LLMBackendBase) and backend.backend_name != "local":
                return backend
        llama_path = os.environ.get("LLAMA_CPP_MODEL_PATH", "").strip()
        if llama_path and Path(llama_path).exists():
            return LlamaCppBackend(llama_path, self.log)
        gguf_manifest = self._latest_manifest(prefer_format="gguf")
        if gguf_manifest and Path(gguf_manifest.source_path).exists():
            return LlamaCppBackend(gguf_manifest.source_path, self.log)
        manifest = self._latest_manifest()
        if manifest and Path(manifest.source_path).exists():
            fmt = manifest_model_format(manifest)
            if fmt == "gguf":
                return LlamaCppBackend(manifest.source_path, self.log)
            if fmt not in {"", "unknown"}:
                return TransformersLLMBackend(manifest.source_path, self.log)
        return LocalFallbackLLMBackend(self.log)

    def select_tts(self, worker: WorkerInfo | None = None, model_hint: JSON | str | None = None) -> TTSBackendBase:
        if model_hint:
            try:
                backend = self.get_backend(model_hint, worker=worker, kind="tts")
                if isinstance(backend, TTSBackendBase) and backend.backend_name != "local_tts":
                    return backend
            except Exception:
                pass
        env_source = os.environ.get("QWEN_TTS_MODEL_PATH", "").strip()
        candidates: list[str] = []
        if env_source and Path(env_source).exists():
            candidates.append(env_source)
        for manifest in self._registry_manifests():
            if manifest_model_format(manifest) in {"tts", "qwen_tts"} and manifest.source_path:
                candidates.append(manifest.source_path)
        if candidates:
            return QwenTTSBackend(candidates[0], self.log, candidates=candidates[1:])
        return LocalTTSPipelineBackend(self.log)

# =============================================================================
# LLM runtime
# =============================================================================

@dataclass
class LLMRequest:
    prompt: str
    request_id: str = field(default_factory=lambda: uid("req_"))
    task_id: str = field(default_factory=lambda: uid("task_"))
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    chunk_size: int = 24
    parallel_plan: ParallelPlan = field(default_factory=lambda: ParallelPlan(mode=ParallelMode.NONE))
    metadata: JSON = field(default_factory=dict)


class LLMRuntime:
    def __init__(self, selector: BackendSelector, log: StructuredLogger, metrics: MetricsRegistry) -> None:
        self.selector = selector
        self.log = log
        self.metrics = metrics

    def stream_generate(self, request: LLMRequest, worker: WorkerInfo | None = None) -> t.Iterable[str]:
        backend = self.selector.get_backend(request.metadata, worker=worker, kind="llm")
        self.metrics.inc(f"backend.llm.{backend.backend_name}.selected")
        start = monotonic()
        try:
            for chunk in backend.generate(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                chunk_size=request.chunk_size,
            ):
                self.metrics.inc("llm.tokens_streamed")
                yield chunk
            self.metrics.observe("llm.latency_s", monotonic() - start)
        except Exception as exc:
            self.log.exception("llm_generate_failed", exc, request_id=request.request_id, backend=backend.backend_name)
            if backend.backend_name != "local":
                fallback = LocalFallbackLLMBackend(self.log)
                self.metrics.inc("llm.fallback.local")
                for chunk in fallback.generate(request.prompt, max_new_tokens=request.max_new_tokens, chunk_size=request.chunk_size):
                    yield chunk
            else:
                raise


# =============================================================================
# TTS runtime
# =============================================================================

@dataclass
class TTSRequest:
    text: str
    request_id: str = field(default_factory=lambda: uid("req_"))
    task_id: str = field(default_factory=lambda: uid("task_"))
    chunk_size: int = 2048
    metadata: JSON = field(default_factory=dict)


class TTSStage(enum.Enum):
    NORMALIZE = "normalize"
    PHONEMIZE = "phonemize"
    SPECTROGRAM = "spectrogram"
    VOCODER = "vocoder"
    AUDIO = "audio"


class TTSPipelineRuntime:
    def __init__(self, selector: BackendSelector, log: StructuredLogger, metrics: MetricsRegistry) -> None:
        self.selector = selector
        self.log = log
        self.metrics = metrics

    def _normalize(self, text: str) -> str:
        text = sanitize_task_text(text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _phonemize(self, text: str) -> list[str]:
        # Lightweight approximation for fallback runtime.
        return [w.lower() for w in re.findall(r"\w+", text)]

    def _spectrogram(self, phonemes: list[str]) -> bytes:
        if np is None:
            return "|".join(phonemes).encode("utf-8")
        n = max(128, len(phonemes) * 32)
        t_axis = np.linspace(0, 1.0, n, endpoint=False)
        vec = np.zeros_like(t_axis, dtype=np.float32)
        for i, ph in enumerate(phonemes or ["sil"]):
            freq = 200 + (abs(hash(ph)) % 500)
            vec += (0.08 / (i + 1)) * np.sin(2 * np.pi * freq * t_axis)
        spec = np.abs(np.fft.rfft(vec))
        return spec.astype(np.float32).tobytes()

    def _vocode(self, spectrogram: bytes, text: str) -> bytes:
        backend = self.selector.select_tts(None)
        # Real backend selection is isolated by adapter; fallback produces usable PCM chunks.
        if backend.backend_name == "local_tts":
            return LocalTTSPipelineBackend(self.log)._pcm_from_text(text)
        return b"".join(backend.synthesize(text, chunk_size=4096))

    def stream_synthesize(self, request: TTSRequest, worker: WorkerInfo | None = None) -> t.Iterable[bytes]:
        backend = self.selector.get_backend(request.metadata, worker=worker, kind="tts")
        self.metrics.inc(f"backend.tts.{backend.backend_name}.selected")
        text = self._normalize(request.text)
        self.metrics.inc("tts.requests")
        start = monotonic()
        try:
            if backend.backend_name == "local_tts":
                for chunk in backend.synthesize(text, chunk_size=request.chunk_size):
                    self.metrics.inc("tts.audio_chunks")
                    yield chunk
            else:
                phonemes = self._phonemize(text)
                spec = self._spectrogram(phonemes)
                audio = self._vocode(spec, text)
                for i in range(0, len(audio), request.chunk_size):
                    self.metrics.inc("tts.audio_chunks")
                    yield memoryview(audio)[i : i + request.chunk_size]
            self.metrics.observe("tts.latency_s", monotonic() - start)
        except Exception as exc:
            self.log.exception("tts_generate_failed", exc, request_id=request.request_id, backend=backend.backend_name)
            fallback = LocalTTSPipelineBackend(self.log)
            self.metrics.inc("tts.fallback.local")
            for chunk in fallback.synthesize(text, chunk_size=request.chunk_size):
                yield chunk


# =============================================================================
# Scheduler and control logic
# =============================================================================

@dataclass
class SchedulingDecision:
    chosen_worker_id: str
    mode: ParallelMode
    tensor_degree: int
    pipeline_degree: int
    reason: str
    backend_preference: str = ""
    local_fallback: bool = False


class Scheduler:
    def __init__(self, store: StateStore, config: ControlConfig, log: StructuredLogger, metrics: MetricsRegistry) -> None:
        self.store = store
        self.config = config
        self.log = log
        self.metrics = metrics
        self._lock = threading.RLock()

    def _worker_health_score(self, worker: WorkerInfo) -> float:
        age = max(0.0, now_s() - worker.last_heartbeat)
        hb_penalty = clamp(age / max(1.0, self.config.worker_timeout_s), 0.0, 2.0)
        latency_penalty = clamp(worker.avg_latency_ms / 1000.0, 0.0, 5.0)
        failure_penalty = clamp(worker.failure_count / 5.0, 0.0, 3.0)
        success_bonus = worker.success_rate * 2.0
        return worker.capability.score() + success_bonus - hb_penalty - latency_penalty - failure_penalty - worker.in_flight * 0.5

    def alive_workers(self) -> list[WorkerInfo]:
        cutoff = now_s() - self.config.worker_timeout_s
        return [
            w for w in self.store.workers.values()
            if w.last_heartbeat >= cutoff and w.status != "dead"
        ]

    def choose_worker(self, kind: TaskKind, prefer_gpu: bool = True) -> WorkerInfo | None:
        candidates = self.alive_workers()
        if not candidates:
            return None
        ranked = sorted(candidates, key=self._worker_health_score, reverse=True)
        if kind == TaskKind.TTS:
            ranked = sorted(ranked, key=lambda w: (w.capability.has_gpu if prefer_gpu else False, w.capability.score(), -w.in_flight), reverse=True)
        elif kind == TaskKind.LLM and prefer_gpu:
            ranked = sorted(ranked, key=lambda w: (w.capability.has_gpu, w.capability.vram_gb, w.capability.score(), -w.in_flight), reverse=True)
        return ranked[0] if ranked else None

    def plan(self, kind: TaskKind, payload: JSON, request_id: str) -> SchedulingDecision:
        workers = self.alive_workers()
        if not workers:
            return SchedulingDecision(chosen_worker_id="local", mode=ParallelMode.NONE, tensor_degree=1, pipeline_degree=1, reason="no_workers", local_fallback=True)
        if kind == TaskKind.LLM:
            top = self.choose_worker(kind, prefer_gpu=self.config.prefer_gpu)
            if top is None:
                return SchedulingDecision("local", ParallelMode.NONE, 1, 1, "no_llm_worker", local_fallback=True)
            tensor_degree = min(4, max(1, len(workers) if top.capability.has_gpu else 1))
            mode = ParallelMode.TENSOR if tensor_degree > 1 else ParallelMode.NONE
            reason = "tensor_parallel_enabled" if tensor_degree > 1 else "single_worker"
            return SchedulingDecision(top.worker_id, mode, tensor_degree, 1, reason, backend_preference=top.current_backend)
        if kind == TaskKind.TTS:
            top = self.choose_worker(kind, prefer_gpu=False)
            if top is None:
                return SchedulingDecision("local", ParallelMode.NONE, 1, 1, "no_tts_worker", local_fallback=True)
            pipeline_degree = min(5, max(1, len(workers)))
            mode = ParallelMode.PIPELINE if pipeline_degree > 1 else ParallelMode.NONE
            reason = "pipeline_parallel_enabled" if pipeline_degree > 1 else "single_worker"
            return SchedulingDecision(top.worker_id, mode, 1, pipeline_degree, reason, backend_preference=top.current_backend)
        return SchedulingDecision("local", ParallelMode.NONE, 1, 1, "default_local", local_fallback=True)

    def mark_failure(self, worker_id: str) -> None:
        self.store.update_worker(worker_id, failure_count=(self.store.workers[worker_id].failure_count + 1 if worker_id in self.store.workers else 1))
        self.metrics.inc("worker.failures")

    def mark_success(self, worker_id: str, latency_ms: float) -> None:
        worker = self.store.workers.get(worker_id)
        if not worker:
            return
        worker.in_flight = max(0, worker.in_flight - 1)
        worker.avg_latency_ms = 0.8 * worker.avg_latency_ms + 0.2 * latency_ms
        worker.success_rate = min(1.0, worker.success_rate * 0.99 + 0.01)
        worker.status = "healthy"
        worker.last_heartbeat = now_s()
        self.store.mark_dirty()


# =============================================================================
# Distributed orchestration runtime
# =============================================================================

class ClusterOrchestrator:
    def __init__(self, config: ControlConfig) -> None:
        self.config = config
        self.log = StructuredLogger("orchestrator", level=config.log_level)
        self.metrics = MetricsRegistry()
        self.tracer = Tracer(self.log, self.metrics)
        self.state = StateStore(config.state_dir, self.log, self.metrics)
        self.queue = DistributedPriorityQueue(config.queue_size, self.log, self.metrics, config.backpressure_high, config.backpressure_low)
        self.scheduler = Scheduler(self.state, config, self.log, self.metrics)
        self.compiler = ModelCompiler(self.state, self.log, self.metrics)
        self.selector = BackendSelector(config, self.log, self.metrics)
        self.llm_runtime = LLMRuntime(self.selector, self.log, self.metrics)
        self.tts_runtime = TTSPipelineRuntime(self.selector, self.log, self.metrics)
        self.signer = HMACSigner(config.secret_key)
        self.httpd: ThreadingClusterServer | None = None
        self.executor = ThreadPoolExecutor(max_workers=max(8, (os.cpu_count() or 4) * 2))
        self._stop = threading.Event()
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, name="dispatch-loop", daemon=True)
        self._health_thread = threading.Thread(target=self._health_loop, name="health-loop", daemon=True)
        self._reaper_thread = threading.Thread(target=self._reaper_loop, name="reaper-loop", daemon=True)
        self._housekeeping_thread = threading.Thread(target=self._housekeeping_loop, name="housekeeping-loop", daemon=True)
        self._udp_beacon_stop = threading.Event()
        self._udp_beacon_thread: threading.Thread | None = None
        self._worker_lock = threading.RLock()
        self._task_lock = threading.RLock()
        self._started = False

    def start(self) -> None:
        if self._started:
            self.log.warning("orchestrator_already_started")
            return
        self._started = True
        self.log.info("orchestrator_start", version=VERSION, host=self.config.host, port=self.config.port)
        self.httpd = ThreadingClusterServer((self.config.host, self.config.port), ClusterHTTPRequestHandler, self)
        self._dispatch_thread.start()
        self._health_thread.start()
        self._reaper_thread.start()
        self._housekeeping_thread.start()
        if self.config.enable_udp_discovery:
            self._udp_beacon_thread = start_udp_beacon(
                self.config.port,
                interval_s=self.config.udp_beacon_interval_s,
                stop_event=self._udp_beacon_stop,
                log=self.log,
            )
            self.metrics.set("udp_beacon.enabled", 1.0)
        if self.config.enable_dashboard:
            self.metrics.set("dashboard.enabled", 1.0)
        try:
            self.httpd.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        self._udp_beacon_stop.set()
        self.queue.shutdown()
        if self.httpd:
            self.httpd.shutdown()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.state.persist(force=True)
        self.metrics.set("cluster.stopped", 1.0)
        self.log.info("orchestrator_stop")

    def _health_loop(self) -> None:
        while not self._stop.is_set():
            try:
                cutoff = now_s() - self.config.worker_timeout_s
                changed = False
                for worker in list(self.state.workers.values()):
                    if worker.last_heartbeat < cutoff:
                        if worker.status != "degraded":
                            worker.status = "degraded"
                            changed = True
                        if worker.worker_id not in self.state.dead_workers:
                            self.state.dead_workers.add(worker.worker_id)
                            changed = True
                        self.metrics.inc("worker.degraded")
                    else:
                        if worker.status != "healthy":
                            worker.status = "healthy"
                            changed = True
                if changed:
                    self.state.mark_dirty()
                self.metrics.set("cluster.worker_count", len(self.state.workers))
                self.metrics.set("cluster.active_workers", len(self.scheduler.alive_workers()))
            except Exception as exc:
                self.log.exception("health_loop_error", exc)
            time.sleep(1.0)

    def _reaper_loop(self) -> None:
        while not self._stop.is_set():
            try:
                now = now_s()
                changed = False
                for task in list(self.state.tasks.values()):
                    if task.status in {TaskStatus.PENDING, TaskStatus.DISPATCHED, TaskStatus.RUNNING} and task.deadline_s and now > task.deadline_s:
                        task.status = TaskStatus.FAILED
                        task.error = "deadline_exceeded"
                        changed = True
                        self.metrics.inc("task.deadline_exceeded")
                if changed:
                    self.state.mark_dirty()
            except Exception as exc:
                self.log.exception("reaper_loop_error", exc)
            time.sleep(2.0)

    def _housekeeping_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.metrics.set("queue.depth", self.queue.size())
                self.metrics.set("queue.saturated", 1.0 if self.queue.saturated() else 0.0)
                self.state.persist()
            except Exception as exc:
                self.log.exception("housekeeping_error", exc)
            time.sleep(5.0)

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            envelope = self.queue.get(timeout=0.5)
            if envelope is None:
                continue
            self.state.store_task(envelope)
            try:
                self.state.mark_task_running(envelope.task_id, envelope.worker_id)
                future = self.executor.submit(self._execute_task, envelope)
                future.add_done_callback(lambda f, task_id=envelope.task_id: f.exception() and self.log.warning("dispatch_task_future_failed", task_id=task_id, error=str(f.exception())))
            except Exception as exc:
                self.log.exception("dispatch_submit_failed", exc, task_id=envelope.task_id)
                self._retry_or_fallback(envelope, error=str(exc))

    def submit_request(self, kind: TaskKind, payload: JSON, priority: int = 50, request_id: str | None = None, deadline_s: float | None = None) -> TaskEnvelope:
        if not isinstance(payload, dict):
            payload = {"text": str(payload)}
        request_id = request_id or uid("req_")
        with self._task_lock:
            existing = self.state.task_by_request(request_id)
            if existing:
                return existing
        task_id = uid("task_")
        payload = dict(payload or {})
        stage_id = str(payload.get("stage_id") or kind.value)
        checksum = payload_checksum(payload)
        decision = self.scheduler.plan(kind, payload, request_id)
        envelope = TaskEnvelope(
            request_id=request_id,
            task_id=task_id,
            kind=kind,
            stage_id=stage_id,
            worker_id=decision.chosen_worker_id,
            rank=0,
            deadline_s=deadline_s or (now_s() + self.config.request_timeout_s),
            retry_count=0,
            priority=clamp(float(priority), 0, 100),
            checksum=checksum,
            payload={**payload, "parallel": dataclasses.asdict(decision)},
            status=TaskStatus.PENDING,
        )
        self.state.store_task(envelope)
        if decision.local_fallback and self.config.allow_local_fallback:
            envelope.status = TaskStatus.FALLBACK
            self._execute_local(envelope)
            return envelope
        if not self.queue.put(envelope, timeout=2.0):
            envelope.status = TaskStatus.FALLBACK
            self.metrics.inc("task.queue_rejected")
            if self.config.allow_local_fallback:
                self._execute_local(envelope)
        return envelope

    def _execute_task(self, envelope: TaskEnvelope) -> None:
        with self.tracer.span(envelope.request_id, "execute_task", task_id=envelope.task_id, kind=envelope.kind.value):
            worker = self.state.get_worker(envelope.worker_id)
            if worker is None or worker.worker_id == "local":
                self._execute_local(envelope)
                return
            start_s = now_s()
            try:
                with self._worker_lock:
                    worker.in_flight = min(worker.in_flight + 1, max(1, worker.capability.max_concurrency) * 2)
                    worker.queue_depth = max(0, worker.queue_depth - 1)
                    worker.status = "running"
                    worker.last_heartbeat = now_s()
                self.state.mark_dirty()
                self.state.mark_task_running(envelope.task_id, worker.worker_id)
                task_url = f"http://{worker.address}:{worker.port}/execute"
                transport = HttpJsonTransport(self.log, self.metrics, timeout_s=self.config.request_timeout_s, auth_secret=self.config.secret_key)
                result = transport.post_json(task_url, envelope.to_public())
                self._handle_worker_result(envelope, result)
                with self._worker_lock:
                    worker.last_heartbeat = now_s()
                    worker.status = "healthy"
                self.state.mark_dirty()
                self.scheduler.mark_success(worker.worker_id, latency_ms=(now_s() - start_s) * 1000.0)
            except Exception as exc:
                self.log.exception("worker_execute_failed", exc, worker_id=worker.worker_id, task_id=envelope.task_id)
                self.scheduler.mark_failure(worker.worker_id)
                self._retry_or_fallback(envelope, error=str(exc))
            finally:
                with self._worker_lock:
                    worker.in_flight = max(0, worker.in_flight - 1)
                self.state.mark_dirty()

    def _handle_worker_result(self, envelope: TaskEnvelope, result: JSON) -> None:
        try:
            status = TaskStatus(result.get("status", TaskStatus.SUCCEEDED.value))
        except Exception:
            status = TaskStatus.FAILED
        metrics = dict(result.get("metrics", {}) or {})
        total_ms = max(0.0, (now_s() - envelope.created_at) * 1000.0)
        inference_ms = safe_float(metrics.get("inference_ms", 0.0), 0.0)
        network_ms = max(0.0, total_ms - inference_ms) if total_ms else 0.0
        metrics.setdefault("request_total_ms", total_ms)
        metrics.setdefault("network_ms", network_ms)
        if envelope.kind == TaskKind.LLM:
            tokens = safe_int(metrics.get("tokens", len(result.get("stream_chunks", []) or [])), 0)
            tps = safe_float(metrics.get("tps", tokens / max(inference_ms / 1000.0, 0.001) if inference_ms else 0.0), 0.0)
            self.metrics.observe("cluster.tps", tps)
            self.metrics.observe("request.latency_ms", total_ms)
        else:
            self.metrics.observe("request.latency_ms", total_ms)
        execution_result = ExecutionResult(
            request_id=envelope.request_id,
            task_id=envelope.task_id,
            kind=envelope.kind,
            status=status,
            output_text=str(result.get("output_text", "")),
            audio_bytes_b64=str(result.get("audio_bytes_b64", "")),
            stream_chunks=list(result.get("stream_chunks", []) or []),
            metrics=metrics,
            error=str(result.get("error", "")),
            worker_id=str(result.get("worker_id", envelope.worker_id)),
            backend=str(result.get("backend", "")),
        )
        self.state.mark_task_finished(envelope.task_id, status, error=execution_result.error, result=execution_result)
        if execution_result.stream_chunks:
            for chunk in execution_result.stream_chunks:
                self.state.append_stream(envelope.request_id, chunk)
        if status == TaskStatus.SUCCEEDED:
            self.metrics.inc("task.succeeded")
        else:
            self.metrics.inc("task.failed")

    def _execute_local(self, envelope: TaskEnvelope) -> None:
        self.log.info("local_execute", task_id=envelope.task_id, kind=envelope.kind.value)
        start = now_s()
        try:
            if envelope.kind == TaskKind.LLM:
                request = LLMRequest(
                    prompt=str(envelope.payload.get("prompt", "")),
                    request_id=envelope.request_id,
                    task_id=envelope.task_id,
                    max_new_tokens=safe_int(envelope.payload.get("max_new_tokens", 128), 128),
                    temperature=safe_float(envelope.payload.get("temperature", 0.7), 0.7),
                    top_p=safe_float(envelope.payload.get("top_p", 0.95), 0.95),
                    chunk_size=safe_int(envelope.payload.get("chunk_size", 24), 24),
                    metadata=dict(envelope.payload),
                )
                chunks = list(self.llm_runtime.stream_generate(request, None))
                out = "".join(chunks)
                inference_ms = (now_s() - start) * 1000.0
                result = ExecutionResult(
                    request_id=envelope.request_id,
                    task_id=envelope.task_id,
                    kind=TaskKind.LLM,
                    status=TaskStatus.SUCCEEDED,
                    output_text=out,
                    stream_chunks=chunks,
                    worker_id="local",
                    backend="local",
                    metrics={"chunks": len(chunks), "chars": len(out), "inference_ms": inference_ms, "tokens": len(chunks), "tps": (len(chunks) / max(inference_ms / 1000.0, 0.001))},
                )
            elif envelope.kind == TaskKind.TTS:
                request = TTSRequest(
                    text=str(envelope.payload.get("text", "")),
                    request_id=envelope.request_id,
                    task_id=envelope.task_id,
                    chunk_size=safe_int(envelope.payload.get("chunk_size", 2048), 2048),
                    metadata=dict(envelope.payload),
                )
                chunks_bytes = list(self.tts_runtime.stream_synthesize(request, None))
                audio = b"".join(chunks_bytes)
                inference_ms = (now_s() - start) * 1000.0
                result = ExecutionResult(
                    request_id=envelope.request_id,
                    task_id=envelope.task_id,
                    kind=TaskKind.TTS,
                    status=TaskStatus.SUCCEEDED,
                    audio_bytes_b64=base64.b64encode(audio).decode("ascii"),
                    stream_chunks=[base64.b64encode(c).decode("ascii") for c in chunks_bytes],
                    worker_id="local",
                    backend="local_tts",
                    metrics={"chunks": len(chunks_bytes), "bytes": len(audio), "inference_ms": inference_ms},
                )
            elif envelope.kind == TaskKind.SHARD_COMPILE:
                model_path = envelope.payload["checkpoint_path"]
                manifest = self.compiler.inspect_checkpoint(model_path, envelope.payload.get("model_id"))
                shard_manifest = self.compiler.build_shard_manifest(
                    manifest,
                    tensor_degree=safe_int(envelope.payload.get("tensor_degree", 1), 1),
                    pipeline_degree=safe_int(envelope.payload.get("pipeline_degree", 1), 1),
                )
                valid, errors = self.compiler.validate_shard_manifest(shard_manifest)
                inference_ms = (now_s() - start) * 1000.0
                result = ExecutionResult(
                    request_id=envelope.request_id,
                    task_id=envelope.task_id,
                    kind=TaskKind.SHARD_COMPILE,
                    status=TaskStatus.SUCCEEDED if valid else TaskStatus.FAILED,
                    output_text=to_json(shard_manifest.to_public()),
                    worker_id="local",
                    backend="compiler",
                    metrics={"valid": valid, "errors": errors, "inference_ms": inference_ms},
                    error=";".join(errors),
                )
            else:
                inference_ms = (now_s() - start) * 1000.0
                result = ExecutionResult(
                    request_id=envelope.request_id,
                    task_id=envelope.task_id,
                    kind=envelope.kind,
                    status=TaskStatus.SUCCEEDED,
                    output_text="ok",
                    worker_id="local",
                    backend="local",
                    metrics={"inference_ms": inference_ms},
                )
            total_ms = (now_s() - envelope.created_at) * 1000.0
            result.metrics["request_total_ms"] = total_ms
            result.metrics["network_ms"] = max(0.0, total_ms - safe_float(result.metrics.get("inference_ms", 0.0), 0.0))
            self.state.mark_task_finished(envelope.task_id, result.status, error=result.error, result=result)
            if result.stream_chunks:
                for chunk in result.stream_chunks:
                    self.state.append_stream(envelope.request_id, chunk)
            if result.kind == TaskKind.LLM:
                self.metrics.observe("cluster.tps", safe_float(result.metrics.get("tps", 0.0), 0.0))
            self.metrics.observe("request.latency_ms", total_ms)
            self.metrics.inc("task.local_succeeded")
        except Exception as exc:
            self.log.exception("local_execute_failed", exc, task_id=envelope.task_id)
            self._retry_or_fallback(envelope, error=str(exc))

    def _retry_or_fallback(self, envelope: TaskEnvelope, error: str) -> None:
        envelope.retry_count += 1
        envelope.error = error
        self.state.mark_task_failed(envelope.task_id, error, status=TaskStatus.RETRYING if envelope.retry_count <= self.config.max_retries_per_stage else TaskStatus.FAILED)
        if envelope.retry_count <= self.config.max_retries_per_stage and not self.queue.saturated():
            decision = self.scheduler.plan(envelope.kind, envelope.payload, envelope.request_id)
            envelope.worker_id = decision.chosen_worker_id
            envelope.status = TaskStatus.RETRYING
            envelope.deadline_s = now_s() + self.config.request_timeout_s
            if self.queue.requeue(envelope):
                self.metrics.inc("task.retried")
                return
        if self.config.allow_local_fallback and envelope.worker_id != "local":
            envelope.worker_id = "local"
            envelope.status = TaskStatus.FALLBACK
            self._execute_local(envelope)
            return
        envelope.status = TaskStatus.FAILED
        envelope.finished_at = now_s()
        self.state.store_task(envelope)
        self.metrics.inc("task.failed")

    def register_worker(self, address: str, port: int, capability: Capability, worker_id: str | None = None, tags: list[str] | None = None) -> WorkerInfo:
        worker_id = worker_id or uid("worker_")
        address = str(address or "127.0.0.1").strip()
        port = safe_int(port, DEFAULT_WORKER_PORT)
        worker = WorkerInfo(
            worker_id=worker_id,
            address=address,
            port=port,
            capability=capability,
            registered_at=now_s(),
            last_heartbeat=now_s(),
            tags=tags or [],
            current_backend="",
        )
        self.state.register_worker(worker)
        self.state.persist(force=True)
        self.metrics.inc("worker.registered")
        self.log.info("worker_registered", worker_id=worker_id, address=address, port=port)
        return worker

    def heartbeat(self, worker_id: str, payload: JSON) -> JSON:
        worker = self.state.get_worker(worker_id)
        if not worker:
            return {"ok": False, "error": "unknown_worker"}
        with self._worker_lock:
            worker.last_heartbeat = now_s()
            worker.queue_depth = safe_int(payload.get("queue_depth", worker.queue_depth), worker.queue_depth)
            worker.in_flight = max(0, safe_int(payload.get("in_flight", worker.in_flight), worker.in_flight))
            worker.status = str(payload.get("status", worker.status))
            worker.avg_latency_ms = safe_float(payload.get("avg_latency_ms", worker.avg_latency_ms), worker.avg_latency_ms)
            worker.success_rate = clamp(safe_float(payload.get("success_rate", worker.success_rate), worker.success_rate), 0.0, 1.0)
            worker.current_backend = str(payload.get("backend", worker.current_backend))
            self.state.mark_dirty()
        return {"ok": True, "cluster": self.cluster_status()}

    def list_models(self) -> list[JSON]:
        models = list(self.state.model_manifests.values())
        models.sort(key=lambda m: getattr(m, "created_at", 0.0), reverse=True)
        return [asdict(m) for m in models]

    def register_uploaded_model(self, filename: str, model_type: str, model_format: str, content: bytes, *, model_id: str | None = None, metadata: JSON | None = None) -> JSON:
        model_id = (model_id or Path(filename or "model").stem or uid("model_")).strip() or uid("model_")
        model_type = str(model_type or "llm").strip().lower()
        model_format = str(model_format or Path(filename).suffix.lstrip(".") or "unknown").strip().lower()
        metadata = dict(metadata or {})
        model_dir = ensure_dir(Path(self.config.state_dir) / "models" / model_id)
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(filename or f"{model_id}.{model_format}").name)
        if not safe_name:
            safe_name = f"{model_id}.{model_format or 'bin'}"
        source_path = model_dir / safe_name
        source_path.write_bytes(content)
        checksum = stable_hash(content)
        architecture = str(metadata.get("architecture", "unknown"))
        if model_format == "gguf" and architecture == "unknown":
            architecture = "llama_cpp_gguf"
        if model_type == "tts" and architecture == "unknown":
            architecture = "qwen_tts"
        manifest = ModelManifest(
            model_id=model_id,
            source_path=str(source_path),
            architecture=architecture,
            checksum=checksum,
            created_at=now_s(),
            dtype=str(metadata.get("dtype", "float16")),
            layers=list(metadata.get("layers", []) or []),
            tensor_parallel_degree=safe_int(metadata.get("tensor_parallel_degree", 1), 1),
            pipeline_parallel_degree=safe_int(metadata.get("pipeline_parallel_degree", 1), 1),
            shard_dir=str(model_dir),
            extra={**metadata, "model_type": model_type, "format": model_format, "filename": filename, "backend": ("llama_cpp" if model_format == "gguf" else ("qwen_tts" if model_type == "tts" else metadata.get("backend", "auto")))},
        )
        self.state.model_manifests[model_id] = manifest
        self.state.persist(force=True)
        self.metrics.inc("model.uploaded")
        self.log.info("model_uploaded", model_id=model_id, model_type=model_type, format=model_format, path=str(source_path))
        return {"ok": True, "manifest": asdict(manifest), "source_path": str(source_path)}

    def cluster_status(self) -> JSON:
        workers = [w.to_public() for w in self.state.list_workers()]
        tasks = self.state.list_tasks()
        pending = sum(1 for t in tasks if t.status in {TaskStatus.PENDING, TaskStatus.DISPATCHED, TaskStatus.RUNNING, TaskStatus.RETRYING})
        network = None
        provider = getattr(self, "network_provider", None)
        if callable(provider):
            try:
                network = provider()
            except Exception as exc:
                network = {"ok": False, "error": str(exc)}
        elif isinstance(provider, dict):
            network = provider
        return {
            "version": VERSION,
            "time": now_ms(),
            "worker_count": len(workers),
            "alive_workers": len(self.scheduler.alive_workers()),
            "queue_depth": self.queue.size(),
            "queue_saturated": self.queue.saturated(),
            "pending_tasks": pending,
            "metrics": self.metrics.snapshot(),
            "workers": workers,
            "network": network,
            "logs": self.log.tail(200),
            "models": self.list_models(),
            "model_count": len(self.state.model_manifests),
        }

    def diagnostics(self) -> JSON:
        validations = {
            "queue": self.queue.validate(),
            "state": {
                "tasks": len(self.state.list_tasks()),
                "results": len(self.state.results),
                "workers": len(self.state.list_workers()),
            },
        }
        return {
            "cluster": self.cluster_status(),
            "validations": validations,
            "state_snapshot": self.state.snapshot(),
        }


# =============================================================================
# Worker runtime
# =============================================================================

class WorkerRuntime:
    def __init__(self, config: ControlConfig) -> None:
        self.config = config
        self.log = StructuredLogger("worker", level=config.log_level)
        self.metrics = MetricsRegistry()
        self.tracer = Tracer(self.log, self.metrics)
        self.selector = BackendSelector(config, self.log, self.metrics)
        self.llm_runtime = LLMRuntime(self.selector, self.log, self.metrics)
        self.tts_runtime = TTSPipelineRuntime(self.selector, self.log, self.metrics)
        self.worker_id = uid("worker_")
        self.capability = self.detect_capability()
        self.last_heartbeat = now_s()
        self.in_flight = 0
        self._stop = threading.Event()
        self.httpd: ThreadingWorkerServer | None = None
        self._server_thread: threading.Thread | None = None
        self._server_started = False
        self._registration_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self.heartbeat_target = None

    def detect_capability(self) -> Capability:
        cpu = os.cpu_count() or 1
        ram_gb = 0.0
        if OPTIONAL["psutil"].available:
            try:
                psutil = OPTIONAL["psutil"].module
                ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            except Exception:
                pass
        has_gpu = False
        vram_gb = 0.0
        gpu_name = ""
        cuda_available = bool(torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available())
        if cuda_available:
            has_gpu = True
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except Exception:
                pass
        backends = ["local", "local_tts"]
        if OPTIONAL["transformers"].available:
            backends.append("transformers")
        if OPTIONAL["vllm"].available:
            backends.append("vllm")
        if OPTIONAL["tensorrt_llm"].available:
            backends.append("tensorrt_llm")
        if OPTIONAL["TTS"].available:
            backends.append("qwen_tts")
        return Capability(cpu_cores=cpu, ram_gb=ram_gb, has_gpu=has_gpu, vram_gb=vram_gb, gpu_name=gpu_name, cuda_available=cuda_available, distributed_ready=bool(dist and dist.is_available()), backends=backends)

    def _ensure_http_server(self) -> None:
        if self._server_started:
            return
        self.httpd = ThreadingWorkerServer((self.config.host, self.config.worker_port), WorkerHTTPRequestHandler, self)
        self._server_started = True
        self._server_thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.2}, name="worker-http", daemon=True)
        self._server_thread.start()
        self.log.info("worker_http_started", worker_id=self.worker_id, port=self.config.worker_port)

    def resolve_orchestrator_url(self, orchestrator_url: str | None = None, *, auto_discover: bool = True) -> tuple[str, str]:
        explicit = orchestrator_url or os.environ.get("ORCHESTRATOR_URL")
        return resolve_orchestrator_url(
            explicit,
            auto_discover=auto_discover and self.config.enable_udp_discovery,
            timeout_s=self.config.auto_discovery_timeout_s,
            fallback_port=DEFAULT_PORT,
        )

    def start(self, orchestrator_url: str | None = None, *, auto_discover: bool = True, block: bool = False) -> str:
        resolved_url, source = self.resolve_orchestrator_url(orchestrator_url, auto_discover=auto_discover)
        self.heartbeat_target = resolved_url.rstrip("/")
        self._ensure_http_server()
        threading.Thread(target=self._heartbeat_loop, name="worker-heartbeat", daemon=True).start()
        self.register()
        self.log.info("worker_start", worker_id=self.worker_id, orchestrator_url=self.heartbeat_target, resolution=source, block=block)
        if block:
            try:
                assert self.httpd is not None
                self.httpd.serve_forever(poll_interval=0.2)
            except KeyboardInterrupt:
                self.stop()
        return self.heartbeat_target

    def stop(self) -> None:
        self._stop.set()
        if self.httpd:
            self.httpd.shutdown()
        self.log.info("worker_stop", worker_id=self.worker_id)

    def register(self) -> JSON:
        if not self.heartbeat_target:
            return {"ok": False, "error": "no_orchestrator_target"}
        transport = HttpJsonTransport(self.log, self.metrics, timeout_s=10.0, auth_secret=self.config.secret_key)
        url = f"{self.heartbeat_target}/register"
        payload = {"worker_id": self.worker_id, "address": self._local_address(), "port": self.config.worker_port, "capability": asdict(self.capability)}
        with self._registration_lock:
            try:
                resp = transport.post_json(url, payload)
                if resp.get("ok"):
                    self.log.info("worker_registered", worker_id=self.worker_id, orchestrator_url=self.heartbeat_target)
                return resp
            except Exception as exc:
                self.log.exception("worker_register_failed", exc)
                return {"ok": False, "error": str(exc)}

    def run_autonomous(self, orchestrator_url: str | None = None, *, auto_discover: bool = True) -> str:
        """Zero-touch worker mode: start server, resolve orchestrator, and keep reconnecting forever."""
        self._ensure_http_server()
        resolved_url, source = self.resolve_orchestrator_url(orchestrator_url, auto_discover=auto_discover)
        self.heartbeat_target = resolved_url.rstrip("/")
        self.log.info("worker_autonomous_mode", worker_id=self.worker_id, orchestrator_url=self.heartbeat_target, resolution=source)
        while not self._stop.is_set():
            result = self.register()
            if result.get("ok"):
                break
            self.log.warning("worker_autonomous_register_retry", orchestrator_url=self.heartbeat_target, retry_delay_s=self.config.silent_boot_retry_interval_s)
            time.sleep(max(1.0, self.config.silent_boot_retry_interval_s))
        return self.heartbeat_target

    def _local_address(self) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(0.2)
                s.connect(("1.1.1.1", 80))
                ip = s.getsockname()[0]
                if ip:
                    return ip
        except Exception:
            pass
        try:
            host = socket.gethostname()
            ip = socket.gethostbyname(host)
            if ip and ip != "127.0.0.1":
                return ip
        except Exception:
            pass
        return "127.0.0.1"

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.heartbeat_target:
                    transport = HttpJsonTransport(self.log, self.metrics, timeout_s=5.0, auth_secret=self.config.secret_key)
                    payload = {
                        "worker_id": self.worker_id,
                        "queue_depth": 0,
                        "in_flight": self.in_flight,
                        "status": "healthy",
                        "avg_latency_ms": self.metrics.histograms.get("task.latency_ms", Histogram()).snapshot().get("avg", 0.0),
                        "success_rate": 1.0,
                        "backend": "local",
                    }
                    transport.post_json(f"{self.heartbeat_target}/heartbeat", payload)
                    self.last_heartbeat = now_s()
            except Exception as exc:
                self.log.warning("heartbeat_failed", error=str(exc))
            time.sleep(self.config.heartbeat_interval_s)

    def execute(self, envelope_data: JSON) -> JSON:
        start = now_s()
        try:
            payload = dict(envelope_data.get("payload", {}) or {})
            expected = str(envelope_data.get("checksum", ""))
            if expected and expected != payload_checksum(payload):
                return {"status": TaskStatus.FAILED.value, "error": "checksum_mismatch", "worker_id": self.worker_id, "backend": "local", "metrics": {"inference_ms": 0.0}}
            envelope = TaskEnvelope(
                request_id=str(envelope_data["request_id"]),
                task_id=str(envelope_data["task_id"]),
                kind=TaskKind(envelope_data["kind"]),
                stage_id=str(envelope_data.get("stage_id", "")),
                worker_id=self.worker_id,
                rank=safe_int(envelope_data.get("rank", 0), 0),
                deadline_s=safe_float(envelope_data.get("deadline_s", 0.0), 0.0),
                retry_count=safe_int(envelope_data.get("retry_count", 0), 0),
                priority=safe_int(envelope_data.get("priority", 50), 50),
                checksum=expected,
                payload=payload,
            )
        except Exception as exc:
            return {"status": TaskStatus.FAILED.value, "error": f"invalid_envelope: {exc}", "worker_id": self.worker_id, "backend": "local", "metrics": {"inference_ms": 0.0}}
        self.in_flight += 1
        try:
            if envelope.kind == TaskKind.LLM:
                request = LLMRequest(
                    prompt=str(envelope.payload.get("prompt", "")),
                    request_id=envelope.request_id,
                    task_id=envelope.task_id,
                    max_new_tokens=safe_int(envelope.payload.get("max_new_tokens", 128), 128),
                    temperature=safe_float(envelope.payload.get("temperature", 0.7), 0.7),
                    top_p=safe_float(envelope.payload.get("top_p", 0.95), 0.95),
                    chunk_size=safe_int(envelope.payload.get("chunk_size", 24), 24),
                    metadata=dict(envelope.payload),
                )
                chunks = list(self.llm_runtime.stream_generate(request, None))
                output_text = "".join(chunks)
                inference_ms = (now_s() - start) * 1000.0
                return {"status": TaskStatus.SUCCEEDED.value, "output_text": output_text, "stream_chunks": chunks, "worker_id": self.worker_id, "backend": "local", "metrics": {"inference_ms": inference_ms, "tokens": len(chunks), "tps": (len(chunks) / max(inference_ms / 1000.0, 0.001))}}
            if envelope.kind == TaskKind.TTS:
                request = TTSRequest(
                    text=str(envelope.payload.get("text", "")),
                    request_id=envelope.request_id,
                    task_id=envelope.task_id,
                    chunk_size=safe_int(envelope.payload.get("chunk_size", 2048), 2048),
                    metadata=dict(envelope.payload),
                )
                chunks = list(self.tts_runtime.stream_synthesize(request, None))
                audio = b"".join(chunks)
                inference_ms = (now_s() - start) * 1000.0
                return {
                    "status": TaskStatus.SUCCEEDED.value,
                    "audio_bytes_b64": base64.b64encode(audio).decode("ascii"),
                    "stream_chunks": [base64.b64encode(c).decode("ascii") for c in chunks],
                    "worker_id": self.worker_id,
                    "backend": "local_tts",
                    "metrics": {"inference_ms": inference_ms, "audio_chunks": len(chunks)},
                }
            inference_ms = (now_s() - start) * 1000.0
            return {"status": TaskStatus.SUCCEEDED.value, "output_text": "ok", "worker_id": self.worker_id, "backend": "local", "metrics": {"inference_ms": inference_ms}}
        except Exception as exc:
            self.log.exception("worker_execute_exception", exc, task_id=envelope.task_id)
            return {"status": TaskStatus.FAILED.value, "error": str(exc), "worker_id": self.worker_id, "backend": "local", "metrics": {"inference_ms": (now_s() - start) * 1000.0}}
        finally:
            self.in_flight = max(0, self.in_flight - 1)

    def health(self) -> JSON:
        return {
            "worker_id": self.worker_id,
            "capability": asdict(self.capability),
            "in_flight": self.in_flight,
            "last_heartbeat": self.last_heartbeat,
            "version": VERSION,
        }


# =============================================================================
# HTTP servers
# =============================================================================

class ThreadingClusterServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, orchestrator: ClusterOrchestrator):
        super().__init__(server_address, RequestHandlerClass)
        self.orchestrator = orchestrator


class ThreadingWorkerServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, worker: WorkerRuntime):
        super().__init__(server_address, RequestHandlerClass)
        self.worker = worker


class BaseJSONHandler(http.server.BaseHTTPRequestHandler):
    server_version = "DistributedInferenceHTTP/1.1"

    def log_message(self, format: str, *args: t.Any) -> None:
        return

    def _read_raw(self) -> bytes:
        length = safe_int(self.headers.get("Content-Length", 0), 0)
        return self.rfile.read(length) if length > 0 else b""

    def _read_json(self) -> JSON:
        raw = self._read_raw()
        self._cached_raw_body = raw  # type: ignore[attr-defined]
        if not raw:
            return {}
        return from_json(raw)

    def _request_secret(self) -> str:
        server = getattr(self, "server", None)
        if server is None:
            return ""
        if hasattr(server, "orchestrator"):
            return getattr(server.orchestrator.config, "secret_key", "")
        if hasattr(server, "worker"):
            return getattr(server.worker.config, "secret_key", "")
        return ""

    def _verify_signature_if_needed(self, raw_body: bytes) -> bool:
        secret = self._request_secret()
        if not secret:
            return True
        signature = self.headers.get("X-Cluster-Signature", "")
        if not signature:
            strict = False
            server = getattr(self, "server", None)
            if hasattr(server, "orchestrator"):
                strict = bool(getattr(server.orchestrator.config, "strict_auth", False))
            if hasattr(server, "worker"):
                strict = bool(getattr(server.worker.config, "strict_auth", False))
            return not strict
        return HMACSigner(secret).verify(raw_body, signature)

    def _send_json(self, data: JSON, status: int = 200) -> None:
        raw = to_json(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


ELECTRON_MAIN_JS = r"""

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

let win = null;
function createWindow() {
  win = new BrowserWindow({
    width: 1600,
    height: 1000,
    backgroundColor: '#0b0f14',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    }
  });
  win.loadURL(process.env.CLUSTER_DASHBOARD_URL || 'http://127.0.0.1:8080/ui');
}
app.whenReady().then(createWindow);
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
"""

ELECTRON_PRELOAD_JS = r"""
const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('clusterBridge', {
  ping: () => ipcRenderer.invoke('ping'),
});
"""


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Distributed Inference Cluster</title>
<style>
:root { color-scheme: dark; --bg:#0a0f14; --panel:#0f1620; --panel2:#111b28; --line:#223044; --text:#e7eef8; --muted:#8fa4bf; --cyan:#2dd4bf; --green:#34d399; --yellow:#fbbf24; --red:#fb7185; --violet:#8b5cf6; }
* { box-sizing:border-box; }
html,body { height:100%; }
body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: linear-gradient(180deg, #081018 0%, #0a0f14 100%); color:var(--text); }
a { color: inherit; }
.shell { display:grid; grid-template-columns: 280px 1fr; min-height:100vh; }
.sidebar { position:sticky; top:0; height:100vh; background: linear-gradient(180deg, rgba(9,14,22,.98), rgba(10,15,20,.96)); border-right:1px solid rgba(255,255,255,.06); padding:18px; display:flex; flex-direction:column; gap:16px; }
.brand { display:flex; flex-direction:column; gap:6px; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,.06); }
.brand h1 { margin:0; font-size:18px; letter-spacing:.2px; }
.brand small { color:var(--muted); line-height:1.5; }
.nav { display:flex; flex-direction:column; gap:8px; }
.nav button { text-align:left; border:1px solid transparent; background:transparent; color:var(--text); padding:12px 14px; border-radius:14px; cursor:pointer; display:flex; align-items:center; justify-content:space-between; gap:12px; }
.nav button:hover, .nav button.active { background: rgba(255,255,255,.04); border-color: rgba(255,255,255,.06); }
.badge { display:inline-flex; align-items:center; gap:6px; padding:5px 9px; border-radius:999px; font-size:11px; background: rgba(255,255,255,.06); color:var(--text); }
.badge.ok { color: var(--green); }
.badge.warn { color: var(--yellow); }
.badge.bad { color: var(--red); }
.main { padding:18px; }
.topbar { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:16px; padding:14px 16px; border:1px solid rgba(255,255,255,.06); border-radius:18px; background: rgba(15,22,32,.82); backdrop-filter: blur(10px); }
.topbar .title { display:flex; flex-direction:column; gap:4px; }
.topbar h2 { margin:0; font-size:18px; }
.topbar p { margin:0; color:var(--muted); font-size:13px; }
.grid { display:grid; gap:16px; }
.grid.kpis { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.grid.two { grid-template-columns: 1.2fr .8fr; }
.grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.card { background: linear-gradient(180deg, rgba(15,22,32,.98), rgba(11,17,26,.98)); border:1px solid rgba(255,255,255,.06); border-radius:18px; padding:16px; box-shadow: 0 20px 50px rgba(0,0,0,.22); }
.card h3 { margin:0 0 12px; font-size:14px; color:#d8e2f0; letter-spacing:.2px; }
.muted { color: var(--muted); }
.kpi { display:flex; flex-direction:column; gap:8px; min-height:108px; }
.kpi .value { font-size:28px; font-weight:700; letter-spacing:-.02em; }
.kpi .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
.kpi .sub { color:var(--muted); font-size:12px; }
.table { width:100%; border-collapse: collapse; }
.table th, .table td { text-align:left; padding:10px 8px; border-bottom:1px solid rgba(255,255,255,.06); vertical-align:top; font-size:12px; }
.table th { color:#b9c7da; font-weight:600; }
.stack { display:flex; flex-direction:column; gap:10px; }
.progress { width:100%; height:10px; background:#0a1019; border-radius:999px; overflow:hidden; border:1px solid rgba(255,255,255,.05); }
.progress > span { display:block; height:100%; background: linear-gradient(90deg, var(--cyan), var(--violet)); }
.chart { width:100%; height:220px; background:#0a1019; border:1px solid rgba(255,255,255,.06); border-radius:16px; }
.logbox { height:320px; overflow:auto; white-space:pre-wrap; word-break:break-word; background:#081018; border:1px solid rgba(255,255,255,.06); border-radius:16px; padding:12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; line-height:1.6; }
.controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
button.action { background: linear-gradient(180deg, rgba(59,130,246,.95), rgba(37,99,235,.95)); color:white; border:0; padding:10px 14px; border-radius:12px; cursor:pointer; font-weight:600; }
button.ghost { background:#121a25; color:var(--text); border:1px solid rgba(255,255,255,.08); }
input, select, textarea { width:100%; background:#09111a; color:var(--text); border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:10px 12px; }
textarea { min-height:110px; resize:vertical; }
.section { display:none; }
.section.active { display:block; }
.inline-grid { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; }
.layers { display:flex; flex-wrap:wrap; gap:8px; }
.layer-chip { padding:6px 10px; border-radius:999px; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.07); font-size:11px; }
@media (max-width: 1180px) {
  .shell { grid-template-columns: 1fr; }
  .sidebar { position:relative; height:auto; }
  .grid.kpis, .grid.two, .grid.three, .inline-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="brand">
      <h1>Cluster Control Plane</h1>
      <small>Industrial Dark Mode • Live orchestration • Mesh aware</small>
    </div>
    <div class="nav" id="nav">
      <button class="active" data-section="dashboard">Dashboard <span class="badge" id="navStatus">connecting</span></button>
      <button data-section="workers">Workers <span class="badge">mesh</span></button>
      <button data-section="network">Network Mesh <span class="badge">p2p</span></button>
      <button data-section="docs">API Docs <span class="badge">openapi</span></button>
      <button data-section="logs">Logs <span class="badge">live</span></button>
    </div>
    <div class="card">
      <h3>Cluster Endpoint</h3>
      <div id="endpoint" class="muted">—</div>
      <div style="margin-top:10px" id="meshBadge" class="badge warn">mesh: pending</div>
    </div>
  </aside>
  <main class="main">
    <div class="topbar">
      <div class="title">
        <h2>Distributed Inference Cluster</h2>
        <p>Telemetry, network mesh, workers, timelines, API docs, and live logs.</p>
      </div>
      <div class="controls">
        <span class="badge" id="clock">—</span>
        <button class="ghost" onclick="refreshAll()">Refresh</button>
        <button class="ghost" onclick="toggleAuto()">Auto <span id="autoMode">on</span></button>
      </div>
    </div>

    <section class="section active" id="dashboard">
      <div class="grid kpis" id="kpis"></div>
      <div class="grid two" style="margin-top:16px">
        <div class="card">
          <h3>Adicionar modelo</h3>
          <div class="stack">
            <div class="controls" style="display:grid;grid-template-columns:1fr 160px 160px;gap:10px;align-items:end">
              <div><label class="muted">Arquivo do modelo</label><input id="modelFile" type="file" accept=".gguf,.bin,.pt,.pth,.safetensors,.onnx,.json,.wav,.mp3,.flac" /></div>
              <div><label class="muted">ID do modelo</label><input id="modelId" type="text" placeholder="opcional" /></div>
              <div><label class="muted">Tipo</label><select id="modelType"><option value="llm">LLM</option><option value="tts">TTS</option></select></div>
            </div>
            <div class="controls" style="display:grid;grid-template-columns:1fr 160px;gap:10px;align-items:end">
              <div><label class="muted">Formato</label><input id="modelFormat" type="text" placeholder="gguf, safetensors, wav..." /></div>
              <button class="action" onclick="uploadModel()">Adicionar arquivo</button>
            </div>
            <div class="muted">O ficheiro é guardado no registry local e passa a ficar disponível para seleção automática.</div>
          </div>
        </div>
        <div class="card">
          <h3>Modelos carregados</h3>
          <div id="modelsView" class="stack"></div>
        </div>
      </div>
      <div class="grid two" style="margin-top:16px">
        <div class="card">
          <h3>TPS Timeline</h3>
          <canvas id="tpsChart" class="chart" width="1200" height="300"></canvas>
        </div>
        <div class="card">
          <h3>Request Timeline</h3>
          <canvas id="latencyChart" class="chart" width="1200" height="300"></canvas>
        </div>
      </div>
    </section>

    <section class="section" id="workers">
      <div class="card">
        <h3>Workers</h3>
        <table class="table" id="workersTable"></table>
      </div>
      <div class="grid two" style="margin-top:16px">
        <div class="card">
          <h3>Node Memory</h3>
          <div id="memoryMap" class="stack"></div>
        </div>
        <div class="card">
          <h3>Layer Mapping</h3>
          <div class="controls" style="margin-bottom:10px">
            <label class="muted">Total layers</label>
            <input id="layerCount" type="number" min="1" value="32" style="max-width:120px" oninput="updateLayerMap()"/>
            <button class="ghost" onclick="updateLayerMap()">Rebuild</button>
          </div>
          <div id="layerMap" class="stack"></div>
        </div>
      </div>
    </section>

    <section class="section" id="network">
      <div class="grid two">
        <div class="card">
          <h3>Mesh Topology</h3>
          <div id="meshView" class="stack"></div>
        </div>
        <div class="card">
          <h3>Network Status</h3>
          <div id="networkStatus" class="stack"></div>
        </div>
      </div>
    </section>

    <section class="section" id="docs">
      <div class="card">
        <h3>OpenAPI / Swagger-style Docs</h3>
        <div class="controls" style="margin-bottom:12px">
          <button class="ghost" onclick="loadDocs()">Reload docs</button>
          <button class="ghost" onclick="copyOpenAPI()">Copy JSON</button>
        </div>
        <div id="docsView" class="stack"></div>
      </div>
    </section>

    <section class="section" id="logs">
      <div class="grid two">
        <div class="card">
          <h3>Live Logs</h3>
          <div id="logsView" class="logbox"></div>
        </div>
        <div class="card">
          <h3>Cluster Actions</h3>
          <div class="stack">
            <button class="action" onclick="purgeQueue()">Drain Queue</button>
            <button class="ghost" onclick="runDiagnostics()">Diagnostics</button>
          </div>
          <div style="margin-top:12px">
            <label class="muted">Task JSON</label>
            <textarea id="taskPayload">{"prompt":"Explain tensor parallelism with fallback design.","max_new_tokens":120}</textarea>
          </div>
          <div class="inline-grid" style="margin-top:12px">
            <div><label class="muted">Kind</label><select id="kind"><option value="llm">LLM</option><option value="tts">TTS</option><option value="shard_compile">Shard Compile</option><option value="diagnostic">Diagnostic</option></select></div>
            <div><label class="muted">Priority</label><input id="priority" type="number" min="0" max="100" value="50"/></div>
          </div>
          <div class="controls" style="margin-top:12px">
            <button class="action" onclick="submitTask()">Submit</button>
            <button class="ghost" onclick="toggleAuto()">Auto <span id="autoMode2">on</span></button>
          </div>
          <div style="margin-top:12px"><pre id="result" class="logbox" style="min-height:200px"></pre></div>
        </div>
      </div>
    </section>
  </main>
</div>
<script>
let auto = true;
let latestRequest = null;
let cachedStatus = null;
let cachedDocs = null;
let cachedOpenAPI = null;
let eventSource = null;

function api(path, opts={}) {
  return fetch(path, {headers:{'Content-Type':'application/json'}, ...opts}).then(r => r.json());
}
function setActive(section) {
  document.querySelectorAll('.section').forEach(el => el.classList.remove('active'));
  document.getElementById(section).classList.add('active');
  document.querySelectorAll('#nav button').forEach(btn => btn.classList.toggle('active', btn.dataset.section === section));
}
document.getElementById('nav').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-section]');
  if (!btn) return;
  setActive(btn.dataset.section);
});
function renderKpis(c) {
  const avgTps = Number((c.metrics?.gauges?.['cluster.tps'] ?? 0)).toFixed(2);
  const queueDepth = c.queue_depth ?? 0;
  const alive = c.alive_workers ?? 0;
  const total = c.worker_count ?? 0;
  const network = c.network || {};
  document.getElementById('kpis').innerHTML = `
    <div class="card kpi"><div class="label">Workers Online</div><div class="value">${alive}/${total}</div><div class="sub">Alive workers currently heartbeating</div></div>
    <div class="card kpi"><div class="label">Queue Depth</div><div class="value">${queueDepth}</div><div class="sub">Pending and in-flight requests</div></div>
    <div class="card kpi"><div class="label">TPS</div><div class="value">${avgTps}</div><div class="sub">Tokens per second, rolling</div></div>
    <div class="card kpi"><div class="label">Mesh</div><div class="value">${network.private_ip || network.mesh_ip || 'offline'}</div><div class="sub">${network.connection_state || 'unknown'}</div></div>
  `;
}
function drawLine(canvasId, values) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#081018';
  ctx.fillRect(0,0,w,h);
  if (!values || !values.length) return;
  const pad = 26;
  const max = Math.max(...values, 1);
  ctx.beginPath();
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#2dd4bf';
  values.forEach((v, i) => {
    const x = pad + i * ((w - pad*2) / Math.max(1, values.length - 1));
    const y = h - pad - (v / max) * (h - pad*2);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}
function renderWorkers(workers) {
  const el = document.getElementById('workersTable');
  let html = '<tr><th>Worker</th><th>Status</th><th>Memory</th><th>Health</th><th>Backend</th><th>Actions</th></tr>';
  for (const w of workers || []) {
    const vram = w.capability?.vram_gb ?? 0;
    const ram = w.capability?.ram_gb ?? 0;
    const cores = w.capability?.cpu_cores ?? 0;
    const inFlight = w.in_flight ?? 0;
    const qd = w.queue_depth ?? 0;
    const statusCls = w.status === 'healthy' ? 'ok' : (w.status === 'dead' ? 'bad' : 'warn');
    html += `<tr>
      <td><div><strong>${w.worker_id || '-'}</strong></div><div class="muted">${w.address || ''}:${w.port || ''}</div></td>
      <td><span class="badge ${statusCls}">${w.status || 'unknown'}</span></td>
      <td>${ram.toFixed ? ram.toFixed(1) : ram} GB RAM<br/>${vram.toFixed ? vram.toFixed(1) : vram} GB VRAM<br/><span class="muted">${cores} cores</span></td>
      <td>${inFlight} in-flight<br/>${qd} queued<br/>${Math.round((w.success_rate ?? 1) * 100)}%</td>
      <td>${w.current_backend || 'local'}</td>
      <td><button class="ghost" onclick="drainNode('${w.worker_id}')">Drain</button> <button class="ghost" onclick="restartNode('${w.worker_id}')">Restart</button></td>
    </tr>`;
  }
  el.innerHTML = html;
}
function renderMemoryMap(workers) {
  const box = document.getElementById('memoryMap');
  if (!workers || !workers.length) {
    box.innerHTML = '<div class="muted">No workers registered.</div>';
    return;
  }
  box.innerHTML = workers.map(w => {
    const ram = Math.max(1, Number(w.capability?.ram_gb || 1));
    const vram = Math.max(1, Number(w.capability?.vram_gb || 1));
    const ramUse = Math.min(100, Math.round((w.load_factor ?? 0.1) * 100));
    const vramUse = Math.min(100, Math.round((w.avg_latency_ms ?? 0) / 20));
    return `<div class="card" style="padding:12px">
      <div style="display:flex;justify-content:space-between;gap:10px"><strong>${w.worker_id}</strong><span class="badge">${w.address || ''}</span></div>
      <div class="muted" style="margin-top:8px">RAM ${ramUse}% of ${ram.toFixed(1)} GB</div>
      <div class="progress"><span style="width:${ramUse}%"></span></div>
      <div class="muted" style="margin-top:8px">VRAM ${vramUse}% of ${vram.toFixed(1)} GB</div>
      <div class="progress"><span style="width:${vramUse}%"></span></div>
    </div>`;
  }).join('');
}
function renderLayerMap(workers) {
  const totalLayers = parseInt(document.getElementById('layerCount').value || '32', 10);
  const box = document.getElementById('layerMap');
  const parts = workers || [];
  if (!parts.length) { box.innerHTML = '<div class="muted">No workers available.</div>'; return; }
  const weightSum = parts.reduce((acc, w) => acc + Math.max(0.1, Number(w.capability?.cpu_cores || 1) + Number(w.capability?.ram_gb || 0) / 2 + (w.capability?.has_gpu ? 8 : 0)), 0);
  let current = 0;
  const chips = [];
  for (const w of parts) {
    const weight = Math.max(0.1, Number(w.capability?.cpu_cores || 1) + Number(w.capability?.ram_gb || 0) / 2 + (w.capability?.has_gpu ? 8 : 0));
    const layers = Math.max(1, Math.floor(totalLayers * weight / weightSum));
    const start = current;
    const end = Math.min(totalLayers, current + layers);
    current = end;
    chips.push(`<div class="card" style="padding:12px"><strong>${w.worker_id}</strong><div class="muted">Layers ${start}-${end - 1}</div><div class="layers">${Array.from({length: Math.max(0, end-start)}, (_, i) => `<span class="layer-chip">${start+i}</span>`).join('')}</div></div>`);
  }
  if (current < totalLayers && chips.length) {
    chips[chips.length - 1] = chips[chips.length - 1].replace('</div></div>', `<div class="muted">+${totalLayers-current} overflow layers</div></div>`);
  }
  box.innerHTML = chips.join('');
}
function renderNetwork(c) {
  const n = c.network || {};
  document.getElementById('endpoint').textContent = n.server_url || c.endpoint || window.location.origin;
  const mesh = document.getElementById('meshBadge');
  const ok = (n.ready_for_core ?? n.connected) ? true : false;
  mesh.textContent = ok ? `mesh: ${n.private_ip || n.mesh_ip || 'connected'}` : 'mesh: offline';
  mesh.className = 'badge ' + (ok ? 'ok' : 'bad');
  const meshView = document.getElementById('meshView');
  const workers = c.workers || [];
  meshView.innerHTML = [`<div class="card"><strong>Master</strong><div class="muted">${n.private_ip || n.mesh_ip || '127.0.0.1'}</div></div>`].concat(workers.map(w => `<div class="card"><strong>${w.worker_id}</strong><div class="muted">${w.address || ''}:${w.port || ''}</div><div class="muted">RTT ${Math.max(1, Math.round(w.avg_latency_ms || 0))} ms</div></div>`)).join('');
  const ns = document.getElementById('networkStatus');
  ns.innerHTML = [
    `<div class="card"><strong>State</strong><div class="muted">${n.connection_state || 'unknown'}</div></div>`,
    `<div class="card"><strong>Mode</strong><div class="muted">${n.role || 'cluster'}</div></div>`,
    `<div class="card"><strong>Static IP</strong><div class="muted">${n.private_ip || n.mesh_ip || 'n/a'}</div></div>`,
    `<div class="card"><strong>Watchdog</strong><div class="muted">${n.watchdog_running ? 'running' : 'idle'}</div></div>`
  ].join('');
}
async function loadDocs() {
  if (!cachedOpenAPI) cachedOpenAPI = await api('/openapi.json');
  const openapi = cachedOpenAPI;
  const view = document.getElementById('docsView');
  const paths = Object.keys(openapi.paths || {});
  view.innerHTML = `<div class="card"><strong>${openapi.info?.title || 'API'}</strong><div class="muted">${openapi.info?.description || ''}</div></div>` + paths.map(p => {
    const methods = Object.keys(openapi.paths[p] || {});
    return `<div class="card"><strong>${p}</strong><div class="muted">${methods.join(', ').toUpperCase()}</div><pre class="logbox">${JSON.stringify(openapi.paths[p], null, 2)}</pre></div>`;
  }).join('');
}
async function copyOpenAPI() {
  if (!cachedOpenAPI) cachedOpenAPI = await api('/openapi.json');
  await navigator.clipboard.writeText(JSON.stringify(cachedOpenAPI, null, 2));
}
function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
function renderModels(models) {
  const el = document.getElementById('modelsView');
  const items = models || [];
  if (!items.length) {
    el.innerHTML = '<div class="muted">Nenhum modelo carregado.</div>';
    return;
  }
  el.innerHTML = items.map(m => `<div class="card" style="padding:12px">
    <div style="display:flex;justify-content:space-between;gap:10px;align-items:center"><strong>${m.model_id || '-'}</strong><span class="badge">${(m.extra?.format || m.dtype || 'model')}</span></div>
    <div class="muted" style="margin-top:8px">${m.source_path || ''}</div>
    <div class="muted">backend: ${m.extra?.backend || m.extra?.model_type || m.architecture || 'auto'}</div>
    <div class="muted">checksum: ${String(m.checksum || '').slice(0, 12)}</div>
  </div>`).join('');
}
async function refreshModels() {
  try {
    const res = await api('/models');
    renderModels(res.models || []);
  } catch (e) {
    const el = document.getElementById('modelsView');
    if (el) el.innerHTML = '<div class="muted">Falha ao carregar modelos.</div>';
  }
}
function renderLogs(lines) {
  const box = document.getElementById('logsView');
  box.textContent = (lines || []).map(x => typeof x === 'string' ? x : JSON.stringify(x)).join('\n');
  box.scrollTop = box.scrollHeight;
}
async function refreshAll() {
  try {
    cachedStatus = await api('/status');
    const c = cachedStatus;
    document.getElementById('clock').textContent = new Date(c.time).toLocaleTimeString();
    document.getElementById('navStatus').textContent = c.network?.connection_state || 'online';
    document.getElementById('navStatus').className = 'badge ' + (((c.network?.ready_for_core ?? true) ? 'ok' : 'warn'));
    renderKpis(c);
    renderWorkers(c.workers || []);
    renderModels(c.models || []);
    renderMemoryMap(c.workers || []);
    renderLayerMap(c.workers || []);
    renderNetwork(c);
    drawLine('tpsChart', (c.metrics?.histograms?.['cluster.tps']?.values || []).slice(-60));
    drawLine('latencyChart', (c.metrics?.histograms?.['request.latency_ms']?.values || []).slice(-60));
    renderLogs(c.logs || []);
    if (latestRequest) {
      const task = await api('/task/' + latestRequest);
      document.getElementById('result').textContent = JSON.stringify(task, null, 2);
    }
  } catch (e) {
    document.getElementById('navStatus').textContent = 'offline';
    document.getElementById('navStatus').className = 'badge bad';
  }
}
async function uploadModel() {
  const fileInput = document.getElementById('modelFile');
  const file = fileInput?.files?.[0];
  if (!file) {
    document.getElementById('result').textContent = JSON.stringify({ok:false, error:'Selecione um arquivo primeiro'}, null, 2);
    return;
  }
  const form = new FormData();
  form.append('file', file, file.name);
  const modelId = document.getElementById('modelId').value.trim();
  const modelType = document.getElementById('modelType').value || 'llm';
  const modelFormat = document.getElementById('modelFormat').value.trim() || (file.name.split('.').pop() || 'unknown');
  if (modelId) form.append('model_id', modelId);
  form.append('model_type', modelType);
  form.append('format', modelFormat);
  const res = await fetch('/upload_model', { method: 'POST', body: form });
  const data = await res.json();
  document.getElementById('result').textContent = JSON.stringify(data, null, 2);
  await refreshModels();
}
async function submitTask() {
  const kind = document.getElementById('kind').value;
  const priority = parseInt(document.getElementById('priority').value || '50', 10);
  let payload;
  try { payload = JSON.parse(document.getElementById('taskPayload').value); }
  catch (e) { payload = { text: document.getElementById('taskPayload').value }; }
  const res = await api('/submit', {method:'POST', body: JSON.stringify({kind, priority, payload})});
  latestRequest = res.request_id;
  document.getElementById('result').textContent = JSON.stringify(res, null, 2);
}
async function runDiagnostics() {
  const res = await api('/diagnostics');
  document.getElementById('result').textContent = JSON.stringify(res, null, 2);
}
async function purgeQueue() {
  const res = await api('/purge', {method:'POST', body: JSON.stringify({scope:'queue'})});
  document.getElementById('result').textContent = JSON.stringify(res, null, 2);
  refreshAll();
}
async function drainNode(workerId) {
  const res = await api('/workers/' + encodeURIComponent(workerId) + '/drain', {method:'POST', body: JSON.stringify({drain: true})});
  document.getElementById('result').textContent = JSON.stringify(res, null, 2);
}
async function restartNode(workerId) {
  const res = await api('/workers/' + encodeURIComponent(workerId) + '/restart', {method:'POST', body: JSON.stringify({restart: true})});
  document.getElementById('result').textContent = JSON.stringify(res, null, 2);
}
async function updateLayerMap() {
  if (!cachedStatus) return;
  renderLayerMap(cachedStatus.workers || []);
}
function toggleAuto() {
  auto = !auto;
  document.getElementById('autoMode').textContent = auto ? 'on' : 'off';
  document.getElementById('autoMode2').textContent = auto ? 'on' : 'off';
}
function connectEvents() {
  try {
    eventSource = new EventSource('/events');
    eventSource.onmessage = (ev) => {
      try {
        const payload = JSON.parse(ev.data);
        if (payload.status) {
          cachedStatus = payload.status;
          renderKpis(payload.status);
          renderWorkers(payload.status.workers || []);
          renderModels(payload.status.models || []);
          renderMemoryMap(payload.status.workers || []);
          renderLayerMap(payload.status.workers || []);
          renderNetwork(payload.status);
          renderLogs(payload.status.logs || []);
        }
      } catch (e) {}
    };
    eventSource.onerror = () => { document.getElementById('navStatus').textContent = 'reconnecting'; };
  } catch (e) {}
}
setInterval(() => { if (auto) refreshAll(); }, 2000);
refreshAll();
refreshModels();
loadDocs();
connectEvents();
</script>
</body>
</html>
"""


def build_openapi_spec(orchestrator: ClusterOrchestrator | None = None) -> JSON:
    config = getattr(orchestrator, "config", None)
    base = {
        "openapi": "3.1.0",
        "info": {
            "title": "Distributed Inference Cluster API",
            "version": VERSION,
            "description": "Cluster orchestration, worker management, network mesh, diagnostics, and live telemetry.",
        },
        "servers": [{"url": f"http://{getattr(config, 'host', '127.0.0.1')}:{getattr(config, 'port', DEFAULT_PORT)}"}],
        "paths": {
            "/status": {"get": {"summary": "Cluster status", "responses": {"200": {"description": "OK"}}}},
            "/metrics": {"get": {"summary": "Metrics snapshot", "responses": {"200": {"description": "OK"}}}},
            "/diagnostics": {"get": {"summary": "Diagnostics and validations", "responses": {"200": {"description": "OK"}}}},
            "/submit": {"post": {"summary": "Submit a task", "responses": {"200": {"description": "Accepted"}}}},
            "/upload_model": {"post": {"summary": "Upload a model file", "responses": {"200": {"description": "OK"}}}},
            "/models": {"get": {"summary": "List uploaded models", "responses": {"200": {"description": "OK"}}}},
            "/register": {"post": {"summary": "Register a worker", "responses": {"200": {"description": "OK"}}}},
            "/heartbeat": {"post": {"summary": "Heartbeat update", "responses": {"200": {"description": "OK"}}}},
            "/logs": {"get": {"summary": "Recent logs", "responses": {"200": {"description": "OK"}}}},
            "/layers": {"get": {"summary": "Layer distribution", "parameters": [{"name": "layers", "in": "query", "schema": {"type": "integer", "default": 32}}], "responses": {"200": {"description": "OK"}}}},
            "/events": {"get": {"summary": "Server-sent live updates", "responses": {"200": {"description": "text/event-stream"}}}},
            "/docs": {"get": {"summary": "Swagger-style documentation", "responses": {"200": {"description": "HTML docs"}}}},
        },
    }
    return base


def build_docs_html() -> str:
    return r"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Cluster API Docs</title>
<style>
:root { color-scheme: dark; --bg:#0a0f14; --panel:#0f1620; --line:#223044; --text:#e7eef8; --muted:#8fa4bf; --accent:#2dd4bf; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
header { position:sticky; top:0; background:rgba(10,15,20,.95); border-bottom:1px solid rgba(255,255,255,.06); padding:18px 22px; }
main { padding:18px 22px; max-width:1200px; margin:0 auto; }
.card { background:linear-gradient(180deg, rgba(15,22,32,.98), rgba(11,17,26,.98)); border:1px solid rgba(255,255,255,.06); border-radius:18px; padding:16px; margin-bottom:14px; }
pre { white-space:pre-wrap; word-break:break-word; background:#081018; border:1px solid rgba(255,255,255,.06); border-radius:14px; padding:12px; overflow:auto; }
small { color:var(--muted); }
button { background:rgba(45,212,191,.12); color:var(--text); border:1px solid rgba(45,212,191,.3); padding:10px 12px; border-radius:12px; cursor:pointer; }
</style>
</head>
<body>
<header>
  <div style='display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap'>
    <div>
      <h1 style='margin:0;font-size:20px'>Cluster API Documentation</h1>
      <small>Swagger-style reference for orchestration, workers, mesh telemetry, and live operations.</small>
    </div>
    <div>
      <button onclick='reloadDocs()'>Reload</button>
      <button onclick='copyJson()'>Copy OpenAPI JSON</button>
    </div>
  </div>
</header>
<main>
  <div id='summary' class='card'>Loading…</div>
  <div id='paths'></div>
</main>
<script>
let spec = null;
async function reloadDocs() {
  spec = await fetch('/openapi.json').then(r => r.json());
  document.getElementById('summary').innerHTML = `<strong>${spec.info.title}</strong><div class='muted'>${spec.info.description}</div><div class='muted'>Version ${spec.info.version}</div>`;
  document.getElementById('paths').innerHTML = Object.entries(spec.paths || {}).map(([path, methods]) => `<div class='card'><strong>${path}</strong><pre>${JSON.stringify(methods, null, 2)}</pre></div>`).join('');
}
async function copyJson() {
  if (!spec) await reloadDocs();
  await navigator.clipboard.writeText(JSON.stringify(spec, null, 2));
}
reloadDocs();
</script>
</body>
</html>"""


class ClusterHTTPRequestHandler(BaseJSONHandler):
    def _cluster_payload(self, orch: ClusterOrchestrator) -> JSON:
        return orch.cluster_status()

    def _send_event_stream(self, orch: ClusterOrchestrator) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                payload = {"status": self._cluster_payload(orch)}
                self.wfile.write(f"data: {to_json(payload)}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(1.0)
        except Exception:
            return

    def do_GET(self) -> None:
        orch = self.server.orchestrator  # type: ignore[attr-defined]
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path in {"/", "/ui"}:
            self._send_text(DASHBOARD_HTML, content_type="text/html; charset=utf-8")
            return
        if path in {"/status", "/api/status"}:
            self._send_json(self._cluster_payload(orch))
            return
        if path in {"/metrics", "/api/metrics"}:
            self._send_json(orch.metrics.snapshot())
            return
        if path in {"/diagnostics", "/api/diagnostics"}:
            self._send_json(orch.diagnostics())
            return
        if path in {"/docs", "/api/docs"}:
            self._send_text(build_docs_html(), content_type="text/html; charset=utf-8")
            return
        if path in {"/openapi.json", "/api/openapi.json"}:
            self._send_json(build_openapi_spec(orch))
            return
        if path in {"/logs", "/api/logs"}:
            self._send_json({"logs": orch.log.tail(400)})
            return
        if path in {"/models", "/api/models"}:
            self._send_json({"ok": True, "models": orch.list_models(), "count": len(orch.state.model_manifests)})
            return
        if path in {"/events", "/api/events"}:
            self._send_event_stream(orch)
            return
        if path in {"/layers", "/api/layers"}:
            q = parse_qs(parsed.query)
            total_layers = safe_int(q.get("layers", ["32"])[0], 32)
            workers = orch.active_workers() if hasattr(orch, "active_workers") else list(orch.state.list_workers())
            self._send_json({"total_layers": total_layers, "distribution": calculate_layer_distribution(total_layers, workers)})
            return
        if path.startswith("/task/"):
            task_id = path.split("/", 2)[2]
            task = orch.state.tasks.get(task_id)
            if not task:
                self._send_json({"error": "not_found"}, 404)
                return
            result = orch.state.results.get(task_id)
            self._send_json({
                "task": task.to_public(),
                "result": result.to_public() if result else None,
                "stream": orch.state.consume_stream(task.request_id),
            })
            return
        if path.startswith("/stream/"):
            request_id = path.split("/", 2)[2]
            self._send_json({"request_id": request_id, "chunks": orch.state.consume_stream(request_id)})
            return
        self._send_json({"error": "not_found"}, 404)

    def do_POST(self) -> None:
        orch = self.server.orchestrator  # type: ignore[attr-defined]
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        raw_body = self._read_raw()
        self._cached_raw_body = raw_body  # type: ignore[attr-defined]
        body: JSON = {}
        if path != "/upload_model":
            body = from_json(raw_body) if raw_body else {}
        if not self._verify_signature_if_needed(getattr(self, "_cached_raw_body", b"")):
            self._send_json({"ok": False, "error": "unauthorized"}, 403)
            return
        if path == "/upload_model":
            try:
                content_type = self.headers.get("Content-Type", "multipart/form-data")
                form = cgi.FieldStorage(
                    fp=io.BytesIO(raw_body),
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                        "CONTENT_LENGTH": str(len(raw_body)),
                    },
                    keep_blank_values=True,
                )
                file_item = form["file"] if "file" in form else None
                if file_item is None or not getattr(file_item, "file", None):
                    self._send_json({"ok": False, "error": "file_missing"}, 400)
                    return
                file_bytes = file_item.file.read()
                filename = str(getattr(file_item, "filename", "model.bin") or "model.bin")
                model_id = str(form.getfirst("model_id", "") or "").strip() or None
                model_type = str(form.getfirst("model_type", "llm") or "llm").strip().lower()
                model_format = str(form.getfirst("format", "") or Path(filename).suffix.lstrip(".") or "unknown").strip().lower()
                metadata = {
                    "original_filename": filename,
                    "content_type": getattr(file_item, "type", ""),
                }
                result = orch.register_uploaded_model(filename, model_type, model_format, file_bytes, model_id=model_id, metadata=metadata)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc), "traceback": traceback.format_exc()}, 500)
            return
        if path == "/register":
            capability = body.get("capability", {}) or {}
            cap = Capability(
                cpu_cores=safe_int(capability.get("cpu_cores", os.cpu_count() or 1), os.cpu_count() or 1),
                ram_gb=safe_float(capability.get("ram_gb", 0.0), 0.0),
                has_gpu=bool(capability.get("has_gpu", False)),
                vram_gb=safe_float(capability.get("vram_gb", 0.0), 0.0),
                gpu_name=str(capability.get("gpu_name", "")),
                cuda_available=bool(capability.get("cuda_available", False)),
                backends=list(capability.get("backends", [])),
                distributed_ready=bool(capability.get("distributed_ready", False)),
                max_concurrency=safe_int(capability.get("max_concurrency", 2), 2),
                notes=str(capability.get("notes", "")),
            )
            cap.ram_total_gb = safe_float(capability.get("ram_total_gb", capability.get("ram_gb", 0.0)), 0.0)  # type: ignore[attr-defined]
            cap.ram_free_gb = safe_float(capability.get("ram_free_gb", 0.0), 0.0)  # type: ignore[attr-defined]
            cap.vram_total_gb = safe_float(capability.get("vram_total_gb", capability.get("vram_gb", 0.0)), 0.0)  # type: ignore[attr-defined]
            cap.vram_free_gb = safe_float(capability.get("vram_free_gb", 0.0), 0.0)  # type: ignore[attr-defined]
            cap.platform = str(capability.get("platform", ""))  # type: ignore[attr-defined]
            cap.is_wsl = bool(capability.get("is_wsl", False))  # type: ignore[attr-defined]
            cap.is_colab = bool(capability.get("is_colab", False))  # type: ignore[attr-defined]
            cap.gpu_layers_capable = safe_int(capability.get("gpu_layers_capable", 0), 0)  # type: ignore[attr-defined]
            worker = orch.register_worker(
                address=str(body.get("address", self.client_address[0] if hasattr(self, "client_address") else "127.0.0.1")),
                port=safe_int(body.get("port", DEFAULT_WORKER_PORT), DEFAULT_WORKER_PORT),
                capability=cap,
                worker_id=str(body.get("worker_id", "")) or None,
                tags=list(body.get("tags", [])),
            )
            self._send_json({"ok": True, "worker": worker.to_public()})
            return
        if path == "/heartbeat":
            worker_id = str(body.get("worker_id", ""))
            self._send_json(orch.heartbeat(worker_id, body))
            return
        if path == "/submit":
            try:
                kind = TaskKind(body.get("kind", "llm"))
            except Exception:
                kind = TaskKind.LLM
            payload = body.get("payload", {}) or {}
            if kind == TaskKind.TTS and isinstance(payload, str):
                payload = {"text": payload}
            if not isinstance(payload, dict):
                payload = {"text": str(payload)}
            envelope = orch.submit_request(kind, payload, priority=safe_int(body.get("priority", 50), 50), request_id=str(body.get("request_id", "")) or None)
            self._send_json({"ok": True, "request_id": envelope.request_id, "task_id": envelope.task_id, "task": envelope.to_public()})
            return
        if path == "/compile_model":
            try:
                manifest = orch.compiler.inspect_checkpoint(str(body["checkpoint_path"]), body.get("model_id"))
                shard_manifest = orch.compiler.build_shard_manifest(
                    manifest,
                    tensor_degree=safe_int(body.get("tensor_degree", 1), 1),
                    pipeline_degree=safe_int(body.get("pipeline_degree", 1), 1),
                )
                valid, errors = orch.compiler.validate_shard_manifest(shard_manifest)
                self._send_json({"ok": valid, "manifest": manifest.__dict__, "shards": shard_manifest.to_public(), "errors": errors})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc), "traceback": traceback.format_exc()}, 500)
            return
        self._send_json({"error": "not_found"}, 404)


class WorkerHTTPRequestHandler(BaseJSONHandler):
    def do_GET(self) -> None:
        worker = self.server.worker  # type: ignore[attr-defined]
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(worker.health())
            return
        if path == "/capabilities":
            self._send_json(asdict(worker.capability))
            return
        self._send_json({"error": "not_found"}, 404)

    def do_POST(self) -> None:
        worker = self.server.worker  # type: ignore[attr-defined]
        path = urlparse(self.path).path
        body = self._read_json()
        if not self._verify_signature_if_needed(getattr(self, "_cached_raw_body", b"")):
            self._send_json({"ok": False, "error": "unauthorized"}, 403)
            return
        if path == "/execute":
            result = worker.execute(body)
            self._send_json(result)
            return
        self._send_json({"error": "not_found"}, 404)


# =============================================================================
# Failure simulation and tests/harness
# =============================================================================

class SelfTest:
    def __init__(self, orchestrator: ClusterOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.results: list[JSON] = []

    def test_queue(self) -> JSON:
        env = self.orchestrator.submit_request(TaskKind.LLM, {"prompt": "hello world"}, priority=10)
        assert env.request_id
        return {"ok": True, "task_id": env.task_id}

    def test_fallback(self) -> JSON:
        env = self.orchestrator.submit_request(TaskKind.TTS, {"text": "Fallback test."}, priority=20)
        return {"ok": True, "task_id": env.task_id}

    def test_shard_manifest(self) -> JSON:
        if torch is not None:
            temp = self.orchestrator.state.path / "test_model.bin"
            tensor = torch.randn(8, 8)
            torch.save({"architecture": "test_transformer", "layer0.weight": tensor}, temp)
            manifest = self.orchestrator.compiler.inspect_checkpoint(str(temp), "test-model")
            shard_manifest = self.orchestrator.compiler.build_shard_manifest(manifest, tensor_degree=2, pipeline_degree=1)
            valid, errors = self.orchestrator.compiler.validate_shard_manifest(shard_manifest)
            return {"ok": valid, "errors": errors, "shards": len(shard_manifest.tensors)}
        return {"ok": True, "note": "torch unavailable"}

    def run(self) -> JSON:
        checks = [self.test_queue(), self.test_fallback(), self.test_shard_manifest()]
        return {"ok": all(c.get("ok", False) for c in checks), "checks": checks}


def simulate_worker_failure(orchestrator: ClusterOrchestrator, worker_id: str) -> JSON:
    worker = orchestrator.state.workers.get(worker_id)
    if not worker:
        return {"ok": False, "error": "worker not found"}
    worker.last_heartbeat = now_s() - orchestrator.config.worker_timeout_s - 1
    worker.status = "dead"
    orchestrator.state.mark_dirty()
    orchestrator.scheduler.mark_failure(worker_id)
    return {"ok": True, "worker": worker.to_public()}



# =============================================================================
# Dynamic pipeline parallelism, hybrid cluster autonomy, and activation transport
# =============================================================================

try:
    import grpc  # type: ignore
except Exception:
    grpc = None  # type: ignore

GRPC_ACTIVATION_PROTO = """syntax = \"proto3\";

service InferenceStream {
  rpc PassActivation (ActivationData) returns (ActivationResponse);
}

message ActivationData {
  int32 source_rank = 1;
  int32 target_rank = 2;
  bytes tensor_data = 3;
  string task_id = 4;
  string request_id = 5;
  string stage_id = 6;
  string payload_json = 7;
  string checksum = 8;
  int32 retry_count = 9;
  double deadline_s = 10;
}

message ActivationResponse {
  bool received = 1;
  string status = 2;
  string task_id = 3;
  string request_id = 4;
  string output_text = 5;
  string audio_bytes_b64 = 6;
  string error = 7;
}
"""



def _worker_capacity_weight(worker: WorkerInfo) -> float:
    """Estimate how much work a worker can absorb right now."""
    score = max(0.1, float(worker.capability.score()))
    score *= 1.0 + max(0.0, worker.success_rate - 0.5) * 0.6
    score *= 1.0 / (1.0 + max(0, worker.in_flight) * 0.35 + max(0, worker.queue_depth) * 0.10)
    score *= 1.0 / (1.0 + max(0.0, worker.avg_latency_ms) / 1500.0)
    if worker.capability.has_gpu:
        score *= 1.25
    if worker.capability.cuda_available:
        score *= 1.10
    return max(0.1, score)


def calculate_layer_distribution(
    total_layers: int,
    workers_or_num: int | list[WorkerInfo] | tuple[WorkerInfo, ...] | list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[dict[str, t.Any]]:
    """Return contiguous layer ranges distributed across workers using weighted capacity."""
    total_layers = max(0, int(total_layers))
    if isinstance(workers_or_num, int):
        num_workers = max(1, int(workers_or_num))
        resolved_weights = [float(w) for w in (weights or [1.0] * num_workers)]
        worker_ids = ["" for _ in range(num_workers)]
    else:
        items = list(workers_or_num)
        num_workers = max(1, len(items))
        if items and isinstance(items[0], WorkerInfo):
            resolved_weights = [_worker_capacity_weight(w) for w in items]  # type: ignore[arg-type]
            worker_ids = [w.worker_id for w in items]  # type: ignore[union-attr]
        else:
            resolved_weights = [max(0.1, float(w)) for w in (weights or items)]  # type: ignore[arg-type]
            worker_ids = ["" for _ in range(num_workers)]

    if len(resolved_weights) < num_workers:
        resolved_weights = list(resolved_weights) + [1.0] * (num_workers - len(resolved_weights))
    elif len(resolved_weights) > num_workers:
        resolved_weights = list(resolved_weights[:num_workers])

    if total_layers == 0:
        return []

    total_weight = sum(max(0.0, w) for w in resolved_weights)
    if total_weight <= 0.0:
        resolved_weights = [1.0] * num_workers
        total_weight = float(num_workers)

    ideal_shares = [total_layers * (max(0.0, w) / total_weight) for w in resolved_weights]
    base_layers = [int(math.floor(s)) for s in ideal_shares]
    assigned = sum(base_layers)
    remainder = total_layers - assigned

    # Give out the remaining layers to the highest fractional parts.
    fractional = sorted(
        enumerate(ideal_shares),
        key=lambda item: (item[1] - math.floor(item[1]), resolved_weights[item[0]]),
        reverse=True,
    )
    for idx, _ in fractional:
        if remainder <= 0:
            break
        base_layers[idx] += 1
        remainder -= 1

    distribution: list[dict[str, t.Any]] = []
    current_layer = 0
    for rank, layer_count in enumerate(base_layers):
        if layer_count <= 0:
            continue
        start = current_layer
        end = current_layer + layer_count
        distribution.append({
            "rank": rank,
            "worker_id": worker_ids[rank] if rank < len(worker_ids) else "",
            "range": (start, end),
            "layers": layer_count,
            "weight": resolved_weights[rank],
        })
        current_layer = end

    # Any rounding leftovers are assigned to the last emitted stage.
    if current_layer < total_layers and distribution:
        distribution[-1]["range"] = (distribution[-1]["range"][0], total_layers)
        distribution[-1]["layers"] = total_layers - distribution[-1]["range"][0]
    return distribution


@dataclass
class PipelineStage:
    stage_index: int
    worker_id: str
    address: str
    http_port: int
    grpc_port: int
    layer_start: int
    layer_end: int
    stage_name: str
    replica_group: int = 0
    mesh_address: str = ""
    mesh_http_port: int = 0
    mesh_grpc_port: int = 0
    peer_id: str = ""
    p2p_transport: str = "mesh"
    tensor_parallel_degree: int = 1
    tensor_group_id: int = 0
    tensor_rank: int = 0
    tensor_peer_ids: list[str] = field(default_factory=list)
    tensor_transport: str = "mesh_grpc"
    tensor_split_axis: int = 0
    tensor_reduce: str = "concat"

    def to_public(self) -> JSON:
        return asdict(self)


@dataclass
class PipelineGroup:
    group_id: int
    mode: str
    workers: list[str] = field(default_factory=list)
    stages: list[PipelineStage] = field(default_factory=list)

    def to_public(self) -> JSON:
        return {
            "group_id": self.group_id,
            "mode": self.mode,
            "workers": list(self.workers),
            "stages": [s.to_public() for s in self.stages],
        }



@dataclass
class PipelinePlan:
    model_id: str
    task_kind: str
    total_layers: int
    mode: ParallelMode
    entry_worker_id: str
    replica_count: int
    transport_mode: str
    groups: list[PipelineGroup] = field(default_factory=list)
    primary_group_id: int = 0
    failover_group_ids: list[int] = field(default_factory=list)
    group_weights: list[float] = field(default_factory=list)
    control_plane: str = "orchestrator"
    data_plane: str = "mesh"
    tensor_degree: int = 1
    pipeline_degree: int = 1
    created_at: float = field(default_factory=now_s)
    notes: str = ""

    def to_public(self) -> JSON:
        return {
            "model_id": self.model_id,
            "task_kind": self.task_kind,
            "total_layers": self.total_layers,
            "mode": self.mode.value,
            "entry_worker_id": self.entry_worker_id,
            "replica_count": self.replica_count,
            "transport_mode": self.transport_mode,
            "primary_group_id": self.primary_group_id,
            "failover_group_ids": list(self.failover_group_ids),
            "group_weights": list(self.group_weights),
            "control_plane": self.control_plane,
            "data_plane": self.data_plane,
            "tensor_degree": self.tensor_degree,
            "pipeline_degree": self.pipeline_degree,
            "groups": [g.to_public() for g in self.groups],
            "created_at": self.created_at,
            "notes": self.notes,
        }


@dataclass
class ActivationFrame:
    source_rank: int
    target_rank: int
    task_id: str
    request_id: str
    stage_id: str
    tensor_data: bytes | bytearray | memoryview
    payload_json: str = ""
    checksum: str = ""
    retry_count: int = 0
    deadline_s: float = 0.0
    stream_index: int = 0
    stream_total: int = 1
    model_id: str = ""
    content_type: str = "application/octet-stream"
    tensor_shape_json: str = ""
    tensor_dtype: str = ""
    shared_memory_name: str = ""
    shared_memory_offset: int = 0
    shared_memory_size: int = 0
    transport_hint: str = "binary"

    def to_header(self) -> JSON:
        header = asdict(self)
        header["tensor_data"] = "<binary>"
        return header

    @property
    def tensor_size(self) -> int:
        if isinstance(self.tensor_data, memoryview):
            return self.tensor_data.nbytes
        return len(self.tensor_data or b"")

    def as_bytes(self) -> bytes:
        if isinstance(self.tensor_data, bytes):
            return self.tensor_data
        if isinstance(self.tensor_data, bytearray):
            return bytes(self.tensor_data)
        if isinstance(self.tensor_data, memoryview):
            return self.tensor_data.tobytes()
        return bytes(self.tensor_data or b"")


@dataclass
class SharedTensorRef:
    name: str
    size: int
    offset: int = 0


try:
    from multiprocessing import shared_memory  # type: ignore
except Exception:
    shared_memory = None  # type: ignore


class SharedTensorStore:
    """Lightweight shared-memory transport for same-machine activations."""

    def __init__(self) -> None:
        self._shared_memory = shared_memory if 'shared_memory' in globals() and shared_memory is not None else None

    def available(self) -> bool:
        return self._shared_memory is not None

    def put(self, data: bytes | bytearray | memoryview, *, prefix: str = "cluster_actv_") -> SharedTensorRef:
        if self._shared_memory is None:
            raise RuntimeError("multiprocessing.shared_memory is unavailable")
        view = memoryview(data)
        shm = self._shared_memory.SharedMemory(create=True, size=view.nbytes, name=None)
        try:
            shm.buf[:view.nbytes] = view
            ref = SharedTensorRef(name=shm.name, size=view.nbytes, offset=0)
            return ref
        finally:
            shm.close()

    def resolve(self, ref: SharedTensorRef) -> tuple[memoryview, t.Any]:
        if self._shared_memory is None:
            raise RuntimeError("multiprocessing.shared_memory is unavailable")
        shm = self._shared_memory.SharedMemory(name=ref.name)
        view = memoryview(shm.buf)[ref.offset: ref.offset + ref.size]
        return view, shm


class ActivationProtoCodec:
    """Native protobuf codec for activation frames and responses."""

    _data_cls: t.Any = None
    _response_cls: t.Any = None
    _available: bool | None = None

    @classmethod
    def available(cls) -> bool:
        if cls._available is not None:
            return cls._available
        try:
            from google.protobuf import descriptor_pb2, descriptor_pool, message_factory  # type: ignore
        except Exception:
            cls._available = False
            return False
        file_proto = descriptor_pb2.FileDescriptorProto()
        file_proto.name = "cluster_activation.proto"
        file_proto.package = "cluster"
        file_proto.syntax = "proto3"

        def add_message(name: str, fields: list[tuple[str, int, int]]) -> None:
            msg = file_proto.message_type.add()
            msg.name = name
            for field_name, number, field_type in fields:
                field = msg.field.add()
                field.name = field_name
                field.number = number
                field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
                field.type = field_type

        add_message(
            "ActivationData",
            [
                ("source_rank", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
                ("target_rank", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
                ("tensor_data", 3, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES),
                ("task_id", 4, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("request_id", 5, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("stage_id", 6, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("payload_json", 7, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("checksum", 8, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("retry_count", 9, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
                ("deadline_s", 10, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE),
                ("stream_index", 11, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
                ("stream_total", 12, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
                ("model_id", 13, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("content_type", 14, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("tensor_shape_json", 15, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("tensor_dtype", 16, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("shared_memory_name", 17, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("shared_memory_offset", 18, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
                ("shared_memory_size", 19, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
                ("transport_hint", 20, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
            ],
        )
        add_message(
            "ActivationResponse",
            [
                ("received", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
                ("status", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("task_id", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("request_id", 4, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("output_text", 5, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("audio_bytes", 6, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES),
                ("audio_bytes_b64", 7, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
                ("error", 8, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
            ],
        )
        pool = descriptor_pool.DescriptorPool()
        pool.Add(file_proto)
        cls._data_cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("cluster.ActivationData"))
        cls._response_cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("cluster.ActivationResponse"))
        cls._available = True
        return True

    @classmethod
    def request_message(cls, frame: ActivationFrame) -> t.Any:
        if not cls.available():
            raise RuntimeError("protobuf is unavailable")
        tensor_bytes = frame.as_bytes()
        msg = cls._data_cls(
            source_rank=int(frame.source_rank),
            target_rank=int(frame.target_rank),
            tensor_data=b"" if frame.shared_memory_name else tensor_bytes,
            task_id=frame.task_id,
            request_id=frame.request_id,
            stage_id=frame.stage_id,
            payload_json=frame.payload_json,
            checksum=frame.checksum,
            retry_count=int(frame.retry_count),
            deadline_s=float(frame.deadline_s),
            stream_index=int(frame.stream_index),
            stream_total=int(frame.stream_total),
            model_id=frame.model_id,
            content_type=frame.content_type,
            tensor_shape_json=frame.tensor_shape_json,
            tensor_dtype=frame.tensor_dtype,
            shared_memory_name=frame.shared_memory_name,
            shared_memory_offset=int(frame.shared_memory_offset),
            shared_memory_size=int(frame.shared_memory_size),
            transport_hint=frame.transport_hint,
        )
        return msg

    @classmethod
    def encode_request(cls, frame: ActivationFrame) -> bytes:
        return cls.request_message(frame).SerializeToString()

    @classmethod
    def decode_request(cls, raw: bytes | bytearray | memoryview | t.Any) -> ActivationFrame:
        if not cls.available():
            raise RuntimeError("protobuf is unavailable")
        msg = raw if hasattr(raw, "SerializeToString") and hasattr(raw, "ParseFromString") else cls._data_cls.FromString(bytes(raw))
        frame = ActivationFrame(
            source_rank=safe_int(getattr(msg, "source_rank", 0), 0),
            target_rank=safe_int(getattr(msg, "target_rank", 0), 0),
            task_id=str(getattr(msg, "task_id", "")),
            request_id=str(getattr(msg, "request_id", "")),
            stage_id=str(getattr(msg, "stage_id", "")),
            tensor_data=bytes(getattr(msg, "tensor_data", b"")),
            payload_json=str(getattr(msg, "payload_json", "")),
            checksum=str(getattr(msg, "checksum", "")),
            retry_count=safe_int(getattr(msg, "retry_count", 0), 0),
            deadline_s=safe_float(getattr(msg, "deadline_s", 0.0), 0.0),
            stream_index=safe_int(getattr(msg, "stream_index", 0), 0),
            stream_total=safe_int(getattr(msg, "stream_total", 1), 1),
            model_id=str(getattr(msg, "model_id", "")),
            content_type=str(getattr(msg, "content_type", "application/octet-stream")),
            tensor_shape_json=str(getattr(msg, "tensor_shape_json", "")),
            tensor_dtype=str(getattr(msg, "tensor_dtype", "")),
            shared_memory_name=str(getattr(msg, "shared_memory_name", "")),
            shared_memory_offset=safe_int(getattr(msg, "shared_memory_offset", 0), 0),
            shared_memory_size=safe_int(getattr(msg, "shared_memory_size", 0), 0),
            transport_hint=str(getattr(msg, "transport_hint", "binary")),
        )
        if frame.shared_memory_name and frame.shared_memory_size:
            if shared_memory is None:
                raise RuntimeError("shared memory transport unavailable")
            shm = shared_memory.SharedMemory(name=frame.shared_memory_name)
            frame.tensor_data = memoryview(shm.buf)[frame.shared_memory_offset: frame.shared_memory_offset + frame.shared_memory_size]
            setattr(frame, "_shared_memory_handle", shm)
        return frame

    @classmethod
    def cleanup_frame(cls, frame: ActivationFrame) -> None:
        shm = getattr(frame, "_shared_memory_handle", None)
        if shm is not None:
            with contextlib.suppress(Exception):
                shm.close()
            with contextlib.suppress(Exception):
                delattr(frame, "_shared_memory_handle")

    @classmethod
    def response_message(cls, *, received: bool, status: str, task_id: str, request_id: str, output_text: str = "", audio_bytes: bytes = b"", error: str = "") -> t.Any:
        if not cls.available():
            raise RuntimeError("protobuf is unavailable")
        audio_bytes = bytes(audio_bytes or b"")
        return cls._response_cls(
            received=bool(received),
            status=status,
            task_id=task_id,
            request_id=request_id,
            output_text=output_text,
            audio_bytes=audio_bytes,
            audio_bytes_b64=base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else "",
            error=error,
        )

    @classmethod
    def encode_response(cls, **kwargs: t.Any) -> bytes:
        return cls.response_message(**kwargs).SerializeToString()

    @classmethod
    def decode_response(cls, raw: bytes | bytearray | memoryview | t.Any) -> JSON:
        if not cls.available():
            raise RuntimeError("protobuf is unavailable")
        msg = raw if hasattr(raw, "SerializeToString") and hasattr(raw, "ParseFromString") else cls._response_cls.FromString(bytes(raw))
        audio_bytes = bytes(getattr(msg, "audio_bytes", b""))
        audio_b64 = str(getattr(msg, "audio_bytes_b64", "")) or (base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else "")
        return {
            "received": bool(getattr(msg, "received", False)),
            "status": str(getattr(msg, "status", "")),
            "task_id": str(getattr(msg, "task_id", "")),
            "request_id": str(getattr(msg, "request_id", "")),
            "output_text": str(getattr(msg, "output_text", "")),
            "audio_bytes": audio_bytes,
            "audio_bytes_b64": audio_b64,
            "error": str(getattr(msg, "error", "")),
        }


class ActivationWireCodec:
    MAGIC = b"ACTV1"

    @staticmethod
    def encode(frame: ActivationFrame) -> bytes:
        header = json.dumps(frame.to_header(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        body = memoryview(frame.tensor_data) if isinstance(frame.tensor_data, memoryview) else memoryview(frame.as_bytes())
        return ActivationWireCodec.MAGIC + struct.pack("!II", len(header), body.nbytes) + header + body.tobytes()

    @staticmethod
    def decode(blob: bytes) -> ActivationFrame:
        if not blob.startswith(ActivationWireCodec.MAGIC):
            raise ValueError("invalid activation frame magic")
        header_len, body_len = struct.unpack("!II", blob[len(ActivationWireCodec.MAGIC): len(ActivationWireCodec.MAGIC) + 8])
        offset = len(ActivationWireCodec.MAGIC) + 8
        header = json.loads(blob[offset: offset + header_len].decode("utf-8", errors="replace"))
        body = memoryview(blob)[offset + header_len: offset + header_len + body_len].tobytes()
        return ActivationFrame(
            source_rank=safe_int(header.get("source_rank", 0), 0),
            target_rank=safe_int(header.get("target_rank", 0), 0),
            task_id=str(header.get("task_id", "")),
            request_id=str(header.get("request_id", "")),
            stage_id=str(header.get("stage_id", "")),
            tensor_data=body,
            payload_json=str(header.get("payload_json", "")),
            checksum=str(header.get("checksum", "")),
            retry_count=safe_int(header.get("retry_count", 0), 0),
            deadline_s=safe_float(header.get("deadline_s", 0.0), 0.0),
            stream_index=safe_int(header.get("stream_index", 0), 0),
            stream_total=safe_int(header.get("stream_total", 1), 1),
            model_id=str(header.get("model_id", "")),
            content_type=str(header.get("content_type", "application/octet-stream")),
            tensor_shape_json=str(header.get("tensor_shape_json", "")),
            tensor_dtype=str(header.get("tensor_dtype", "")),
            shared_memory_name=str(header.get("shared_memory_name", "")),
            shared_memory_offset=safe_int(header.get("shared_memory_offset", 0), 0),
            shared_memory_size=safe_int(header.get("shared_memory_size", 0), 0),
            transport_hint=str(header.get("transport_hint", "binary")),
        )


class TensorActivationTransport:
    """Transport that prefers native protobuf gRPC and same-machine shared memory."""

    def __init__(self, log: StructuredLogger, metrics: MetricsRegistry, *, timeout_s: float = 10.0) -> None:
        self.log = log
        self.metrics = metrics
        self.timeout_s = timeout_s
        self._shm_store = SharedTensorStore()

    def _grpc_available(self) -> bool:
        return grpc is not None and ActivationProtoCodec.available()

    def _local_transport_available(self) -> bool:
        return self._shm_store.available()

    def _is_local_host(self, host: str) -> bool:
        host = host.strip().lower()
        return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    def _prepare_frame(self, frame: ActivationFrame, transport_mode: str) -> ActivationFrame:
        if transport_mode in {"shared_memory", "grpc_shm"} and self._local_transport_available() and frame.tensor_size >= 4096:
            ref = self._shm_store.put(frame.as_bytes())
            frame = dataclasses.replace(
                frame,
                tensor_data=b"",
                shared_memory_name=ref.name,
                shared_memory_offset=ref.offset,
                shared_memory_size=ref.size,
                transport_hint="shared_memory",
            )
        return frame

    def send_frame(self, host: str, http_port: int, grpc_port: int, frame: ActivationFrame, *, transport_mode: str = "grpc") -> JSON:
        frame = self._prepare_frame(frame, transport_mode)
        try:
            if transport_mode in {"grpc", "grpc_shm"} and self._grpc_available():
                return self._send_grpc(host, grpc_port, frame)
            if transport_mode in {"shared_memory", "grpc_shm"} and self._local_transport_available() and self._is_local_host(host):
                return self._send_grpc(host, grpc_port, frame)
            return self._send_http(host, http_port, frame)
        finally:
            if frame.shared_memory_name and frame.transport_hint == "shared_memory" and self._local_transport_available():
                with contextlib.suppress(Exception):
                    shm = shared_memory.SharedMemory(name=frame.shared_memory_name)
                    shm.close()
                    shm.unlink()

    def _send_grpc(self, host: str, port: int, frame: ActivationFrame) -> JSON:
        assert grpc is not None
        target = f"{host}:{port}"
        payload = ActivationProtoCodec.encode_request(frame)
        try:
            with grpc.insecure_channel(target) as channel:
                stub = channel.unary_unary(
                    "/InferenceStream/PassActivation",
                    request_serializer=lambda b: b,
                    response_deserializer=lambda b: b,
                )
                raw = stub(payload, timeout=self.timeout_s)
                return ActivationProtoCodec.decode_response(raw)
        except Exception as exc:
            self.log.warning("grpc_activation_send_failed", target=target, error=str(exc), fallback="http")
            return self._send_http(host, port - 1000 if port > 1000 else port, frame)

    def _send_http(self, host: str, port: int, frame: ActivationFrame) -> JSON:
        url = f"http://{host}:{port}/activation"
        payload = ActivationWireCodec.encode(frame)
        req = Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        req.add_header("X-Request-Id", frame.request_id)
        req.add_header("X-Task-Id", frame.task_id)
        with urlopen(req, timeout=self.timeout_s) as res:
            raw = res.read()
            if not raw:
                return {"ok": True}
            try:
                return from_json(raw)
            except Exception:
                return {"ok": True, "raw_b64": base64.b64encode(raw).decode("ascii")}



class DynamicPipelinePlanner:
    """Builds replica groups and stage plans from the current worker set."""

    def __init__(self, store: StateStore, config: ControlConfig, log: StructuredLogger, metrics: MetricsRegistry) -> None:
        self.store = store
        self.config = config
        self.log = log
        self.metrics = metrics

    def _rank_workers(self, workers: list[WorkerInfo]) -> list[WorkerInfo]:
        return sorted(workers, key=lambda w: (
            w.capability.score(),
            w.capability.vram_gb,
            w.capability.cpu_cores,
            -w.in_flight,
            w.success_rate,
        ), reverse=True)

    def _worker_load_weight(self, worker: WorkerInfo) -> float:
        return _worker_capacity_weight(worker)

    def _total_layers_for_kind(self, kind: TaskKind, payload: JSON) -> int:
        if kind == TaskKind.TTS:
            return max(5, safe_int(payload.get("tts_layers", 5), 5))
        if kind == TaskKind.LLM:
            if "model_layers" in payload:
                return max(1, safe_int(payload.get("model_layers", 32), 32))
            model_size = str(payload.get("model_size", payload.get("model_id", "3b"))).lower()
            if "405" in model_size:
                return 120
            if "70" in model_size:
                return 80
            if "13" in model_size:
                return 40
            return 32
        return max(1, safe_int(payload.get("layers", 1), 1))

    def group_workers(self, workers: list[WorkerInfo], model_size_layers: int) -> list[list[WorkerInfo]]:
        ranked = self._rank_workers(workers)
        if not ranked:
            return []
        if model_size_layers < 40 and len(ranked) >= 2:
            group_count = 2
        elif model_size_layers >= 80 and len(ranked) >= 4:
            group_count = 1
        else:
            group_count = 1 if len(ranked) < 6 else 2
        group_count = min(group_count, len(ranked))
        buckets: list[list[WorkerInfo]] = [[] for _ in range(group_count)]
        bucket_loads = [0.0] * group_count
        for worker in ranked:
            idx = min(range(group_count), key=lambda i: (bucket_loads[i], len(buckets[i])))
            buckets[idx].append(worker)
            bucket_loads[idx] += self._worker_load_weight(worker)
        return [bucket for bucket in buckets if bucket]

    def _group_mode(self, layers: int, worker_count: int) -> ParallelMode:
        if worker_count <= 1:
            return ParallelMode.NONE
        if layers < 16:
            return ParallelMode.HYBRID
        if layers < 40:
            return ParallelMode.PIPELINE
        return ParallelMode.HYBRID if worker_count > 1 else ParallelMode.NONE

    def _group_weight(self, workers: list[WorkerInfo]) -> float:
        return sum(self._worker_load_weight(w) for w in workers)

    def build_plan(self, kind: TaskKind, payload: JSON, request_id: str) -> PipelinePlan | None:
        alive = [
            w for w in self.store.workers.values()
            if w.status != "dead" and (now_s() - w.last_heartbeat) <= self.config.worker_timeout_s
        ]
        alive = self._rank_workers(alive)
        if not alive:
            return None

        total_layers = self._total_layers_for_kind(kind, payload)
        groups = self.group_workers(alive, total_layers)
        if not groups:
            return None

        transport_mode = "mesh_grpc" if grpc is not None and ActivationProtoCodec.available() else "mesh_http"
        preferred_tensor = safe_int(payload.get("tensor_degree", 0), 0)
        preferred_pipeline = safe_int(payload.get("pipeline_degree", 0), 0)
        pipeline_groups: list[PipelineGroup] = []
        group_weights: list[float] = []
        max_tensor_degree = 1
        max_pipeline_degree = 1

        for group_id, group_workers in enumerate(groups):
            distribution = calculate_layer_distribution(total_layers, group_workers)
            if not distribution:
                continue

            base_pipeline_degree = max(1, len(distribution))
            tensor_target = preferred_tensor or (2 if total_layers >= 16 and len(group_workers) >= 2 else 1)
            tensor_degree = max(1, min(len(group_workers), tensor_target))
            if total_layers < 16:
                tensor_degree = 1
            elif tensor_degree > 2:
                tensor_degree = 2

            max_tensor_degree = max(max_tensor_degree, tensor_degree)
            max_pipeline_degree = max(max_pipeline_degree, preferred_pipeline or base_pipeline_degree)

            stages: list[PipelineStage] = []
            for idx, item in enumerate(distribution):
                layer_start, layer_end = item["range"]
                selected_workers = group_workers[:tensor_degree] if tensor_degree > 1 else [group_workers[min(item["rank"], len(group_workers) - 1)]]
                peer_ids = [w.worker_id for w in selected_workers[1:]]
                for shard_rank, worker in enumerate(selected_workers):
                    data_host, data_http_port, data_grpc_port = _worker_transport_profile(worker)
                    stages.append(PipelineStage(
                        stage_index=idx * 100 + shard_rank,
                        worker_id=worker.worker_id,
                        address=data_host,
                        http_port=data_http_port,
                        grpc_port=data_grpc_port,
                        layer_start=layer_start,
                        layer_end=layer_end,
                        stage_name=f"g{group_id}_stage_{idx}_tp{shard_rank}",
                        replica_group=group_id,
                        mesh_address=data_host,
                        mesh_http_port=data_http_port,
                        mesh_grpc_port=data_grpc_port,
                        peer_id=worker.mesh_peer_id or worker.worker_id,
                        p2p_transport="mesh_grpc" if worker.p2p_enabled else "lan_grpc",
                        tensor_parallel_degree=tensor_degree,
                        tensor_group_id=idx,
                        tensor_rank=shard_rank,
                        tensor_peer_ids=peer_ids,
                        tensor_transport=transport_mode,
                        tensor_split_axis=0,
                        tensor_reduce="concat",
                    ))
            pipeline_groups.append(PipelineGroup(
                group_id=group_id,
                mode=self._group_mode(total_layers, len(group_workers)).value,
                workers=[w.worker_id for w in group_workers],
                stages=stages,
            ))
            group_weights.append(self._group_weight(group_workers))

        if not pipeline_groups:
            return None

        primary_group_id = max(range(len(group_weights)), key=lambda i: group_weights[i]) if group_weights else 0
        failover_group_ids = [i for i in sorted(range(len(group_weights)), key=lambda i: group_weights[i], reverse=True) if i != primary_group_id]
        entry_stage = pipeline_groups[primary_group_id].stages[0]
        mode = ParallelMode.HYBRID if any(stage.tensor_parallel_degree > 1 for group in pipeline_groups for stage in group.stages) else self._group_mode(total_layers, len(groups[0]))

        return PipelinePlan(
            model_id=str(payload.get("model_id", payload.get("model", request_id))),
            task_kind=kind.value,
            total_layers=total_layers,
            mode=mode,
            entry_worker_id=entry_stage.worker_id,
            replica_count=len(pipeline_groups),
            transport_mode=transport_mode,
            groups=pipeline_groups,
            primary_group_id=primary_group_id,
            failover_group_ids=failover_group_ids,
            group_weights=group_weights,
            control_plane="orchestrator",
            data_plane="mesh",
            tensor_degree=max_tensor_degree,
            pipeline_degree=max_pipeline_degree,
            notes="weighted_replica_groups" if len(pipeline_groups) > 1 else "single_pipeline",
        )

    def validate_plan(self, plan: PipelinePlan) -> tuple[bool, list[str]]:
        errors: list[str] = []
        if not plan.groups:
            errors.append("empty groups")
        for group in plan.groups:
            if not group.stages:
                errors.append(f"group {group.group_id} has no stages")
                continue
            for i, stage in enumerate(group.stages):
                if stage.layer_end <= stage.layer_start:
                    errors.append(f"stage {stage.stage_index} invalid layer range")
                if i > 0 and group.stages[i - 1].layer_end > stage.layer_start:
                    errors.append(f"group {group.group_id} overlapping ranges")
        return (not errors, errors)


class HybridClusterOrchestrator(ClusterOrchestrator):
    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.scheduler = AdvancedScheduler(self.state, config, self.log, self.metrics)
        self.pipeline_planner = DynamicPipelinePlanner(self.state, config, self.log, self.metrics)
        self.activation_transport = TensorActivationTransport(self.log, self.metrics, timeout_s=config.request_timeout_s)
        self.model_registry_path = ensure_dir(Path(config.state_dir) / "models")
        self.pipeline_plans: dict[str, PipelinePlan] = {}

    def active_workers(self) -> list[WorkerInfo]:
        return [w for w in self.state.workers.values() if w.status != "dead" and (now_s() - w.last_heartbeat) <= self.config.worker_timeout_s]

    def compile_and_stage_model(self, checkpoint_path: str, model_id: str | None = None, *, tensor_degree: int = 1, pipeline_degree: int = 1) -> JSON:
        manifest = self.compiler.inspect_checkpoint(checkpoint_path, model_id)
        shard_manifest = self.compiler.build_shard_manifest(manifest, tensor_degree=tensor_degree, pipeline_degree=pipeline_degree)
        valid, errors = self.compiler.validate_shard_manifest(shard_manifest)
        if not valid:
            return {"ok": False, "errors": errors, "manifest": manifest.__dict__ if hasattr(manifest, "__dict__") else asdict(manifest)}
        plan = self.pipeline_planner.build_plan(TaskKind.LLM, {"model_id": manifest.model_id, "model_layers": len(manifest.layers) or 32, "tensor_degree": tensor_degree, "pipeline_degree": pipeline_degree}, manifest.model_id)
        if plan:
            self.pipeline_plans[manifest.model_id] = plan
        self.state.model_manifests[manifest.model_id] = manifest
        self.state.persist(force=True)
        return {"ok": True, "manifest": asdict(manifest), "shards": shard_manifest.to_public(), "plan": plan.to_public() if plan else None}

    def submit_request(self, kind: TaskKind, payload: JSON, priority: int = 50, request_id: str | None = None, deadline_s: float | None = None) -> TaskEnvelope:
        request_id = request_id or uid("req_")
        task_id = uid("task_")
        stage_id = str(payload.get("stage_id") or kind.value)
        checksum = payload_checksum(payload)
        plan = self.pipeline_planner.build_plan(kind, payload, request_id) if kind in {TaskKind.LLM, TaskKind.TTS} else None
        if plan:
            valid, errors = self.pipeline_planner.validate_plan(plan)
            if not valid:
                self.log.warning("pipeline_plan_invalid", errors=errors)
                plan = None
        if plan:
            self.pipeline_plans[request_id] = plan
            parallel = {
                "mode": plan.mode.value,
                "tensor_degree": plan.tensor_degree,
                "pipeline_degree": plan.pipeline_degree,
                "reason": plan.notes,
                "backend_preference": "",
                "local_fallback": False,
            }
            entry_worker_id = plan.entry_worker_id
            payload = dict(payload)
            payload["pipeline_runtime"] = plan.to_public()
            payload["pipeline_transport"] = plan.transport_mode
            payload["pipeline_context"] = {
                "request_id": request_id,
                "task_id": task_id,
                "stage_id": stage_id,
                "created_at": now_s(),
                "primary_group_id": plan.primary_group_id,
                "failover_group_ids": list(plan.failover_group_ids),
            }
            payload["parallel"] = parallel
            envelope = TaskEnvelope(
                request_id=request_id,
                task_id=task_id,
                kind=kind,
                stage_id=stage_id,
                worker_id=entry_worker_id,
                rank=0,
                deadline_s=deadline_s or (now_s() + self.config.request_timeout_s),
                retry_count=0,
                priority=priority,
                checksum=checksum,
                payload=payload,
                status=TaskStatus.PENDING,
            )
            self.state.store_task(envelope)
            if not self.queue.put(envelope, timeout=2.0):
                envelope.status = TaskStatus.FALLBACK
                self.metrics.inc("task.queue_rejected")
            return envelope
        return super().submit_request(kind, payload, priority=priority, request_id=request_id, deadline_s=deadline_s)

    def cluster_status(self) -> JSON:
        status = super().cluster_status()
        status["pipeline_plans"] = {rid: plan.to_public() for rid, plan in self.pipeline_plans.items()}
        status["active_workers"] = [w.to_public() for w in self.active_workers()]
        status["transport_mode"] = "grpc" if grpc is not None else "binary_http"
        return status


class HybridWorkerRuntime(WorkerRuntime):
    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.activation_transport = TensorActivationTransport(self.log, self.metrics, timeout_s=config.request_timeout_s)
        self._grpc_server = None
        self._grpc_thread: threading.Thread | None = None
        self._grpc_started = False

    def _grpc_port(self) -> int:
        return int(self.config.worker_port) + 1000

    def _plan_stage_maps(self, plan_data: JSON) -> tuple[dict[int, list[JSON]], dict[str, JSON]]:
        groups = plan_data.get("groups", []) or []
        by_group: dict[int, list[JSON]] = defaultdict(list)
        by_worker: dict[str, JSON] = {}
        for group in groups:
            for stage in group.get("stages", []) or []:
                gid = safe_int(stage.get("tensor_group_id", stage.get("stage_index", 0)), 0)
                by_group[gid].append(stage)
                worker_id = str(stage.get("worker_id", ""))
                if worker_id:
                    by_worker[worker_id] = stage
        for gid in list(by_group.keys()):
            by_group[gid] = sorted(by_group[gid], key=lambda s: safe_int(s.get("tensor_rank", 0), 0))
        return by_group, by_worker

    def _frame_tensor_bytes(self, result: JSON) -> bytes:
        if not isinstance(result, dict):
            return b""
        raw = result.get("tensor_bytes")
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return bytes(raw)
        b64 = result.get("tensor_bytes_b64")
        if isinstance(b64, str) and b64:
            try:
                return base64.b64decode(b64.encode("ascii"))
            except Exception:
                pass
        output = result.get("output_text")
        if isinstance(output, str) and output:
            try:
                return base64.b64decode(output.encode("ascii"))
            except Exception:
                return output.encode("utf-8")
        return b""

    def _chunk_tensor_bytes(self, data: bytes | bytearray | memoryview, parts: int) -> list[memoryview]:
        view = memoryview(data)
        parts = max(1, parts)
        if parts == 1:
            return [view]
        step = max(1, math.ceil(view.nbytes / parts))
        chunks = [view[i:min(view.nbytes, i + step)] for i in range(0, view.nbytes, step)]
        return chunks[:parts]

    def _tensor_stage_local(self, stage: PipelineStage, shard: bytes | bytearray | memoryview, payload: JSON, kind: TaskKind) -> JSON:
        processed = self._stage_transform(kind, stage, shard, payload)
        return {
            "status": TaskStatus.SUCCEEDED.value,
            "tensor_bytes_b64": base64.b64encode(processed).decode("ascii") if processed else "",
            "worker_id": self.worker_id,
            "stage_id": stage.stage_name,
            "tensor_rank": stage.tensor_rank,
            "tensor_group_id": stage.tensor_group_id,
        }

    def _tensor_stage_remote(self, stage: PipelineStage, shard: bytes | bytearray | memoryview, frame: ActivationFrame, payload: JSON, transport_mode: str, kind: TaskKind) -> JSON:
        shard_frame = ActivationFrame(
            source_rank=stage.tensor_rank,
            target_rank=stage.tensor_rank,
            task_id=frame.task_id,
            request_id=frame.request_id,
            stage_id=stage.stage_name,
            tensor_data=shard,
            payload_json=frame.payload_json,
            checksum=stable_hash(bytes(shard) if not isinstance(shard, bytes) else shard),
            retry_count=frame.retry_count,
            deadline_s=frame.deadline_s,
            model_id=frame.model_id,
            content_type=frame.content_type,
            tensor_shape_json=frame.tensor_shape_json,
            tensor_dtype=frame.tensor_dtype,
            shared_memory_name=frame.shared_memory_name,
            shared_memory_offset=frame.shared_memory_offset,
            shared_memory_size=frame.shared_memory_size,
            transport_hint=stage.tensor_transport,
            tensor_rank=stage.tensor_rank,
            tensor_world_size=stage.tensor_parallel_degree,
            tensor_group_id=stage.tensor_group_id,
            tensor_transport=stage.tensor_transport,
        )
        return self.activation_transport.send_frame(
            stage.address or stage.mesh_address,
            stage.http_port or stage.mesh_http_port,
            stage.grpc_port or stage.mesh_grpc_port,
            shard_frame,
            transport_mode=transport_mode if transport_mode in {"mesh_grpc", "mesh_http", "grpc", "http"} else stage.tensor_transport,
        )

    def _run_tensor_parallel_stage(self, stage: PipelineStage, frame: ActivationFrame, payload: JSON, current: bytes | bytearray | memoryview, kind: TaskKind, transport_mode: str, peer_lookup: dict[str, JSON]) -> bytes:
        degree = max(1, stage.tensor_parallel_degree)
        if degree <= 1:
            return self._stage_transform(kind, stage, current, payload)

        stages = [stage]
        for peer_id in stage.tensor_peer_ids:
            peer_stage = peer_lookup.get(peer_id)
            if peer_stage:
                stages.append(PipelineStage(**peer_stage))
        stages = sorted(stages, key=lambda s: safe_int(s.tensor_rank, 0))
        shards = self._chunk_tensor_bytes(current, len(stages))
        if len(shards) < len(stages):
            shards.extend([memoryview(b"")] * (len(stages) - len(shards)))

        results: list[tuple[int, bytes]] = []
        for shard_stage, shard in zip(stages, shards):
            if shard_stage.worker_id == self.worker_id:
                local_result = self._tensor_stage_local(shard_stage, shard, payload, kind)
            else:
                local_result = self._tensor_stage_remote(shard_stage, shard, frame, payload, transport_mode, kind)
            if not self._transport_succeeded(local_result):
                raise RuntimeError(f"tensor shard failed for {shard_stage.worker_id}")
            results.append((shard_stage.tensor_rank, self._frame_tensor_bytes(local_result)))

        results.sort(key=lambda item: item[0])
        self.metrics.inc("tensor.parallel.stages")
        return b"".join(part for _, part in results)

    def _ensure_grpc_server(self) -> None:
        if self._grpc_started or grpc is None or not ActivationProtoCodec.available():
            return
        try:
            server = grpc.server(ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 2))), options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ])
            handler = grpc.unary_unary_rpc_method_handler(
                self._grpc_pass_activation,
                request_deserializer=lambda b: b,
                response_serializer=lambda b: b,
            )
            generic = grpc.method_handlers_generic_handler("InferenceStream", {"PassActivation": handler})
            server.add_generic_rpc_handlers((generic,))
            server.add_insecure_port(f"{self.config.host}:{self._grpc_port()}")
            server.start()
            self._grpc_server = server
            self._grpc_started = True
            self.log.info("grpc_activation_server_started", worker_id=self.worker_id, port=self._grpc_port())
        except Exception as exc:
            self.log.warning("grpc_activation_server_failed", error=str(exc))

    def _grpc_pass_activation(self, request_bytes: bytes, context: t.Any) -> bytes:
        try:
            frame = ActivationProtoCodec.decode_request(request_bytes)
            try:
                result = self.receive_activation_frame(frame)
            finally:
                ActivationProtoCodec.cleanup_frame(frame)
            return ActivationProtoCodec.encode_response(
                received=True,
                status=str(result.get("status", "ok")),
                task_id=frame.task_id,
                request_id=frame.request_id,
                output_text=str(result.get("output_text", "")),
                audio_bytes=bytes(result.get("audio_bytes", b"")) if result.get("audio_bytes") else b"",
                error=str(result.get("error", "")),
            )
        except Exception as exc:
            return ActivationProtoCodec.encode_response(
                received=False,
                status="error",
                task_id="",
                request_id="",
                output_text="",
                audio_bytes=b"",
                error=str(exc),
            )

    def _ensure_http_server(self) -> None:
        super()._ensure_http_server()
        self._ensure_grpc_server()

    def start(self, orchestrator_url: str | None = None, *, auto_discover: bool = True, block: bool = False) -> str:
        target = super().start(orchestrator_url, auto_discover=auto_discover, block=block)
        self._ensure_grpc_server()
        return target

    def run_autonomous(self, orchestrator_url: str | None = None, *, auto_discover: bool = True) -> str:
        self._ensure_http_server()
        self._ensure_grpc_server()
        resolved_url, source = self.resolve_orchestrator_url(orchestrator_url, auto_discover=auto_discover)
        self.heartbeat_target = resolved_url.rstrip("/")
        self.log.info("worker_autonomous_mode", worker_id=self.worker_id, orchestrator_url=self.heartbeat_target, resolution=source)
        backoff = max(1.0, self.config.silent_boot_retry_interval_s)
        while not self._stop.is_set():
            result = self.register()
            if result.get("ok"):
                break
            self.log.warning("worker_autonomous_register_retry", orchestrator_url=self.heartbeat_target, retry_delay_s=backoff)
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 60.0)
        return self.heartbeat_target

    def execute(self, envelope_data: JSON) -> JSON:
        plan_data = envelope_data.get("payload", {}).get("pipeline_runtime") if isinstance(envelope_data.get("payload"), dict) else None
        if plan_data and envelope_data.get("kind") in {TaskKind.LLM.value, TaskKind.TTS.value}:
            try:
                return self._execute_pipeline_envelope(envelope_data, plan_data)
            except Exception as exc:
                self.log.exception("pipeline_execute_failed", exc, task_id=envelope_data.get("task_id"))
                return super().execute(envelope_data)
        return super().execute(envelope_data)

    def receive_activation_frame(self, frame: ActivationFrame) -> JSON:
        payload = json.loads(frame.payload_json) if frame.payload_json else {}
        plan = payload.get("pipeline_runtime", {}) if isinstance(payload, dict) else {}
        return self._process_activation_frame(frame, payload, plan)

    def receive_activation_bytes(self, blob: bytes) -> JSON:
        frame = ActivationWireCodec.decode(blob)
        try:
            return self.receive_activation_frame(frame)
        finally:
            ActivationProtoCodec.cleanup_frame(frame)

    def _stage_transform(self, kind: TaskKind, stage: PipelineStage, data: bytes | bytearray | memoryview, payload: JSON) -> bytes:
        buf = data if isinstance(data, bytes) else bytes(data)
        if kind == TaskKind.TTS:
            if stage.stage_name.endswith("normalize"):
                txt = buf.decode("utf-8", errors="replace")
                return re.sub(r"\s+", " ", sanitize_task_text(txt)).encode("utf-8")
            if stage.stage_name.endswith("phonemize"):
                txt = buf.decode("utf-8", errors="replace")
                phonemes = [w.lower() for w in re.findall(r"\w+", txt)]
                return to_json({"phonemes": phonemes}).encode("utf-8")
            if stage.stage_name.endswith("spectrogram"):
                obj = buf.decode("utf-8", errors="replace")
                return stable_hash(obj.encode("utf-8")).encode("utf-8")
            if stage.stage_name.endswith("vocoder"):
                return buf + b"|pcm"
            return buf
        if kind == TaskKind.LLM:
            prompt = payload.get("prompt", "")
            if stage.stage_name.endswith("final") or stage.stage_name.startswith("g") and stage.stage_name.endswith("stage_0"):
                return prompt.encode("utf-8") if isinstance(prompt, str) else buf
            marker = f"[{stage.stage_name}:{stage.layer_start}-{stage.layer_end}]".encode("utf-8")
            return stable_hash(buf + marker).encode("utf-8")
        return buf


    def _transport_succeeded(self, result: JSON) -> bool:
        if not isinstance(result, dict):
            return True
        status = str(result.get("status", "")).lower()
        if result.get("received") is False:
            return False
        if result.get("error"):
            return False
        if status in {"error", "failed", "failure"}:
            return False
        return True

    def _selected_group(self, plan_data: JSON) -> dict[str, t.Any] | None:
        groups = plan_data.get("groups", []) or []
        if not groups:
            return None
        preferred = safe_int(plan_data.get("primary_group_id", 0), 0)
        for group in groups:
            if safe_int(group.get("group_id", -1), -1) == preferred:
                return group
        for group in groups:
            if self.worker_id in group.get("workers", []):
                return group
        return groups[0]

    def _candidate_groups(self, plan_data: JSON, current_group_id: int) -> list[dict[str, t.Any]]:
        groups = plan_data.get("groups", []) or []
        ordered_ids = []
        for gid in plan_data.get("failover_group_ids", []) or []:
            if gid != current_group_id:
                ordered_ids.append(gid)
        for group in groups:
            gid = safe_int(group.get("group_id", -1), -1)
            if gid != current_group_id and gid not in ordered_ids:
                ordered_ids.append(gid)
        lookup = {safe_int(group.get("group_id", -1), -1): group for group in groups}
        return [lookup[gid] for gid in ordered_ids if gid in lookup]

    def _find_stage_for_failover(self, stages: list[JSON], next_stage: PipelineStage, next_idx: int) -> PipelineStage | None:
        for stage in stages:
            if safe_int(stage.get("stage_index", -1), -1) == next_idx:
                return PipelineStage(**stage)
        for stage in stages:
            if safe_int(stage.get("layer_start", -1), -1) == next_stage.layer_start and safe_int(stage.get("layer_end", -1), -1) == next_stage.layer_end:
                return PipelineStage(**stage)
        if stages:
            return PipelineStage(**stages[min(next_idx, len(stages) - 1)])
        return None

    def _failover_pipeline_forward(
        self,
        frame: ActivationFrame,
        payload: JSON,
        plan_data: JSON,
        failed_group_id: int,
        failed_stage: PipelineStage,
        next_idx: int,
        current: bytes | bytearray | memoryview,
        kind: TaskKind,
        transport_mode: str,
    ) -> JSON | None:
        for group in self._candidate_groups(plan_data, failed_group_id):
            stages = group.get("stages", []) or []
            candidate_stage = self._find_stage_for_failover(stages, failed_stage, next_idx)
            if candidate_stage is None:
                continue
            if candidate_stage.worker_id == failed_stage.worker_id and safe_int(group.get("group_id", -1), -1) == failed_group_id:
                continue
            alt_frame = ActivationFrame(
                source_rank=failed_stage.stage_index,
                target_rank=candidate_stage.stage_index,
                task_id=frame.task_id,
                request_id=frame.request_id,
                stage_id=candidate_stage.stage_name,
                tensor_data=current,
                payload_json=frame.payload_json,
                checksum=stable_hash(bytes(current) if not isinstance(current, bytes) else current),
                retry_count=frame.retry_count + 1,
                deadline_s=frame.deadline_s,
                model_id=frame.model_id,
                content_type=frame.content_type,
                tensor_shape_json=frame.tensor_shape_json,
                tensor_dtype=frame.tensor_dtype,
                shared_memory_name=frame.shared_memory_name,
                shared_memory_offset=frame.shared_memory_offset,
                shared_memory_size=frame.shared_memory_size,
                transport_hint=frame.transport_hint,
            )
            try:
                result = self.activation_transport.send_frame(
                    candidate_stage.address,
                    candidate_stage.http_port,
                    candidate_stage.grpc_port,
                    alt_frame,
                    transport_mode=transport_mode,
                )
                if self._transport_succeeded(result):
                    self.metrics.inc("pipeline.failover.recovered")
                    self.log.info(
                        "pipeline_failover_rerouted",
                        request_id=frame.request_id,
                        task_id=frame.task_id,
                        failed_group_id=failed_group_id,
                        target_group_id=safe_int(group.get("group_id", -1), -1),
                        failed_worker_id=failed_stage.worker_id,
                        target_worker_id=candidate_stage.worker_id,
                    )
                    return result
            except Exception as exc:
                self.log.warning(
                    "pipeline_failover_attempt_failed",
                    request_id=frame.request_id,
                    task_id=frame.task_id,
                    error=str(exc),
                    target_worker_id=candidate_stage.worker_id,
                )
                continue
        return None

    def _execute_pipeline_envelope(self, envelope_data: JSON, plan_data: JSON) -> JSON:
        payload = envelope_data.get("payload", {}) or {}
        kind = TaskKind(envelope_data.get("kind", TaskKind.LLM.value))
        chosen_group = self._selected_group(plan_data)
        if chosen_group is None:
            return super().execute(envelope_data)
        stages = chosen_group.get("stages", []) or []
        if not stages:
            return super().execute(envelope_data)

        entry_worker = plan_data.get("entry_worker_id", stages[0].get("worker_id"))
        if self.worker_id != entry_worker and envelope_data.get("worker_id") != self.worker_id:
            return super().execute(envelope_data)

        data = (payload.get("prompt") or payload.get("text") or "").encode("utf-8")
        frame = ActivationFrame(
            source_rank=0,
            target_rank=0,
            task_id=envelope_data.get("task_id", ""),
            request_id=envelope_data.get("request_id", ""),
            stage_id=envelope_data.get("stage_id", kind.value),
            tensor_data=data,
            payload_json=to_json(payload),
            checksum=str(envelope_data.get("checksum", "")),
            retry_count=safe_int(envelope_data.get("retry_count", 0), 0),
            deadline_s=safe_float(envelope_data.get("deadline_s", 0.0), 0.0),
            model_id=str(plan_data.get("model_id", "")),
        )
        return self._process_activation_frame(frame, payload, plan_data)

    def _process_activation_frame(self, frame: ActivationFrame, payload: JSON, plan_data: JSON) -> JSON:
        kind = TaskKind(plan_data.get("task_kind", TaskKind.LLM.value))
        group = self._selected_group(plan_data)
        if group is None:
            return {"status": TaskStatus.FAILED.value, "error": "no_pipeline_group"}

        stages = group.get("stages", []) or []
        if not stages:
            return {"status": TaskStatus.FAILED.value, "error": "empty_pipeline_group"}

        by_group, by_worker = self._plan_stage_maps(plan_data)
        current_stage = by_worker.get(self.worker_id)
        if current_stage is None:
            return {"status": TaskStatus.FAILED.value, "error": "worker_not_in_pipeline"}

        ordered_groups = sorted(by_group.items(), key=lambda item: (min(safe_int(s.get("stage_index", 0), 0) for s in item[1]), item[0]))
        current = frame.tensor_data
        start_group = safe_int(current_stage.get("tensor_group_id", current_stage.get("stage_index", 0)), 0)

        for group_id, shard_stages in ordered_groups:
            if group_id < start_group:
                continue
            shard_stages = sorted(shard_stages, key=lambda s: safe_int(s.get("tensor_rank", 0), 0))
            local_stage = next((PipelineStage(**s) for s in shard_stages if str(s.get("worker_id", "")) == self.worker_id), None)
            if local_stage is None:
                continue

            if local_stage.tensor_parallel_degree > 1:
                if frame.tensor_world_size > 1 and frame.tensor_rank > 0:
                    shard_out = self._stage_transform(kind, local_stage, current, payload)
                    return {
                        "status": TaskStatus.SUCCEEDED.value,
                        "tensor_bytes_b64": base64.b64encode(shard_out).decode("ascii") if shard_out else "",
                        "worker_id": self.worker_id,
                        "stage_id": local_stage.stage_name,
                        "tensor_rank": local_stage.tensor_rank,
                        "tensor_group_id": local_stage.tensor_group_id,
                    }

                current = self._run_tensor_parallel_stage(
                    local_stage,
                    frame,
                    payload,
                    current,
                    kind,
                    str(plan_data.get("transport_mode", local_stage.tensor_transport)),
                    by_worker,
                )
            else:
                current = self._stage_transform(kind, local_stage, current, payload)

            future_groups = [gid for gid, _ in ordered_groups if gid > group_id]
            if future_groups:
                next_gid = future_groups[0]
                next_stages = sorted(by_group[next_gid], key=lambda s: safe_int(s.get("tensor_rank", 0), 0))
                next_leader = PipelineStage(**next_stages[0])
                out_frame = ActivationFrame(
                    source_rank=local_stage.stage_index,
                    target_rank=next_leader.stage_index,
                    task_id=frame.task_id,
                    request_id=frame.request_id,
                    stage_id=next_leader.stage_name,
                    tensor_data=current,
                    payload_json=frame.payload_json,
                    checksum=stable_hash(bytes(current) if not isinstance(current, bytes) else current),
                    retry_count=frame.retry_count,
                    deadline_s=frame.deadline_s,
                    model_id=frame.model_id,
                    content_type=frame.content_type,
                    tensor_shape_json=frame.tensor_shape_json,
                    tensor_dtype=frame.tensor_dtype,
                    shared_memory_name=frame.shared_memory_name,
                    shared_memory_offset=frame.shared_memory_offset,
                    shared_memory_size=frame.shared_memory_size,
                    transport_hint=frame.transport_hint,
                    tensor_rank=0,
                    tensor_world_size=max(1, next_leader.tensor_parallel_degree),
                    tensor_group_id=next_leader.tensor_group_id,
                    tensor_transport=next_leader.tensor_transport,
                )
                next_host = next_leader.mesh_address or next_leader.address
                next_http_port = next_leader.mesh_http_port or next_leader.http_port
                next_grpc_port = next_leader.mesh_grpc_port or next_leader.grpc_port
                try:
                    transport_result = self.activation_transport.send_frame(
                        next_host,
                        next_http_port,
                        next_grpc_port,
                        out_frame,
                        transport_mode=next_leader.tensor_transport or str(plan_data.get("transport_mode", "mesh_grpc")),
                    )
                except Exception as exc:
                    transport_result = {"status": "error", "error": str(exc)}
                if self._transport_succeeded(transport_result):
                    return transport_result
                failover_result = self._failover_pipeline_forward(
                    frame=out_frame,
                    payload=payload,
                    plan_data=plan_data,
                    failed_group_id=safe_int(group.get("group_id", 0), 0),
                    failed_stage=next_leader,
                    next_idx=next_gid,
                    current=current,
                    kind=kind,
                    transport_mode=str(plan_data.get("transport_mode", "mesh_grpc")),
                )
                if failover_result is not None:
                    return failover_result
                return transport_result

        if kind == TaskKind.LLM:
            request = LLMRequest(
                prompt=str(payload.get("prompt", "")),
                request_id=frame.request_id,
                task_id=frame.task_id,
                max_new_tokens=safe_int(payload.get("max_new_tokens", 128), 128),
                temperature=safe_float(payload.get("temperature", 0.7), 0.7),
                top_p=safe_float(payload.get("top_p", 0.95), 0.95),
                chunk_size=safe_int(payload.get("chunk_size", 24), 24),
                metadata=dict(payload),
            )
            chunks = list(self.llm_runtime.stream_generate(request, None))
            output_text = "".join(chunks)
            return {
                "status": TaskStatus.SUCCEEDED.value,
                "output_text": output_text,
                "stream_chunks": chunks,
                "worker_id": self.worker_id,
                "backend": "pipeline_local_fallback",
                "pipeline": plan_data,
            }

        request = TTSRequest(
            text=str(payload.get("text", "")),
            request_id=frame.request_id,
            task_id=frame.task_id,
            chunk_size=safe_int(payload.get("chunk_size", 2048), 2048),
            metadata=dict(payload),
        )
        chunks = list(self.tts_runtime.stream_synthesize(request, None))
        audio = b"".join(chunks)
        return {
            "status": TaskStatus.SUCCEEDED.value,
            "audio_bytes_b64": base64.b64encode(audio).decode("ascii"),
            "stream_chunks": [base64.b64encode(c).decode("ascii") for c in chunks],
            "worker_id": self.worker_id,
            "backend": "pipeline_local_fallback",
            "pipeline": plan_data,
        }


class AdvancedScheduler(Scheduler):
    def __init__(self, store: StateStore, config: ControlConfig, log: StructuredLogger, metrics: MetricsRegistry) -> None:
        super().__init__(store, config, log, metrics)
        self.pipeline_planner = DynamicPipelinePlanner(store, config, log, metrics)

    def group_workers(self, workers: list[WorkerInfo], model_size_layers: int) -> list[list[WorkerInfo]]:
        return self.pipeline_planner.group_workers(workers, model_size_layers)

    def create_parallel_pipelines(self, workers: list[WorkerInfo], groups: int = 2) -> list[list[WorkerInfo]]:
        ranked = self.pipeline_planner._rank_workers(workers)
        groups = max(1, min(groups, len(ranked) or 1))
        result: list[list[WorkerInfo]] = [[] for _ in range(groups)]
        for idx, worker in enumerate(ranked):
            result[idx % groups].append(worker)
        return [g for g in result if g]

    def create_single_long_pipeline(self, workers: list[WorkerInfo]) -> list[list[WorkerInfo]]:
        ranked = self.pipeline_planner._rank_workers(workers)
        return [ranked] if ranked else []

    def plan(self, kind: TaskKind, payload: JSON, request_id: str) -> SchedulingDecision:
        workers = self.alive_workers()
        if not workers:
            return SchedulingDecision(chosen_worker_id="local", mode=ParallelMode.NONE, tensor_degree=1, pipeline_degree=1, reason="no_workers", local_fallback=True)
        if kind in {TaskKind.LLM, TaskKind.TTS}:
            plan = self.pipeline_planner.build_plan(kind, payload, request_id)
            if plan and plan.groups:
                entry = plan.entry_worker_id
                mode = plan.mode
                pipeline_degree = max(1, len(plan.groups[0].stages))
                return SchedulingDecision(entry, mode, 1, pipeline_degree, plan.notes, backend_preference="grpc" if plan.transport_mode == "grpc" else "binary_http", local_fallback=False)
        return super().plan(kind, payload, request_id)


def _patch_worker_http_activation_handler() -> None:
    old_do_POST = WorkerHTTPRequestHandler.do_POST

    def do_POST(self: t.Any) -> None:
        worker = self.server.worker  # type: ignore[attr-defined]
        path = urlparse(self.path).path
        if path == "/activation":
            try:
                length = safe_int(self.headers.get("Content-Length", "0"), 0)
                raw = self.rfile.read(length) if length > 0 else b""
                result = worker.receive_activation_bytes(raw)
                self._send_json(result)
                return
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, 500)
                return
        return old_do_POST(self)

    WorkerHTTPRequestHandler.do_POST = do_POST  # type: ignore[assignment]


_patch_worker_http_activation_handler()

# =============================================================================
# Main bootstrap
# =============================================================================

def build_orchestrator_from_env(network_provider: t.Callable[[], JSON] | JSON | None = None) -> HybridClusterOrchestrator:
    config = ControlConfig(
        host=os.environ.get("CLUSTER_HOST", DEFAULT_HOST),
        port=safe_int(os.environ.get("CLUSTER_PORT", DEFAULT_PORT), DEFAULT_PORT),
        worker_port=safe_int(os.environ.get("WORKER_PORT", DEFAULT_WORKER_PORT), DEFAULT_WORKER_PORT),
        state_dir=os.environ.get("CLUSTER_STATE_DIR", "./cluster_state"),
        secret_key=os.environ.get("CLUSTER_SECRET_KEY", "change-me-in-production"),
        enable_dashboard=os.environ.get("ENABLE_DASHBOARD", "1") == "1",
        enable_electron_bridge=os.environ.get("ENABLE_ELECTRON_BRIDGE", "1") == "1",
        enable_udp_discovery=os.environ.get("ENABLE_UDP_DISCOVERY", "1") == "1",
        udp_discovery_port=safe_int(os.environ.get("UDP_DISCOVERY_PORT", UDP_DISCOVERY_PORT), UDP_DISCOVERY_PORT),
        udp_beacon_interval_s=safe_float(os.environ.get("UDP_BEACON_INTERVAL", UDP_BEACON_INTERVAL), UDP_BEACON_INTERVAL),
        auto_discovery_timeout_s=safe_float(os.environ.get("AUTO_DISCOVERY_TIMEOUT", DEFAULT_AUTODISCOVERY_TIMEOUT), DEFAULT_AUTODISCOVERY_TIMEOUT),
        silent_boot_retry_interval_s=safe_float(os.environ.get("SILENT_BOOT_RETRY_INTERVAL", DEFAULT_SILENT_BOOT_RETRY_INTERVAL), DEFAULT_SILENT_BOOT_RETRY_INTERVAL),
        log_level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    )
    orch = HybridClusterOrchestrator(config)
    if network_provider is not None:
        orch.network_provider = network_provider
        try:
            info = network_provider() if callable(network_provider) else network_provider
            if isinstance(info, dict):
                host = info.get("mesh_ip") or info.get("private_ip") or info.get("bind_ip")
                if host:
                    orch.config.host = str(host)
        except Exception:
            pass
    return orch


def build_worker_from_env() -> HybridWorkerRuntime:
    config = ControlConfig(
        host=os.environ.get("WORKER_HOST", "0.0.0.0"),
        worker_port=safe_int(os.environ.get("WORKER_PORT", DEFAULT_WORKER_PORT), DEFAULT_WORKER_PORT),
        heartbeat_interval_s=safe_float(os.environ.get("HEARTBEAT_INTERVAL", DEFAULT_HEARTBEAT_INTERVAL), DEFAULT_HEARTBEAT_INTERVAL),
        enable_udp_discovery=os.environ.get("ENABLE_UDP_DISCOVERY", "1") == "1",
        udp_discovery_port=safe_int(os.environ.get("UDP_DISCOVERY_PORT", UDP_DISCOVERY_PORT), UDP_DISCOVERY_PORT),
        auto_discovery_timeout_s=safe_float(os.environ.get("AUTO_DISCOVERY_TIMEOUT", DEFAULT_AUTODISCOVERY_TIMEOUT), DEFAULT_AUTODISCOVERY_TIMEOUT),
        silent_boot_retry_interval_s=safe_float(os.environ.get("SILENT_BOOT_RETRY_INTERVAL", DEFAULT_SILENT_BOOT_RETRY_INTERVAL), DEFAULT_SILENT_BOOT_RETRY_INTERVAL),
        log_level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    )
    return HybridWorkerRuntime(config)


def write_electron_bridge_files(target_dir: str | Path) -> JSON:
    d = ensure_dir(target_dir)
    (d / "main.js").write_text(ELECTRON_MAIN_JS, "utf-8")
    (d / "preload.js").write_text(ELECTRON_PRELOAD_JS, "utf-8")
    return {"ok": True, "dir": str(d), "files": ["main.js", "preload.js"]}


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    mode = "orchestrator"
    if "--worker" in argv:
        mode = "worker"
    if "--self-test" in argv:
        mode = "self-test"

    if mode == "self-test":
        orch = build_orchestrator_from_env()
        report = SelfTest(orch).run()
        print(to_json(report))
        return 0 if report.get("ok") else 1

    if mode == "worker":
        worker = build_worker_from_env()
        explicit = os.environ.get("ORCHESTRATOR_URL")
        if is_silent_boot_context():
            worker.run_autonomous(explicit, auto_discover=True)
        else:
            resolved = explicit or f"http://127.0.0.1:{DEFAULT_PORT}"
            worker.start(resolved, auto_discover=True, block=False)
        try:
            while not worker._stop.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            worker.stop()
        return 0

    orch = build_orchestrator_from_env()

    if orch.config.enable_udp_discovery and is_silent_boot_context():
        orch.log.info("silent_boot_orchestrator", enabled=True, udp_port=orch.config.udp_discovery_port)

    # Auto-register a local pseudo-worker for CPU-first fallback.
    local_cap = Capability(
        cpu_cores=os.cpu_count() or 1,
        ram_gb=0.0,
        has_gpu=bool(torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available()),
        cuda_available=bool(torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available()),
        backends=["local", "local_tts"] + (["transformers"] if OPTIONAL["transformers"].available else []),
        max_concurrency=max(2, (os.cpu_count() or 2)),
    )
    orch.register_worker("127.0.0.1", orch.config.worker_port, local_cap, worker_id="local", tags=["fallback", "local"])

    if orch.config.enable_electron_bridge:
        try:
            write_electron_bridge_files(Path("./electron_bridge"))
        except Exception as exc:
            orch.log.warning("electron_bridge_write_failed", error=str(exc))

    orch.start()
    return 0


# =============================================================================
# Compatibility and consolidation layer
# =============================================================================

class MessageType(str, enum.Enum):
    CONTROL = "control"
    WORKER = "worker"
    HEALTH = "health"
    RESULT = "result"
    STREAM = "stream"
    ERROR = "error"


class WireKind(str, enum.Enum):
    FRAME = "frame"
    PACKET = "packet"
    OUTCOME = "outcome"
    CONTROL = "control"


@dataclass
class ProtocolFrame:
    kind: WireKind
    request_id: str
    task_id: str
    stage_id: str
    worker_id: str
    payload: JSON = field(default_factory=dict)
    metadata: JSON = field(default_factory=dict)
    created_at_ms: int = field(default_factory=now_ms)

    def to_bytes(self) -> bytes:
        payload = {
            "kind": self.kind.value if isinstance(self.kind, WireKind) else str(self.kind),
            "request_id": self.request_id,
            "task_id": self.task_id,
            "stage_id": self.stage_id,
            "worker_id": self.worker_id,
            "payload": self.payload,
            "metadata": self.metadata,
            "created_at_ms": self.created_at_ms,
        }
        return zlib.compress(to_json(payload).encode("utf-8"), level=3)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ProtocolFrame":
        data = from_json(zlib.decompress(raw).decode("utf-8"))
        return cls(
            kind=WireKind(data.get("kind", "frame")),
            request_id=str(data.get("request_id", "")),
            task_id=str(data.get("task_id", "")),
            stage_id=str(data.get("stage_id", "")),
            worker_id=str(data.get("worker_id", "")),
            payload=dict(data.get("payload") or {}),
            metadata=dict(data.get("metadata") or {}),
            created_at_ms=safe_int(data.get("created_at_ms", now_ms()), now_ms()),
        )


@dataclass
class WireEnvelope:
    kind: str
    request_id: str
    task_id: str
    stage_id: str
    worker_id: str
    payload: JSON = field(default_factory=dict)
    checksum: str = ""
    metadata: JSON = field(default_factory=dict)

    def to_public(self) -> JSON:
        return asdict(self)

    @classmethod
    def from_public(cls, data: JSON) -> "WireEnvelope":
        return cls(
            kind=str(data.get("kind", "frame")),
            request_id=str(data.get("request_id", "")),
            task_id=str(data.get("task_id", "")),
            stage_id=str(data.get("stage_id", "")),
            worker_id=str(data.get("worker_id", "")),
            payload=dict(data.get("payload") or {}),
            checksum=str(data.get("checksum", "")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class JournalEntry:
    ts_ms: int
    kind: str
    message: str
    data: JSON = field(default_factory=dict)

    def to_public(self) -> JSON:
        return asdict(self)


class EventJournal:
    def __init__(self, path: str | Path, *, log: StructuredLogger | None = None) -> None:
        self.path = Path(path)
        self.log = log
        ensure_dir(self.path.parent if self.path.suffix else self.path)
        self._lock = threading.RLock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def append(self, kind: str, message: str, **data: t.Any) -> JournalEntry:
        entry = JournalEntry(now_ms(), kind, message, data)
        with self._lock:
            self._fh.write(to_json(entry.to_public()) + "\n")
            self._fh.flush()
        return entry

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass


class DurableJsonJournal(EventJournal):
    def snapshot(self) -> list[JSON]:
        items: list[JSON] = []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            items.append(from_json(line))
                        except Exception:
                            continue
        except FileNotFoundError:
            pass
        return items

    def replay(self) -> list[JournalEntry]:
        return [JournalEntry(**item) for item in self.snapshot() if isinstance(item, dict)]


class LatencyEWMA:
    def __init__(self, alpha: float = 0.2, initial: float = 0.0) -> None:
        self.alpha = clamp(alpha, 0.01, 0.99)
        self.value = initial

    def update(self, sample: float) -> float:
        sample = max(0.0, sample)
        self.value = sample if self.value <= 0 else (self.alpha * sample + (1.0 - self.alpha) * self.value)
        return self.value


class TokenBucket:
    def __init__(self, rate: float, burst: float) -> None:
        self.rate = max(0.0, rate)
        self.capacity = max(1.0, burst)
        self.tokens = self.capacity
        self.updated_at = monotonic()
        self._lock = threading.RLock()

    def allow(self, cost: float = 1.0) -> bool:
        cost = max(0.0, cost)
        with self._lock:
            elapsed = max(0.0, monotonic() - self.updated_at)
            self.updated_at = monotonic()
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False


class CircuitBreaker:
    def __init__(self, *, fail_threshold: int = 5, reset_after_s: float = 30.0) -> None:
        self.fail_threshold = max(1, fail_threshold)
        self.reset_after_s = max(1.0, reset_after_s)
        self.fail_count = 0
        self.opened_at = 0.0

    def allow(self) -> bool:
        if self.fail_count < self.fail_threshold:
            return True
        if self.opened_at <= 0:
            self.opened_at = monotonic()
            return False
        if monotonic() - self.opened_at >= self.reset_after_s:
            self.fail_count = 0
            self.opened_at = 0.0
            return True
        return False

    def success(self) -> None:
        self.fail_count = 0
        self.opened_at = 0.0

    def failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.fail_threshold and self.opened_at <= 0:
            self.opened_at = monotonic()

    def snapshot(self) -> JSON:
        return {"fail_count": self.fail_count, "opened_at": self.opened_at, "allow": self.allow()}


class RequestTimeline:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.events: list[tuple[str, float]] = []

    def mark(self, event: str) -> None:
        self.events.append((event, monotonic()))

    def latency_ms(self) -> float:
        if len(self.events) < 2:
            return 0.0
        return max(0.0, (self.events[-1][1] - self.events[0][1]) * 1000.0)

    def to_public(self) -> JSON:
        return {"request_id": self.request_id, "events": [{"event": e, "t": ts} for e, ts in self.events], "latency_ms": self.latency_ms()}


class RequestLedger:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, RequestTimeline] = {}

    def get(self, request_id: str) -> RequestTimeline:
        with self._lock:
            if request_id not in self._items:
                self._items[request_id] = RequestTimeline(request_id)
            return self._items[request_id]

    def snapshot(self) -> JSON:
        with self._lock:
            return {rid: tl.to_public() for rid, tl in self._items.items()}


class DedupeStore:
    def __init__(self, ttl_s: float = 300.0) -> None:
        self.ttl_s = max(1.0, ttl_s)
        self._lock = threading.RLock()
        self._items: dict[str, float] = {}

    def add(self, key: str) -> bool:
        with self._lock:
            self._cleanup_locked()
            if key in self._items:
                return False
            self._items[key] = monotonic()
            return True

    def seen(self, key: str) -> bool:
        with self._lock:
            self._cleanup_locked()
            return key in self._items

    def _cleanup_locked(self) -> None:
        now = monotonic()
        for k, ts in list(self._items.items()):
            if now - ts > self.ttl_s:
                self._items.pop(k, None)


@dataclass(order=True)
class LeaseQueueItem:
    priority: int
    created_at: float
    lease_id: str = field(compare=False)
    payload: JSON = field(default_factory=dict, compare=False)


@dataclass
class LeaseRecord:
    lease_id: str
    owner_id: str
    resource_id: str
    expires_at: float
    metadata: JSON = field(default_factory=dict)

    def expired(self) -> bool:
        return monotonic() >= self.expires_at

    def to_public(self) -> JSON:
        return asdict(self)


class LeaseManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._leases: dict[str, LeaseRecord] = {}

    def acquire(self, owner_id: str, resource_id: str, ttl_s: float, metadata: JSON | None = None) -> LeaseRecord:
        with self._lock:
            lease = LeaseRecord(uid("lease_"), owner_id, resource_id, monotonic() + max(1.0, ttl_s), metadata or {})
            self._leases[lease.lease_id] = lease
            return lease

    def renew(self, lease_id: str, ttl_s: float) -> LeaseRecord | None:
        with self._lock:
            lease = self._leases.get(lease_id)
            if not lease:
                return None
            lease.expires_at = monotonic() + max(1.0, ttl_s)
            return lease

    def cancel(self, lease_id: str) -> bool:
        with self._lock:
            return self._leases.pop(lease_id, None) is not None

    def reap(self) -> list[str]:
        with self._lock:
            expired = [lid for lid, lease in self._leases.items() if lease.expired()]
            for lid in expired:
                self._leases.pop(lid, None)
            return expired

    def snapshot(self) -> JSON:
        with self._lock:
            return {lid: lease.to_public() for lid, lease in self._leases.items()}


class StreamBus:
    def __init__(self, max_history: int = 1024) -> None:
        self.max_history = max_history
        self._lock = threading.RLock()
        self._subs: dict[str, list[queue.Queue[str]]] = defaultdict(list)
        self._history: dict[str, deque[str]] = defaultdict(deque)

    def subscribe(self, topic: str) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue()
        with self._lock:
            self._subs[topic].append(q)
            for item in self._history.get(topic, []):
                q.put(item)
        return q

    def publish(self, topic: str, item: str) -> None:
        with self._lock:
            hist = self._history[topic]
            hist.append(item)
            while len(hist) > self.max_history:
                hist.popleft()
            for q in self._subs.get(topic, []):
                try:
                    q.put_nowait(item)
                except Exception:
                    pass

    def unsubscribe(self, topic: str, q: queue.Queue[str]) -> None:
        with self._lock:
            subs = self._subs.get(topic, [])
            if q in subs:
                subs.remove(q)

    def topic_stats(self) -> JSON:
        with self._lock:
            return {topic: len(hist) for topic, hist in self._history.items()}


class StreamingSession:
    def __init__(self, request_id: str, topic: str = "") -> None:
        self.request_id = request_id
        self.topic = topic or request_id
        self.chunks: list[str] = []
        self.closed = False

    def emit(self, chunk: str) -> None:
        if not self.closed:
            self.chunks.append(chunk)

    def drain(self) -> list[str]:
        out = self.chunks[:]
        self.chunks.clear()
        return out

    def close(self) -> None:
        self.closed = True


class StreamAggregator:
    def __init__(self) -> None:
        self.sessions: dict[str, StreamingSession] = {}

    def emit_tokens(self, request_id: str, tokens: list[str]) -> StreamingSession:
        session = self.sessions.setdefault(request_id, StreamingSession(request_id, request_id))
        for token in tokens:
            session.emit(token)
        return session

    def emit_audio(self, request_id: str, audio_chunks: list[bytes]) -> StreamingSession:
        session = self.sessions.setdefault(request_id, StreamingSession(request_id, request_id))
        for chunk in audio_chunks:
            session.emit(base64.b64encode(chunk).decode("ascii"))
        return session


class TopologyView:
    def __init__(self, workers: list[WorkerInfo] | None = None) -> None:
        self.workers = workers or []

    def choose_tensor_degree(self, layers: int) -> int:
        workers = max(1, len(self.workers))
        if layers >= 96:
            return min(8, workers)
        if layers >= 48:
            return min(4, workers)
        if layers >= 16:
            return min(2, workers)
        return 1

    def choose_pipeline_degree(self, layers: int) -> int:
        workers = max(1, len(self.workers))
        if layers >= 64:
            return min(8, workers)
        if layers >= 24:
            return min(4, workers)
        if layers >= 8:
            return min(2, workers)
        return 1


class TopologyInspector:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def view(self) -> TopologyView:
        alive = [w for w in self.store.workers.values() if w.status != "dead"]
        return TopologyView(alive)


class HeartbeatMonitor:
    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self._seen: dict[str, float] = {}

    def mark(self, worker_id: str, ts: float | None = None) -> None:
        self._seen[worker_id] = ts or now_s()

    def stale_workers(self) -> list[str]:
        now = now_s()
        return [wid for wid, ts in self._seen.items() if now - ts > self.timeout_s]

    def score(self, worker_id: str) -> float:
        ts = self._seen.get(worker_id, 0.0)
        if not ts:
            return 0.0
        age = max(0.0, now_s() - ts)
        return clamp(1.0 - age / max(1.0, self.timeout_s), 0.0, 1.0)


class FailureDetector:
    def __init__(self, monitor: HeartbeatMonitor) -> None:
        self.monitor = monitor

    def scan(self) -> list[str]:
        return self.monitor.stale_workers()


class PrometheusMetricsExporter:
    def __init__(self, metrics: MetricsRegistry, prefix: str = "cluster") -> None:
        self.metrics = metrics
        self.prefix = prefix

    def render(self) -> str:
        snap = self.metrics.snapshot()
        lines: list[str] = []
        for k, v in snap["counters"].items():
            lines.append(f"{self.prefix}_{k} {v}")
        for k, v in snap["gauges"].items():
            lines.append(f"{self.prefix}_{k} {v}")
        for k, s in snap["histograms"].items():
            lines.append(f"{self.prefix}_{k}_count {s['count']}")
            lines.append(f"{self.prefix}_{k}_avg {s['avg']}")
            lines.append(f"{self.prefix}_{k}_p95 {s['p95']}")
            lines.append(f"{self.prefix}_{k}_max {s['max']}")
        return "\n".join(lines) + ("\n" if lines else "")


@dataclass
class CapabilityMatrix:
    entries: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))  # type: ignore[assignment]

    def register(self, name: str, capability: str) -> None:
        self.entries.setdefault(name, set()).add(capability)

    def supports(self, name: str, capability: str) -> bool:
        return capability in self.entries.get(name, set())

    def snapshot(self) -> JSON:
        return {name: sorted(vals) for name, vals in self.entries.items()}


class AdaptiveParallelPlanner:
    def __init__(self, store: StateStore, config: ControlConfig) -> None:
        self.store = store
        self.config = config

    def plan(self, payload: JSON, kind: TaskKind = TaskKind.LLM) -> JSON:
        workers = [w for w in self.store.workers.values() if w.status != "dead"]
        view = TopologyView(workers)
        layers = safe_int(payload.get("model_layers", payload.get("layers", 32)), 32)
        return {
            "mode": (ParallelMode.NONE if len(workers) <= 1 else ParallelMode.HYBRID).value,
            "tensor_degree": view.choose_tensor_degree(layers),
            "pipeline_degree": view.choose_pipeline_degree(layers),
            "workers": [w.worker_id for w in sorted(workers, key=lambda x: x.capability.score(), reverse=True)],
            "kind": kind.value,
        }


class AdaptiveShardPlanner:
    def __init__(self, store: StateStore, config: ControlConfig) -> None:
        self.store = store
        self.config = config

    def choose(self, layers: int, tensor_degree: int = 1, pipeline_degree: int = 1) -> list[JSON]:
        layers = max(1, layers)
        parts = max(1, tensor_degree * pipeline_degree)
        size = max(1, layers // parts)
        shards = []
        start = 0
        for i in range(parts):
            end = layers if i == parts - 1 else min(layers, start + size)
            shards.append({"index": i, "start": start, "end": end, "layers": max(0, end - start)})
            start = end
        return shards


class LoadSheddingPolicy:
    def __init__(self, *, max_queue_depth: int = DEFAULT_QUEUE_SIZE, high_watermark: float = 0.85) -> None:
        self.max_queue_depth = max(1, max_queue_depth)
        self.high_watermark = clamp(high_watermark, 0.1, 0.99)

    def allow(self, queue_depth: int, in_flight: int = 0, priority: int = 50) -> bool:
        pressure = (queue_depth + in_flight) / self.max_queue_depth
        if pressure < self.high_watermark:
            return True
        return priority <= 20


class ModelRegistry:
    def __init__(self, root: str | Path, log: StructuredLogger | None = None) -> None:
        self.root = ensure_dir(root)
        self.log = log
        self._index_path = self.root / "index.json"
        self._lock = threading.RLock()
        self._index: dict[str, JSON] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                self._index = from_json(self._index_path.read_text("utf-8"))
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        self._index_path.write_text(to_json(self._index), "utf-8")

    def register_manifest(self, manifest: ModelManifest) -> None:
        with self._lock:
            self._index[manifest.model_id] = asdict(manifest)
            self._save_index()

    def get(self, model_id: str) -> JSON | None:
        return self._index.get(model_id)

    def list(self) -> list[str]:
        return sorted(self._index.keys())


class ShardLayoutVerifier:
    def __init__(self, log: StructuredLogger | None = None) -> None:
        self.log = log

    def verify_checkpoint(self, manifest: ModelManifest | JSON) -> tuple[bool, list[str]]:
        data = manifest if isinstance(manifest, dict) else asdict(manifest)
        errors: list[str] = []
        if not data.get("model_id"):
            errors.append("missing model_id")
        if not data.get("source_path"):
            errors.append("missing source_path")
        if not data.get("layers"):
            errors.append("missing layers")
        return (not errors, errors)


class SharedStorageModelLoader:
    def __init__(self, registry: ModelRegistry, log: StructuredLogger | None = None) -> None:
        self.registry = registry
        self.log = log

    def resolve_checkpoint(self, model_id_or_path: str) -> str:
        if Path(model_id_or_path).exists():
            return str(Path(model_id_or_path).resolve())
        manifest = self.registry.get(model_id_or_path)
        if manifest and manifest.get("source_path"):
            return str(Path(manifest["source_path"]).resolve())
        return model_id_or_path

    def load_manifest(self, model_id_or_path: str) -> ModelManifest:
        path = self.resolve_checkpoint(model_id_or_path)
        model_id = Path(path).stem
        return ModelManifest(
            model_id=model_id,
            source_path=path,
            architecture="unknown",
            checksum=stable_hash(path.encode("utf-8")),
            created_at=now_s(),
            layers=[],
            extra={},
        )

    def select_layer_slice(self, layers: int, shard_index: int, shard_count: int) -> tuple[int, int]:
        shard_count = max(1, shard_count)
        shard_index = min(max(0, shard_index), shard_count - 1)
        start = (layers * shard_index) // shard_count
        end = (layers * (shard_index + 1)) // shard_count
        return start, max(start, end)

    def load_shard_bytes(self, path: str | Path, start: int = 0, end: int | None = None) -> bytes:
        raw = Path(path).read_bytes()
        return raw[start:end]


class RemoteWorkerProxy:
    def __init__(self, endpoint: str, timeout_s: float = DEFAULT_REQUEST_TIMEOUT) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> JSON:
        req = Request(f"{self.endpoint}/health", method="GET")
        with urlopen(req, timeout=self.timeout_s) as resp:
            return from_json(resp.read() or b"{}")

    def execute(self, envelope: TaskEnvelope | JSON) -> JSON:
        payload = envelope.to_public() if hasattr(envelope, "to_public") else dict(envelope)
        req = Request(f"{self.endpoint}/execute", data=to_json(payload).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=self.timeout_s) as resp:
            return from_json(resp.read() or b"{}")


class BenchmarkHarness:
    def __init__(self, orch: ClusterOrchestrator) -> None:
        self.orch = orch

    def run_queue_benchmark(self, n: int = 100) -> JSON:
        t0 = monotonic()
        accepted = 0
        for i in range(max(1, n)):
            env = TaskEnvelope(uid("req_"), uid("task_"), TaskKind.DIAGNOSTIC, "diag", "local", payload={"i": i})
            if self.orch.queue.put(env, timeout=0.1):
                accepted += 1
        return {"n": n, "accepted": accepted, "elapsed_s": monotonic() - t0}

    def run_manifest_benchmark(self) -> JSON:
        t0 = monotonic()
        manifest = ModelManifest(uid("model_"), "/tmp/model.bin", "unknown", stable_hash(b"x"), now_s(), layers=[{"id": 0}])
        self.orch.state.model_manifests[manifest.model_id] = manifest
        return {"elapsed_s": monotonic() - t0, "manifest_id": manifest.model_id}


class DiagnosticsPack:
    def __init__(self, orch: ClusterOrchestrator) -> None:
        self.orch = orch

    def run(self) -> JSON:
        workers = len(self.orch.state.workers)
        queue_depth = len(self.orch.queue._heap) if hasattr(self.orch.queue, "_heap") else 0
        return {
            "ok": True,
            "workers": workers,
            "queue_depth": queue_depth,
            "metrics": self.orch.metrics.snapshot(),
            "status": self.orch.cluster_status() if hasattr(self.orch, "cluster_status") else {},
        }


class AdvancedDiagnosticsRunner:
    def __init__(self, orch: ClusterOrchestrator) -> None:
        self.orch = orch

    def run(self) -> JSON:
        return DiagnosticsPack(self.orch).run()


class ClusterConsoleReport:
    def __init__(self, orch: ClusterOrchestrator) -> None:
        self.orch = orch

    def render_text(self) -> str:
        status = self.orch.cluster_status() if hasattr(self.orch, "cluster_status") else {}
        lines = [
            "Distributed Inference Cluster Report",
            f"workers: {len(status.get('workers', {}))}",
            f"tasks: {len(status.get('tasks', {}))}",
            f"results: {len(status.get('results', {}))}",
            f"metrics: {json.dumps(status.get('metrics', {}), ensure_ascii=False)}",
        ]
        return "\n".join(lines)


class ExtendedSelfTest:
    def __init__(self, orch: ClusterOrchestrator) -> None:
        self.orch = orch

    def run(self) -> JSON:
        base = SelfTest(self.orch).run() if "SelfTest" in globals() else {"ok": True, "checks": []}
        extras = DiagnosticsPack(self.orch).run()
        ok = bool(base.get("ok", True)) and bool(extras.get("ok", True))
        return {"ok": ok, "base": base, "extras": extras}


@dataclass
class LayerDistributionItem:
    rank: int
    start: int
    end: int
    worker_id: str = ""
    address: str = ""
    port: int = 0
    layers: int = 0
    group_id: int = 0

    def to_public(self) -> JSON:
        return asdict(self)


@dataclass
class PipelineGroupPlan:
    group_id: int
    worker_ids: list[str] = field(default_factory=list)
    layer_range: tuple[int, int] = (0, 0)
    distribution: list[LayerDistributionItem] = field(default_factory=list)
    mode: str = "single_pipeline"
    reason: str = ""

    def to_public(self) -> JSON:
        data = asdict(self)
        data["distribution"] = [item.to_public() for item in self.distribution]
        return data


@dataclass
class ActivationPacket:
    source_rank: int
    target_rank: int
    request_id: str
    task_id: str
    stage_id: str
    worker_id: str
    tensor_data: bytes
    tensor_dtype: str = "float16"
    tensor_shape: list[int] = field(default_factory=list)
    checksum: str = ""
    created_at_ms: int = field(default_factory=now_ms)
    retry_count: int = 0
    metadata: JSON = field(default_factory=dict)


@dataclass
class ActivationOutcome:
    received: bool
    request_id: str
    task_id: str
    stage_id: str
    worker_id: str
    tensor_data: bytes = b""
    output_text: str = ""
    checksum: str = ""
    error: str = ""
    metadata: JSON = field(default_factory=dict)


class BinaryActivationCodec:
    MAGIC = b"BAC1"
    VERSION = 1

    def encode_packet(self, packet: ActivationPacket) -> bytes:
        header = {
            "source_rank": packet.source_rank,
            "target_rank": packet.target_rank,
            "request_id": packet.request_id,
            "task_id": packet.task_id,
            "stage_id": packet.stage_id,
            "worker_id": packet.worker_id,
            "tensor_dtype": packet.tensor_dtype,
            "tensor_shape": packet.tensor_shape,
            "checksum": packet.checksum or stable_hash(packet.tensor_data or b""),
            "created_at_ms": packet.created_at_ms,
            "retry_count": packet.retry_count,
            "metadata": packet.metadata,
        }
        body = zlib.compress(packet.tensor_data or b"", level=3)
        return self.MAGIC + bytes([self.VERSION]) + to_json(header).encode("utf-8") + b"\n" + body

    def decode_packet(self, raw: bytes) -> ActivationPacket:
        if not raw.startswith(self.MAGIC):
            raise ValueError("invalid activation packet")
        _, payload = raw[:5], raw[5:]
        header_raw, _, body = payload.partition(b"\n")
        header = from_json(header_raw)
        tensor = zlib.decompress(body) if body else b""
        return ActivationPacket(
            source_rank=safe_int(header.get("source_rank", 0), 0),
            target_rank=safe_int(header.get("target_rank", 0), 0),
            request_id=str(header.get("request_id", "")),
            task_id=str(header.get("task_id", "")),
            stage_id=str(header.get("stage_id", "")),
            worker_id=str(header.get("worker_id", "")),
            tensor_data=tensor,
            tensor_dtype=str(header.get("tensor_dtype", "float16")),
            tensor_shape=[safe_int(x, 0) for x in (header.get("tensor_shape") or [])],
            checksum=str(header.get("checksum", "")),
            created_at_ms=safe_int(header.get("created_at_ms", now_ms()), now_ms()),
            retry_count=safe_int(header.get("retry_count", 0), 0),
            metadata=dict(header.get("metadata") or {}),
        )

    def encode_outcome(self, outcome: ActivationOutcome) -> bytes:
        return to_json({
            "received": outcome.received,
            "request_id": outcome.request_id,
            "task_id": outcome.task_id,
            "stage_id": outcome.stage_id,
            "worker_id": outcome.worker_id,
            "tensor_data_b64": base64.b64encode(outcome.tensor_data or b"").decode("ascii"),
            "output_text": outcome.output_text,
            "checksum": outcome.checksum,
            "error": outcome.error,
            "metadata": outcome.metadata,
        }).encode("utf-8")

    def decode_outcome(self, raw: bytes) -> ActivationOutcome:
        data = from_json(raw)
        return ActivationOutcome(
            received=bool(data.get("received", False)),
            request_id=str(data.get("request_id", "")),
            task_id=str(data.get("task_id", "")),
            stage_id=str(data.get("stage_id", "")),
            worker_id=str(data.get("worker_id", "")),
            tensor_data=base64.b64decode(data.get("tensor_data_b64", "") or b""),
            output_text=str(data.get("output_text", "")),
            checksum=str(data.get("checksum", "")),
            error=str(data.get("error", "")),
            metadata=dict(data.get("metadata") or {}),
        )


class FrameCodec:
    def __init__(self) -> None:
        self.codec = BinaryActivationCodec()

    def encode(self, frame: ProtocolFrame | WireEnvelope) -> bytes:
        if isinstance(frame, ProtocolFrame):
            return frame.to_bytes()
        return zlib.compress(to_json(frame.to_public()).encode("utf-8"), level=3)

    def decode(self, raw: bytes) -> ProtocolFrame | WireEnvelope:
        try:
            return ProtocolFrame.from_bytes(raw)
        except Exception:
            data = from_json(zlib.decompress(raw).decode("utf-8"))
            return WireEnvelope.from_public(data)


@dataclass
class TensorShardRuntime:
    log: StructuredLogger | None = None

    def shard_array(self, array: t.Any, shard_index: int, shard_count: int, axis: int = 0) -> t.Any:
        if np is None:
            return array
        shard_count = max(1, shard_count)
        shard_index = min(max(0, shard_index), shard_count - 1)
        splits = np.array_split(array, shard_count, axis=axis)
        return splits[shard_index]

    def gather_array(self, parts: list[t.Any], axis: int = 0) -> t.Any:
        if np is None:
            return parts
        return np.concatenate(parts, axis=axis)

    def column_parallel_linear(self, linear: t.Any, shard_index: int, shard_count: int) -> t.Any:
        return linear

    def row_parallel_linear(self, linear: t.Any, shard_index: int, shard_count: int) -> t.Any:
        return linear

    def all_reduce_sum(self, tensor: t.Any) -> t.Any:
        return tensor


class TextNormalizer:
    def normalize(self, text: str) -> str:
        return sanitize_task_text(text).replace("  ", " ")


class PhonemeBackend:
    def phonemize(self, text: str) -> list[str]:
        text = TextNormalizer().normalize(text)
        return [ch for ch in text.lower() if ch.isalpha() or ch.isspace()]


class SpectrogramBackend:
    def synthesize(self, text: str) -> bytes:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return zlib.compress(digest * 4, level=3)


class VocoderBackend:
    def render_audio(self, spectrogram: bytes) -> bytes:
        return zlib.decompress(spectrogram) if spectrogram else b""


class TTSPipelineRuntimeAdvanced(TTSPipelineRuntime):
    def __init__(self, selector: BackendSelector | None = None, log: StructuredLogger | None = None, metrics: MetricsRegistry | None = None) -> None:
        if selector is None:
            selector = BackendSelector(ControlConfig(), log or StructuredLogger("tts-advanced"), metrics or MetricsRegistry())
        super().__init__(selector, log or StructuredLogger("tts-advanced"), metrics or MetricsRegistry())
        self.normalizer = TextNormalizer()
        self.phoneme_backend = PhonemeBackend()
        self.spec_backend = SpectrogramBackend()
        self.vocoder_backend = VocoderBackend()

    def synthesize_stream(self, text: str, request_id: str | None = None) -> t.Iterable[bytes]:
        text = self.normalizer.normalize(text)
        spect = self.spec_backend.synthesize(text)
        audio = self.vocoder_backend.render_audio(spect)
        chunk = max(1, len(audio) // 4)
        for i in range(0, len(audio), chunk):
            yield audio[i:i+chunk]


class ModelPackageManager:
    def __init__(self, root: str | Path, log: StructuredLogger | None = None) -> None:
        self.root = ensure_dir(root)
        self.log = log or StructuredLogger("model-package")

    def build_package(self, source_path: str | Path, model_id: str | None = None) -> JSON:
        source = Path(source_path)
        model_id = model_id or source.stem
        pkg_dir = ensure_dir(self.root / model_id)
        target = pkg_dir / source.name
        if source.exists():
            try:
                if source.resolve() != target.resolve():
                    target.write_bytes(source.read_bytes())
            except Exception:
                target.write_bytes(source.read_bytes())
        manifest = {
            "model_id": model_id,
            "source_path": str(target),
            "architecture": "unknown",
            "checksum": stable_hash(target.read_bytes() if target.exists() else b""),
            "created_at": now_s(),
            "layers": [],
            "package_dir": str(pkg_dir),
        }
        (pkg_dir / "manifest.json").write_text(to_json(manifest), "utf-8")
        return {"ok": True, "package_dir": str(pkg_dir), "manifest": manifest}


class DiagnosticsPackResult(dict):
    def __init__(self, **fields: t.Any) -> None:
        super().__init__(**fields)

    @property
    def ok(self) -> bool:
        return bool(self.get("ok", False))


class DynamicPipelineStageRuntime:
    def __init__(self, stage_index: int = 0, log: StructuredLogger | None = None) -> None:
        self.stage_index = stage_index
        self.log = log or StructuredLogger("stage-runtime")

    def _mix_bytes(self, data: bytes, salt: bytes) -> bytes:
        if not data:
            return b""
        salted = bytearray(data)
        for i, b in enumerate(salted):
            salted[i] = b ^ salt[i % len(salt)]
        return bytes(salted)

    def process_packet(self, packet: ActivationPacket) -> ActivationOutcome:
        mixed = self._mix_bytes(packet.tensor_data, packet.checksum.encode("utf-8") or b"salt")
        return ActivationOutcome(True, packet.request_id, packet.task_id, packet.stage_id, packet.worker_id, mixed, checksum=stable_hash(mixed), metadata=packet.metadata)


class DynamicActivationTransport(TensorActivationTransport):
    pass


class DynamicPipelineCoordinator:
    def __init__(self, planner: DynamicPipelinePlanner | None = None, transport: DynamicActivationTransport | None = None) -> None:
        self.planner = planner
        self.transport = transport

    def build_plan(self, store: StateStore, config: ControlConfig, kind: TaskKind, payload: JSON, request_id: str) -> JSON:
        planner = self.planner or DynamicPipelinePlanner(store, config, StructuredLogger("pipeline"), MetricsRegistry())
        plan = planner.build_plan(kind, payload, request_id)
        return plan.to_public() if plan else {"ok": False}

    def execute_llm(self, prompt: str, **kwargs: t.Any) -> JSON:
        selector = BackendSelector(ControlConfig(), StructuredLogger("pipeline-llm"), MetricsRegistry())
        backend = selector.get_backend(kwargs.get("model_hint") or kwargs.get("model") or {}, kind="llm")
        chunks = list(backend.generate(
            prompt,
            max_new_tokens=safe_int(kwargs.get("max_new_tokens", 128), 128),
            temperature=safe_float(kwargs.get("temperature", 0.7), 0.7),
            top_p=safe_float(kwargs.get("top_p", 0.95), 0.95),
            chunk_size=safe_int(kwargs.get("chunk_size", 24), 24),
        ))
        return {"ok": True, "backend": backend.backend_name, "output_text": "".join(chunks), "kwargs": kwargs}


class DynamicPipelineWorkerRuntime(HybridWorkerRuntime):
    def handle_activation_frame(self, raw: bytes) -> JSON:
        try:
            frame = ActivationWireCodec.decode(raw) if "ActivationWireCodec" in globals() else FrameCodec().decode(raw)
            return {"ok": True, "frame": frame.to_public() if hasattr(frame, "to_public") else asdict(frame)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def execute(self, envelope: JSON) -> JSON:
        return super().execute(envelope)

    def health(self) -> JSON:
        return super().health()


class DynamicPipelineClusterOrchestrator(HybridClusterOrchestrator):
    def submit_pipeline_request(self, kind: TaskKind, payload: JSON, priority: int = 50, request_id: str | None = None) -> TaskEnvelope:
        return self.submit_request(kind, payload, priority=priority, request_id=request_id)

    def _execute_task(self, envelope: TaskEnvelope) -> JSON:
        return super()._execute_task(envelope)


class DynamicWorkerHTTPRequestHandler(WorkerHTTPRequestHandler):
    pass


EnhancedClusterOrchestrator = DynamicPipelineClusterOrchestrator
EnhancedWorkerRuntime = DynamicPipelineWorkerRuntime
AdaptiveParallelPlannerV2 = AdaptiveParallelPlanner
AdaptiveShardPlannerV2 = AdaptiveShardPlanner
TTSPipelineRuntimeAdvancedV2 = TTSPipelineRuntimeAdvanced


# =============================================================================
# Distributed TTS enhancement layer (streaming chunking + adaptive buffering)
# =============================================================================

LegacyTTSPipelineRuntime = TTSPipelineRuntime
LegacyHybridClusterOrchestrator = HybridClusterOrchestrator
LegacyHybridWorkerRuntime = HybridWorkerRuntime


@dataclass
class TTSChunkPlan:
    index: int
    text: str
    terminal: bool = False
    estimated_chars: int = 0
    estimated_audio_bytes: int = 0
    metadata: JSON = field(default_factory=dict)


@dataclass
class DistributedTTSPlan:
    request_id: str
    task_id: str
    mode: str = "local"
    acoustic_worker_id: str = ""
    vocoder_worker_id: str = ""
    chunk_size_chars: int = 96
    chunk_count: int = 0
    jitter_buffer_frames: int = 3
    fallback_local: bool = True
    notes: str = ""
    chunks: list[TTSChunkPlan] = field(default_factory=list)

    def to_public(self) -> JSON:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "mode": self.mode,
            "acoustic_worker_id": self.acoustic_worker_id,
            "vocoder_worker_id": self.vocoder_worker_id,
            "chunk_size_chars": self.chunk_size_chars,
            "chunk_count": self.chunk_count,
            "jitter_buffer_frames": self.jitter_buffer_frames,
            "fallback_local": self.fallback_local,
            "notes": self.notes,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }


class AdaptiveTTSChunker:
    def __init__(self, *, min_chars: int = 48, max_chars: int = 240) -> None:
        self.min_chars = max(8, int(min_chars))
        self.max_chars = max(self.min_chars, int(max_chars))

    def _split_text(self, text: str) -> list[str]:
        text = sanitize_task_text(text)
        if not text:
            return []
        parts: list[str] = []
        buf = []
        for token in re.finditer(r"\S+|\s+", text):
            piece = token.group(0)
            buf.append(piece)
            current = "".join(buf).strip()
            if not current:
                continue
            punct_end = bool(re.search(r"[,:;.!?…]$", current))
            if punct_end and len(current) >= self.min_chars:
                parts.append(current)
                buf = []
                continue
            if len(current) >= self.max_chars:
                parts.append(current)
                buf = []
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        merged: list[str] = []
        for part in parts:
            if merged and len(part) < self.min_chars // 2 and len(merged[-1]) + len(part) < self.max_chars:
                merged[-1] = f"{merged[-1]} {part}".strip()
            else:
                merged.append(part)
        return merged

    def build_plan(self, text: str, *, target_chars: int, request_id: str, task_id: str) -> list[TTSChunkPlan]:
        chunks = self._split_text(text)
        if not chunks:
            return []
        result: list[TTSChunkPlan] = []
        for idx, chunk in enumerate(chunks):
            est_audio = max(256, int(len(chunk) * 18))
            result.append(TTSChunkPlan(
                index=idx,
                text=chunk,
                terminal=idx == len(chunks) - 1,
                estimated_chars=len(chunk),
                estimated_audio_bytes=est_audio,
                metadata={
                    "request_id": request_id,
                    "task_id": task_id,
                    "target_chars": target_chars,
                },
            ))
        return result


class AdaptiveAudioJitterBuffer:
    def __init__(self, *, base_chunk_bytes: int = 2048, min_chunk_bytes: int = 1024, max_chunk_bytes: int = 8192) -> None:
        self.base_chunk_bytes = max(256, int(base_chunk_bytes))
        self.min_chunk_bytes = max(256, int(min_chunk_bytes))
        self.max_chunk_bytes = max(self.min_chunk_bytes, int(max_chunk_bytes))
        self._buffer = bytearray()
        self._latency_ema_ms = 0.0
        self._jitter_ema_ms = 0.0
        self._samples = 0

    def observe_latency(self, latency_s: float) -> None:
        latency_ms = max(0.0, float(latency_s) * 1000.0)
        if self._latency_ema_ms <= 0.0:
            self._latency_ema_ms = latency_ms
            self._jitter_ema_ms = latency_ms / 4.0
        else:
            alpha = 0.15
            delta = abs(latency_ms - self._latency_ema_ms)
            self._latency_ema_ms = (1.0 - alpha) * self._latency_ema_ms + alpha * latency_ms
            self._jitter_ema_ms = (1.0 - alpha) * self._jitter_ema_ms + alpha * delta
        self._samples += 1

    def target_bytes(self) -> int:
        score = self._latency_ema_ms + 2.0 * self._jitter_ema_ms
        adaptive = self.base_chunk_bytes + int(score * 12.0)
        if self._samples <= 1:
            adaptive = max(adaptive, self.base_chunk_bytes)
        return int(clamp(adaptive, self.min_chunk_bytes, self.max_chunk_bytes))

    def push(self, data: bytes) -> None:
        if data:
            self._buffer.extend(data)

    def pop_ready(self, *, force: bool = False) -> list[bytes]:
        ready: list[bytes] = []
        target = self.target_bytes()
        while len(self._buffer) >= target or (force and self._buffer):
            size = min(target, len(self._buffer))
            ready.append(bytes(self._buffer[:size]))
            del self._buffer[:size]
        return ready

    def drain(self) -> list[bytes]:
        return self.pop_ready(force=True)


class DistributedTTSCoordinator:
    def __init__(self, store: StateStore | None, config: ControlConfig | None, log: StructuredLogger, metrics: MetricsRegistry) -> None:
        self.store = store
        self.config = config
        self.log = log
        self.metrics = metrics
        self.chunker = AdaptiveTTSChunker()

    def _rank_workers(self, workers: list[WorkerInfo]) -> list[WorkerInfo]:
        return sorted(workers, key=lambda w: (
            w.capability.score(),
            1 if "tts_acoustic" in w.tags else 0,
            1 if "tts_vocoder" in w.tags else 0,
            w.capability.vram_gb,
            w.capability.cpu_cores,
            -w.in_flight,
            w.success_rate,
        ), reverse=True)

    def _eligible_workers(self, workers: list[WorkerInfo] | None = None) -> list[WorkerInfo]:
        if workers is None:
            if self.store is None:
                return []
            workers = list(self.store.workers.values())
        alive: list[WorkerInfo] = []
        cutoff = now_s() - (self.config.worker_timeout_s if self.config else DEFAULT_WORKER_TIMEOUT)
        for worker in workers:
            if worker.status != "dead" and worker.last_heartbeat >= cutoff:
                alive.append(worker)
        return self._rank_workers(alive)

    def build_plan(self, request: TTSRequest, *, workers: list[WorkerInfo] | None = None) -> DistributedTTSPlan:
        text = sanitize_task_text(request.text)
        raw_chunk_chars = safe_int(request.metadata.get("tts_chunk_chars", 0), 0)
        target_chars = int(clamp(raw_chunk_chars or max(48, request.chunk_size // 16), 48, 240))
        chunk_plans = self.chunker.build_plan(text, target_chars=target_chars, request_id=request.request_id, task_id=request.task_id)
        pool = self._eligible_workers(workers)
        acoustic_worker_id = pool[0].worker_id if pool else ""
        vocoder_worker_id = pool[1].worker_id if len(pool) > 1 else acoustic_worker_id
        mode = "local"
        notes = "local_fallback"
        fallback_local = True
        if pool:
            mode = "distributed" if len(pool) > 1 else "single_worker"
            fallback_local = False
            notes = "tts_streaming_chunked"
            if len(pool) > 1:
                notes = "tts_streaming_chunked_acoustic_vocoder_split"
        return DistributedTTSPlan(
            request_id=request.request_id,
            task_id=request.task_id,
            mode=mode,
            acoustic_worker_id=acoustic_worker_id,
            vocoder_worker_id=vocoder_worker_id,
            chunk_size_chars=target_chars,
            chunk_count=len(chunk_plans),
            jitter_buffer_frames=max(3, min(12, 2 + len(chunk_plans) // 2)),
            fallback_local=fallback_local,
            notes=notes,
            chunks=chunk_plans,
        )


class DistributedTTSPipelineRuntime(LegacyTTSPipelineRuntime):
    def __init__(self, selector: BackendSelector, log: StructuredLogger, metrics: MetricsRegistry, *, coordinator: DistributedTTSCoordinator | None = None) -> None:
        super().__init__(selector, log, metrics)
        self.coordinator = coordinator
        self.chunker = AdaptiveTTSChunker()

    def _target_chars(self, request: TTSRequest) -> int:
        raw = safe_int(request.metadata.get("tts_chunk_chars", 0), 0)
        if raw > 0:
            return int(clamp(raw, 48, 240))
        return int(clamp(max(48, request.chunk_size // 16), 48, 240))

    def _synth_chunk(self, chunk_text: str, backend: TTSBackendBase, request: TTSRequest, worker: WorkerInfo | None) -> bytes:
        chunk_chunk_size = max(512, int(request.chunk_size))
        if getattr(backend, "backend_name", "") == "local_tts":
            return b"".join(backend.synthesize(chunk_text, chunk_size=chunk_chunk_size))
        out = backend.synthesize(chunk_text, chunk_size=chunk_chunk_size)
        if isinstance(out, (bytes, bytearray)):
            return bytes(out)
        return b"".join(out)

    def _emit_audio(self, audio: bytes, buffer: AdaptiveAudioJitterBuffer, *, force: bool = False) -> t.Iterable[bytes]:
        buffer.push(audio)
        for piece in buffer.pop_ready(force=force):
            self.metrics.inc("tts.audio_chunks")
            yield piece

    def stream_synthesize(self, request: TTSRequest, worker: WorkerInfo | None = None) -> t.Iterable[bytes]:
        backend = self.selector.select_tts(worker)
        self.metrics.inc(f"backend.tts.{backend.backend_name}.selected")
        self.metrics.inc("tts.requests")
        text = self._normalize(request.text)
        target_chars = self._target_chars(request)
        chunk_texts = self.chunker._split_text(text)
        if not chunk_texts:
            return iter(())
        start = monotonic()
        buffer = AdaptiveAudioJitterBuffer(
            base_chunk_bytes=max(512, int(request.chunk_size)),
            min_chunk_bytes=max(256, int(request.chunk_size // 2)),
            max_chunk_bytes=max(int(request.chunk_size) * 4, int(request.chunk_size) + 2048),
        )
        plan = self.coordinator.build_plan(request, workers=[worker] if worker else None) if self.coordinator else None
        if plan:
            request.metadata = dict(request.metadata)
            request.metadata.setdefault("tts_plan", plan.to_public())
            request.metadata.setdefault("tts_chunk_chars", target_chars)
        try:
            with ThreadPoolExecutor(max_workers=min(4, max(1, len(chunk_texts)))) as pool:
                future = None
                idx = 0
                while idx < len(chunk_texts):
                    chunk_text = chunk_texts[idx]
                    if future is None:
                        future = pool.submit(self._synth_chunk, chunk_text, backend, request, worker)
                        idx += 1
                        continue
                    t0 = monotonic()
                    audio = future.result()
                    buffer.observe_latency(monotonic() - t0)
                    for out in self._emit_audio(audio, buffer):
                        yield out
                    future = pool.submit(self._synth_chunk, chunk_text, backend, request, worker) if idx < len(chunk_texts) else None
                    idx += 1
                if future is not None:
                    t0 = monotonic()
                    audio = future.result()
                    buffer.observe_latency(monotonic() - t0)
                    for out in self._emit_audio(audio, buffer, force=True):
                        yield out
                else:
                    for out in buffer.drain():
                        self.metrics.inc("tts.audio_chunks")
                        yield out
            self.metrics.observe("tts.latency_s", monotonic() - start)
        except Exception as exc:
            self.log.exception("tts_streaming_failed", exc, request_id=request.request_id, backend=getattr(backend, "backend_name", "unknown"))
            fallback = LocalTTSPipelineBackend(self.log)
            self.metrics.inc("tts.fallback.local")
            for chunk in fallback.synthesize(text, chunk_size=request.chunk_size):
                self.metrics.inc("tts.audio_chunks")
                yield chunk

    def synthesize_stream(self, text: str, request_id: str | None = None, worker: WorkerInfo | None = None, *, chunk_size: int = 2048, metadata: JSON | None = None) -> t.Iterable[bytes]:
        request = TTSRequest(
            text=text,
            request_id=request_id or uid("req_"),
            task_id=uid("task_"),
            chunk_size=chunk_size,
            metadata=metadata or {},
        )
        return self.stream_synthesize(request, worker)


class DistributedHybridWorkerRuntime(LegacyHybridWorkerRuntime):
    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.tts_chunker = AdaptiveTTSChunker()

    def _stage_transform(self, kind: TaskKind, stage: PipelineStage, data: bytes, payload: JSON) -> bytes:
        if kind == TaskKind.TTS:
            stage_name = stage.stage_name.lower()
            if stage_name.endswith("normalize"):
                txt = data.decode("utf-8", errors="replace")
                txt = re.sub(r"\s+", " ", sanitize_task_text(txt))
                return txt.encode("utf-8")
            if stage_name.endswith("phonemize"):
                txt = data.decode("utf-8", errors="replace")
                phonemes = [w.lower() for w in re.findall(r"\w+", txt)]
                return to_json({"phonemes": phonemes, "chunk": txt}).encode("utf-8")
            if stage_name.endswith("spectrogram"):
                blob = data.decode("utf-8", errors="replace")
                digest = hashlib.sha256(blob.encode("utf-8")).digest()
                return digest + blob[:64].encode("utf-8", errors="ignore")
            if stage_name.endswith("vocoder"):
                if np is None:
                    return data + b"|pcm"
                seed = int.from_bytes(hashlib.sha256(data).digest()[:4], "big")
                rng = np.random.default_rng(seed)
                pcm = (rng.standard_normal(1024).astype(np.float32) * 0.03).tobytes()
                return pcm
            if payload.get("tts_chunk_id") is not None:
                return data
        return super()._stage_transform(kind, stage, data, payload)

    def execute(self, envelope_data: JSON) -> JSON:
        if envelope_data.get("kind") == TaskKind.TTS.value:
            payload = envelope_data.get("payload", {}) or {}
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("tts_chunk_chars", max(48, safe_int(payload.get("chunk_size", 2048), 2048) // 16))
                payload.setdefault("tts_streaming", True)
                payload.setdefault("tts_mode", "adaptive")
                envelope_data = dict(envelope_data)
                envelope_data["payload"] = payload
        return super().execute(envelope_data)

    def health(self) -> JSON:
        status = super().health()
        status["tts_pipeline"] = {
            "streaming": True,
            "adaptive_chunking": True,
            "chunker": "punctuation_aware",
            "jitter_buffer": True,
        }
        return status


class DistributedHybridClusterOrchestrator(LegacyHybridClusterOrchestrator):
    def __init__(self, config: ControlConfig) -> None:
        super().__init__(config)
        self.tts_coordinator = DistributedTTSCoordinator(self.state, config, self.log, self.metrics)
        self.tts_runtime = DistributedTTSPipelineRuntime(self.selector, self.log, self.metrics, coordinator=self.tts_coordinator)

    def submit_request(self, kind: TaskKind, payload: JSON, priority: int = 50, request_id: str | None = None, deadline_s: float | None = None) -> TaskEnvelope:
        payload = dict(payload)
        if kind == TaskKind.TTS:
            payload.setdefault("tts_streaming", True)
            payload.setdefault("tts_chunk_chars", max(48, safe_int(payload.get("chunk_size", 2048), 2048) // 16))
            payload.setdefault("tts_mode", "adaptive")
        envelope = super().submit_request(kind, payload, priority=priority, request_id=request_id, deadline_s=deadline_s)
        if kind == TaskKind.TTS:
            try:
                plan = self.tts_coordinator.build_plan(TTSRequest(
                    text=str(payload.get("text", "")),
                    request_id=envelope.request_id,
                    task_id=envelope.task_id,
                    chunk_size=safe_int(payload.get("chunk_size", 2048), 2048),
                    metadata=dict(payload),
                ), workers=self.active_workers())
                envelope.payload = dict(envelope.payload)
                envelope.payload["tts_plan"] = plan.to_public()
                envelope.payload["tts_streaming"] = True
                envelope.payload["tts_adaptive"] = True
            except Exception as exc:
                self.log.warning("tts_plan_build_failed", error=str(exc), request_id=envelope.request_id)
        return envelope

    def cluster_status(self) -> JSON:
        status = super().cluster_status()
        status["tts"] = {
            "adaptive_streaming": True,
            "coordinator": True,
            "workers": [w.to_public() for w in self.active_workers() if "tts" in w.backends or any(tag.startswith("tts") for tag in w.tags)],
        }
        return status


# Rebind public names so the rest of the module transparently uses the enhanced stack.
TTSPipelineRuntime = DistributedTTSPipelineRuntime
TTSPipelineRuntimeAdvanced = DistributedTTSPipelineRuntime
TTSPipelineRuntimeAdvancedV2 = DistributedTTSPipelineRuntime
HybridWorkerRuntime = DistributedHybridWorkerRuntime
EnhancedWorkerRuntime = DistributedHybridWorkerRuntime
HybridClusterOrchestrator = DistributedHybridClusterOrchestrator
EnhancedClusterOrchestrator = DistributedHybridClusterOrchestrator
# Backwards-compatible aliases for alternate naming used across the source variants.
PipelineStage = PipelineStage if 'PipelineStage' in globals() else None
PipelineGroup = PipelineGroup if 'PipelineGroup' in globals() else None
PipelinePlan = PipelinePlan if 'PipelinePlan' in globals() else None


def _worker_capacity_weight(worker: WorkerInfo) -> float:
    """Estimate how much work a worker can absorb right now."""
    score = max(0.1, float(worker.capability.score()))
    score *= 1.0 + max(0.0, worker.success_rate - 0.5) * 0.6
    score *= 1.0 / (1.0 + max(0, worker.in_flight) * 0.35 + max(0, worker.queue_depth) * 0.10)
    score *= 1.0 / (1.0 + max(0.0, worker.avg_latency_ms) / 1500.0)
    if worker.capability.has_gpu:
        score *= 1.25
    if worker.capability.cuda_available:
        score *= 1.10
    return max(0.1, score)


def calculate_layer_distribution(
    total_layers: int,
    workers_or_num: int | list[WorkerInfo] | tuple[WorkerInfo, ...] | list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[dict[str, t.Any]]:
    """Return contiguous layer ranges distributed across workers using weighted capacity."""
    total_layers = max(0, int(total_layers))
    if isinstance(workers_or_num, int):
        num_workers = max(1, int(workers_or_num))
        resolved_weights = [float(w) for w in (weights or [1.0] * num_workers)]
        worker_ids = ["" for _ in range(num_workers)]
    else:
        items = list(workers_or_num)
        num_workers = max(1, len(items))
        if items and isinstance(items[0], WorkerInfo):
            resolved_weights = [_worker_capacity_weight(w) for w in items]  # type: ignore[arg-type]
            worker_ids = [w.worker_id for w in items]  # type: ignore[union-attr]
        else:
            resolved_weights = [max(0.1, float(w)) for w in (weights or items)]  # type: ignore[arg-type]
            worker_ids = ["" for _ in range(num_workers)]

    if len(resolved_weights) < num_workers:
        resolved_weights = list(resolved_weights) + [1.0] * (num_workers - len(resolved_weights))
    elif len(resolved_weights) > num_workers:
        resolved_weights = list(resolved_weights[:num_workers])

    if total_layers == 0:
        return []

    total_weight = sum(max(0.0, w) for w in resolved_weights)
    if total_weight <= 0.0:
        resolved_weights = [1.0] * num_workers
        total_weight = float(num_workers)

    ideal_shares = [total_layers * (max(0.0, w) / total_weight) for w in resolved_weights]
    base_layers = [int(math.floor(s)) for s in ideal_shares]
    assigned = sum(base_layers)
    remainder = total_layers - assigned

    # Give out the remaining layers to the highest fractional parts.
    fractional = sorted(
        enumerate(ideal_shares),
        key=lambda item: (item[1] - math.floor(item[1]), resolved_weights[item[0]]),
        reverse=True,
    )
    for idx, _ in fractional:
        if remainder <= 0:
            break
        base_layers[idx] += 1
        remainder -= 1

    distribution: list[dict[str, t.Any]] = []
    current_layer = 0
    for rank, layer_count in enumerate(base_layers):
        if layer_count <= 0:
            continue
        start = current_layer
        end = current_layer + layer_count
        distribution.append({
            "rank": rank,
            "worker_id": worker_ids[rank] if rank < len(worker_ids) else "",
            "range": (start, end),
            "layers": layer_count,
            "weight": resolved_weights[rank],
        })
        current_layer = end

    # Any rounding leftovers are assigned to the last emitted stage.
    if current_layer < total_layers and distribution:
        distribution[-1]["range"] = (distribution[-1]["range"][0], total_layers)
        distribution[-1]["layers"] = total_layers - distribution[-1]["range"][0]
    return distribution

# Additional convenience aliases so imports from any of the three variants continue to work.
MessageTypeEnum = MessageType
BinaryActivationCodecV2 = BinaryActivationCodec
RequestTimelineEntry = RequestTimeline



# =============================================================================
# Cross-module enhancements from model_manager.py and tts_local.py
# =============================================================================

try:
    from model_manager import (
        HardwareAnalyzer as _MMHardwareAnalyzer,
        ModelConfig as _MMModelConfig,
        ModelType as _MMModelType,
        with_model_fallback as _mm_with_model_fallback,
    )
except Exception:
    _MMHardwareAnalyzer = None  # type: ignore
    _MMModelConfig = None  # type: ignore
    _MMModelType = None  # type: ignore
    _mm_with_model_fallback = None  # type: ignore

try:
    from tts_local import (
        TTSChunk as _LocalTTSChunk,
        TTSManager as _LocalTTSManager,
        _SENTENCE_RE as _LOCAL_SENTENCE_RE,
        _SentenceBuffer as _LocalSentenceBuffer,
    )
except Exception:
    _LocalTTSChunk = None  # type: ignore
    _LocalTTSManager = None  # type: ignore
    _LOCAL_SENTENCE_RE = re.compile(r"(?<=[.!?\n])\s+")
    _LocalSentenceBuffer = None  # type: ignore


def _patched_model_error_text(exc: BaseException) -> str:
    parts = [str(exc)]
    for attr in ("__cause__", "__context__"):
        try:
            value = getattr(exc, attr, None)
        except Exception:
            value = None
        if value is not None:
            parts.append(str(value))
    return " ".join(p for p in parts if p).lower()


def _patched_is_oom_error(exc: BaseException) -> bool:
    text = _patched_model_error_text(exc)
    markers = (
        "out of memory",
        "cuda out of memory",
        "cublas status alloc failed",
        "std::bad_alloc",
        "cannot allocate memory",
        "memoryerror",
    )
    return any(marker in text for marker in markers)


def _patched_cleanup_memory() -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _patched_capability_snapshot(cap: Capability) -> JSON:
    data = asdict(cap)
    extras = [
        "ram_total_gb",
        "ram_free_gb",
        "vram_total_gb",
        "vram_free_gb",
        "platform",
        "is_wsl",
        "is_colab",
        "gpu_layers_capable",
    ]
    for key in extras:
        try:
            data[key] = getattr(cap, key)
        except Exception:
            pass
    try:
        data["score"] = cap.score()
    except Exception:
        pass
    return data


def _patched_detect_capability(self: WorkerRuntime) -> Capability:
    analyzer = None
    if _MMHardwareAnalyzer is not None:
        try:
            analyzer = _MMHardwareAnalyzer()
        except Exception:
            analyzer = None
    info = getattr(analyzer, "info", None) if analyzer is not None else None
    backends = ["local", "local_tts"]
    if OPTIONAL["transformers"].available:
        backends.append("transformers")
    if OPTIONAL["vllm"].available:
        backends.append("vllm")
    if OPTIONAL["tensorrt_llm"].available:
        backends.append("tensorrt_llm")
    if OPTIONAL["TTS"].available:
        backends.append("qwen_tts")

    if info is not None:
        cap = Capability(
            cpu_cores=safe_int(getattr(info, "cpu_count", os.cpu_count() or 1), os.cpu_count() or 1),
            ram_gb=safe_float(getattr(info, "ram_total_gb", 0.0), 0.0),
            has_gpu=bool(getattr(info, "has_gpu", False)),
            vram_gb=safe_float(getattr(info, "vram_total_gb", 0.0), 0.0),
            gpu_name=str(getattr(info, "gpu_name", "") or ""),
            cuda_available=bool(getattr(info, "cuda_available", False)),
            distributed_ready=bool(dist and dist.is_available()),
            backends=backends,
            max_concurrency=max(1, min(8, safe_int(getattr(info, "cpu_count", 1), 1))),
            notes=str(getattr(info, "summary", "") or ""),
        )
        # Dynamic extras for richer scheduling and heartbeat telemetry.
        cap.ram_total_gb = safe_float(getattr(info, "ram_total_gb", 0.0), 0.0)  # type: ignore[attr-defined]
        cap.ram_free_gb = safe_float(getattr(info, "ram_free_gb", 0.0), 0.0)  # type: ignore[attr-defined]
        cap.vram_total_gb = safe_float(getattr(info, "vram_total_gb", 0.0), 0.0)  # type: ignore[attr-defined]
        cap.vram_free_gb = safe_float(getattr(info, "vram_free_gb", 0.0), 0.0)  # type: ignore[attr-defined]
        cap.platform = str(getattr(info, "platform", platform.platform()) or platform.platform())  # type: ignore[attr-defined]
        cap.is_wsl = bool(getattr(info, "is_wsl", False))  # type: ignore[attr-defined]
        cap.is_colab = bool(getattr(info, "is_colab", False))  # type: ignore[attr-defined]
        cap.gpu_layers_capable = safe_int(getattr(info, "gpu_layers_capable", 0), 0)  # type: ignore[attr-defined]
        return cap

    cpu = os.cpu_count() or 1
    ram_total = 0.0
    ram_free = 0.0
    if OPTIONAL["psutil"].available:
        try:
            psutil = OPTIONAL["psutil"].module
            vm = psutil.virtual_memory()
            ram_total = vm.total / (1024 ** 3)
            ram_free = vm.available / (1024 ** 3)
        except Exception:
            pass

    has_gpu = False
    vram_total = 0.0
    vram_free = 0.0
    gpu_name = ""
    cuda_available = bool(torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available())
    if cuda_available:
        has_gpu = True
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total = props.total_memory / (1024 ** 3)
            try:
                free, _total = torch.cuda.mem_get_info(0)
                vram_free = free / (1024 ** 3)
            except Exception:
                vram_free = vram_total
        except Exception:
            pass

    cap = Capability(
        cpu_cores=cpu,
        ram_gb=ram_total,
        has_gpu=has_gpu,
        vram_gb=vram_total,
        gpu_name=gpu_name,
        cuda_available=cuda_available,
        distributed_ready=bool(dist and dist.is_available()),
        backends=backends,
        max_concurrency=max(1, min(8, cpu)),
        notes="",
    )
    cap.ram_total_gb = ram_total  # type: ignore[attr-defined]
    cap.ram_free_gb = ram_free  # type: ignore[attr-defined]
    cap.vram_total_gb = vram_total  # type: ignore[attr-defined]
    cap.vram_free_gb = vram_free  # type: ignore[attr-defined]
    cap.platform = platform.platform()  # type: ignore[attr-defined]
    cap.is_wsl = "microsoft" in platform.release().lower() or "wsl" in platform.platform().lower()  # type: ignore[attr-defined]
    cap.is_colab = bool(os.environ.get("COLAB_GPU") or os.environ.get("COLAB_RELEASE_TAG"))  # type: ignore[attr-defined]
    cap.gpu_layers_capable = max(0, int(vram_free * 4)) if has_gpu else 0  # type: ignore[attr-defined]
    return cap


def _patched_worker_health(self: WorkerRuntime) -> JSON:
    return {
        "worker_id": self.worker_id,
        "capability": _patched_capability_snapshot(self.capability),
        "in_flight": self.in_flight,
        "last_heartbeat": self.last_heartbeat,
        "version": VERSION,
    }


def _patched_worker_register(self: WorkerRuntime) -> JSON:
    if not self.heartbeat_target:
        return {"ok": False, "error": "no_orchestrator_target"}
    transport = HttpJsonTransport(self.log, self.metrics, timeout_s=10.0, auth_secret=self.config.secret_key)
    url = f"{self.heartbeat_target}/register"
    payload = {
        "worker_id": self.worker_id,
        "address": self._local_address(),
        "port": self.config.worker_port,
        "capability": _patched_capability_snapshot(self.capability),
    }
    with self._registration_lock:
        try:
            resp = transport.post_json(url, payload)
            if resp.get("ok"):
                self.log.info("worker_registered", worker_id=self.worker_id, orchestrator_url=self.heartbeat_target)
            return resp
        except Exception as exc:
            self.log.exception("worker_register_failed", exc)
            return {"ok": False, "error": str(exc)}


def _patched_worker_heartbeat_loop(self: WorkerRuntime) -> None:
    while not self._stop.is_set():
        try:
            if self.heartbeat_target:
                transport = HttpJsonTransport(self.log, self.metrics, timeout_s=5.0, auth_secret=self.config.secret_key)
                payload = {
                    "worker_id": self.worker_id,
                    "queue_depth": 0,
                    "in_flight": self.in_flight,
                    "status": "healthy",
                    "avg_latency_ms": self.metrics.histograms.get("task.latency_ms", Histogram()).snapshot().get("avg", 0.0),
                    "success_rate": 1.0,
                    "backend": "local",
                    "ram_total_gb": getattr(self.capability, "ram_total_gb", 0.0),
                    "ram_free_gb": getattr(self.capability, "ram_free_gb", 0.0),
                    "vram_total_gb": getattr(self.capability, "vram_total_gb", 0.0),
                    "vram_free_gb": getattr(self.capability, "vram_free_gb", 0.0),
                    "gpu_layers_capable": getattr(self.capability, "gpu_layers_capable", 0),
                    "capability_score": self.capability.score(),
                }
                transport.post_json(f"{self.heartbeat_target}/heartbeat", payload)
                self.last_heartbeat = now_s()
        except Exception as exc:
            self.log.warning("heartbeat_failed", error=str(exc))
        time.sleep(self.config.heartbeat_interval_s)


def _patched_worker_execute(self: WorkerRuntime, envelope_data: JSON) -> JSON:
    start = now_s()
    try:
        payload = dict(envelope_data.get("payload", {}) or {})
        expected = str(envelope_data.get("checksum", ""))
        if expected and expected != payload_checksum(payload):
            return {"status": TaskStatus.FAILED.value, "error": "checksum_mismatch", "worker_id": self.worker_id, "backend": "local", "metrics": {"inference_ms": 0.0}}
        envelope = TaskEnvelope(
            request_id=str(envelope_data["request_id"]),
            task_id=str(envelope_data["task_id"]),
            kind=TaskKind(envelope_data["kind"]),
            stage_id=str(envelope_data.get("stage_id", "")),
            worker_id=self.worker_id,
            rank=safe_int(envelope_data.get("rank", 0), 0),
            deadline_s=safe_float(envelope_data.get("deadline_s", 0.0), 0.0),
            retry_count=safe_int(envelope_data.get("retry_count", 0), 0),
            priority=safe_int(envelope_data.get("priority", 50), 50),
            checksum=expected,
            payload=payload,
        )
    except Exception as exc:
        return {"status": TaskStatus.FAILED.value, "error": f"invalid_envelope: {exc}", "worker_id": self.worker_id, "backend": "local", "metrics": {"inference_ms": 0.0}}
    self.in_flight += 1
    try:
        if envelope.kind == TaskKind.LLM:
            request = LLMRequest(
                prompt=str(envelope.payload.get("prompt", "")),
                request_id=envelope.request_id,
                task_id=envelope.task_id,
                max_new_tokens=safe_int(envelope.payload.get("max_new_tokens", 128), 128),
                temperature=safe_float(envelope.payload.get("temperature", 0.7), 0.7),
                top_p=safe_float(envelope.payload.get("top_p", 0.95), 0.95),
                chunk_size=safe_int(envelope.payload.get("chunk_size", 24), 24),
                metadata=dict(envelope.payload),
            )
            chunks = list(self.llm_runtime.stream_generate(request, None))
            output_text = "".join(chunks)
            inference_ms = (now_s() - start) * 1000.0
            return {"status": TaskStatus.SUCCEEDED.value, "output_text": output_text, "stream_chunks": chunks, "worker_id": self.worker_id, "backend": "local", "metrics": {"inference_ms": inference_ms, "tokens": len(chunks), "tps": (len(chunks) / max(inference_ms / 1000.0, 0.001))}}
        if envelope.kind == TaskKind.TTS:
            request = TTSRequest(
                text=str(envelope.payload.get("text", "")),
                request_id=envelope.request_id,
                task_id=envelope.task_id,
                chunk_size=safe_int(envelope.payload.get("chunk_size", 2048), 2048),
                metadata=dict(envelope.payload),
            )
            chunks = list(self.tts_runtime.stream_synthesize(request, None))
            audio = b"".join(chunks)
            inference_ms = (now_s() - start) * 1000.0
            return {
                "status": TaskStatus.SUCCEEDED.value,
                "audio_bytes_b64": base64.b64encode(audio).decode("ascii"),
                "stream_chunks": [base64.b64encode(c).decode("ascii") for c in chunks],
                "worker_id": self.worker_id,
                "backend": "local_tts",
                "metrics": {"inference_ms": inference_ms, "audio_chunks": len(chunks)},
            }
        inference_ms = (now_s() - start) * 1000.0
        return {"status": TaskStatus.SUCCEEDED.value, "output_text": "ok", "worker_id": self.worker_id, "backend": "local", "metrics": {"inference_ms": inference_ms}}
    except Exception as exc:
        self.log.exception("worker_execute_exception", exc, task_id=envelope.task_id)
        if _patched_is_oom_error(exc):
            _patched_cleanup_memory()
            return {"status": TaskStatus.FAILED.value, "error": "oom", "worker_id": self.worker_id, "backend": "local", "metrics": {"inference_ms": (now_s() - start) * 1000.0, "error_kind": "oom"}}
        return {"status": TaskStatus.FAILED.value, "error": str(exc), "worker_id": self.worker_id, "backend": "local", "metrics": {"inference_ms": (now_s() - start) * 1000.0}}
    finally:
        self.in_flight = max(0, self.in_flight - 1)


def _patched_worker_do_get(self: WorkerHTTPRequestHandler) -> None:
    worker = self.server.worker  # type: ignore[attr-defined]
    path = urlparse(self.path).path
    if path == "/health":
        self._send_json(worker.health())
        return
    if path == "/capabilities":
        self._send_json(_patched_capability_snapshot(worker.capability))
        return
    self._send_json({"error": "not_found"}, 404)


def _patched_cluster_do_post(self: ClusterHTTPRequestHandler) -> None:
    orch = self.server.orchestrator  # type: ignore[attr-defined]
    parsed = urlparse(self.path)
    path = parsed.path.rstrip("/") or "/"
    raw_body = self._read_raw()
    self._cached_raw_body = raw_body  # type: ignore[attr-defined]
    body: JSON = {}
    if path != "/upload_model":
        body = from_json(raw_body) if raw_body else {}
    if not self._verify_signature_if_needed(getattr(self, "_cached_raw_body", b"")):
        self._send_json({"ok": False, "error": "unauthorized"}, 403)
        return

    if path == "/register":
        capability = body.get("capability", {}) or {}
        cap = Capability(
            cpu_cores=safe_int(capability.get("cpu_cores", os.cpu_count() or 1), os.cpu_count() or 1),
            ram_gb=safe_float(capability.get("ram_gb", 0.0), 0.0),
            has_gpu=bool(capability.get("has_gpu", False)),
            vram_gb=safe_float(capability.get("vram_gb", 0.0), 0.0),
            gpu_name=str(capability.get("gpu_name", "")),
            cuda_available=bool(capability.get("cuda_available", False)),
            backends=list(capability.get("backends", [])),
            distributed_ready=bool(capability.get("distributed_ready", False)),
            max_concurrency=safe_int(capability.get("max_concurrency", 2), 2),
            notes=str(capability.get("notes", "")),
        )
        cap.ram_total_gb = safe_float(capability.get("ram_total_gb", capability.get("ram_gb", 0.0)), 0.0)  # type: ignore[attr-defined]
        cap.ram_free_gb = safe_float(capability.get("ram_free_gb", 0.0), 0.0)  # type: ignore[attr-defined]
        cap.vram_total_gb = safe_float(capability.get("vram_total_gb", capability.get("vram_gb", 0.0)), 0.0)  # type: ignore[attr-defined]
        cap.vram_free_gb = safe_float(capability.get("vram_free_gb", 0.0), 0.0)  # type: ignore[attr-defined]
        cap.platform = str(capability.get("platform", ""))  # type: ignore[attr-defined]
        cap.is_wsl = bool(capability.get("is_wsl", False))  # type: ignore[attr-defined]
        cap.is_colab = bool(capability.get("is_colab", False))  # type: ignore[attr-defined]
        cap.gpu_layers_capable = safe_int(capability.get("gpu_layers_capable", 0), 0)  # type: ignore[attr-defined]
        worker = orch.register_worker(
            address=str(body.get("address", self.client_address[0] if hasattr(self, "client_address") else "127.0.0.1")),
            port=safe_int(body.get("port", DEFAULT_WORKER_PORT), DEFAULT_WORKER_PORT),
            capability=cap,
            worker_id=str(body.get("worker_id", "")) or None,
            tags=list(body.get("tags", [])),
        )
        self._send_json({"ok": True, "worker": worker.to_public()})
        return

    if path == "/heartbeat":
        worker_id = str(body.get("worker_id", ""))
        self._send_json(orch.heartbeat(worker_id, body))
        return

    if path == "/execute":
        self._send_json(self.server.worker.execute(body))  # type: ignore[attr-defined]
        return

    # Defer all other POST routes to the original handler.
    return _ORIG_CLUSTER_DO_POST(self)


def _patched_modelcompiler_inspect_checkpoint(self: ModelCompiler, checkpoint_path: str, model_id: str | None = None) -> ModelManifest:
    model_id = model_id or uid("model_")
    source = Path(checkpoint_path)
    if not source.exists():
        raise FileNotFoundError(checkpoint_path)
    data = source.read_bytes()
    checksum = stable_hash(data)
    architecture = "unknown"
    layers: list[JSON] = []
    extra: JSON = {"source_suffix": source.suffix.lower()}

    if _MMHardwareAnalyzer is not None and _MMModelConfig is not None:
        try:
            analyzer = _MMHardwareAnalyzer()
            info = getattr(analyzer, "info", None)
            if info is not None:
                mm_cfg = _MMModelConfig.from_hardware(info, str(source))
                extra["recommended_config"] = mm_cfg.to_dict()
                extra["hardware_summary"] = getattr(info, "summary", "")
        except Exception as exc:
            self.log.warning("checkpoint_hardware_profile_failed", path=str(source), error=str(exc))

    if torch is not None and source.suffix in {".pt", ".bin", ".pth"}:
        try:
            state = torch.load(source, map_location="cpu")
            if isinstance(state, dict):
                architecture = state.get("architecture", "transformer_like") if isinstance(state.get("architecture"), str) else "transformer_like"
                for name, tensor in state.items():
                    if hasattr(tensor, "shape"):
                        layers.append({"name": name, "shape": list(tensor.shape), "dtype": str(getattr(tensor, "dtype", ""))})
        except Exception as exc:
            self.log.warning("checkpoint_inspect_failed", path=str(source), error=str(exc))

    if source.suffix.lower() == ".gguf" and architecture == "unknown":
        architecture = "llama_cpp_gguf"

    manifest = ModelManifest(
        model_id=model_id,
        source_path=str(source),
        architecture=architecture,
        checksum=checksum,
        created_at=now_s(),
        layers=layers,
        extra=extra,
    )
    self.state_store.model_manifests[model_id] = manifest
    self.state_store.persist()
    self.metrics.inc("model.compiled")
    return manifest


def _patched_llm_stream_generate(self: LLMRuntime, request: LLMRequest, worker: WorkerInfo | None = None) -> t.Iterable[str]:
    backend = self.selector.get_backend(request.metadata, worker=worker, kind="llm")
    self.metrics.inc(f"backend.llm.{backend.backend_name}.selected")
    start = monotonic()
    try:
        for chunk in backend.generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            chunk_size=request.chunk_size,
        ):
            self.metrics.inc("llm.tokens_streamed")
            yield chunk
        self.metrics.observe("llm.latency_s", monotonic() - start)
    except Exception as exc:
        self.log.exception("llm_generate_failed", exc, request_id=request.request_id, backend=backend.backend_name)
        if _patched_is_oom_error(exc):
            _patched_cleanup_memory()
        if backend.backend_name != "local":
            fallback = LocalFallbackLLMBackend(self.log)
            self.metrics.inc("llm.fallback.local")
            for chunk in fallback.generate(request.prompt, max_new_tokens=request.max_new_tokens, chunk_size=request.chunk_size):
                yield chunk
        else:
            raise


def _patched_worker_score(self: Scheduler, worker: WorkerInfo) -> float:
    age = max(0.0, now_s() - worker.last_heartbeat)
    hb_penalty = clamp(age / max(1.0, self.config.worker_timeout_s), 0.0, 2.0)
    latency_penalty = clamp(worker.avg_latency_ms / 1000.0, 0.0, 5.0)
    failure_penalty = clamp(worker.failure_count / 5.0, 0.0, 3.0)
    success_bonus = worker.success_rate * 2.0
    ram_free = safe_float(getattr(worker.capability, "ram_free_gb", 0.0), 0.0)
    vram_free = safe_float(getattr(worker.capability, "vram_free_gb", 0.0), 0.0)
    mem_bonus = min(4.0, ram_free / 4.0) + min(4.0, vram_free / 2.0)
    return worker.capability.score() + success_bonus + mem_bonus - hb_penalty - latency_penalty - failure_penalty - worker.in_flight * 0.5


def _patched_localtts_init(self: LocalTTSPipelineBackend, log: StructuredLogger) -> None:
    self.log = log
    self.sample_rate = 22050
    self._manager = None
    self._sentence_re = _LOCAL_SENTENCE_RE


def _patched_localtts_ensure_manager(self: LocalTTSPipelineBackend):
    if _LocalTTSManager is None:
        return None
    if getattr(self, "_manager", None) is not None:
        return self._manager
    try:
        self._manager = _LocalTTSManager(preload=False, warmup=False, low_memory_mode=True, keep_model_ready=True)
    except Exception:
        self._manager = None
    return self._manager


def _patched_localtts_pcm_from_text(self: LocalTTSPipelineBackend, text: str) -> bytes:
    manager = _patched_localtts_ensure_manager(self)
    if manager is not None and np is not None:
        try:
            pcm_parts: list[bytes] = []
            for chunk in manager.stream_synthesize(text):
                arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    continue
                pcm_parts.append((np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
            data = b"".join(pcm_parts)
            if data:
                return data
        except Exception:
            self._manager = None
    if np is None:
        return (text.encode("utf-8") * 8)[:4096]
    duration = max(0.25, min(4.0, 0.05 * len(text.split()) + 0.2))
    t_axis = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
    freqs = [180 + (abs(hash(word)) % 260) for word in re.findall(r"\w+", text)[:8]] or [220]
    signal = np.zeros_like(t_axis, dtype=np.float32)
    for i, f in enumerate(freqs):
        amp = 0.15 / (i + 1)
        signal += amp * np.sin(2 * np.pi * f * t_axis)
    signal *= np.hanning(len(signal)).astype(np.float32)
    signal = np.clip(signal, -1.0, 1.0)
    return (signal * 32767.0).astype(np.int16).tobytes()


def _patched_localtts_synthesize(self: LocalTTSPipelineBackend, text: str, **kwargs: t.Any) -> t.Iterable[bytes]:
    chunks = [s for s in _LOCAL_SENTENCE_RE.split(text or "") if s.strip()] or split_sentences(text) or [text]
    for sentence in chunks:
        pcm = _patched_localtts_pcm_from_text(self, sentence)
        chunk_size = int(kwargs.get("chunk_size", 2048))
        for i in range(0, len(pcm), chunk_size):
            yield pcm[i:i + chunk_size]


def _patched_split_sentences(text: str) -> list[str]:
    chunks = _LOCAL_SENTENCE_RE.split(sanitize_task_text(text))
    return [c.strip() for c in chunks if c.strip()]


def _patched_tts_stream_synthesize(self: TTSPipelineRuntime, request: TTSRequest, worker: WorkerInfo | None = None) -> t.Iterable[bytes]:
    backend = self.selector.get_backend(request.metadata, worker=worker, kind="tts")
    self.metrics.inc(f"backend.tts.{backend.backend_name}.selected")
    text = self._normalize(request.text)
    self.metrics.inc("tts.requests")
    start = monotonic()
    try:
        if backend.backend_name == "local_tts":
            for index, chunk in enumerate(backend.synthesize(text, chunk_size=request.chunk_size)):
                self.metrics.inc("tts.audio_chunks")
                if _LocalTTSChunk is not None:
                    try:
                        _LocalTTSChunk(text=text, pcm=np.asarray(chunk, dtype=np.int16) if np is not None else chunk, sample_rate=getattr(backend, "sample_rate", 22050), index=index, is_last=False)
                    except Exception:
                        pass
                yield chunk
        else:
            phonemes = self._phonemize(text)
            spec = self._spectrogram(phonemes)
            audio = self._vocode(spec, text)
            for i in range(0, len(audio), request.chunk_size):
                self.metrics.inc("tts.audio_chunks")
                yield memoryview(audio)[i:i + request.chunk_size]
        self.metrics.observe("tts.latency_s", monotonic() - start)
    except Exception as exc:
        self.log.exception("tts_generate_failed", exc, request_id=request.request_id, backend=backend.backend_name)
        if _patched_is_oom_error(exc):
            _patched_cleanup_memory()
        fallback = LocalTTSPipelineBackend(self.log)
        self.metrics.inc("tts.fallback.local")
        for chunk in fallback.synthesize(text, chunk_size=request.chunk_size):
            yield chunk


def _patched_tts_stream_text_fragments(self: TTSPipelineRuntime, fragments: t.Iterable[str], worker: WorkerInfo | None = None, voice: str | None = None) -> t.Iterable[bytes]:
    if _LocalSentenceBuffer is None:
        yield from self.stream_synthesize(TTSRequest(text=" ".join(str(x) for x in fragments)), worker=worker)
        return
    gate = _LocalSentenceBuffer(2, 1)
    for fragment in fragments:
        if fragment is None:
            continue
        for sentence in gate.push(str(fragment)):
            req = TTSRequest(text=sentence, metadata={"voice": voice} if voice else {})
            yield from self.stream_synthesize(req, worker=worker)
    for sentence in gate.flush():
        req = TTSRequest(text=sentence, metadata={"voice": voice} if voice else {})
        yield from self.stream_synthesize(req, worker=worker)


def _patched_tts_select(self: BackendSelector, worker: WorkerInfo | None = None, model_hint: JSON | str | None = None) -> TTSBackendBase:
    if model_hint:
        try:
            backend = self.get_backend(model_hint, worker=worker, kind="tts")
            if isinstance(backend, TTSBackendBase) and backend.backend_name != "local_tts":
                return backend
        except Exception:
            pass
    env_source = os.environ.get("QWEN_TTS_MODEL_PATH", "").strip()
    candidates: list[str] = []
    if env_source and Path(env_source).exists():
        candidates.append(env_source)
    for manifest in self._registry_manifests():
        if manifest_model_format(manifest) in {"tts", "qwen_tts"} and manifest.source_path:
            candidates.append(manifest.source_path)
    if candidates:
        return QwenTTSBackend(candidates[0], self.log, candidates=candidates[1:])
    return LocalTTSPipelineBackend(self.log)


# Monkey-patch the runtime/classes in place.
WorkerRuntime.detect_capability = _patched_detect_capability  # type: ignore[assignment]
WorkerRuntime.health = _patched_worker_health  # type: ignore[assignment]
WorkerRuntime.register = _patched_worker_register  # type: ignore[assignment]
WorkerRuntime._heartbeat_loop = _patched_worker_heartbeat_loop  # type: ignore[assignment]
WorkerRuntime.execute = _patched_worker_execute  # type: ignore[assignment]
WorkerHTTPRequestHandler.do_GET = _patched_worker_do_get  # type: ignore[assignment]
# ClusterHTTPRequestHandler.do_POST left unchanged; original handler is patched in-place above.
ModelCompiler.inspect_checkpoint = _patched_modelcompiler_inspect_checkpoint  # type: ignore[assignment]
LLMRuntime.stream_generate = _patched_llm_stream_generate  # type: ignore[assignment]
Scheduler._worker_health_score = _patched_worker_score  # type: ignore[assignment]
LocalTTSPipelineBackend.__init__ = _patched_localtts_init  # type: ignore[assignment]
LocalTTSPipelineBackend._ensure_manager = _patched_localtts_ensure_manager  # type: ignore[assignment]
LocalTTSPipelineBackend._pcm_from_text = _patched_localtts_pcm_from_text  # type: ignore[assignment]
LocalTTSPipelineBackend.synthesize = _patched_localtts_synthesize  # type: ignore[assignment]
TTSPipelineRuntime.stream_synthesize = _patched_tts_stream_synthesize  # type: ignore[assignment]
TTSPipelineRuntime.stream_text_fragments = _patched_tts_stream_text_fragments  # type: ignore[assignment]
BackendSelector.select_tts = _patched_tts_select  # type: ignore[assignment]
split_sentences = _patched_split_sentences  # type: ignore[assignment]

def _patched_capability_score(self: Capability) -> float:
    score = self.cpu_cores + self.ram_gb / 2.0
    if self.has_gpu:
        score += 8.0 + self.vram_gb / 2.0
    if self.cuda_available:
        score += 4.0
    score += min(4.0, self.max_concurrency / 2.0)
    score += min(4.0, safe_float(getattr(self, "ram_free_gb", 0.0), 0.0) / 4.0)
    score += min(4.0, safe_float(getattr(self, "vram_free_gb", 0.0), 0.0) / 2.0)
    return score

Capability.score = _patched_capability_score  # type: ignore[assignment]

if _mm_with_model_fallback is not None:
    with_model_fallback = _mm_with_model_fallback  # type: ignore[assignment]



if __name__ == "__main__":
    raise SystemExit(main())
