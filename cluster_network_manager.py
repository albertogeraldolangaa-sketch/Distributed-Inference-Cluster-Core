from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from dataclasses import dataclass, field
import enum
import http.server
import json
import logging
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

__all__ = ["ClusterNetwork", "NetworkRole", "ConnectionState", "ClusterNetworkStatus"]

# =============================================================================
# Small helpers
# =============================================================================

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_CYAN = "\033[36m"


def _now() -> float:
    return time.time()


def _mono() -> float:
    return time.monotonic()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on", "y"}:
        return True
    if s in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _mask_secret(value: str | None, keep: int = 4) -> str | None:
    if not value:
        return None
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}…{value[-keep:]}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _atomic_write_text(path: Path, text: str, mode: int | None = None) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    if mode is not None:
        with contextlib.suppress(Exception):
            os.chmod(path, mode)


def _download(url: str, destination: Path, timeout: float = 60.0) -> None:
    _ensure_dir(destination.parent)
    req = urllib.request.Request(url, headers={"User-Agent": "ClusterNetwork/2.0"})
    tmp = destination.with_suffix(destination.suffix + ".download")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, destination)


def _load_dotenv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            data[key] = value
    return data


def _which(*names: str) -> Optional[str]:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def _system() -> str:
    return platform.system().lower()


def _machine() -> str:
    return platform.machine().lower()


def _is_windows() -> bool:
    return _system().startswith("win")


def _has_admin_privileges() -> bool:
    if _is_windows():
        try:
            import ctypes  # noqa: WPS433

            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    try:
        return os.geteuid() == 0  # type: ignore[attr-defined]
    except Exception:
        return False


def _default_node_name() -> str:
    host = socket.gethostname().strip() or "node"
    host = re.sub(r"[^a-zA-Z0-9._-]+", "-", host)
    return host[:63] or "node"


def _normalize_server_url(value: Any) -> str | None:
    if not value:
        return None
    s = str(value).strip().rstrip("/")
    if not s:
        return None
    if not re.match(r"^https?://", s, re.I):
        s = "http://" + s
    return s


def _validate_subprocess_args(args: list[str]) -> None:
    if not args:
        raise ValueError("Empty subprocess args")
    for a in args:
        if not isinstance(a, str):
            raise TypeError("Subprocess args must be strings")
        if "\x00" in a:
            raise ValueError("NUL byte found in subprocess arg")


def _parse_env_file_candidates(dotenv_path: str | os.PathLike[str] | None, state_dir: Path | None) -> dict[str, str]:
    env: dict[str, str] = {}
    candidates: list[Path] = []
    if dotenv_path:
        candidates.append(Path(dotenv_path))
    candidates.append(Path.cwd() / ".env")
    if state_dir is not None:
        candidates.append(state_dir / ".env")
    for p in candidates:
        if p.exists():
            env.update(_load_dotenv(p))
    return env


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str)


# =============================================================================
# Colored logger
# =============================================================================

class ColoredLogger:
    def __init__(self, name: str = "cluster.network", level: int = logging.INFO, use_color: bool | None = None) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
        for handler in self.logger.handlers:
            handler.setFormatter(logging.Formatter("%(message)s"))
        self.use_color = sys.stdout.isatty() if use_color is None else use_color
        self._lock = threading.RLock()

    def _emit(self, tag: str, color: str, message: str, **fields: Any) -> None:
        payload = {
            "ts": round(_now(), 3),
            "tag": tag,
            "msg": message,
            **fields,
        }
        line = _json_dump(payload)
        if self.use_color:
            line = f"{color}{ANSI_BOLD}[{tag}]{ANSI_RESET} {line}"
        with self._lock:
            self.logger.info(line)

    def network(self, message: str, **fields: Any) -> None:
        self._emit("NETWORK", ANSI_CYAN, message, **fields)

    def status(self, message: str, **fields: Any) -> None:
        self._emit("STATUS", ANSI_GREEN, message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._emit("ERROR", ANSI_RED, message, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        self._emit("WARN", ANSI_YELLOW, message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._emit("INFO", ANSI_BLUE, message, **fields)


# =============================================================================
# Public state model
# =============================================================================

class NetworkRole(str, enum.Enum):
    MASTER = "master"
    WORKER = "worker"
    AUTO = "auto"


class ConnectionState(str, enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RESTARTING = "restarting"
    ERROR = "error"


@dataclass
class ClusterNetworkStatus:
    role: str
    connection_state: str
    private_ip: str | None
    process_active: bool
    server_url: str | None
    node_name: str | None
    pid: int | None = None
    last_error: str | None = None
    backoff_s: float = 0.0
    uptime_s: float = 0.0
    watchdog_running: bool = False
    status_endpoint: str | None = None
    binary: str | None = None
    aux_binary: str | None = None
    secret_note: str | None = None


@dataclass
class _ManagedProc:
    name: str
    args: list[str]
    proc: subprocess.Popen[str]
    started_at: float = field(default_factory=_now)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def terminate(self, timeout: float = 8.0) -> None:
        if self.proc.poll() is not None:
            return
        with contextlib.suppress(Exception):
            self.proc.terminate()
        end = _mono() + timeout
        while _mono() < end:
            if self.proc.poll() is not None:
                return
            time.sleep(0.1)
        with contextlib.suppress(Exception):
            self.proc.kill()


class _StatusHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        owner: "ClusterNetwork" = self.server.owner  # type: ignore[attr-defined]
        if self.path.rstrip("/") in {"", "/", "/status"}:
            body = _json_dump(owner.status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.rstrip("/") == "/health":
            body = json.dumps({"ok": owner.is_connected()}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


class _ThreadingHTTPServer(http.server.ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], RequestHandlerClass, owner: "ClusterNetwork") -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.owner = owner


# =============================================================================
# ClusterNetwork
# =============================================================================

class ClusterNetwork:
    """Private-network manager for distributed inference clusters.

    Responsibilities:
    - detect whether the machine is the Headscale control-plane host (master)
      or a Tailscale client node (worker)
    - bootstrap the private overlay network
    - supervise the daemon process with a watchdog
    - expose a simple status snapshot for the rest of the system
    - keep secrets out of normal logs

    This module intentionally does not touch scheduler, API, LLM, TTS, or UI.
    """

    def __init__(
        self,
        *,
        mode: str | NetworkRole | None = None,
        server_url: str | None = None,
        auth_key: str | None = None,
        node_name: str | None = None,
        state_dir: str | os.PathLike[str] | None = None,
        config_dir: str | os.PathLike[str] | None = None,
        bin_dir: str | os.PathLike[str] | None = None,
        log_level: int | str = logging.INFO,
        auto_download: bool = True,
        watchdog_interval: float = 3.0,
        restart_backoff: float = 2.0,
        use_userspace_networking: bool | None = None,
        mask_secrets_in_logs: bool = True,
        dotenv_path: str | os.PathLike[str] | None = None,
        master_user_name: str | None = None,
        expose_status_endpoint: bool = True,
        status_bind_host: str = "127.0.0.1",
        status_bind_port: int = 0,
        test_mode: bool = False,
        **overrides: Any,
    ) -> None:
        self.state_dir = Path(state_dir or os.environ.get("CLUSTER_NETWORK_STATE_DIR", "./cluster_network_state")).expanduser()
        self.config_dir = Path(config_dir or os.environ.get("CLUSTER_NETWORK_CONFIG_DIR", str(self.state_dir / "config"))).expanduser()
        self.bin_dir = Path(bin_dir or os.environ.get("CLUSTER_NETWORK_BIN_DIR", str(self.state_dir / "bin"))).expanduser()
        self.public_dir = self.state_dir / "public"
        self.secret_dir = self.state_dir / "secret"
        self.dotenv_path = dotenv_path
        self._env = _parse_env_file_candidates(dotenv_path, self.state_dir)

        def pick(explicit: Any, env_keys: list[str], default: Any = None) -> Any:
            if explicit is not None:
                return explicit
            for key in env_keys:
                if key in self._env:
                    return self._env[key]
                if key in os.environ:
                    return os.environ[key]
            return default

        raw_mode = pick(mode, ["CLUSTER_NETWORK_MODE", "NETWORK_MODE", "ROLE"], "auto")
        if isinstance(raw_mode, NetworkRole):
            raw_mode = raw_mode.value
        self.mode = str(raw_mode).strip().lower()
        if self.mode not in {"auto", "master", "worker"}:
            raise ValueError("mode must be one of: auto, master, worker")

        self.server_url = _normalize_server_url(pick(server_url, ["CLUSTER_NETWORK_SERVER_URL", "HEADSCALE_SERVER_URL", "TAILSCALE_LOGIN_SERVER"], None))
        self.auth_key = str(pick(auth_key, ["CLUSTER_NETWORK_AUTH_KEY", "HEADSCALE_AUTH_KEY", "TAILSCALE_AUTH_KEY"], "") or "")
        self.node_name = str(pick(node_name, ["CLUSTER_NETWORK_NODE_NAME", "HEADSCALE_NODE_NAME", "TAILSCALE_NODE_NAME"], _default_node_name()) or _default_node_name())
        self.master_user_name = str(pick(master_user_name, ["CLUSTER_NETWORK_MASTER_USER", "HEADSCALE_USER_NAME"], "cluster") or "cluster")
        self.auto_download = _coerce_bool(pick(auto_download, ["CLUSTER_NETWORK_AUTO_DOWNLOAD"], auto_download), True)
        self.watchdog_interval = max(0.5, _safe_float(pick(watchdog_interval, ["CLUSTER_NETWORK_WATCHDOG_INTERVAL"], watchdog_interval), watchdog_interval))
        self.restart_backoff = max(0.5, _safe_float(pick(restart_backoff, ["CLUSTER_NETWORK_RESTART_BACKOFF"], restart_backoff), restart_backoff))
        self.use_userspace_networking = _coerce_bool(
            pick(use_userspace_networking, ["CLUSTER_NETWORK_USE_USERSPACE_NETWORKING", "TAILSCALE_USE_USERSPACE_NETWORKING"], None),
            default=not _has_admin_privileges(),
        )
        self.mask_secrets_in_logs = _coerce_bool(pick(mask_secrets_in_logs, ["CLUSTER_NETWORK_MASK_SECRETS_IN_LOGS"], mask_secrets_in_logs), True)
        self.expose_status_endpoint = _coerce_bool(pick(expose_status_endpoint, ["CLUSTER_NETWORK_EXPOSE_STATUS_ENDPOINT"], expose_status_endpoint), True)
        self.status_bind_host = str(pick(status_bind_host, ["CLUSTER_NETWORK_STATUS_BIND_HOST"], status_bind_host) or "127.0.0.1")
        self.status_bind_port = _safe_int(pick(status_bind_port, ["CLUSTER_NETWORK_STATUS_BIND_PORT"], status_bind_port), status_bind_port)
        self.test_mode = _coerce_bool(pick(test_mode, ["CLUSTER_NETWORK_TEST_MODE"], test_mode), False)
        self.overrides = overrides

        if isinstance(log_level, str):
            self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        else:
            self.log_level = int(log_level)

        self.log = ColoredLogger(level=self.log_level)

        _ensure_dir(self.state_dir)
        _ensure_dir(self.config_dir)
        _ensure_dir(self.bin_dir)
        _ensure_dir(self.public_dir)
        _ensure_dir(self.secret_dir)

        self._role = self._detect_role()
        self._connection_state = ConnectionState.STOPPED
        self._connected = False
        self._private_ip: str | None = None
        self._bind_ip: str | None = None
        self._start_ts = _now()
        self._last_error: str | None = None
        self._backoff_current = float(self.restart_backoff)
        self._failure_count = 0
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._started = False
        self._lock = threading.RLock()
        self._main_proc: _ManagedProc | None = None
        self._aux_proc: _ManagedProc | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._status_server: _ThreadingHTTPServer | None = None
        self._status_server_thread: threading.Thread | None = None
        self._status_endpoint_url: str | None = None
        self._install_signal_handlers = True
        self._master_preauth_key: str | None = None
        self._headscale_config_path: Path | None = None
        self._tailscale_state_path: Path | None = None
        self._tailscaled_socket_path: Path | None = None
        self._headscale_bin: str | None = None
        self._tailscale_bin: str | None = None
        self._tailscaled_bin: str | None = None

        self.log.network(
            "init",
            role=self._role,
            mode=self.mode,
            server_url=self.server_url,
            node_name=self.node_name,
            state_dir=str(self.state_dir),
            config_dir=str(self.config_dir),
            bin_dir=str(self.bin_dir),
            auto_download=self.auto_download,
            userspace=self.use_userspace_networking,
            test_mode=self.test_mode,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stop_event.clear()
            self._connection_state = ConnectionState.STARTING
            if self._install_signal_handlers and threading.current_thread() is threading.main_thread():
                self._install_signals()

        if self._role == "auto":
            self._role = self._detect_role()

        if self._role == "master":
            self._start_master()
        else:
            self._start_worker()

        self._loop_thread = threading.Thread(target=self._run_watchdog_loop, name="cluster-network-watchdog", daemon=True)
        self._loop_thread.start()

        # Wait briefly so the object is usable immediately.
        self._ready_event.wait(timeout=15.0)
        self.log.status("started", role=self._role, state=self._connection_state.value, connected=self._connected)

    def stop(self) -> None:
        self._stop_event.set()
        self._connection_state = ConnectionState.STOPPED
        self._connected = False
        self._ready_event.set()

        with self._lock:
            procs = [p for p in (self._aux_proc, self._main_proc) if p is not None]
            self._aux_proc = None
            self._main_proc = None

        for proc in procs:
            with contextlib.suppress(Exception):
                proc.terminate(timeout=8.0)

        if self._status_server is not None:
            with contextlib.suppress(Exception):
                self._status_server.shutdown()
                self._status_server.server_close()
            self._status_server = None

        if self._watchdog_thread and self._watchdog_thread.is_alive() and threading.current_thread() is not self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
        if self._loop_thread and self._loop_thread.is_alive() and threading.current_thread() is not self._loop_thread:
            self._loop_thread.join(timeout=2.0)

        self.log.network("stopped", role=self._role)

    def restart(self) -> None:
        self.log.warn("restart_requested", role=self._role)
        self.stop()
        time.sleep(min(self.restart_backoff, 2.0))
        with self._lock:
            self._started = False
        self.start()

    def get_ip(self) -> str | None:
        if self._private_ip:
            return self._private_ip
        if self._role == "master" and self._bind_ip:
            return self._bind_ip
        self._private_ip = self._query_private_ip()
        return self._private_ip or self._bind_ip

    def status(self) -> dict[str, Any]:
        proc_active = self._main_proc is not None and self._main_proc.alive()
        pid = self._main_proc.proc.pid if self._main_proc is not None else None
        status = dataclasses.asdict(
            ClusterNetworkStatus(
                role=self._role,
                connection_state=self._connection_state.value,
                private_ip=self.get_ip(),
                process_active=proc_active,
                server_url=self.server_url,
                node_name=self.node_name,
                pid=pid,
                last_error=self._last_error,
                backoff_s=self._backoff_current,
                uptime_s=max(0.0, _now() - self._start_ts),
                watchdog_running=bool(self._watchdog_thread and self._watchdog_thread.is_alive()),
                status_endpoint=self._status_endpoint_url,
                binary=self._main_proc.args[0] if self._main_proc else self._main_binary(),
                aux_binary=self._aux_proc.args[0] if self._aux_proc else self._aux_binary(),
                secret_note=("masked" if self.mask_secrets_in_logs else None),
            )
        )
        return status

    def is_connected(self) -> bool:
        if self._role == "master":
            return self._main_proc is not None and self._main_proc.alive()
        if not self._connected:
            return False
        if self._main_proc is None or not self._main_proc.alive():
            return False
        return self._worker_backend_healthy()

    def server_state(self) -> dict[str, Any]:
        return {
            "role": self._role,
            "state": self._connection_state.value,
            "connected": self.is_connected(),
            "private_ip": self.get_ip(),
            "server_url": self.server_url,
            "node_name": self.node_name,
            "preauth_key": _mask_secret(self._master_preauth_key) if self.mask_secrets_in_logs else self._master_preauth_key,
        }

    def bind_for_services(self) -> str:
        return self.get_ip() or self._bind_ip or "0.0.0.0"

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "ClusterNetwork":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Role detection / environment
    # ------------------------------------------------------------------

    def _detect_role(self) -> str:
        if self.mode in {"master", "worker"}:
            return self.mode
        # Heuristic: if a server URL and auth key are provided, assume worker.
        if self.auth_key or self.server_url:
            return "worker"
        return "master"

    def _has_privilege_for_tun(self) -> bool:
        return _has_admin_privileges()

    # ------------------------------------------------------------------
    # Watchdog (async control, running in a dedicated background thread)
    # ------------------------------------------------------------------

    def _run_watchdog_loop(self) -> None:
        self._watchdog_thread = threading.current_thread()
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._watchdog_main())
        finally:
            with contextlib.suppress(Exception):
                self._loop.close()

    async def _watchdog_main(self) -> None:
        self._ready_event.set()
        while not self._stop_event.is_set():
            try:
                if self._role == "master":
                    await self._watch_master_once()
                else:
                    await self._watch_worker_once()
                self._failure_count = 0
            except Exception as exc:
                self._last_error = str(exc)
                self._connection_state = ConnectionState.ERROR
                self.log.error("watchdog_error", role=self._role, error=str(exc))
                self._failure_count += 1
                self._backoff_current = min(max(self.restart_backoff, self._backoff_current * 2.0), 60.0)
            await asyncio.sleep(self.watchdog_interval)

    async def _watch_master_once(self) -> None:
        if self._main_proc is None or not self._main_proc.alive():
            self.log.warn("master_daemon_stopped", action="restart")
            self._restart_master()
            self._connection_state = ConnectionState.RESTARTING
            await asyncio.sleep(self._backoff_current)
            return

        if self.test_mode:
            self._connection_state = ConnectionState.CONNECTED
            self._connected = True
            return

        if self._master_healthcheck():
            self._connection_state = ConnectionState.CONNECTED
            self._connected = True
            self._backoff_current = float(self.restart_backoff)
        else:
            self._connection_state = ConnectionState.DISCONNECTED
            self._connected = False
            self.log.warn("master_health_failed", pid=self._main_proc.proc.pid if self._main_proc else None)
            self._restart_master()
            await asyncio.sleep(self._backoff_current)

    async def _watch_worker_once(self) -> None:
        if self._main_proc is None or not self._main_proc.alive():
            self.log.warn("worker_daemon_stopped", action="restart")
            self._restart_worker()
            self._connection_state = ConnectionState.RESTARTING
            await asyncio.sleep(self._backoff_current)
            return

        if self.test_mode:
            self._connection_state = ConnectionState.CONNECTED
            self._connected = True
            return

        if self._worker_backend_healthy():
            self._connection_state = ConnectionState.CONNECTED
            self._connected = True
            self._private_ip = self._query_private_ip() or self._private_ip
            self._backoff_current = float(self.restart_backoff)
        else:
            self._connection_state = ConnectionState.DISCONNECTED
            self._connected = False
            self.log.warn("worker_connection_lost", node_name=self.node_name)
            if not self._attempt_worker_reconnect():
                self._restart_worker()
            await asyncio.sleep(self._backoff_current)

    def _restart_master(self) -> None:
        self._connection_state = ConnectionState.RESTARTING
        self._stop_processes(keep_status_server=True)
        self._start_master(spawn_watchdog=False)

    def _restart_worker(self) -> None:
        self._connection_state = ConnectionState.RESTARTING
        self._stop_processes(keep_status_server=True)
        self._start_worker(spawn_watchdog=False)

    def _stop_processes(self, keep_status_server: bool = False) -> None:
        with self._lock:
            procs = [p for p in (self._aux_proc, self._main_proc) if p is not None]
            self._aux_proc = None
            self._main_proc = None
        for proc in procs:
            with contextlib.suppress(Exception):
                proc.terminate(timeout=8.0)
        if not keep_status_server and self._status_server is not None:
            with contextlib.suppress(Exception):
                self._status_server.shutdown()
                self._status_server.server_close()
            self._status_server = None
            self._status_endpoint_url = None

    # ------------------------------------------------------------------
    # Master bootstrap
    # ------------------------------------------------------------------

    def _start_master(self, spawn_watchdog: bool = True) -> None:
        self._connection_state = ConnectionState.STARTING
        self._ensure_master_layout()
        if self.test_mode:
            self._start_master_test_mode()
            if self.expose_status_endpoint:
                self._ensure_status_server()
            self._connected = True
            self._connection_state = ConnectionState.CONNECTED
            self._ready_event.set()
            return

        headscale_bin = self._resolve_headscale_binary()
        self._headscale_bin = headscale_bin
        self._headscale_config_path = self._write_headscale_config()

        # Validate the config when the command supports it.
        with contextlib.suppress(Exception):
            self._run_command([headscale_bin, "--config", str(self._headscale_config_path), "configtest"], timeout=20.0)

        # Ensure a user exists before starting the daemon.
        self._ensure_headscale_user(headscale_bin, self._headscale_config_path)

        self._main_proc = self._spawn_process(
            "headscale",
            [headscale_bin, "--config", str(self._headscale_config_path), "serve"],
            cwd=str(self.state_dir),
        )
        self.log.network(
            "master_process_started",
            pid=self._main_proc.proc.pid,
            binary=headscale_bin,
            config=str(self._headscale_config_path),
        )

        if self.expose_status_endpoint:
            self._ensure_status_server()

        self._wait_for_master_boot()
        self._master_preauth_key = self._create_preauth_key(headscale_bin, self._headscale_config_path)
        self._private_ip = self._discover_local_ip()
        self._bind_ip = self._private_ip
        self._connected = True
        self._connection_state = ConnectionState.CONNECTED
        self._backoff_current = float(self.restart_backoff)
        self._ready_event.set()
        if self._master_preauth_key:
            self.log.status(
                "master_ready",
                preauth_key=(_mask_secret(self._master_preauth_key) if self.mask_secrets_in_logs else self._master_preauth_key),
                server_url=self.server_url,
            )

    def _start_master_test_mode(self) -> None:
        self._bind_ip = self._discover_local_ip() or "127.0.0.1"
        self._private_ip = self._bind_ip
        self._master_preauth_key = "hskey-test-" + uuid.uuid4().hex
        self._main_proc = self._spawn_test_sleep_proc("headscale-test")
        self.log.network("master_test_daemon_started", pid=self._main_proc.proc.pid)

    def _ensure_master_layout(self) -> None:
        _ensure_dir(self.state_dir)
        _ensure_dir(self.config_dir)
        _ensure_dir(self.bin_dir)
        _ensure_dir(self.public_dir)
        _ensure_dir(self.secret_dir)
        _ensure_dir(self.state_dir / "db")
        _ensure_dir(self.state_dir / "logs")

    def _write_headscale_config(self) -> Path:
        config_path = self.config_dir / "config.yaml"
        db_path = self.state_dir / "db" / "headscale.sqlite"
        noise_key_path = self.secret_dir / "noise_private.key"
        if not self.server_url:
            self.server_url = "http://127.0.0.1:8080"

        # Minimal, practical Headscale YAML based on the official config schema.
        # Keep the file compact to reduce incompatibility risks between versions.
        yaml = f"""# Generated by ClusterNetwork
server_url: {self.server_url}
listen_addr: 0.0.0.0:8080
metrics_listen_addr: 127.0.0.1:9090
grpc_listen_addr: 127.0.0.1:50443
grpc_allow_insecure: false
noise:
  private_key_path: {noise_key_path.as_posix()}
prefixes:
  v4: 100.64.0.0/10
  v6: fd7a:115c:a1e0::/48
  allocation: sequential
database:
  type: sqlite
  sqlite:
    path: {db_path.as_posix()}
dns:
  magic_dns: true
  base_domain: cluster.internal
  override_local_dns: true
  nameservers:
    global:
      - 1.1.1.1
      - 1.0.0.1
derp:
  server:
    enabled: false
"""
        _atomic_write_text(config_path, yaml, mode=0o600)
        return config_path

    def _ensure_headscale_user(self, headscale_bin: str, config_path: Path) -> None:
        code, out, err = self._run_command([headscale_bin, "--config", str(config_path), "users", "create", self.master_user_name], timeout=30.0)
        text = (out + "\n" + err).strip()
        if code == 0 or re.search(r"exists|already", text, re.I):
            self.log.network("headscale_user_ready", user=self.master_user_name)
        else:
            self.log.warn("headscale_user_create_failed", code=code, error=text[:500])

    def _wait_for_master_boot(self) -> None:
        deadline = _mono() + 30.0
        while _mono() < deadline and not self._stop_event.is_set():
            if self._master_healthcheck():
                return
            time.sleep(0.5)
        self.log.warn("master_boot_timeout", server_url=self.server_url)

    def _create_preauth_key(self, headscale_bin: str, config_path: Path) -> str | None:
        base = [headscale_bin, "--config", str(config_path), "preauthkeys", "create", "--user", self.master_user_name]
        attempts = [base + ["--reusable", "--expiration", "24h"], base]
        for args in attempts:
            code, out, err = self._run_command(args, timeout=30.0)
            text = (out + "\n" + err).strip()
            if code == 0:
                key = self._extract_auth_key(text)
                if key:
                    secret_path = self.secret_dir / "headscale_preauth.key"
                    _atomic_write_text(secret_path, key, mode=0o600)
                    self.log.warn("secret_persisted_to_disk", path=str(secret_path), secret_kind="preauth_key")
                    return key
                self.log.error("preauthkey_parse_failed", output=text[:500])
                return None
            if re.search(r"unknown|flag|usage", text, re.I) and args is not attempts[-1]:
                continue
            self.log.error("preauthkey_create_failed", error=text[:700])
        return None

    def _master_healthcheck(self) -> bool:
        if self.test_mode:
            return self._main_proc is not None and self._main_proc.alive()
        if not self._main_proc or not self._main_proc.alive():
            return False
        if not self._headscale_config_path:
            return False
        try:
            parsed = urllib.parse.urlparse(self.server_url or "http://127.0.0.1:8080")
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Worker bootstrap
    # ------------------------------------------------------------------

    def _start_worker(self, spawn_watchdog: bool = True) -> None:
        self._connection_state = ConnectionState.STARTING
        if not self.server_url:
            raise ValueError("worker mode requires server_url")
        if not self.auth_key:
            raise ValueError("worker mode requires auth_key")

        self._tailscale_state_path = self.state_dir / "tailscaled.state"
        self._tailscaled_socket_path = self.state_dir / ("tailscaled.sock" if not _is_windows() else "tailscaled.sock")

        if self.test_mode:
            self._start_worker_test_mode()
            self._connected = True
            self._connection_state = ConnectionState.CONNECTED
            self._ready_event.set()
            return

        self._tailscale_bin = self._resolve_tailscale_binary()
        self._tailscaled_bin = self._resolve_tailscaled_binary() if self._resolve_tailscaled_binary(allow_missing=True) else None

        # Start the daemon when we have a daemon binary.
        if self._tailscaled_bin:
            self._start_tailscaled(self._tailscaled_bin)
        else:
            self.log.warn("tailscaled_binary_missing", action="using_existing_service_or_system_install")

        if not self._attempt_worker_up(self._tailscale_bin):
            raise RuntimeError(f"Worker failed to join tailnet: {self._last_error or 'unknown error'}")

        self._private_ip = self._query_private_ip()
        self._connected = True
        self._connection_state = ConnectionState.CONNECTED
        self._backoff_current = float(self.restart_backoff)
        self._ready_event.set()
        self.log.status("worker_connected", node_name=self.node_name, ip=self._private_ip)

    def _start_worker_test_mode(self) -> None:
        self._private_ip = f"100.64.0.{(os.getpid() % 200) + 10}"
        self._bind_ip = self._private_ip
        self._main_proc = self._spawn_test_sleep_proc("tailscale-test")
        self._aux_proc = self._spawn_test_sleep_proc("tailscaled-test")
        self.log.network("worker_test_daemons_started", pid=self._main_proc.proc.pid, ip=self._private_ip)

    def _start_tailscaled(self, tailscaled_bin: str) -> None:
        args = [tailscaled_bin, "--state", str(self._tailscale_state_path)]
        if self._tailscaled_socket_path and not _is_windows():
            args += ["--socket", str(self._tailscaled_socket_path)]
        if self.use_userspace_networking or not self._has_privilege_for_tun():
            args += ["--tun=userspace-networking"]
        self._aux_proc = self._spawn_process("tailscaled", args, cwd=str(self.state_dir))
        self.log.network("tailscaled_started", pid=self._aux_proc.proc.pid, binary=tailscaled_bin, userspace=self.use_userspace_networking)
        time.sleep(0.8)

    def _attempt_worker_up(self, tailscale_bin: str | None = None) -> bool:
        tailscale_bin = tailscale_bin or self._resolve_tailscale_binary()
        args = [
            tailscale_bin,
            "up",
            "--login-server",
            self.server_url or "",
            "--auth-key",
            self.auth_key,
            "--hostname",
            self.node_name,
        ]
        if self.use_userspace_networking and not _is_windows():
            args.append("--tun=userspace-networking")
        code, out, err = self._run_command(args, timeout=90.0)
        text = (out + "\n" + err).strip()
        if code == 0:
            self._connected = True
            self._connection_state = ConnectionState.CONNECTED
            return True
        self._last_error = text[:1200]
        self.log.warn(
            "worker_up_failed",
            code=code,
            error=text[:700],
            auth_key=(_mask_secret(self.auth_key) if self.mask_secrets_in_logs else self.auth_key),
        )
        return False

    def _attempt_worker_reconnect(self) -> bool:
        if self._main_proc is not None and self._main_proc.alive() and self._attempt_worker_up(self._tailscale_bin):
            return True
        return self._restart_worker_and_retry()

    def _restart_worker_and_retry(self) -> bool:
        self._restart_worker()
        if self._main_proc is None:
            return False
        time.sleep(min(self._backoff_current, 5.0))
        return self._attempt_worker_up(self._tailscale_bin)

    def _worker_backend_healthy(self) -> bool:
        if self.test_mode:
            return self._main_proc is not None and self._main_proc.alive()
        if self._main_proc is None or not self._main_proc.alive():
            return False
        tailscale_bin = self._tailscale_bin or _which("tailscale", "tailscale.exe")
        if not tailscale_bin:
            return False
        code, out, err = self._run_command([tailscale_bin, "status", "--json"], timeout=20.0)
        if code != 0:
            self._last_error = (out + "\n" + err).strip()[:1000]
            return False
        try:
            data = json.loads(out)
        except Exception:
            return False
        backend_state = str(data.get("BackendState", data.get("backendState", ""))).lower()
        if backend_state and backend_state not in {"running", "synchronized", "synced"}:
            return False
        return bool(self._query_private_ip())

    # ------------------------------------------------------------------
    # Process and command helpers
    # ------------------------------------------------------------------

    def _spawn_process(self, name: str, args: list[str], cwd: str | None = None) -> _ManagedProc:
        _validate_subprocess_args(args)
        kwargs: dict[str, Any] = {
            "cwd": cwd,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        if _is_windows():
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(args, **kwargs)
        return _ManagedProc(name=name, args=args, proc=proc)

    def _spawn_test_sleep_proc(self, name: str) -> _ManagedProc:
        script = "import time; time.sleep(3600)"
        args = [sys.executable, "-c", script]
        return self._spawn_process(name, args, cwd=str(self.state_dir))

    def _run_command(self, args: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
        _validate_subprocess_args(args)
        try:
            completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            return completed.returncode, completed.stdout or "", completed.stderr or ""
        except FileNotFoundError as exc:
            return 127, "", str(exc)
        except subprocess.TimeoutExpired as exc:
            return 124, exc.stdout or "", exc.stderr or "timeout"

    def _main_binary(self) -> str | None:
        if self._role == "master":
            return self._headscale_bin or _which("headscale", "headscale.exe")
        return self._tailscale_bin or _which("tailscale", "tailscale.exe")

    def _aux_binary(self) -> str | None:
        return self._tailscaled_bin or _which("tailscaled", "tailscaled.exe")

    # ------------------------------------------------------------------
    # Binary resolution / bootstrap
    # ------------------------------------------------------------------

    def _resolve_headscale_binary(self) -> str:
        if self.test_mode:
            return sys.executable
        existing = _which("headscale", "headscale.exe")
        if existing:
            self.log.network("headscale_binary_found", path=existing)
            return existing
        if not self.auto_download:
            raise FileNotFoundError(
                "headscale binary not found. Install the official Headscale binary for your OS or enable auto_download."
            )
        self.log.warn("headscale_binary_missing", action="attempt_download")
        return self._download_headscale_binary()

    def _resolve_tailscale_binary(self) -> str:
        if self.test_mode:
            return sys.executable
        existing = _which("tailscale", "tailscale.exe")
        if existing:
            self.log.network("tailscale_binary_found", path=existing)
            return existing
        if not self.auto_download:
            raise FileNotFoundError(
                "tailscale binary not found. Install the official Tailscale client for your OS or enable auto_download."
            )
        self.log.warn("tailscale_binary_missing", action="attempt_download")
        return self._download_tailscale_binary()

    def _resolve_tailscaled_binary(self, allow_missing: bool = False) -> str | None:
        if self.test_mode:
            return sys.executable
        existing = _which("tailscaled", "tailscaled.exe")
        if existing:
            self.log.network("tailscaled_binary_found", path=existing)
            return existing
        if allow_missing:
            return None
        if not self.auto_download:
            raise FileNotFoundError(
                "tailscaled binary not found. Install the official Tailscale client for your OS or enable auto_download."
            )
        self.log.warn("tailscaled_binary_missing", action="attempt_download")
        return self._download_tailscale_binary()

    def _download_headscale_binary(self) -> str:
        repo_api = "https://api.github.com/repos/juanfont/headscale/releases/latest"
        raw = self._fetch_url(repo_api)
        release = json.loads(raw.decode("utf-8", errors="replace"))
        assets = release.get("assets", []) or []
        system = _system()
        machine = _machine()
        if machine in {"x86_64", "amd64"}:
            arch = "amd64"
        elif machine in {"aarch64", "arm64"}:
            arch = "arm64"
        else:
            arch = machine

        chosen = None
        for asset in assets:
            name = str(asset.get("name", ""))
            lower = name.lower()
            if "headscale" not in lower:
                continue
            if system == "windows" and not lower.endswith(".zip") and not lower.endswith(".exe"):
                continue
            if system == "linux" and not any(ext in lower for ext in ("linux", ".tar.gz", ".tgz", ".zip", "amd64", "arm64")):
                continue
            if arch not in lower and "any" not in lower and "linux" in lower:
                continue
            chosen = asset
            break

        if not chosen:
            raise RuntimeError(
                f"Could not find a compatible Headscale asset in the latest release for {system}/{arch}."
            )

        name = str(chosen.get("name", "headscale"))
        url = str(chosen.get("browser_download_url", ""))
        dest = self.bin_dir / name
        _download(url, dest)
        with contextlib.suppress(Exception):
            os.chmod(dest, 0o755)
        self.log.network("headscale_downloaded", path=str(dest), asset=name)
        return str(dest)

    def _download_tailscale_binary(self) -> str:
        # Best effort only: Tailscale distributes platform installers/packages,
        # so this path tries to obtain a usable package asset when possible.
        system = _system()
        machine = _machine()
        if machine in {"x86_64", "amd64"}:
            arch = "amd64"
        elif machine in {"aarch64", "arm64"}:
            arch = "arm64"
        else:
            arch = machine

        if system == "windows":
            raise RuntimeError(
                "Automatic Windows bootstrap is not handled here. Install the official Tailscale MSI, then rerun."
            )

        # Linux: try the stable package metadata endpoint first.
        meta_raw = self._fetch_url("https://pkgs.tailscale.com/stable/?mode=json")
        try:
            meta = json.loads(meta_raw.decode("utf-8", errors="replace"))
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Tailscale package metadata: {exc}") from exc

        version = meta.get("Version") or meta.get("version")
        candidates: list[str] = []
        if version:
            candidates.extend(
                [
                    f"https://pkgs.tailscale.com/stable/tailscale_{version}_{arch}.tgz",
                    f"https://pkgs.tailscale.com/stable/tailscale_{version}_{arch}.deb",
                    f"https://pkgs.tailscale.com/stable/tailscale_{version}_{arch}.rpm",
                ]
            )
        for url in candidates:
            try:
                dest = self.bin_dir / Path(urllib.parse.urlparse(url).path).name
                _download(url, dest)
                self.log.network("tailscale_downloaded", path=str(dest), url=url)
                with contextlib.suppress(Exception):
                    os.chmod(dest, 0o755)
                return str(dest)
            except Exception:
                continue

        raise RuntimeError(
            "Automatic Tailscale bootstrap was not possible for this system. Install the official client package and retry."
        )

    def _fetch_url(self, url: str, timeout: float = 60.0) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "ClusterNetwork/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    # ------------------------------------------------------------------
    # Status server / discovery
    # ------------------------------------------------------------------

    def _ensure_status_server(self) -> None:
        if self._status_server is not None:
            return
        server = _ThreadingHTTPServer((self.status_bind_host, self.status_bind_port), _StatusHandler, self)
        self._status_server = server
        host, port = server.server_address
        self._status_endpoint_url = f"http://{host}:{port}/status"
        thread = threading.Thread(target=server.serve_forever, name="cluster-network-status", daemon=True)
        self._status_server_thread = thread
        thread.start()
        self.log.network("status_server_started", url=self._status_endpoint_url)

    def _discover_local_ip(self) -> str | None:
        # Best effort private LAN IP; useful for binding local services.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return None

    def _query_private_ip(self) -> str | None:
        if self.test_mode:
            return self._private_ip or self._bind_ip
        tailscale_bin = self._tailscale_bin or _which("tailscale", "tailscale.exe")
        if not tailscale_bin:
            return self._private_ip

        code, out, err = self._run_command([tailscale_bin, "status", "--json"], timeout=20.0)
        if code == 0 and out.strip():
            try:
                data = json.loads(out)
                self_block = data.get("Self") or data.get("self") or {}
                ips = self_block.get("TailscaleIPs") or self_block.get("tailscale_ips") or self_block.get("IPs") or []
                for ip in ips:
                    s = str(ip)
                    if re.match(r"^100(?:\.\d{1,3}){3}$", s):
                        return s
                if ips:
                    return str(ips[0])
            except Exception:
                pass

        code, out, err = self._run_command([tailscale_bin, "ip", "-4"], timeout=20.0)
        text = (out + "\n" + err).strip()
        if code == 0 and text:
            match = re.search(r"\b(100(?:\.\d{1,3}){3})\b", text)
            if match:
                return match.group(1)
            for token in text.split():
                token = token.strip()
                if re.match(r"^100(?:\.\d{1,3}){3}$", token):
                    return token
        return self._private_ip

    def _worker_fetch_status_json(self) -> dict[str, Any] | None:
        tailscale_bin = self._tailscale_bin or _which("tailscale", "tailscale.exe")
        if not tailscale_bin:
            return None
        code, out, err = self._run_command([tailscale_bin, "status", "--json"], timeout=20.0)
        if code != 0:
            self._last_error = (out + "\n" + err).strip()[:1000]
            return None
        try:
            return json.loads(out)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Parsing / secrets / cleanup
    # ------------------------------------------------------------------

    def _extract_auth_key(self, text: str) -> str | None:
        patterns = [
            r"(tskey-[A-Za-z0-9_-]+)",
            r"(hskey-[A-Za-z0-9_-]+)",
            r"(tskey-auth-[A-Za-z0-9_-]+)",
            r"([A-Za-z0-9]{18,}-[A-Za-z0-9_-]{8,})",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1)
        tokens = [t for t in re.split(r"\s+", text.strip()) if t]
        if tokens:
            candidate = tokens[-1]
            if len(candidate) >= 16:
                return candidate
        return None

    def _install_signals(self) -> None:
        def _handler(signum: int, frame: Any) -> None:
            self.log.warn("signal_received", signal=signum)
            self.stop()

        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                with contextlib.suppress(Exception):
                    signal.signal(sig, _handler)

    def mesh_ip_valid(self, value: str | None = None) -> bool:
        ip = value or self.get_ip()
        return bool(ip and re.match(r"^100(?:\.\d{1,3}){3}$", str(ip)))

    def ready_for_core(self) -> bool:
        return self.mesh_ip_valid() and self.is_connected()

    def wait_until_ready(self, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> str:
        deadline = _mono() + max(0.5, float(timeout_s))
        last = None
        while _mono() < deadline and not self._stop_event.is_set():
            if not self._started:
                self.start()
            ip = self.get_ip()
            if self.mesh_ip_valid(ip) and self.ready_for_core():
                self._ready_event.set()
                return str(ip)
            last = ip
            time.sleep(max(0.1, float(poll_interval_s)))
        raise TimeoutError(f"ClusterNetwork did not become ready in {timeout_s}s (last_ip={last!r})")

    def mesh_summary(self) -> dict[str, Any]:
        return {
            "role": self._role,
            "connection_state": self._connection_state.value,
            "private_ip": self._private_ip,
            "mesh_ip": self.get_ip(),
            "bind_ip": self.bind_for_services(),
            "server_url": self.server_url,
            "node_name": self.node_name,
            "connected": self.is_connected(),
            "ready_for_core": self.ready_for_core(),
            "watchdog_running": bool(self._watchdog_thread and self._watchdog_thread.is_alive()),
            "status_endpoint": self._status_endpoint_url,
        }

    # ------------------------------------------------------------------
    # Convenience aliases for the rest of the system
    # ------------------------------------------------------------------

    def connection_state(self) -> str:
        return self._connection_state.value


# =============================================================================
# Integration example helpers
# =============================================================================

def _example_master() -> str:
    return """from cluster_network_manager import ClusterNetwork

network = ClusterNetwork(
    mode=\"master\",
    server_url=\"http://127.0.0.1:8080\",
    node_name=\"headscale-master\",
    auto_download=True,
)
network.start()
print(network.status())
""".strip()


def _example_worker() -> str:
    return """from cluster_network_manager import ClusterNetwork

network = ClusterNetwork(
    mode=\"worker\",
    server_url=\"https://headscale.internal.example\",
    auth_key=\"tskey-auth-REDACTED\",
    node_name=\"worker-01\",
    use_userspace_networking=True,
)
network.start()
print(network.get_ip())
print(network.status())
""".strip()


def _example_integration() -> str:
    return """# distributed_inference_cluster_core.py
from cluster_network_manager import ClusterNetwork

class DistributedInferenceClusterCore:
    def __init__(self, **config):
        self.network = ClusterNetwork(
            mode=config.get(\"network_mode\", \"auto\"),
            server_url=config.get(\"server_url\"),
            auth_key=config.get(\"auth_key\"),
            node_name=config.get(\"node_name\"),
            state_dir=config.get(\"state_dir\"),
            config_dir=config.get(\"config_dir\"),
            bin_dir=config.get(\"bin_dir\"),
        )

    def bootstrap(self):
        self.network.start()
        bind_ip = self.network.bind_for_services()
        # bind LLM/TTS services to bind_ip
        return bind_ip

    def shutdown(self):
        self.network.stop()
""".strip()


# =============================================================================
# Smoke tests and direct execution
# =============================================================================

def _self_test() -> None:
    print("Running ClusterNetwork smoke tests...")
    master = ClusterNetwork(
        mode="master",
        server_url="http://127.0.0.1:8080",
        node_name="unit-master",
        auto_download=False,
        test_mode=True,
        expose_status_endpoint=True,
        watchdog_interval=0.2,
        restart_backoff=0.5,
        mask_secrets_in_logs=True,
    )
    master.start()
    st1 = master.status()
    assert st1["role"] == "master"
    assert st1["process_active"] is True
    assert master.server_state()["connected"] is True
    master.stop()

    worker = ClusterNetwork(
        mode="worker",
        server_url="http://127.0.0.1:8080",
        auth_key="tskey-test-1234567890",
        node_name="unit-worker",
        auto_download=False,
        test_mode=True,
        expose_status_endpoint=False,
        watchdog_interval=0.2,
        restart_backoff=0.5,
        use_userspace_networking=True,
    )
    worker.start()
    st2 = worker.status()
    assert st2["role"] == "worker"
    assert worker.is_connected() is True
    assert isinstance(worker.get_ip(), (str, type(None)))
    worker.stop()

    print("Smoke tests passed.")


if __name__ == "__main__":
    if "--show-examples" in sys.argv:
        print("\n=== MASTER EXAMPLE ===\n")
        print(_example_master())
        print("\n=== WORKER EXAMPLE ===\n")
        print(_example_worker())
        print("\n=== INTEGRATION EXAMPLE ===\n")
        print(_example_integration())
        raise SystemExit(0)

    _self_test()
