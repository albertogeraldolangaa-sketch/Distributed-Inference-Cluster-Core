from __future__ import annotations

import gc
import logging
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


_OOM_ERROR_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "cublas status alloc failed",
    "std::bad_alloc",
    "cannot allocate memory",
    "memoryerror",
)
_TOKEN_ERROR_MARKERS = (
    "token limit",
    "maximum context length",
    "context length",
    "sequence length",
    "too many tokens",
    "max tokens",
    "overflowed",
)


def _exception_text(exc: BaseException) -> str:
    parts = [str(exc)]
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        parts.append(str(cause))
    context = getattr(exc, "__context__", None)
    if context is not None:
        parts.append(str(context))
    return " ".join(p for p in parts if p).lower()


def _classify_model_error(exc: BaseException) -> str:
    text = _exception_text(exc)
    if any(marker in text for marker in _OOM_ERROR_MARKERS):
        return "oom"
    if any(marker in text for marker in _TOKEN_ERROR_MARKERS):
        return "token_limit"
    return "other"


class HardwareProfile(Enum):
    ECONOMY = "economy"
    STANDARD = "standard"
    ULTRA = "ultra"
    COLAB_T4 = "colab_t4"


class ModelStatus(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    FALLBACK = "fallback"


class ModelType(Enum):
    LLAMA = "llama"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    QWEN = "qwen"
    UNKNOWN = "unknown"


@dataclass
class HardwareInfo:
    has_gpu: bool = False
    gpu_name: Optional[str] = None
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    cuda_available: bool = False
    gpu_layers_capable: int = 0
    is_colab: bool = False
    cpu_count: int = 0
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    platform: str = ""
    is_wsl: bool = False

    @property
    def profile(self) -> HardwareProfile:
        if self.is_colab and self.gpu_name and "T4" in self.gpu_name:
            return HardwareProfile.COLAB_T4
        if self.has_gpu and self.vram_total_gb >= 8 and self.ram_total_gb >= 16:
            return HardwareProfile.ULTRA
        if (self.has_gpu and self.vram_total_gb >= 4) or self.ram_total_gb >= 16:
            return HardwareProfile.STANDARD
        return HardwareProfile.ECONOMY

    @property
    def summary(self) -> str:
        lines = [
            "🧠 Hardware Detectado:",
            f"   CPU: {self.cpu_count} núcleos",
            f"   RAM: {self.ram_total_gb:.1f}GB total, {self.ram_free_gb:.1f}GB livre",
        ]
        if self.has_gpu:
            lines.extend([
                f"   GPU: {self.gpu_name}",
                f"   VRAM: {self.vram_total_gb:.1f}GB total, {self.vram_free_gb:.1f}GB livre",
                f"   CUDA: {'✅' if self.cuda_available else '❌'}",
            ])
        else:
            lines.append("   GPU: ❌ Não detectada (usando CPU)")
        lines.append(f"   Perfil: {self.profile.value.upper()}")
        if self.is_colab:
            lines.append("   🚀 Modo Colab otimizado ativado")
        return "\n".join(lines)


@dataclass
class ModelConfig:
    n_ctx: int = 1024
    n_batch: int = 64
    n_gpu_layers: int = 0
    n_threads: int = 4
    max_tokens: int = 150
    use_mmap: bool = True
    use_mlock: bool = False
    offload_kqv: bool = False
    flash_attn: bool = False
    rope_scaling: Optional[Dict[str, Any]] = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.2
    frequency_penalty: float = 0.5
    presence_penalty: float = 0.0
    profile: HardwareProfile = HardwareProfile.ECONOMY
    model_type: ModelType = ModelType.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "offload_kqv": self.offload_kqv,
            "profile": self.profile.value,
            "model_type": self.model_type.value,
            "flash_attn": self.flash_attn,
            "rope_scaling": self.rope_scaling,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    @staticmethod
    def _detect_model_type(model_path: str) -> ModelType:
        name = Path(model_path).stem.lower()
        if "llama" in name:
            return ModelType.LLAMA
        if "deepseek" in name:
            return ModelType.DEEPSEEK
        if "mistral" in name:
            return ModelType.MISTRAL
        if "gemma" in name:
            return ModelType.GEMMA
        if "qwen" in name:
            return ModelType.QWEN
        return ModelType.UNKNOWN

    @classmethod
    def from_hardware(cls, hardware: HardwareInfo, model_path: str) -> "ModelConfig":
        config = cls()
        config.profile = hardware.profile
        config.model_type = cls._detect_model_type(model_path)
        config.n_threads = 2

        if config.model_type == ModelType.LLAMA:
            config.n_ctx = 8192
            config.flash_attn = True
            config.rope_scaling = {"type": "linear", "factor": 2.0}
            if hardware.has_gpu and hardware.vram_free_gb >= 4:
                config.n_gpu_layers = -1 if hardware.vram_free_gb >= 8 else 28 if hardware.vram_free_gb >= 6 else 20
                config.offload_kqv = hardware.vram_free_gb >= 6
        elif config.model_type == ModelType.DEEPSEEK:
            config.n_ctx = 4096
            config.flash_attn = True
            config.rope_scaling = {"type": "linear", "factor": 2.0}
            config.n_gpu_layers = -1 if hardware.has_gpu and hardware.vram_free_gb >= 6 else 25 if hardware.has_gpu and hardware.vram_free_gb >= 4 else 0
        elif config.model_type == ModelType.MISTRAL:
            config.n_ctx = 8192
            config.flash_attn = True
            config.n_gpu_layers = -1 if hardware.has_gpu and hardware.vram_free_gb >= 8 else 24 if hardware.has_gpu and hardware.vram_free_gb >= 4 else 0
        else:
            config.n_gpu_layers = min(40, max(10, int((hardware.vram_free_gb or 0.0) * 4))) if hardware.has_gpu and hardware.vram_free_gb >= 4 else 0

        if hardware.profile == HardwareProfile.COLAB_T4:
            config.n_ctx = min(config.n_ctx, 4096)
            config.n_batch = 1024
            config.use_mlock = False
            config.offload_kqv = True
        elif hardware.profile == HardwareProfile.ULTRA:
            config.n_batch = 1024
            config.use_mlock = hardware.ram_free_gb > 16
        elif hardware.profile == HardwareProfile.STANDARD:
            config.n_batch = 512
        else:
            config.n_ctx = min(config.n_ctx, 2048)
            config.n_batch = 256
            config.flash_attn = False

        if hardware.ram_free_gb < 4:
            config.n_ctx = min(config.n_ctx, 1024)
            config.n_batch = min(config.n_batch, 256)
            config.use_mmap = True
            config.use_mlock = False

        return config


class HardwareAnalyzer:
    def __init__(self):
        self.info = self._analyze()

    def _analyze(self) -> HardwareInfo:
        info = HardwareInfo()
        self._analyze_cpu_ram(info)
        self._analyze_gpu(info)
        self._detect_wsl(info)
        self._detect_colab(info)
        return info

    def _analyze_cpu_ram(self, info: HardwareInfo) -> None:
        try:
            import multiprocessing
            info.cpu_count = multiprocessing.cpu_count()
        except Exception:
            info.cpu_count = os.cpu_count() or 4

        if psutil is not None:
            try:
                mem = psutil.virtual_memory()
                info.ram_total_gb = mem.total / (1024**3)
                info.ram_free_gb = mem.available / (1024**3)
                return
            except Exception:
                pass

        try:
            if os.name == "posix":
                total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
                info.ram_total_gb = total / (1024**3)
                info.ram_free_gb = avail / (1024**3)
            else:
                info.ram_total_gb = 8.0
                info.ram_free_gb = 4.0
        except Exception:
            info.ram_total_gb = 8.0
            info.ram_free_gb = 4.0

    def _analyze_gpu(self, info: HardwareInfo) -> None:
        try:
            import torch  # type: ignore
            info.cuda_available = torch.cuda.is_available()
            if info.cuda_available:
                info.has_gpu = True
                info.gpu_name = torch.cuda.get_device_name(0)
                try:
                    info.vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    free, _total = torch.cuda.mem_get_info(0)
                    info.vram_free_gb = free / (1024**3)
                except Exception:
                    self._get_vram_nvidia_smi(info)
            else:
                self._get_vram_nvidia_smi(info)
        except Exception:
            self._get_vram_nvidia_smi(info)

    def _get_vram_nvidia_smi(self, info: HardwareInfo) -> None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
                if len(parts) >= 3:
                    info.has_gpu = True
                    info.gpu_name = parts[0]
                    info.vram_total_gb = float(parts[1]) / 1024
                    info.vram_free_gb = float(parts[2]) / 1024
        except Exception:
            pass

    def _detect_wsl(self, info: HardwareInfo) -> None:
        try:
            if os.path.exists("/proc/version"):
                with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
                    if "microsoft" in f.read().lower():
                        info.is_wsl = True
        except Exception:
            pass

    def _detect_colab(self, info: HardwareInfo) -> None:
        try:
            import sys
            info.is_colab = "google.colab" in sys.modules
        except Exception:
            pass

    def get_optimized_config(self, model_path: str) -> ModelConfig:
        return ModelConfig.from_hardware(self.info, model_path)

    def log_summary(self) -> None:
        logger.info("\n" + "=" * 60)
        for line in self.info.summary.split("\n"):
            logger.info(line)
        logger.info("=" * 60)


class SmartModelLoader:
    def __init__(self, model_path: str, hardware_analyzer: HardwareAnalyzer):
        self.model_path = model_path
        self.hardware = hardware_analyzer
        self.config = hardware_analyzer.get_optimized_config(model_path)
        self._lock = threading.RLock()
        self._model = None
        self._status = ModelStatus.UNLOADED
        self._load_time = None
        self._load_error = None
        self.stats = {
            "load_attempts": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "total_inference_time": 0.0,
            "inference_count": 0,
            "total_tokens_generated": 0,
        }
        logger.info("📦 SmartModelLoader inicializado")
        logger.info("   Modelo: %s", Path(model_path).name)
        logger.info("   Tipo: %s", self.config.model_type.value.upper())
        logger.info("   Perfil: %s", self.config.profile.value)
        logger.info("   Config: %s", self.config.to_dict())

    @property
    def status(self) -> ModelStatus:
        with self._lock:
            return self._status

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._status == ModelStatus.LOADED and self._model is not None

    @property
    def is_loading(self) -> bool:
        with self._lock:
            return self._status == ModelStatus.LOADING

    def _load_model_impl(self) -> Tuple[bool, Optional[Any], Optional[str]]:
        if not self.model_path or not os.path.exists(self.model_path):
            return False, None, f"Modelo não encontrado: {self.model_path}"
        try:
            from llama_cpp import Llama  # type: ignore
            logger.info("📦 Carregando modelo %s", Path(self.model_path).name)
            start = time.time()
            kwargs: Dict[str, Any] = {
                "model_path": self.model_path,
                "n_ctx": self.config.n_ctx,
                "n_batch": self.config.n_batch,
                "n_gpu_layers": self.config.n_gpu_layers,
                "n_threads": self.config.n_threads,
                "use_mmap": self.config.use_mmap,
                "use_mlock": self.config.use_mlock,
                "offload_kqv": self.config.offload_kqv,
                "verbose": False,
            }
            if self.config.flash_attn:
                kwargs["flash_attn"] = True
            if self.config.rope_scaling:
                kwargs["rope_scaling"] = self.config.rope_scaling
            if self.config.model_type == ModelType.LLAMA:
                kwargs["chat_format"] = "llama-3"
            model = Llama(**kwargs)
            logger.info("✅ Modelo carregado em %.2fs", time.time() - start)
            return True, model, None
        except ImportError as e:
            return False, None, f"llama-cpp-python não instalado: {e}"
        except Exception as e:
            return False, None, f"Erro ao carregar modelo: {e}"

    def _warmup_model(self, model) -> bool:
        try:
            model("Olá, tudo bem?", max_tokens=5, temperature=0.1, echo=False)
            return True
        except Exception as e:
            logger.warning("⚠️ Warmup falhou: %s", e)
            return False

    def get_model(self) -> Optional[Any]:
        with self._lock:
            if self._status == ModelStatus.LOADED and self._model is not None:
                return self._model
            if self._status == ModelStatus.LOADING:
                self._lock.release()
                try:
                    for _ in range(60):
                        time.sleep(1)
                        with self._lock:
                            if self._status != ModelStatus.LOADING:
                                break
                finally:
                    self._lock.acquire()
                return self._model if self._status == ModelStatus.LOADED else None
            if self._status == ModelStatus.FAILED:
                return None
            self.stats["load_attempts"] += 1
            self._status = ModelStatus.LOADING
            self._load_error = None

        success, model, error = self._load_model_impl()
        with self._lock:
            if success:
                self._warmup_model(model)
                self._model = model
                self._status = ModelStatus.LOADED
                self._load_time = time.time()
                self.stats["successful_loads"] += 1
                logger.info("✅ Modelo pronto para uso")
                return model
            self._status = ModelStatus.FAILED
            self._load_error = error
            self.stats["failed_loads"] += 1
            logger.error("❌ Falha no carregamento: %s", error)
            return None

    def _trim_history_messages(self, messages: List[Dict[str, str]], limit: int = 8) -> List[Dict[str, str]]:
        msgs = [m for m in (messages or []) if isinstance(m, dict) and m.get("role") and m.get("content")]
        if len(msgs) <= limit:
            return msgs
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        tail = [m for m in msgs if m.get("role") != "system"][-max(1, limit - len(system_msgs)):]
        return (system_msgs[:1] + tail)[-limit:]

    def _build_chat_prompt(self, messages: List[Dict[str, str]], limit: int = 8) -> str:
        msgs = self._trim_history_messages(messages, limit=limit)
        lines: List[str] = []
        for m in msgs:
            role = str(m.get("role", "user")).strip().lower()
            content = " ".join(str(m.get("content", "")).split())
            if not content:
                continue
            if role == "system":
                lines.append(f"### Sistema:\n{content}")
            elif role == "assistant":
                lines.append(f"### Assistente:\n{content}")
            else:
                lines.append(f"### Usuário:\n{content}")
        lines.append("### Assistente:")
        return "\n\n".join(lines)

    def _invoke(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs) -> Optional[Dict[str, Any]]:
        model = self.get_model()
        if model is None:
            return None
        try:
            start = time.time()
            result = model(
                prompt,
                max_tokens=int(max_tokens or self.config.max_tokens),
                temperature=float(temperature if temperature is not None else self.config.temperature),
                top_p=float(self.config.top_p),
                top_k=int(self.config.top_k),
                repeat_penalty=float(self.config.repeat_penalty),
                stream=False,
                **kwargs,
            )
            dt = time.time() - start
            self.stats["inference_count"] += 1
            self.stats["total_inference_time"] += dt
            try:
                text = result["choices"][0]["text"]
                self.stats["total_tokens_generated"] += max(0, len(str(text).split()))
            except Exception:
                pass
            return result
        except Exception as e:
            logger.exception("Falha na inferência: %s", e)
            return None

    def generate(self, prompt: str, **kwargs) -> str:
        logger.info('💡 Gerando texto LLM | chars=%d', len((prompt or '')))
        result = self._invoke(prompt, **kwargs)
        if result is None:
            return ""
        try:
            return str(result["choices"][0]["text"]).strip()
        except Exception:
            return ""

    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        prompt = self._build_chat_prompt(messages, limit=kwargs.pop("history_limit", 8))
        result = self._invoke(prompt, **kwargs)
        if result is not None:
            return result
        return {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
            "usage": {},
        }

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        model = self.get_model()
        if model is None:
            return
        try:
            stream = model(
                prompt,
                max_tokens=int(kwargs.get("max_tokens", self.config.max_tokens)),
                temperature=float(kwargs.get("temperature", self.config.temperature)),
                top_p=float(kwargs.get("top_p", self.config.top_p)),
                top_k=int(kwargs.get("top_k", self.config.top_k)),
                repeat_penalty=float(kwargs.get("repeat_penalty", self.config.repeat_penalty)),
                stream=True,
            )
            for part in stream:
                token = part["choices"][0].get("text", "")
                if token:
                    yield token
        except Exception as e:
            logger.exception("Falha no stream: %s", e)

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        prompt = self._build_chat_prompt(messages, limit=kwargs.pop("history_limit", 8))
        yield from self.generate_stream(prompt, **kwargs)

    def unload_model(self) -> None:
        with self._lock:
            self._model = None
            self._status = ModelStatus.UNLOADED
            gc.collect()
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        avg = self.stats["total_inference_time"] / self.stats["inference_count"] if self.stats["inference_count"] else 0.0
        return {
            "model_path": self.model_path,
            "status": self.status.value,
            "load_time": self._load_time,
            "load_error": self._load_error,
            "config": self.config.to_dict(),
            "stats": {**self.stats, "avg_inference_time": f"{avg:.2f}s"},
        }


class ModelManager:
    def __init__(self, models_folder: str = "models"):
        self.models_folder = Path(models_folder)
        self.hardware = HardwareAnalyzer()
        self.loaders: Dict[str, SmartModelLoader] = {}
        self.default_loader: Optional[SmartModelLoader] = None
        self.history_limit = 8
        self.max_prompt_chars = 4096
        self.hardware.log_summary()
        self.available_models = self._find_models()
        if self.available_models:
            preferred = sorted(
                self.available_models.items(),
                key=lambda kv: (
                    0 if "qwen" in kv[0].lower() else 1,
                    0 if ("qwen" in kv[0].lower() and ("0.5" in kv[0].lower() or "500m" in kv[0].lower() or "0_5" in kv[0].lower() or "0-5" in kv[0].lower() or "0p5" in kv[0].lower())) else 1,
                    len(kv[0]),
                ),
            )
            default_model = preferred[0][1]
            logger.info("🎯 Selecionando %s como modelo padrão", Path(default_model).name)
            self.default_loader = self._create_loader(default_model)
        else:
            logger.warning("⚠️ Nenhum modelo .gguf encontrado na pasta 'models/'")

    def _find_models(self) -> Dict[str, str]:
        models: Dict[str, str] = {}
        if self.models_folder.exists():
            candidates = [p for p in self.models_folder.iterdir() if p.suffix.lower() == ".gguf"]
            candidates.sort(key=lambda p: p.name.lower())
            for filename in candidates:
                name = filename.stem.replace("_", " ").title()
                models[name] = str(filename)
        return models

    def _create_loader(self, model_path: str) -> SmartModelLoader:
        return SmartModelLoader(model_path, self.hardware)

    def get_loader(self, model_name: Optional[str] = None) -> Optional[SmartModelLoader]:
        if model_name and model_name in self.available_models:
            loader = self.loaders.get(model_name)
            if loader is None:
                loader = self._create_loader(self.available_models[model_name])
                self.loaders[model_name] = loader
            return loader
        return self.default_loader

    def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        loader = self.get_loader(model_name)
        return loader.get_model() if loader else None

    def _trim_history(self, messages: List[Dict[str, str]], limit: Optional[int] = None) -> List[Dict[str, str]]:
        loader = self.get_loader()
        if loader is None:
            return messages
        return loader._trim_history_messages(messages, limit=limit or self.history_limit)

    def generate(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> Optional[str]:
        loader = self.get_loader(model_name)
        if loader:
            return loader.generate(prompt[: self.max_prompt_chars], **kwargs)
        return self.generate_fallback_response(prompt)

    def stream(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> Iterator[str]:
        loader = self.get_loader(model_name)
        if loader:
            yield from loader.generate_stream(prompt[: self.max_prompt_chars], **kwargs)
            return
        yield self.generate_fallback_response(prompt)

    def chat(self, messages: List[Dict[str, str]], model_name: Optional[str] = None, **kwargs) -> Optional[Dict]:
        loader = self.get_loader(model_name)
        if loader:
            trimmed = self._trim_history(messages)
            return loader.create_chat_completion(trimmed, **kwargs)
        return None

    def chat_stream(self, messages: List[Dict[str, str]], model_name: Optional[str] = None, **kwargs) -> Iterator[str]:
        logger.info('💡 Gerando texto LLM em streaming | mensagens=%d', len(messages or []))
        loader = self.get_loader(model_name)
        if loader:
            trimmed = self._trim_history(messages)
            yield from loader.chat_stream(trimmed, **kwargs)
            return
        yield self.generate_fallback_response(" ".join(m.get("content", "") for m in messages[-2:]))

    def unload_all(self) -> None:
        for name, loader in list(self.loaders.items()):
            logger.info("📤 Descarregando %s...", name)
            loader.unload_model()
        if self.default_loader and self.default_loader not in self.loaders.values():
            self.default_loader.unload_model()
        logger.info("✅ Todos os modelos descarregados")

    def get_status(self) -> Dict[str, Any]:
        return {
            "hardware": {
                "has_gpu": self.hardware.info.has_gpu,
                "gpu_name": self.hardware.info.gpu_name,
                "vram_free_gb": self.hardware.info.vram_free_gb,
                "ram_free_gb": self.hardware.info.ram_free_gb,
                "profile": self.hardware.info.profile.value,
                "is_colab": self.hardware.info.is_colab,
            },
            "models": {
                "available": list(self.available_models.keys()),
                "loaders": {name: loader.get_stats() for name, loader in self.loaders.items()},
                "default": self.default_loader.get_stats() if self.default_loader else None,
            },
            "history_limit": self.history_limit,
        }

    def generate_fallback_response(self, prompt: str) -> str:
        prompt = " ".join((prompt or "").split())
        logger.info('💡 Fallback LLM ativo | chars=%d', len(prompt))
        choices = [
            "Estou a processar isso.",
            "Estou a analisar a tua mensagem.",
            "Boa pergunta.",
            "Vou responder de forma objetiva.",
        ]
        return random.choice(choices)


def with_model_fallback(model_manager: ModelManager):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            prompt = kwargs.get("prompt", "")
            try:
                model = model_manager.get_model()
                if model is None:
                    logger.warning("⚠️ Modelo não disponível, usando fallback")
                    return model_manager.generate_fallback_response(prompt)
                return func(*args, **kwargs, model=model)
            except Exception as e:
                kind = _classify_model_error(e)
                logger.error("❌ Erro ao usar modelo (%s): %s", kind, e)
                if kind in {"oom", "token_limit"}:
                    try:
                        gc.collect()
                    except Exception:
                        pass
                    try:
                        import torch  # type: ignore
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return ""
                return model_manager.generate_fallback_response(prompt)
        return wrapper
    return decorator


__all__ = [
    "HardwareProfile",
    "ModelStatus",
    "ModelType",
    "HardwareInfo",
    "ModelConfig",
    "HardwareAnalyzer",
    "SmartModelLoader",
    "ModelManager",
    "with_model_fallback",
]
