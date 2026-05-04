
from __future__ import annotations

import gc
import logging
import os
import queue
import re
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable, Generator, Iterable, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    from qwen_tts import Qwen3TTSModel  # type: ignore
    QWEN_TTS_AVAILABLE = True
except Exception:
    Qwen3TTSModel = None  # type: ignore
    QWEN_TTS_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:
    BitsAndBytesConfig = None  # type: ignore


_SENTENCE_RE = re.compile(r"(?<=[.!?\n])\s+")


@dataclass
class TTSChunk:
    text: str
    pcm: np.ndarray
    sample_rate: int
    index: int = 0
    is_last: bool = False


@dataclass
class TTSConfig:
    models_folder: Optional[str] = None
    default_speaker: str = "serena"
    sample_rate: int = 16000
    chunk_seconds: float = 0.22
    max_chars_per_chunk: int = 120
    max_chars_per_sentence: int = 220
    max_threads: int = 1
    preload: bool = False
    warmup: bool = False
    allow_cuda: bool = True
    gpu_partial: bool = True
    gpu_max_memory_gb: float = 1.5
    cpu_offload_gb: float = 10.0
    low_memory_mode: bool = True
    prefer_int8_cpu: bool = True
    keep_model_ready: bool = True
    language: str = "portuguese"
    pre_roll_sentences: int = 2
    start_after_sentences: int = 1
    silence_fallback_s: float = 0.10


@dataclass
class VisemeFrame:
    label: str = "silence"
    openness: float = 0.0
    rounding: float = 0.0
    closure: float = 1.0
    lip_spread: float = 0.0
    jaw: float = 0.0
    energy: float = 0.0
    timestamp: float = 0.0
    duration: float = 0.0
    is_vowel: bool = False
    is_lip_close: bool = False


def _clean_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _split_sentences(text: str) -> list[str]:
    t = _clean_text(text)
    if not t:
        return []
    parts = [p.strip() for p in _SENTENCE_RE.split(t) if p.strip()]
    if parts:
        return [p if p[-1] in ".!?\n" else p + "." for p in parts]
    return [t if t[-1] in ".!?\n" else t + "."]


def _safe_float_audio(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(x, -1.0, 1.0)


def _to_pcm16(audio: np.ndarray) -> np.ndarray:
    x = _safe_float_audio(audio)
    return (x * 32767.0).astype(np.int16, copy=False)


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    audio = _safe_float_audio(audio)
    if audio.size == 0 or src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr:
        return audio
    ratio = float(dst_sr) / float(src_sr)
    new_len = max(1, int(round(audio.size * ratio)))
    if new_len == audio.size:
        return audio
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=True, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True, dtype=np.float32)
    return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)


def _format_gib(value_gb: float) -> str:
    return f"{max(0.5, float(value_gb)):.1f}GiB"


def _chunk_split(text: str, limit: int) -> list[str]:
    t = _clean_text(text)
    if not t:
        return []
    if len(t) <= limit:
        return [t]
    out: list[str] = []
    buff: list[str] = []
    buff_len = 0
    for token in t.split():
        extra = len(token) + (1 if buff else 0)
        if buff and (buff_len + extra > limit):
            out.append(" ".join(buff))
            buff = [token]
            buff_len = len(token)
        else:
            buff.append(token)
            buff_len += extra
    if buff:
        out.append(" ".join(buff))
    return out


def _is_vowel(ch: str) -> bool:
    return ch.lower() in "aeiouáàâãéêíóôõúü"


def _text_char_weight(ch: str) -> tuple[float, float, float, float, bool, bool, str]:
    ch_l = ch.lower()
    if ch_l in " \t\r\n":
        return 0.02, 0.0, 0.0, 0.98, False, False, "silence"
    if ch_l in ".!?;:":
        return 0.04, 0.0, 0.0, 0.96, False, False, "pause"
    if ch_l in ",-—":
        return 0.08, 0.0, 0.0, 0.92, False, False, "pause_short"
    if ch_l in "bmp":
        return 0.10, 0.04, 0.0, 0.90, False, True, "lip_close"
    if ch_l in "fv":
        return 0.26, 0.05, 0.10, 0.72, False, False, "labiodental"
    if ch_l in "osuóôõ":
        return 0.46, 0.40, 0.04, 0.56, True, False, "round_vowel"
    if ch_l in "eiéê":
        return 0.34, 0.02, 0.50, 0.36, True, False, "spread_vowel"
    if _is_vowel(ch_l):
        return 0.38, 0.12, 0.16, 0.48, True, False, "vowel"
    if ch_l in "kgqjxçsz":
        return 0.22, 0.08, 0.12, 0.58, False, False, "fricative"
    return 0.18, 0.06, 0.08, 0.64, False, False, "consonant"


def _viseme_from_text(text: str, chunk_index: int, chunk_total: int, energy: float, timestamp: float) -> VisemeFrame:
    t = _clean_text(text)
    if not t:
        return VisemeFrame(timestamp=timestamp, energy=energy, duration=0.0)

    visible = [c for c in t if not c.isspace()]
    if not visible:
        return VisemeFrame(timestamp=timestamp, energy=energy, duration=0.0)

    if chunk_total <= 1:
        idx = min(len(visible) - 1, max(0, len(visible) // 2))
    else:
        ratio = chunk_index / max(1, chunk_total - 1)
        idx = min(len(visible) - 1, int(round(ratio * (len(visible) - 1))))

    window = visible[max(0, idx - 1): min(len(visible), idx + 2)]
    openness = 0.0
    rounding = 0.0
    spread = 0.0
    closure = 0.0
    vowel_count = 0
    lip_close = False
    labels: list[str] = []

    for ch in window:
        open_w, round_w, spread_w, close_w, is_vowel, is_lip_close, label = _text_char_weight(ch)
        openness += open_w
        rounding += round_w
        spread += spread_w
        closure += close_w
        lip_close = lip_close or is_lip_close
        vowel_count += 1 if is_vowel else 0
        labels.append(label)

    n = max(1, len(window))
    openness = openness / n
    rounding = rounding / n
    spread = spread / n
    closure = closure / n

    openness = float(np.clip(openness * (0.78 + 0.55 * energy), 0.0, 1.0))
    rounding = float(np.clip(rounding * (0.65 + 0.35 * energy), 0.0, 1.0))
    closure = float(np.clip(closure * (0.88 - 0.20 * energy), 0.0, 1.0))
    spread = float(np.clip(spread * (0.75 + 0.25 * energy), 0.0, 1.0))
    jaw = float(np.clip((openness * 0.72 + energy * 0.28), 0.0, 1.0))

    if lip_close:
        label = "lip_close"
        openness = min(openness, 0.12)
        closure = max(closure, 0.80)
        jaw = min(jaw, 0.18)
    elif vowel_count:
        if rounding > spread and rounding > 0.22:
            label = "round_vowel"
        elif spread > rounding and spread > 0.22:
            label = "spread_vowel"
        else:
            label = "vowel"
    else:
        label = labels[0] if labels else "consonant"

    if any(c in ".!?" for c in t):
        openness *= 0.95
        closure = min(1.0, closure + 0.04)

    return VisemeFrame(
        label=label,
        openness=openness,
        rounding=rounding,
        closure=closure,
        lip_spread=spread,
        jaw=jaw,
        energy=float(np.clip(energy, 0.0, 1.0)),
        timestamp=timestamp,
        duration=0.0,
        is_vowel=bool(vowel_count),
        is_lip_close=bool(lip_close),
    )


class _SentenceBuffer:
    def __init__(self, pre_roll_sentences: int = 2, start_after_sentences: int = 1):
        self.pre_roll_sentences = max(1, int(pre_roll_sentences))
        self.start_after_sentences = max(1, int(start_after_sentences))
        self.buffer = ""
        self.complete: list[str] = []
        self.started = False

    def push(self, text: str) -> list[str]:
        out: list[str] = []
        text = _clean_text(text)
        if not text:
            return out

        self.buffer = f"{self.buffer} {text}".strip() if self.buffer else text

        if self.buffer and self.buffer[-1] in ".!?\n":
            self.complete.extend(_split_sentences(self.buffer))
            self.buffer = ""
        else:
            parts = _split_sentences(self.buffer)
            if len(parts) > 1:
                self.complete.extend(parts[:-1])
                self.buffer = parts[-1]

        if not self.started and len(self.complete) >= self.start_after_sentences:
            self.started = True

        if self.started:
            while self.complete and len(out) < self.pre_roll_sentences:
                out.append(self.complete.pop(0))
        return out

    def flush(self) -> list[str]:
        out = [s for s in self.complete if s.strip()]
        self.complete.clear()
        tail = _clean_text(self.buffer)
        self.buffer = ""
        if tail:
            out.extend(_split_sentences(tail))
        return [s for s in out if s.strip()]


class TTSManager:
    """TTS local com arranque leve, frases completas e emissão estável."""

    def __init__(self, **kwargs):
        cfg = TTSConfig(**kwargs)
        self.cfg = cfg
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.models_folder = cfg.models_folder or os.path.join(base_path, "models")
        self.model_id = os.path.join(self.models_folder, "Qwen3-TTS-12Hz-0.6B-CustomVoice")
        self._lock = threading.RLock()
        self._model = None
        self._loaded = False
        self._loading = False
        self._stop_event = threading.Event()
        self._sample_rate = int(cfg.sample_rate)
        self._device = "cuda" if (cfg.allow_cuda and torch is not None and torch.cuda.is_available()) else "cpu"
        self._gpu_partial = bool(cfg.gpu_partial)
        self._gpu_max_memory_gb = max(0.5, float(cfg.gpu_max_memory_gb))
        self._cpu_offload_gb = max(2.0, float(cfg.cpu_offload_gb))
        self._dtype = torch.float16 if (torch is not None and self._device == "cuda") else (torch.float32 if torch is not None else None)
        self._prefer_int8_cpu = bool(cfg.prefer_int8_cpu)
        self._keep_model_ready = bool(cfg.keep_model_ready)
        self._echo_reference_sink: Optional[Callable[[np.ndarray], None]] = None
        self._audio_listener: Optional[Callable[[np.ndarray], None]] = None
        self._viseme_listener: Optional[Callable[[VisemeFrame], None]] = None
        self._last_chunk_t = 0.0

        if torch is not None:
            try:
                torch.set_grad_enabled(False)
                torch.set_num_threads(max(1, int(cfg.max_threads)))
                try:
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass
            except Exception:
                pass

        logger.info("TTSManager inicializado em %s%s", self._device, " (GPU parcial)" if self._device == "cuda" and self._gpu_partial else "")
        if cfg.preload:
            self.preload_async()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def register_echo_reference_sink(self, sink: Optional[Callable[[np.ndarray], None]]) -> None:
        self._echo_reference_sink = sink

    def register_audio_listener(self, listener: Optional[Callable[[np.ndarray], None]]) -> None:
        self._audio_listener = listener

    def register_viseme_listener(self, listener: Optional[Callable[[VisemeFrame], None]]) -> None:
        self._viseme_listener = listener

    def _emit_viseme(self, frame: VisemeFrame) -> None:
        if self._viseme_listener is None:
            return
        try:
            self._viseme_listener(frame)
        except Exception:
            pass

    def preload(self) -> bool:
        if self._loaded:
            return True
        if self._loading:
            return False
        with self._lock:
            if self._loaded:
                return True
            self._loading = True
            try:
                return self._load_model()
            finally:
                self._loading = False

    def ensure_loaded(self, timeout_s: Optional[float] = None, retries: int = 0) -> bool:
        """Garante que o TTS esteja carregado antes da primeira síntese."""
        deadline = None if timeout_s is None else (time.time() + max(0.0, float(timeout_s)))
        attempts = max(1, int(retries) + 1)

        for attempt in range(attempts):
            if self._loaded and self._model is not None:
                return True

            if self._loading:
                while self._loading and not self._loaded:
                    if deadline is not None and time.time() >= deadline:
                        break
                    time.sleep(0.05)
                if self._loaded and self._model is not None:
                    return True

            if self.preload():
                return True

            if deadline is not None and time.time() >= deadline:
                break

            if attempt + 1 < attempts:
                time.sleep(0.10)

        return bool(self._loaded and self._model is not None)

    def preload_async(self) -> threading.Thread:
        th = threading.Thread(target=self.preload, daemon=True, name="TTSPreload")
        th.start()
        return th

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._loaded = False
            self._loading = False
            self._stop_event.set()
            self._stop_event = threading.Event()
            gc.collect()
            try:
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    def _load_model(self) -> bool:
        if not QWEN_TTS_AVAILABLE:
            logger.warning("Qwen3-TTS indisponível; fallback silencioso ativo")
            self._loaded = False
            return False
        if not os.path.exists(self.model_id):
            logger.warning("Pasta do modelo TTS não encontrada: %s", self.model_id)
            self._loaded = False
            return False

        def _attempt_load(**kwargs):
            return Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)

        try:
            base_kwargs = dict(
                device_map="auto" if self._device == "cuda" else "cpu",
                dtype=self._dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            if self._device == "cuda" and self._gpu_partial:
                base_kwargs["max_memory"] = {0: _format_gib(self._gpu_max_memory_gb), "cpu": _format_gib(self._cpu_offload_gb)}

            if self._device == "cpu" and self._prefer_int8_cpu:
                quant_kwargs = dict(base_kwargs)
                quant_kwargs["device_map"] = "cpu"
                quant_kwargs["load_in_8bit"] = True
                if BitsAndBytesConfig is not None:
                    try:
                        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    except Exception:
                        pass
                try:
                    self._model = _attempt_load(**quant_kwargs)
                except Exception:
                    self._model = None

            if self._model is None:
                base_cpu_kwargs = dict(base_kwargs)
                self._model = _attempt_load(**base_cpu_kwargs)

            if hasattr(self._model, "eval"):
                self._model.eval()
            self._loaded = True
            if self.cfg.warmup:
                self._warmup_model()
            gc.collect()
            return True
        except Exception as exc:
            logger.exception("Falha ao carregar TTS: %s", exc)
            self._loaded = False
            self._model = None
            return False

    def _warmup_model(self) -> None:
        try:
            _ = next(self.stream_synthesize("Olá.", voice=self.cfg.default_speaker), None)
        except Exception:
            pass

    def split_text(self, text: str) -> list[str]:
        return _split_sentences(text)

    def _generate_audio(self, text: str, speaker: Optional[str] = None):
        if self._model is None:
            raise RuntimeError("Modelo TTS não carregado")
        speaker = speaker or self.cfg.default_speaker
        text = _clean_text(text)
        if hasattr(self._model, "generate_custom_voice"):
            try:
                return self._model.generate_custom_voice(text=text, language=self.cfg.language, speaker=speaker)
            except TypeError:
                return self._model.generate_custom_voice(text=text, language="Auto", speaker=speaker)
        if hasattr(self._model, "generate"):
            try:
                return self._model.generate(text=text, language=self.cfg.language)
            except TypeError:
                return self._model.generate(text=text, language="Auto")
        raise RuntimeError("Modelo TTS não expõe API compatível")

    def _fallback_audio(self, text: str) -> tuple[np.ndarray, int]:
        duration = max(self.cfg.silence_fallback_s, min(1.8, 0.10 + len(text) * 0.018))
        audio = np.zeros(int(self._sample_rate * duration), dtype=np.float32)
        return audio, self._sample_rate

    def _yield_audio_chunks(self, audio_np: np.ndarray, sr: int, source_text: str = "") -> Iterator[np.ndarray]:
        audio_np = _safe_float_audio(audio_np)
        if audio_np.ndim > 1:
            audio_np = audio_np.reshape(-1)
        if sr and sr != self._sample_rate:
            audio_np = _resample_linear(audio_np, sr, self._sample_rate)
            sr = self._sample_rate
        if audio_np.size == 0:
            return
        step = max(256, int(sr * self.cfg.chunk_seconds))
        total = max(1, int(np.ceil(audio_np.size / float(step))))
        for i in range(0, len(audio_np), step):
            if self._stop_event.is_set():
                break
            chunk = np.ascontiguousarray(audio_np[i:i + step], dtype=np.float32)
            if chunk.size == 0:
                continue
            if self._echo_reference_sink is not None:
                try:
                    self._echo_reference_sink(chunk)
                except Exception:
                    pass
            if self._audio_listener is not None:
                try:
                    self._audio_listener(chunk)
                except Exception:
                    pass
            viseme = _viseme_from_text(source_text, i // step, total, float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0, time.time())
            self._emit_viseme(viseme)
            self._last_chunk_t = time.time()
            yield chunk

    def stream_synthesize(self, text: str, voice: Optional[str] = None) -> Iterator[np.ndarray]:
        self._last_chunk_t = time.time()
        if not self.ensure_loaded():
            silence = np.zeros(int(self._sample_rate * self.cfg.silence_fallback_s), dtype=np.float32)
            self._emit_viseme(VisemeFrame(label="silence", openness=0.0, rounding=0.0, closure=1.0, lip_spread=0.0, jaw=0.0, energy=0.0, timestamp=time.time(), duration=self.cfg.silence_fallback_s, is_vowel=False))
            yield silence
            return

        for sentence in self.split_text(text):
            if self._stop_event.is_set():
                break
            try:
                wavs, sr = self._generate_audio(sentence, speaker=voice)
                audio_np = wavs[0] if isinstance(wavs, list) else wavs
                audio_np = np.asarray(audio_np, dtype=np.float32)
            except Exception:
                logger.exception("Erro ao sintetizar frase: %s", sentence)
                audio_np, sr = self._fallback_audio(sentence)
            yield from self._yield_audio_chunks(audio_np, int(sr or self._sample_rate), source_text=sentence)

    def synthesize(self, text: str, voice: Optional[str] = None) -> Generator[np.ndarray, None, None]:
        if not self.ensure_loaded():
            silence = np.zeros(int(self._sample_rate * self.cfg.silence_fallback_s), dtype=np.float32)
            yield silence
            return
        yield from self.stream_synthesize(text, voice=voice)

    def stream_from_queue(self, text_queue: "queue.Queue[str]", voice: Optional[str] = None) -> Iterator[np.ndarray]:
        gate = _SentenceBuffer(self.cfg.pre_roll_sentences, self.cfg.start_after_sentences)
        while True:
            if self._stop_event.is_set():
                break
            try:
                item = text_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                break
            item = _clean_text(str(item))
            if not item:
                continue
            for sentence in gate.push(item):
                yield from self.stream_synthesize(sentence, voice=voice)
        for sentence in gate.flush():
            yield from self.stream_synthesize(sentence, voice=voice)

    def play_stream(
        self,
        audio_stream: Iterable[np.ndarray],
        sample_rate: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        if sd is None:
            for _ in audio_stream:
                if stop_event is not None and stop_event.is_set():
                    break
            return

        sr = int(sample_rate or self._sample_rate)
        try:
            if hasattr(sd, "OutputStream"):
                with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as stream:
                    for chunk in audio_stream:
                        if stop_event is not None and stop_event.is_set():
                            break
                        data = np.asarray(chunk, dtype=np.float32).reshape(-1, 1)
                        if data.size == 0:
                            continue
                        stream.write(data)
                return
        except Exception:
            logger.debug("OutputStream indisponível; usando fallback bloqueante", exc_info=True)

        for chunk in audio_stream:
            if stop_event is not None and stop_event.is_set():
                break
            data = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if data.size == 0:
                continue
            try:
                sd.play(data, samplerate=sr)
                sd.wait()
            except Exception:
                break

    def generate_and_play(self, text: str, voice: Optional[str] = None) -> bool:
        played = False
        if not self.ensure_loaded():
            return False
        for chunk in self.stream_synthesize(text, voice=voice):
            played = True
            if sd is not None:
                try:
                    sd.play(chunk, samplerate=self._sample_rate)
                    sd.wait()
                except Exception:
                    break
        return played

    def save_to_wav(self, text: str, filename: str, voice: Optional[str] = None) -> bool:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        wrote = False
        wf = None
        try:
            for chunk in self.stream_synthesize(text, voice=voice):
                pcm16 = _to_pcm16(chunk)
                if wf is None:
                    wf = wave.open(filename, "wb")
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self._sample_rate)
                wf.writeframes(pcm16.tobytes())
                wrote = True
            return wrote
        finally:
            try:
                if wf is not None:
                    wf.close()
            except Exception:
                pass
            gc.collect()


class StreamingTTSWorker:
    def __init__(self, tts: TTSManager, text_queue: "queue.Queue[str]", audio_queue: "queue.Queue[np.ndarray]", voice: Optional[str] = None):
        self.tts = tts
        self.text_queue = text_queue
        self.audio_queue = audio_queue
        self.voice = voice
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="StreamingTTSWorker")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.tts.stop()

    def _run(self) -> None:
        try:
            for chunk in self.tts.stream_from_queue(self.text_queue, voice=self.voice):
                if self._stop.is_set():
                    break
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    try:
                        _ = self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(chunk)
                    except Exception:
                        pass
        except Exception:
            logger.exception("TTS worker falhou")


create_tts_manager = lambda **kwargs: TTSManager(**kwargs)
LocalTTSManager = TTSManager
create_local_tts_manager = create_tts_manager

__all__ = [
    "TTSChunk",
    "TTSConfig",
    "VisemeFrame",
    "TTSManager",
    "LocalTTSManager",
    "StreamingTTSWorker",
    "create_tts_manager",
    "create_local_tts_manager",
]


# ======================================================================
# Robustness patch: adaptive local loading, safe fallback semantics
# ======================================================================

_TTS_LOCAL_ROBUST_PATCH = True

try:
    import inspect
except Exception:  # pragma: no cover
    inspect = None  # type: ignore


def _tts_filter_kwargs(callable_obj, kwargs: dict) -> dict:
    """Retém somente argumentos compatíveis quando a assinatura é conhecida.

    O objetivo é tolerar APIs de TTS que aceitem subconjuntos diferentes de
    argumentos entre versões, evitando regressões por kwargs incompatíveis.
    """
    if inspect is None:
        return dict(kwargs)
    try:
        sig = inspect.signature(callable_obj)
    except Exception:
        return dict(kwargs)

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    allowed = set(params.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _tts_load_model_robust(self) -> bool:
    """Carrega o modelo com fallback progressivo de kwargs e quantização opcional."""
    if not QWEN_TTS_AVAILABLE:
        logger.warning("Qwen3-TTS indisponível; fallback silencioso ativo")
        self._loaded = False
        return False
    if not os.path.exists(self.model_id):
        logger.warning("Pasta do modelo TTS não encontrada: %s", self.model_id)
        self._loaded = False
        return False

    def _call_from_pretrained(extra_kwargs: dict):
        kwargs = dict(extra_kwargs)
        kwargs = _tts_filter_kwargs(Qwen3TTSModel.from_pretrained, kwargs)
        return Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)

    try:
        base_kwargs = {
            "device_map": "auto" if self._device == "cuda" else "cpu",
            "dtype": self._dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        if self._device == "cuda" and self._gpu_partial:
            base_kwargs["max_memory"] = {
                0: _format_gib(self._gpu_max_memory_gb),
                "cpu": _format_gib(self._cpu_offload_gb),
            }

        candidates: list[dict] = []

        if self._device == "cpu" and self._prefer_int8_cpu:
            quant_kwargs = dict(base_kwargs)
            quant_kwargs["device_map"] = "cpu"
            quant_kwargs["load_in_8bit"] = True
            if BitsAndBytesConfig is not None:
                try:
                    quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                except Exception:
                    pass
            candidates.append(quant_kwargs)

        candidates.append(dict(base_kwargs))
        candidates.append({k: v for k, v in base_kwargs.items() if k not in {"dtype", "max_memory"}})
        candidates.append({k: v for k, v in base_kwargs.items() if k not in {"dtype", "max_memory", "device_map"}})
        candidates.append({"local_files_only": True})
        candidates.append({})

        last_exc = None
        for candidate in candidates:
            try:
                self._model = _call_from_pretrained(candidate)
                if self._model is not None:
                    break
            except TypeError as exc:
                last_exc = exc
                self._model = None
            except Exception as exc:
                last_exc = exc
                self._model = None

        if self._model is None:
            raise RuntimeError(f"Falha ao carregar TTS com variantes compatíveis: {last_exc}")

        if hasattr(self._model, "eval"):
            try:
                self._model.eval()
            except Exception:
                pass
        self._loaded = True
        if self.cfg.warmup:
            self._warmup_model()
        gc.collect()
        return True
    except Exception as exc:
        logger.exception("Falha ao carregar TTS: %s", exc)
        self._loaded = False
        self._model = None
        return False


TTSManager._load_model = _tts_load_model_robust  # type: ignore[assignment]
