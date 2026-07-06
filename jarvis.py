#!/usr/bin/env python3
"""
JARVIS - Asistente de Voz Inteligente para Linux
Pipeline: VAD / Push-to-Talk → Faster-Whisper (STT) → Ollama (LLM) → Piper (TTS)

Modos de entrada:
  - VAD (Voice Activity Detection): Escucha continua, detecta cuando hablas
  - Push-to-talk: Mantén ESPACIO para hablar
  - Texto: Escribe tus consultas

Uso:
  python3 jarvis.py                    # Selección interactiva de modo
  python3 jarvis.py --vad              # Modo VAD (manos libres)
  python3 jarvis.py --ptt              # Modo push-to-talk
  python3 jarvis.py --text             # Modo texto
  python3 jarvis.py --list-devices     # Listar dispositivos de audio
  python3 jarvis.py --test-mic         # Probar micrófono
  python3 jarvis.py --quick "texto"    # Solo sintetizar texto
"""

import os
import sys
import time
import json
import tempfile
import wave
import subprocess
import threading
import argparse
from pathlib import Path

import pyaudio
import numpy as np
import requests

# ─── Configuración ───────────────────────────────────────────────────────────

# Directorio base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(BASE_DIR, "voices")
VOICE_MODEL = os.path.join(VOICE_DIR, "es_ES-davefx-medium.onnx")
VOICE_CONFIG = os.path.join(VOICE_DIR, "es_ES-davefx-medium.onnx.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Audio
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024

# Whisper
WHISPER_MODEL_SIZE = "large-v3"

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("JARVIS_MODEL", "qwen2.5-coder:7b")

# TTS Piper
PIPER_SAMPLE_RATE = 22050  # tasa nativa de Piper
TTS_OUTPUT_RATE = 48000    # tasa de reproducción (la de tu dispositivo)

# ─── VAD (Voice Activity Detection) ──────────────────────────────────────────
# Silero VAD trabaja con chunks de 30ms (480 samples) a 16000 Hz

VAD_CHUNK_SIZE = 480       # 30ms a 16000 Hz
VAD_SPEECH_THRESHOLD = 0.5 # probabilidad mínima para considerar voz
VAD_MIN_SPEECH_CHUNKS = 6  # ~180ms de habla continua para activar grabación
VAD_SILENCE_CHUNKS = 50    # ~1.5s de silencio para detener grabación
VAD_TIMEOUT_SECS = 30      # timeout total de grabación en segundos

# ─── Estilo de Jarvis ────────────────────────────────────────────────────────

JARVIS_SYSTEM_PROMPT = """Eres Jarvis, un asistente de voz inteligente, eficiente y con personalidad.

REGLAS:
- Responde SIEMPRE en español, de forma clara y natural.
- Tus respuestas deben ser BREVES (máximo 3 oraciones). Esto es un asistente de voz, no texto.
- Sé directo, útil y con un toque de personalidad.
- Si no sabes algo, dilo honestamente.
- No uses markdown, emojis, ni formato especial.
- No hagas preguntas retóricas.
- Prioriza la utilidad sobre la formalidad.
- Si te piden hacer algo en el sistema, explica cómo hacerlo de forma segura."""

# ─── Colores para terminal ──────────────────────────────────────────────────

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    CLEAR = '\033[2J\033[H'


# ═══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 1: Audio - Grabación y reproducción
# ═══════════════════════════════════════════════════════════════════════════════

class AudioManager:
    """Maneja la captura y reproducción de audio."""

    def __init__(self, device_index=None):
        self.device_index = device_index
        self.p = pyaudio.PyAudio()

    def list_devices(self):
        """Lista los dispositivos de entrada disponibles."""
        print(f"\n{Colors.BOLD}🎤 Dispositivos de entrada:{Colors.RESET}")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                default = " (default)" if info.get('defaultSampleRate') else ""
                print(f"  [{i}] {info['name']}{default}")
        print()

    def record_until_silence(self, timeout=30, silence_threshold=500, silence_duration=1.5):
        """
        Graba audio hasta que se detecta silencio prolongado.
        Retorna la ruta del archivo WAV y la duración.
        """
        stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK,
        )

        frames = []
        silent_chunks = 0
        started = False
        start_time = time.time()
        max_silent = int(SAMPLE_RATE / CHUNK * silence_duration)
        max_chunks = int(SAMPLE_RATE / CHUNK * timeout)

        for _ in range(max_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()

            if volume > silence_threshold:
                if not started:
                    started = True
                    print(f"{Colors.GREEN}  🔴 Escuchando...{Colors.RESET}")
                silent_chunks = 0
            else:
                silent_chunks += 1

            if started:
                frames.append(data)
                if silent_chunks > max_silent:
                    break

        stream.stop_stream()
        stream.close()

        if not frames:
            return None, 0

        # Guardar WAV
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))

        duration = len(frames) * CHUNK / SAMPLE_RATE
        return tmp.name, duration

    def play_wav(self, wav_path):
        """Reproduce un archivo WAV."""
        wf = wave.open(wav_path, 'rb')
        stream = self.p.open(
            format=self.p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)
        stream.stop_stream()
        stream.close()
        wf.close()

    def cleanup(self):
        self.p.terminate()


# ═══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 2: STT - Faster-Whisper
# ═══════════════════════════════════════════════════════════════════════════════

class SpeechRecognizer:
    """Reconocimiento de voz con Faster-Whisper."""

    def __init__(self, model_size=WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self.model = None

    def load(self):
        """Carga el modelo (bajo demanda)."""
        if self.model is not None:
            return
        print(f"{Colors.DIM}📥 Cargando Faster-Whisper '{self.model_size}'...{Colors.RESET}")
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            self.model_size,
            device="cuda",
            compute_type="float16",
            download_root=MODELS_DIR,
        )
        print(f"{Colors.DIM}   ✅ Whisper listo{Colors.RESET}")

    def transcribe(self, audio_path):
        """Transcribe un archivo WAV. Retorna el texto."""
        if self.model is None:
            self.load()

        print(f"{Colors.YELLOW}🧠 Transcribiendo...{Colors.RESET}", end=" ", flush=True)
        start = time.time()

        segments, info = self.model.transcribe(
            audio_path,
            language="es",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments = list(segments)
        elapsed = time.time() - start

        text = " ".join(seg.text.strip() for seg in segments)
        print(f"{Colors.DIM}({elapsed:.1f}s, {info.language_probability:.0%}){Colors.RESET}")

        return text


# ═══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 3: LLM - Ollama
# ═══════════════════════════════════════════════════════════════════════════════

class JarvisBrain:
    """Cerebro de Jarvis usando Ollama."""

    def __init__(self, model=OLLAMA_MODEL, system_prompt=JARVIS_SYSTEM_PROMPT):
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history = []

    def think(self, user_input):
        """
        Envía el texto al LLM y retorna la respuesta.
        Mantiene contexto de conversación corto (últimas 5 interacciones).
        """
        # Mantener historial acotado
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})

        print(f"{Colors.MAGENTA}🤔 Pensando...{Colors.RESET}", end=" ", flush=True)
        start = time.time()

        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512,
                    }
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            reply = result["message"]["content"].strip()

            elapsed = time.time() - start
            print(f"{Colors.DIM}({elapsed:.1f}s){Colors.RESET}")

            # Actualizar historial
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": reply})

            return reply

        except requests.exceptions.ConnectionError:
            print(f"{Colors.RED}❌ No se pudo conectar a Ollama{Colors.RESET}")
            print(f"   Asegúrate de que Ollama esté corriendo: ollama serve")
            return "Lo siento, no puedo conectar con mi cerebro en este momento."
        except Exception as e:
            print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
            return f"Lo siento, tuve un error: {e}"

    def clear_history(self):
        """Limpia el historial de conversación."""
        self.conversation_history = []
        print(f"{Colors.DIM}   🧹 Historial limpiado{Colors.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 4: TTS - Piper
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceSynthesizer:
    """Síntesis de voz con Piper TTS."""

    def __init__(self, model_path=VOICE_MODEL, config_path=VOICE_CONFIG):
        self.model_path = model_path
        self.config_path = config_path

    def check_voice(self):
        """Verifica que el modelo de voz exista."""
        if not os.path.exists(self.model_path):
            print(f"{Colors.RED}❌ Voz no encontrada: {self.model_path}{Colors.RESET}")
            print(f"   Descarga una voz con: python3 download_voice.py")
            return False
        return True

    def speak(self, text):
        """Convierte texto a voz y lo reproduce."""
        if not self.check_voice():
            return False

        print(f"{Colors.BLUE}🔊 Hablando...{Colors.RESET}", end=" ", flush=True)
        start = time.time()

        try:
            cmd = [
                "piper",
                "--model", self.model_path,
                "--output-raw",
            ]
            if os.path.exists(self.config_path):
                cmd.extend(["--config", self.config_path])

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            audio_bytes, stderr = proc.communicate(
                input=text.encode("utf-8"), timeout=30
            )

            if proc.returncode != 0:
                error_msg = stderr.decode()[:200]
                print(f"{Colors.RED}❌ Piper error: {error_msg}{Colors.RESET}")
                return False

            # Resamplar de 22050 → 48000 Hz para compatibilidad con el dispositivo
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            if PIPER_SAMPLE_RATE != TTS_OUTPUT_RATE:
                src_len = len(audio_array)
                tgt_len = int(src_len * TTS_OUTPUT_RATE / PIPER_SAMPLE_RATE)
                audio_array = np.interp(
                    np.linspace(0, src_len - 1, tgt_len),
                    np.arange(src_len),
                    audio_array.astype(np.float64),
                )

            audio_bytes_out = audio_array.astype(np.int16).tobytes()

            # Reproducir
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=TTS_OUTPUT_RATE,
                output=True,
            )
            stream.write(audio_bytes_out)
            stream.stop_stream()
            stream.close()
            p.terminate()

            elapsed = time.time() - start
            print(f"{Colors.DIM}({elapsed:.1f}s){Colors.RESET}")
            return True

        except subprocess.TimeoutExpired:
            print(f"{Colors.RED}❌ Timeout en Piper{Colors.RESET}")
            return False
        except Exception as e:
            print(f"{Colors.RED}❌ TTS Error: {e}{Colors.RESET}")
            return False

    def speak_nonblocking(self, text):
        """Habla en un hilo separado para no bloquear."""
        thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
        thread.start()
        return thread


# ═══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 5: VAD - Voice Activity Detection con Silero VAD
# ═══════════════════════════════════════════════════════════════════════════════

class VADManager:
    """
    Detección de actividad de voz usando Silero VAD.
    Detecta cuándo una persona empieza y deja de hablar en tiempo real.
    """

    def __init__(self):
        self.model = None
        self._available = False
        self.running = False

    def load(self):
        """Carga el modelo Silero VAD (bajo demanda)."""
        if self.model is not None:
            return

        print(f"{Colors.DIM}📥 Cargando Silero VAD...{Colors.RESET}")
        try:
            import silero_vad
            self.model = silero_vad.load_silero_vad()
            self._available = True
            print(f"{Colors.DIM}   ✅ VAD listo{Colors.RESET}")
        except Exception as e:
            self._available = False
            print(f"{Colors.YELLOW}   ⚠️  VAD no disponible: {e}{Colors.RESET}")

    @property
    def available(self):
        return self._available

    def is_speech(self, audio_chunk):
        """
        Evalúa si un chunk de audio contiene voz.
        audio_chunk: bytes de PCM int16 a 16000 Hz, 480 samples (30ms)
        Retorna: True/False
        """
        if self.model is None:
            return False

        # Convertir bytes a tensor float32 normalizado [-1, 1]
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        import torch
        audio_tensor = torch.from_numpy(audio_float32)

        # Silero VAD espera [batch, samples] o [samples]
        with torch.no_grad():
            prob = self.model(audio_tensor, SAMPLE_RATE).item()

        return prob > VAD_SPEECH_THRESHOLD

    def listen_for_speech(self, stream, timeout_secs=VAD_TIMEOUT_SECS):
        """
        Escucha el stream hasta que detecta voz o se acaba el timeout.
        Bufferiza los chunks que activan la detección para no perder audio.

        Retorna:
          frames: lista de bytes de audio con la voz detectada
          total_chunks: número total de chunks procesados
        """
        speech_chunks = 0
        total_chunks = 0
        start_time = time.time()
        # Buffer circular para no perder los chunks que activan la detección
        pre_buffer = []

        # 1. Esperar a que empiece a hablar
        while self.running:
            if time.time() - start_time > timeout_secs:
                return [], total_chunks

            data = stream.read(VAD_CHUNK_SIZE, exception_on_overflow=False)
            total_chunks += 1

            if self.is_speech(data):
                speech_chunks += 1
                pre_buffer.append(data)
                if speech_chunks >= VAD_MIN_SPEECH_CHUNKS:
                    break
            else:
                speech_chunks = 0
                pre_buffer.clear()
        else:
            return [], total_chunks

        # 2. Grabar hasta silencio prolongado (incluyendo buffer de activación)
        frames = list(pre_buffer)
        silence_chunks = 0

        while self.running:
            if time.time() - start_time > timeout_secs:
                break

            data = stream.read(VAD_CHUNK_SIZE, exception_on_overflow=False)
            total_chunks += 1
            frames.append(data)

            if self.is_speech(data):
                silence_chunks = 0
            else:
                silence_chunks += 1
                if silence_chunks >= VAD_SILENCE_CHUNKS:
                    break

        return frames, total_chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  JARVIS - El asistente completo
# ═══════════════════════════════════════════════════════════════════════════════

class Jarvis:
    """El asistente de voz definitivo."""

    def __init__(self, model=None, whisper_model=None):
        self.audio = AudioManager()
        self.stt = SpeechRecognizer(model_size=whisper_model or WHISPER_MODEL_SIZE)
        self.brain = JarvisBrain(model=model or OLLAMA_MODEL)
        self.tts = VoiceSynthesizer()
        self.vad = VADManager()
        self.running = False
        self._initialized = False
        self._use_push_to_talk = False
        self._mode = "text"  # text | ptt | vad

    def _check_push_to_talk(self):
        """Verifica si push-to-talk está disponible."""
        try:
            import keyboard
            if os.geteuid() == 0:
                self._use_push_to_talk = True
                return True
            if os.access('/dev/input', os.R_OK):
                self._use_push_to_talk = True
                return True
        except ImportError:
            pass
        self._use_push_to_talk = False
        return False

    def initialize(self, mode=None):
        """Inicializa todos los componentes y selecciona modo."""
        print(f"{Colors.CLEAR}{Colors.BOLD}")
        print("╔══════════════════════════════════════════════════╗")
        print("║        🤖  J.A.R.V.I.S.  v1.0                  ║")
        print("║    Just A Rather Very Intelligent System        ║")
        print("╚══════════════════════════════════════════════════╝")
        print(f"{Colors.RESET}")
        print(f"{Colors.CYAN}🧠 Modelo LLM: {self.brain.model}{Colors.RESET}")
        print(f"{Colors.CYAN}🎤 Modelo STT: Whisper {self.stt.model_size}{Colors.RESET}")
        print(f"{Colors.CYAN}🔊 Modelo TTS: Piper (es_ES-davefx-medium){Colors.RESET}")
        print()

        # Verificar push-to-talk
        has_ptt = self._check_push_to_talk()

        # Cargar Whisper
        self.stt.load()

        # Verificar voz
        if not self.tts.check_voice():
            return False

        # Cargar VAD (modo silencioso si no se va a usar)
        self.vad.load()

        if mode:
            self._mode = mode
        else:
            self._mode = self._select_mode(has_ptt)

        if self._mode == "ptt" and not has_ptt:
            print(f"{Colors.YELLOW}⚠️  Push-to-talk no disponible (sin sudo).{Colors.RESET}")
            print(f"{Colors.YELLOW}   Usando VAD como alternativa.{Colors.RESET}")
            if self.vad.available:
                self._mode = "vad"
            else:
                self._mode = "text"

        # Mostrar info del modo
        mode_names = {"vad": "🎙️ VAD (manos libres)", "ptt": "⌨️ Push-to-talk", "text": "📝 Texto"}
        print(f"{Colors.CYAN}🎯 Modo: {mode_names.get(self._mode, self._mode)}{Colors.RESET}")

        self._initialized = True
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Jarvis listo y operativo.{Colors.RESET}")
        return True

    def _select_mode(self, has_ptt):
        """Menú interactivo para seleccionar modo de entrada."""
        print(f"\n{Colors.BOLD}Selecciona modo de entrada:{Colors.RESET}")

        options = []
        if self.vad.available:
            options.append(("1", "VAD", "🎙️  Manos libres — escucha siempre, detecta cuando hablas"))
        if has_ptt:
            options.append(("2", "Push-to-talk", "⌨️   Mantén ESPACIO para hablar"))
        options.append(("3", "Texto", "📝  Escribe tus consultas"))

        for key, name, desc in options:
            print(f"  [{key}] {desc}")

        try:
            choice = input(f"\n{Colors.CYAN}Modo (Enter={options[0][0]}): {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            choice = options[0][0]

        mode_map = {opt[0]: opt[1].lower() for opt in options}
        # Normalizar a clave interna
        internal_map = {
            "1": "vad", "2": "ptt", "3": "text",
            "vad": "vad", "ptt": "ptt", "text": "text",
            "": options[0][1].lower()
        }
        return internal_map.get(choice, options[0][1].lower())

    def process_query(self, text):
        """Procesa una consulta de texto: LLM → TTS."""
        if not text.strip():
            return

        print(f"\n{Colors.BOLD}{Colors.CYAN}🧑 Tú: {Colors.RESET}{text}")

        response = self.brain.think(text)

        if response:
            print(f"{Colors.BOLD}{Colors.GREEN}🤖 Jarvis: {Colors.RESET}{response}")
            self.tts.speak_nonblocking(response)

    # ─── Modo 1: VAD (manos libres) ─────────────────────────────────────────

    def vad_loop(self):
        """
        Modo manos libres con Silero VAD.
        Escucha el micrófono continuamente y detecta cuándo hablas.
        El stream de audio se mantiene abierto durante toda la sesión.
        """
        if not self.vad.available:
            print(f"{Colors.RED}❌ VAD no disponible. Usa otro modo.{Colors.RESET}")
            return

        print(f"\n{Colors.BOLD}{Colors.GREEN}🎙️  MODO VAD — MANOS LIBRES{Colors.RESET}")
        print(f"{Colors.YELLOW}   Solo habla. Jarvis te escucha y responde automáticamente.{Colors.RESET}")
        print(f"{Colors.DIM}   Di 'salir' o 'terminar' para finalizar.{Colors.RESET}")
        print(f"{Colors.DIM}   Ctrl+C para interrumpir.{Colors.RESET}")
        print(f"{Colors.DIM}   (Sensibilidad: {VAD_SPEECH_THRESHOLD}, silencio: {VAD_SILENCE_CHUNKS/ (SAMPLE_RATE/VAD_CHUNK_SIZE):.1f}s){Colors.RESET}\n")

        self.running = True
        self.vad.running = True

        # Abrir stream UNA VEZ para toda la sesión
        stream = self.audio.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self.audio.device_index,
            frames_per_buffer=VAD_CHUNK_SIZE,
        )

        try:
            while self.running:
                try:
                    print(f"{Colors.DIM}  🎤 Esperando...{Colors.RESET}", end="\r")

                    frames, total_chunks = self.vad.listen_for_speech(stream)

                    if not frames:
                        continue

                    # Guardar audio a WAV temporal
                    audio_bytes = b''.join(frames)
                    duration = len(frames) * VAD_CHUNK_SIZE / SAMPLE_RATE

                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    with wave.open(tmp.name, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(audio_bytes)

                    print(f"{Colors.GREEN}  🎤 {duration:.1f}s detectados, transcribiendo...{Colors.RESET}  ")

                    text = self.stt.transcribe(tmp.name)
                    os.unlink(tmp.name)

                    if text:
                        text_lower = text.lower().strip()
                        if text_lower in ("salir", "exit", "quit", "terminar", "adiós jarvis", "hasta luego"):
                            print(f"\n{Colors.CYAN}👋 Hasta luego!{Colors.RESET}")
                            self.running = False
                            break
                        elif "limpiar" in text_lower:
                            self.brain.clear_history()
                            self.tts.speak("Historial de conversación limpiado.")
                            continue

                        self.process_query(text)

                    print()

                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    print(f"\n{Colors.RED}❌ Error en VAD: {e}{Colors.RESET}")
                    time.sleep(0.5)
        finally:
            self.vad.running = False
            stream.stop_stream()
            stream.close()

    # ─── Modo 2: Push-to-talk ───────────────────────────────────────────────

    def interactive_mode(self):
        """Modo interactivo por teclado (sin push-to-talk)."""
        print(f"\n{Colors.YELLOW}Modo texto interactivo. Escribe 'salir' para terminar.{Colors.RESET}\n")

        while self.running:
            try:
                text = input(f"{Colors.CYAN}🧑 Tú: {Colors.RESET}").strip()
                if text.lower() in ("salir", "exit", "quit", "q"):
                    break
                if text.lower() == "limpiar":
                    self.brain.clear_history()
                    continue
                if not text:
                    continue

                response = self.brain.think(text)
                if response:
                    print(f"{Colors.GREEN}🤖 Jarvis: {Colors.RESET}{response}")
                    self.tts.speak_nonblocking(response)
                print()

            except KeyboardInterrupt:
                break
            except EOFError:
                break

    def push_to_talk_loop(self):
        """Bucle principal: Push-to-talk con tecla de espacio."""
        import keyboard

        print(f"\n{Colors.BOLD}{Colors.YELLOW}🎯 MANTÉN PRESIONADA LA TECLA 'ESPACIO' PARA HABLAR{Colors.RESET}")
        print(f"{Colors.YELLOW}   Suelta la tecla para transcribir y procesar{Colors.RESET}")
        print(f"{Colors.DIM}   Di 'salir' o Ctrl+C para terminar{Colors.RESET}\n")

        self.running = True
        while self.running:
            try:
                keyboard.wait('space', suppress=True)
                if not self.running:
                    break

                print(f"{Colors.GREEN}  🎤 Grabando (suelta espacio para detener)...{Colors.RESET}", end="\r")

                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()

                frames = []
                stream = self.audio.p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=self.audio.device_index,
                    frames_per_buffer=CHUNK,
                )

                while keyboard.is_pressed('space') and self.running:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)

                stream.stop_stream()
                stream.close()

                if not frames:
                    continue

                with wave.open(tmp_path, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(self.audio.p.get_sample_size(FORMAT))
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(b''.join(frames))

                duration = len(frames) * CHUNK / SAMPLE_RATE
                print(f"{Colors.GREEN}  ✅ {duration:.1f}s grabados, transcribiendo...{Colors.RESET}        ")

                text = self.stt.transcribe(tmp_path)
                os.unlink(tmp_path)

                if text:
                    text_lower = text.lower().strip()
                    if text_lower in ("salir", "exit", "quit", "terminar", "adiós jarvis"):
                        print(f"\n{Colors.CYAN}👋 Hasta luego!{Colors.RESET}")
                        self.running = False
                        break
                    elif "limpiar" in text_lower:
                        self.brain.clear_history()
                        self.tts.speak("Historial de conversación limpiado.")
                        continue

                    self.process_query(text)
                print()

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")

    def run(self):
        """Punto de entrada principal."""
        if not self._initialized:
            if not self.initialize():
                return

        self.running = True
        try:
            if self._mode == "vad":
                self.vad_loop()
            elif self._mode == "ptt":
                self.push_to_talk_loop()
            else:
                self.interactive_mode()
        finally:
            self.audio.cleanup()
            print(f"\n{Colors.BOLD}{Colors.CYAN}🤖 Jarvis desconectado.{Colors.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Punto de entrada
# ═══════════════════════════════════════════════════════════════════════════════

def test_mic():
    """Prueba rápida del micrófono."""
    audio = AudioManager()
    audio.list_devices()

    try:
        choice = input("Número de dispositivo (Enter=default): ").strip()
        device_idx = int(choice) if choice else None
    except ValueError:
        device_idx = None

    print(f"\n🎤 Probando micrófono durante 3 segundos...")
    audio.device_index = device_idx
    path, dur = audio.record_until_silence(timeout=3, silence_duration=10)
    audio.cleanup()

    if path:
        print(f"✅ Grabados {dur:.1f}s")
        os.unlink(path)
    else:
        print("❌ No se grabó nada")

def main():
    parser = argparse.ArgumentParser(description="JARVIS - Asistente de Voz")
    parser.add_argument("--list-devices", action="store_true", help="Listar dispositivos de audio")
    parser.add_argument("--test-mic", action="store_true", help="Probar micrófono")
    parser.add_argument("--quick", type=str, help="Sintetizar texto rápidamente")
    parser.add_argument("--model", type=str, help=f"Modelo Ollama (default: {OLLAMA_MODEL})")
    parser.add_argument("--whisper", type=str, help=f"Modelo Whisper (default: {WHISPER_MODEL_SIZE})")
    parser.add_argument("--vad", action="store_true", help="Modo VAD (manos libres)")
    parser.add_argument("--ptt", action="store_true", help="Modo push-to-talk")
    parser.add_argument("--text", action="store_true", help="Modo texto")

    args = parser.parse_args()

    if args.list_devices:
        audio = AudioManager()
        audio.list_devices()
        audio.cleanup()
        return

    if args.test_mic:
        test_mic()
        return

    if args.quick:
        tts = VoiceSynthesizer()
        tts.speak(args.quick)
        return

    # Determinar modo
    mode = None
    if args.vad:
        mode = "vad"
    elif args.ptt:
        mode = "ptt"
    elif args.text:
        mode = "text"

    # Iniciar Jarvis
    jarvis = Jarvis(model=args.model, whisper_model=args.whisper)
    try:
        jarvis.initialize(mode=mode)
        jarvis.run()
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}👋 Hasta luego!{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error fatal: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
