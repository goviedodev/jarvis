#!/usr/bin/env python3
"""
Jarvis - Prueba de reconocimiento de voz (STT) con Faster-Whisper
Graba audio desde el micrófono y lo transcribe usando GPU.
"""

import os
import sys
import tempfile
import wave
import time

import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# Configuración
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024
SILENCE_THRESHOLD = 500  # umbral para silencio
SILENCE_DURATION = 1.5   # segundos de silencio para cortar
MODEL_SIZE = "large-v3"  # o "medium", "small", "base", "tiny"

def list_audio_devices():
    """Lista los dispositivos de audio disponibles."""
    p = pyaudio.PyAudio()
    print("\n=== Dispositivos de audio disponibles ===")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} (entradas: {info['maxInputChannels']}, tasa: {int(info['defaultSampleRate'])})")
    p.terminate()

def record_audio(device_index=None, timeout=30):
    """
    Graba audio desde el micrófono.
    Detiene la grabación cuando detecta silencio prolongado.
    Retorna el path al archivo WAV temporal.
    """
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )

    print("\n🎤 Grabando... (habla ahora, o espera 30s para timeout)")
    print("   La grabación se detendrá cuando dejes de hablar por 1.5s")
    print("   Presiona Ctrl+C para detener manualmente.\n")

    frames = []
    silent_chunks = 0
    started = False
    start_time = time.time()

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Detectar si hay voz
            volume = np.abs(audio_data).mean()
            
            if volume > SILENCE_THRESHOLD:
                if not started:
                    started = True
                    print("  🔴 Hablando...")
                silent_chunks = 0
            else:
                silent_chunks += 1

            if started:
                frames.append(data)

            # Detener tras silencio prolongado
            if started and silent_chunks > int(SAMPLE_RATE / CHUNK * SILENCE_DURATION):
                print("  ⏸️  Silencio detectado, deteniendo...")
                break

            # Timeout de seguridad
            if time.time() - start_time > timeout:
                print("  ⏰ Timeout alcanzado")
                break

    except KeyboardInterrupt:
        print("\n  ⏹️  Detenido por el usuario")

    stream.stop_stream()
    stream.close()
    p.terminate()

    if not frames:
        print("  ❌ No se grabó audio")
        return None

    # Guardar a archivo WAV temporal
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp_file.name
    
    with wave.open(tmp_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    duration = len(frames) * CHUNK / SAMPLE_RATE
    print(f"  ✅ Grabados {duration:.1f} segundos de audio")
    return tmp_path

def transcribe(audio_path, model):
    """Transcribe audio usando Faster-Whisper."""
    print(f"\n🧠 Transcribiendo con Faster-Whisper ({MODEL_SIZE})...")
    
    start = time.time()
    segments, info = model.transcribe(
        audio_path,
        language="es",
        beam_size=5,
        vad_filter=True,       # Filtro de actividad de voz
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    
    segments = list(segments)
    elapsed = time.time() - start
    
    print(f"\n📝 Transcripción completa en {elapsed:.1f}s")
    print(f"   Idioma detectado: {info.language} (prob: {info.language_probability:.2%})")
    print(f"   Duración del audio: {info.duration:.1f}s")
    print(f"\n{'='*60}")
    for seg in segments:
        print(f"   {seg.text.strip()}")
    print(f"{'='*60}\n")
    
    return " ".join(seg.text.strip() for seg in segments) if segments else ""

def main():
    print("=" * 60)
    print("  🎤 JARVIS - Prueba de Reconocimiento de Voz (STT)")
    print("  Faster-Whisper + CUDA")
    print("=" * 60)
    
    # Listar dispositivos
    list_audio_devices()
    
    # Seleccionar dispositivo
    device_idx = None
    try:
        choice = input("\nSelecciona el número de dispositivo (Enter para usar el default): ").strip()
        if choice:
            device_idx = int(choice)
    except (ValueError, EOFError):
        pass
    
    # Cargar modelo
    print(f"\n📥 Cargando modelo Faster-Whisper '{MODEL_SIZE}'...")
    print("   (la primera vez descargará el modelo, puede tomar unos minutos)")
    
    model = WhisperModel(
        MODEL_SIZE,
        device="cuda",
        compute_type="float16",
        download_root=os.path.join(os.path.dirname(__file__), "models"),
    )
    print("   ✅ Modelo cargado exitosamente\n")
    
    while True:
        try:
            audio_path = record_audio(device_index=device_idx)
            if audio_path:
                texto = transcribe(audio_path, model)
                if texto:
                    print(f"   📢 Dijiste: \"{texto}\"")
                os.unlink(audio_path)
            
            again = input("\n¿Grabar otra vez? (Enter=si, n=no): ").strip().lower()
            if again == 'n':
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

if __name__ == "__main__":
    main()
