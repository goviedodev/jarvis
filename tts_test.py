#!/usr/bin/env python3
"""
Jarvis - Prueba de síntesis de voz (TTS) con Piper
Convierte texto a voz usando el modelo de español.
"""

import os
import sys
import tempfile
import subprocess
import wave
import pyaudio

# Configuración
VOICE_DIR = os.path.join(os.path.dirname(__file__), "voices")
VOICE_MODEL = os.path.join(VOICE_DIR, "es_ES-davefx-medium.onnx")
VOICE_CONFIG = os.path.join(VOICE_DIR, "es_ES-davefx-medium.onnx.json")

SAMPLE_RATE = 22050  # tasa de Piper

def speak(text, output_file=None):
    """
    Convierte texto a voz usando Piper y lo reproduce.
    Si output_file es None, reproduce directamente.
    """
    if not os.path.exists(VOICE_MODEL):
        print(f"❌ Modelo de voz no encontrado: {VOICE_MODEL}")
        print("   Ejecuta primero el script de descarga de voces.")
        return False
    
    if not os.path.exists(VOICE_CONFIG):
        print(f"⚠️  Archivo de configuración no encontrado: {VOICE_CONFIG}")
        print("   Intentando sin configuración...")
    
    print(f"🔊 Sintetizando: \"{text[:60]}...\"", end=" ", flush=True)
    
    try:
        # Ejecutar Piper
        cmd = [
            "piper",
            "--model", VOICE_MODEL,
            "--output-raw",
        ]
        
        if os.path.exists(VOICE_CONFIG):
            cmd.extend(["--config", VOICE_CONFIG])
        
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Enviar texto y leer audio raw
        audio_bytes, stderr = proc.communicate(input=text.encode("utf-8"), timeout=30)
        
        if proc.returncode != 0:
            print(f"❌ Error de Piper: {stderr.decode()[:200]}")
            return False
        
        print("✅")
        
        # Reproducir
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
        )
        stream.write(audio_bytes)
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Opcionalmente guardar a archivo
        if output_file:
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16 bits
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_bytes)
            print(f"   💾 Audio guardado en: {output_file}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout en Piper")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def list_voices():
    """Lista las voces disponibles."""
    if not os.path.exists(VOICE_DIR):
        print("❌ Directorio de voces no encontrado")
        return
    
    voices = []
    for f in os.listdir(VOICE_DIR):
        if f.endswith(".onnx"):
            voices.append(f)
    
    if voices:
        print("\n🎤 Voces disponibles:")
        for v in voices:
            size_mb = os.path.getsize(os.path.join(VOICE_DIR, v)) / (1024 * 1024)
            print(f"   • {v} ({size_mb:.1f} MB)")
    else:
        print("❌ No se encontraron modelos de voz (.onnx)")

def main():
    print("=" * 60)
    print("  🔊 JARVIS - Prueba de Síntesis de Voz (TTS)")
    print("  Piper TTS + Voz Española")
    print("=" * 60)
    
    list_voices()
    
    # Frases de prueba
    test_phrases = [
        "Hola, soy Jarvis. Estoy listo para ayudarte.",
        "¿En qué puedo ayudarte hoy?",
        "Claro, voy a procesar tu solicitud ahora mismo.",
        "La inteligencia artificial está transformando el mundo.",
        "Este es el modelo de voz Davecz en español de España.",
    ]
    
    print("\n📋 Frases de prueba:")
    for i, phrase in enumerate(test_phrases, 1):
        print(f"   [{i}] {phrase}")
    print(f"   [0] Frase personalizada")
    
    try:
        choice = input("\nSelecciona una frase (Enter=1): ").strip()
        
        if choice == "0":
            text = input("Escribe el texto a sintetizar: ").strip()
            if not text:
                text = test_phrases[0]
        elif choice and choice.isdigit() and 1 <= int(choice) <= len(test_phrases):
            text = test_phrases[int(choice) - 1]
        else:
            text = test_phrases[0]
        
        # Guardar también a archivo
        output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        speak(text, output_file=output_file)
        
        print(f"\n💾 Audio guardado temporalmente en: {output_file}")
        print("   (puedes reproducirlo con: aplay " + output_file + ")")
        
    except KeyboardInterrupt:
        print("\n\n👋 Hasta luego!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
