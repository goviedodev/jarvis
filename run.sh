#!/bin/bash
# =============================================================
#  JARVIS - Asistente de Voz Inteligente
#  Lanzador con entorno virtual
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activar entorno virtual
if [ ! -d "venv" ]; then
    echo "❌ Entorno virtual no encontrado."
    echo "   Ejecuta: python3 -m venv venv"
    echo "   Luego: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate

# Verificar dependencias
echo "🔍 Verificando dependencias..."
python3 -c "import faster_whisper; import pyaudio; import requests" 2>/dev/null || {
    echo "⚠️  Instalando dependencias faltantes..."
    pip install faster-whisper pyaudio requests piper-tts keyboard sounddevice 2>&1 | tail -3
}

# Verificar modelo de voz
if [ ! -f "$SCRIPT_DIR/voices/es_ES-davefx-medium.onnx" ]; then
    echo "⚠️  Voz no encontrada. Descargando..."
    python3 -c "
import urllib.request, os
os.makedirs('voices', exist_ok=True)
base = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium'
for f in ['es_ES-davefx-medium.onnx', 'es_ES-davefx-medium.onnx.json']:
    print(f'  Descargando {f}...')
    urllib.request.urlretrieve(base + '/' + f, f'voices/{f}')
print('  ✅ Voz descargada')
"
fi

# Verificar Ollama
echo "🔍 Verificando Ollama..."
if ! curl -s "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo "⚠️  Ollama no está corriendo."
    echo "   Inicia Ollama con: ollama serve"
    echo "   En otra terminal, o como servicio."
fi

echo ""
echo "=============================================="
echo "  🚀 Lanzando JARVIS..."
echo "  Modelo LLM: ${JARVIS_MODEL:-qwen2.5-coder:7b}"
echo "  STM: Faster-Whisper large-v3 (CUDA)"
echo "  TTS: Piper es_ES-davefx-medium"
echo "=============================================="
echo ""

# Pasar argumentos al script
python3 jarvis.py "$@"
