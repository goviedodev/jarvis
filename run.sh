#!/bin/bash
# =============================================================
#  JARVIS - Asistente de Voz Inteligente
#  Lanzador con entorno virtual
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Detectar modo pdev ──────────────────────────────────────────────
# Si se pasa --pdev, usar pi CLI como backend LLM en lugar de Ollama
USE_PDEV=false
for arg in "$@"; do
    if [ "$arg" = "--pdev" ]; then
        USE_PDEV=true
        break
    fi
done

# ─── Usar el binario de Python del venv directamente ─────────────────────
# NOTA: No usamos 'source venv/bin/activate' porque el venv fue creado en
# otra ubicación y movido, por lo que el script de activación tiene rutas
# absolutas incorrectas. Usar el binario directamente es más robusto.

VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
VENV_PIP="$SCRIPT_DIR/venv/bin/pip"

# Verificar que el venv existe
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Entorno virtual no encontrado en $SCRIPT_DIR/venv"
    echo "   Ejecuta: python3 -m venv venv"
    echo "   Luego: $VENV_PIP install -r requirements.txt"
    exit 1
fi

# ─── Verificar dependencias clave ────────────────────────────────────
# NOTA: Se usa set +e porque las comprobaciones individuales pueden
# fallar sin que eso sea un error fatal del script.
echo "🔍 Verificando dependencias..."

set +e
DEPS_OK=true
"$VENV_PYTHON" -c "import faster_whisper" 2>/dev/null || { echo "   ⚠️  faster-whisper no instalado"; DEPS_OK=false; }
"$VENV_PYTHON" -c "import pyaudio" 2>/dev/null || { echo "   ⚠️  pyaudio no instalado"; DEPS_OK=false; }
"$VENV_PYTHON" -c "import requests" 2>/dev/null || { echo "   ⚠️  requests no instalado"; DEPS_OK=false; }
# Piper se usa como CLI, no como módulo Python
command -v piper >/dev/null 2>&1 || { echo "   ⚠️  piper CLI no encontrado (pip install piper-tts)"; DEPS_OK=false; }

# Si se usa --pdev, verificar que pi CLI esté disponible
if [ "$USE_PDEV" = true ]; then
    command -v pi >/dev/null 2>&1 || { echo "   ⚠️  pi CLI no encontrado (npm i -g @anthropic-ai/pi)"; DEPS_OK=false; }
fi
set -e

if [ "$DEPS_OK" = false ]; then
    echo ""
    echo "⚠️  Faltan dependencias. Instálalas con:"
    echo "   $VENV_PIP install faster-whisper pyaudio requests piper-tts keyboard sounddevice"
    if [ "$USE_PDEV" = true ]; then
        echo "   npm i -g @anthropic-ai/pi  # Para modo --pdev"
    fi
    echo ""
    echo "   Si pyaudio falla al compilar, instala el paquete del sistema:"
    echo "   sudo apt install python3-pyaudio"
    echo "   Luego copia el módulo al venv:"
    echo "   cp -r /usr/lib/python3/dist-packages/pyaudio* \$(dirname \$(dirname \$VENV_PYTHON))/lib/python3.12/site-packages/"
    echo ""
    exit 1
fi

# ─── Verificar modelo de voz ───────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/voices/es_ES-davefx-medium.onnx" ]; then
    echo "⚠️  Voz no encontrada. Descargando..."
    "$VENV_PYTHON" -c "
import urllib.request, os
os.makedirs('voices', exist_ok=True)
base = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium'
for f in ['es_ES-davefx-medium.onnx', 'es_ES-davefx-medium.onnx.json']:
    print(f'  Descargando {f}...')
    urllib.request.urlretrieve(base + '/' + f, f'voices/{f}')
print('  ✅ Voz descargada')
"
fi

# ─── Verificar Ollama (solo si NO se usa --pdev) ─────────────────────
if [ "$USE_PDEV" = false ]; then
    echo "🔍 Verificando Ollama..."
    if ! curl -s "http://localhost:11434/api/tags" > /dev/null 2>&1; then
        echo "⚠️  Ollama no está corriendo."
        echo "   Inicia Ollama con: ollama serve"
        echo "   En otra terminal, o como servicio."
    fi
else
    echo "🔍 Modo pdev activado: usando pi CLI como backend LLM"
fi

echo ""
echo "=============================================="
echo "  🚀 Lanzando JARVIS..."
if [ "$USE_PDEV" = true ]; then
    echo "  Backend LLM: pi CLI (Gemini)"
else
    echo "  Modelo LLM: ${JARVIS_MODEL:-qwen2.5-coder:7b}"
fi
echo "  STT: Whisper large-v3-turbo (CUDA)"
echo "  TTS: Piper es_ES-davefx-medium"
echo "=============================================="
echo ""

# Ejecutar Jarvis con el Python del venv directamente
"$VENV_PYTHON" jarvis.py "$@"
