# 🤖 J.A.R.V.I.S.

**Just A Rather Very Intelligent System** — Asistente de voz inteligente para Linux.

Pipeline completo y local: **VAD → Faster-Whisper (STT) → Ollama (LLM) → Piper (TTS)**

<p align="center">
  <img src="https://img.shields.io/badge/STT-Faster--Whisper%20large--v3--turbo-blue" alt="STT">
  <img src="https://img.shields.io/badge/LLM-Ollama%20%2B%20Qwen%202.5-purple" alt="LLM">
  <img src="https://img.shields.io/badge/TTS-Piper%20es--ES-brightgreen" alt="TTS">
  <img src="https://img.shields.io/badge/VAD-Silero%20VAD-orange" alt="VAD">
  <img src="https://img.shields.io/badge/GPU-CUDA-green" alt="GPU">
</p>

---

## ✨ Características

- **🎙️ 3 modos de entrada:** VAD (manos libres), Push-to-talk, o Texto
- **🔊 2 modos de salida:** Voz (Piper TTS con streaming optimizado) o Escritor (texto en consola en tiempo real)
- **🧠 2 backends de LLM:** Ollama local (default) o `pi` CLI con Gemini (`--pdev`)
- **🎤 Reconocimiento de voz preciso:** Faster-Whisper large-v3-turbo con GPU NVIDIA
- **🔊 Voz natural en español:** Piper TTS con voz Davefx (es_ES)
- **🏠 100% offline:** Todo corre en tu máquina, nada sube a la nube (excepto en modo `--pdev`)
- **🖥️ Aceleración GPU:** CUDA para Whisper y CPU optimizada para VAD/TTS
- **⚡ TTS optimizado:** Double buffering, PyAudio persistente y agrupación de oraciones para eliminar pausas

---

## 📋 Requisitos del sistema

| Componente | Requisito mínimo | Recomendado |
|---|---|---|
| **GPU** | NVIDIA con 6GB VRAM | RTX 3060+ (12GB) |
| **RAM** | 8 GB | 16 GB |
| **Disco** | 10 GB libres | 20 GB (modelos) |
| **SO** | Linux con ALSA/PipeWire | Ubuntu 22.04+ / Arch / Fedora |
| **Micrófono** | Cualquier micrófono | USB o headsets con buena cancelación de ruido |
| **Python** | 3.10+ | 3.12 |
| **CUDA** | 11.8+ | 12.x |

### Dependencias del sistema

```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev python3-pyaudio python3-dev build-essential

# Arch
sudo pacman -S portaudio python-pyaudio base-devel

# Fedora
sudo dnf install portaudio-devel python3-devel gcc-c++
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repo> jarvis
cd jarvis
```

### 2. Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install faster-whisper pyaudio requests piper-tts numpy keyboard sounddevice
pip install silero-vad torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> **Nota:** Si no tienes GPU NVIDIA, instala PyTorch sin CUDA:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```

### 4. Descargar la voz española

```bash
mkdir -p voices
python3 -c "
import urllib.request
base = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium'
for f in ['es_ES-davefx-medium.onnx', 'es_ES-davefx-medium.onnx.json']:
    print(f'Descargando {f}...')
    urllib.request.urlretrieve(base + '/' + f, f'voices/{f}')
print('✅ Voz descargada')
"
```

### 5. Instalar y configurar Ollama

```bash
# Instalar Ollama (si no lo tienes)
curl -fsSL https://ollama.com/install.sh | sh

# Iniciar el servicio
ollama serve

# En otra terminal, descargar el modelo
ollama pull qwen2.5-coder:7b
```

> También puedes usar cualquier otro modelo: `ollama pull llama3.2`, `ollama pull qwen3.5:9b`, etc.

### 6. Instalar `pi` CLI (opcional, solo para `--pdev`)

El modo `--pdev` reemplaza Ollama por el CLI `pi` como backend de LLM. Solo lo necesitas si vas a usar ese modo:

```bash
npm i -g @earendil-works/pi-coding-agent
pi --version
```

> **Ojo:** este modo consulta un modelo en la nube (Gemini por default), así que **no es offline**. Con Ollama, Jarvis sigue siendo 100% local.

### 7. Verificar instalación

```bash
source venv/bin/activate

# Probar la voz
python3 jarvis.py --quick "Hola, soy Jarvis. Estoy listo para ayudarte."

# Listar micrófonos
python3 jarvis.py --list-devices

# Probar micrófono
python3 jarvis.py --test-mic
```

---

## 🎮 Uso

### Arranque rápido

```bash
cd jarvis
source venv/bin/activate

# Modo interactivo (elige VAD, push-to-talk, o texto)
python3 jarvis.py

# O directamente:
python3 jarvis.py --vad     # Manos libres
python3 jarvis.py --ptt     # Push-to-talk (requiere sudo)
python3 jarvis.py --text    # Modo texto
```

### Con el script lanzador

```bash
./run.sh                    # Menú interactivo
./run.sh --vad              # Manos libres
sudo ./run.sh --ptt         # Push-to-talk
```

### Modos de entrada

#### 🎙️ VAD — Manos libres (recomendado)

Jarvis escucha el micrófono continuamente. Cuando detecta que hablas, graba automáticamente y procesa tu consulta. Cuando dejas de hablar ~1.5s, transcribe y responde.

```bash
python3 jarvis.py --vad
```

#### ⌨️ Push-to-talk

Mantén presionada la tecla **ESPACIO** mientras hablas. Suelta para que Jarvis procese. Requiere permisos de root en Linux.

```bash
sudo python3 jarvis.py --ptt
```

#### 📝 Modo texto

Escribe tus consultas directamente en la terminal.

```bash
python3 jarvis.py --text
```

### Modos de salida

#### 🔊 Modo voz (default)

Jarvis responde hablando usando Piper TTS. El audio se procesa con double buffering para minimizar pausas entre oraciones.

```bash
python3 jarvis.py --vad              # VAD + voz
python3 jarvis.py --text             # Texto + voz
sudo python3 jarvis.py --ptt         # PTT + voz
```

#### ✍️ Modo escritor

Jarvis escribe las respuestas en la consola en tiempo real, token por token mientras el LLM las genera. Ideal para entornos silenciosos, sesiones SSH, o cuando prefieres leer en lugar de escuchar.

```bash
python3 jarvis.py --vad --writer     # VAD + escritor
python3 jarvis.py --text --writer    # Texto + escritor
sudo python3 jarvis.py --ptt --writer # PTT + escritor
```

**Casos de uso del modo escritor:**
- Entornos de oficina silenciosos
- Sesiones SSH remotas sin audio
- Pruebas rápidas de prompts sin esperar audio
- Documentación automática (puedes redirigir salida a archivo)
- Integración con tmux/screen

**Ventajas del modo escritor:**
- ⚡ Latencia mínima: primer token en ~200ms (vs ~800ms en TTS)
- 💾 Menos memoria: no carga Piper (~200MB ahorrados)
- 🚀 Inicio más rápido: evita descarga/verificación de modelos TTS

### Backends de LLM

Los modos de entrada y salida son independientes del backend que genera las respuestas. Hay dos:

#### 🧠 Ollama (default)

Modelo local, sin internet, con streaming token a token. Requiere `ollama serve` corriendo.

```bash
python3 jarvis.py --vad
python3 jarvis.py --vad --model llama3.2   # cambiar de modelo en caliente
```

#### ☁️ pdev — pi CLI (`--pdev`)

Delega el razonamiento al CLI `pi` (Gemini por default) en lugar de Ollama. Útil cuando querés respuestas de un modelo más grande del que entra en tu VRAM, o cuando no tenés Ollama corriendo.

```bash
python3 jarvis.py --vad --pdev            # VAD + voz, cerebro en pi
python3 jarvis.py --text --pdev --writer  # texto + escritor, cerebro en pi
./run.sh --pdev                           # run.sh omite el chequeo de Ollama
```

Diferencias respecto a Ollama:

| | Ollama | `--pdev` |
|---|---|---|
| **Privacidad** | 100% local | Consulta un servicio en la nube |
| **Streaming** | Sí, token a token | No: espera la respuesta completa |
| **Historial** | Mensajes `role`/`content` nativos | Se aplana a texto (`Usuario:` / `Jarvis:`) |
| **System prompt** | `JARVIS_SYSTEM_PROMPT` en `jarvis.py` | `prompts/pdev_system.md` |
| **Timeout** | 30s | 60s |
| **Requisito** | `ollama serve` | `pi` en el PATH |

Como `--pdev` no hace streaming, combinarlo con `--writer` imprime la respuesta completa de golpe en vez de token a token, y en modo voz el TTS arranca recién cuando `pi` terminó de responder.

El prompt de personalidad de este modo vive en `prompts/pdev_system.md` y es distinto al de Ollama — si ajustás el tono de Jarvis, acordate de tocar los dos.

### Comandos de voz

Durante la conversación, puedes decir:

| Comando | Efecto |
|---|---|
| `"salir"`, `"terminar"`, `"adiós Jarvis"` | Finaliza la sesión |
| `"limpiar historial"`, `"limpiar"` | Borra el contexto de la conversación |
| `Ctrl+C` | Interrupción de emergencia |

### Opciones de línea de comandos

| Flag | Descripción |
|---|---|
| `--vad` | Modo VAD (manos libres) |
| `--ptt` | Modo push-to-talk |
| `--text` | Modo texto |
| `--writer` | Modo escritor (escribe en consola en lugar de hablar) |
| `--pdev` | Usar `pi` CLI (Gemini) como backend LLM en lugar de Ollama |
| `--model <nombre>` | Modelo Ollama (ej: `qwen3.5:9b`) |
| `--whisper <tamaño>` | Tamaño de Whisper (ej: `medium`, `small`) |
| `--quick "<texto>"` | Sintetizar texto y salir |
| `--list-devices` | Listar dispositivos de audio |
| `--test-mic` | Probar micrófono |

`--vad`, `--ptt` y `--text` son excluyentes entre sí; si no pasás ninguno, Jarvis muestra un menú interactivo. `--writer` y `--pdev` son ortogonales y se combinan con cualquier modo de entrada.

---

## 🏗️ Arquitectura

```
                    ┌──────────────────────────────────────────┐
                    │              JARVIS                       │
                    │                                          │
  🎤 Micrófono ────▶│  ┌──────────┐  ┌──────────┐  ┌───────┐  │
                    │  │   VAD    │─▶│  Whisper │─▶│Ollama │  │
                    │  │  Silero  │  │  STT GPU │  │ o pi  │  │
                    │  └──────────┘  └──────────┘  └───┬───┘  │
                    │                                  │       │
                    │                    ┌─────────────┴─────┐ │
                    │                    │                   │ │
                    │              ┌─────▼─────┐      ┌────▼───┐
                    │              │ Piper TTS │      │Writer  │
                    │              │  (voz)    │      │(texto) │
                    │              └─────┬─────┘      └────┬───┘
                    │                    │                 │
                    └────────────────────┼─────────────────┼───┘
                                         │                 │
                              ┌──────────▼──┐         ┌───▼────────┐
                              │ 🔊 Parlantes│         │ 📺 Consola │
                              └─────────────┘         └────────────┘
```

### Modos de salida

**Modo TTS (voz):**
- Streaming de Ollama → agrupación de oraciones (MIN_TTS_LENGTH=100 chars)
- Cola de síntesis → Worker de Piper → Cola de audio → Worker de reproducción
- Double buffering: sintetiza N+1 mientras reproduce N
- PyAudio persistente: elimina overhead de inicialización
- Resampling automático: 22050 Hz → 48000 Hz

**Modo escritor (texto):**
- Streaming de Ollama → tokens individuales
- Escritura directa a stdout sin buffering
- Latencia mínima: primer token en ~200ms

**Modo pdev (`--pdev`):**
- Subproceso `pi` con el historial aplanado a texto plano
- Sin streaming: la respuesta llega completa y recién ahí se sintetiza o se imprime

### Módulos

Todo el código de la aplicación vive en `jarvis.py`. Cinco módulos más un orquestador:

| Módulo | Tecnología | Función |
|---|---|---|
| `AudioManager` | PyAudio + NumPy | Captura y reproducción de audio |
| `SpeechRecognizer` | Faster-Whisper (CUDA) | Transcripción de voz a texto |
| `JarvisBrain` | Ollama API / `pi` CLI | Procesamiento de lenguaje y razonamiento |
| `VoiceSynthesizer` | Piper TTS | Síntesis de texto a voz |
| `VADManager` | Silero VAD | Detección de actividad de voz |
| `Jarvis` | — | Orquesta los módulos y corre los loops de cada modo |

El historial de conversación se mantiene acotado a 10 mensajes (5 intercambios) para no inflar el contexto del LLM.

---

## ⚙️ Configuración

Puedes ajustar Jarvis mediante variables de entorno:

```bash
# Modelo LLM de Ollama (default: qwen2.5-coder:7b)
export JARVIS_MODEL="qwen3.5:9b"

# Host de Ollama (default: http://localhost:11434)
export OLLAMA_HOST="http://localhost:11434"

# Solo para --pdev: modelo que usa el CLI pi (default: google/gemini-2.5-flash)
export PDEV_MODEL="google/gemini-2.5-flash"

# Solo para --pdev: archivo de system prompt (default: prompts/pdev_system.md)
export PDEV_SYSTEM_PROMPT="prompts/pdev_system.md"
```

El resto de los ajustes son constantes al inicio de `jarvis.py` (no hay archivo de configuración):

| Constante | Default | Descripción |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | Tamaño del modelo Whisper |
| `VAD_SPEECH_THRESHOLD` | `0.5` | Sensibilidad del VAD (0.0 - 1.0) |
| `VAD_MIN_SPEECH_CHUNKS` | `6` | ~180ms de habla continua para activar la grabación |
| `VAD_SILENCE_CHUNKS` | `50` | ~1.5s de silencio para cortar |
| `VAD_TIMEOUT_SECS` | `30` | Timeout total de grabación |
| `TTS_OUTPUT_RATE` | `48000` | Frecuencia de reproducción |

> `VAD_CHUNK_SIZE` (512 samples) **no es ajustable**: es el mínimo que exige Silero VAD a 16 kHz.

---

## 🧪 Pruebas

### Chequeos manuales

```bash
# Verificar que todo compila
source venv/bin/activate
python3 -c "from jarvis import Jarvis, VADManager; print('✅ OK')"

# Probar solo transcripción
python3 stt_test.py

# Probar solo síntesis de voz
python3 tts_test.py

# Benchmark rápido
python3 jarvis.py --quick "Prueba de sonido"
```

### Suite automatizada

El proyecto tiene tests unitarios en `tests/` configurados en `pytest.ini`. Todas las dependencias externas (PyAudio, Whisper, Ollama, Piper, Silero VAD) están mockeadas, así que la suite corre sin hardware, sin GPU y sin red — pero necesita que los paquetes estén instalados para poder importar `jarvis.py`.

`pytest` no viene en la lista de dependencias de runtime, así que hay que instalarlo aparte:

```bash
pip install pytest pytest-cov

pytest                                    # suite completa con reporte de cobertura
pytest tests/test_jarvis_brain.py         # un solo archivo
pytest tests/test_vad_manager.py -k silence   # un solo test
pytest -m unit                            # por marcador
```

Marcadores disponibles: `unit`, `integration`, `e2e`, `slow`, `streaming`.

El reporte HTML de cobertura queda en `tests/coverage_html/`.

---

## 🐛 Solución de problemas

### "paInvalidSampleRate" / Error de audio al hablar

Jarvis hace resampling automático de 22050 Hz → 48000 Hz. Si tu dispositivo usa otra tasa, edita `TTS_OUTPUT_RATE` en `jarvis.py`:

```python
TTS_OUTPUT_RATE = 44100  # o la tasa de tu dispositivo
```

### "No module named 'torch'" / VAD no disponible

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Whisper lento o usa mucha VRAM

Usa un modelo más pequeño:

```bash
python3 jarvis.py --whisper medium --vad
```

### Push-to-talk no funciona sin sudo

La librería `keyboard` requiere root en Linux. Alternativas:
- Usa `sudo python3 jarvis.py --ptt`
- O mejor: usa el modo VAD que no necesita permisos especiales

Si no hay permisos, Jarvis lo detecta solo y cae a VAD (o a modo texto si el VAD tampoco está disponible) en vez de fallar.

### "pi CLI no encontrado" en modo `--pdev`

Instalá el paquete correcto:

```bash
npm i -g @earendil-works/pi-coding-agent
```

### `bad interpreter` al usar `venv/bin/pip` o `source venv/bin/activate`

Pasa cuando el venv se creó en un directorio y después se movió: los scripts de adentro guardan la ruta absoluta del intérprete original en el shebang y quedan rotos. El binario de Python sí funciona, así que llamalo directo:

```bash
./venv/bin/python3 jarvis.py --vad
./venv/bin/python3 -m pip install <paquete>
```

`run.sh` ya hace esto por vos. La alternativa definitiva es recrear el venv en su ubicación final.

---

## 📦 Dependencias

```
faster-whisper    → Transcripción STT (GPU)
pyaudio           → Captura/reproducción de audio
requests          → API Ollama
piper-tts         → Síntesis de voz
numpy             → Procesamiento de audio
keyboard          → Push-to-talk (opcional, requiere sudo)
sounddevice       → Utilidades de audio
silero-vad        → Detección de actividad de voz
torch             → Motor de Silero VAD
torchaudio        → Audio para PyTorch
```

**Externas (no son paquetes de Python):**

```
ollama            → Servidor de LLM local (debe estar corriendo: ollama serve)
piper             → CLI de síntesis de voz, se invoca como subproceso
pi                → CLI de LLM, solo para --pdev (npm i -g @earendil-works/pi-coding-agent)
pytest            → Solo para correr la suite de tests
```

---

## 📄 Licencia

MIT

---

## 🙏 Agradecimientos

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — Transcripción ultrarrápida
- [Ollama](https://ollama.com/) — Gestión de modelos locales
- [Piper TTS](https://github.com/rhasspy/piper) — Síntesis de voz open-source
- [Silero VAD](https://github.com/snakers4/silero-vad) — Detección de actividad de voz
- [Qwen](https://github.com/QwenLM/Qwen) — Modelo de lenguaje
