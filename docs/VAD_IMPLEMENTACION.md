# 🎙️ VAD — Voice Activity Detection en Jarvis

## ¿Qué es VAD y por qué lo necesitamos?

**VAD** (Voice Activity Detection) es la tecnología que permite a una computadora
saber cuándo una persona está hablando y cuándo está en silencio. Es el equivalente
digital de la capacidad humana de saber si alguien nos está hablando o no.

En Jarvis tenemos **tres formas** de decirle al sistema "ahora habla":

| Método | Cómo funciona | Lo bueno | Lo malo |
|---|---|---|---|
| **Push-to-talk** | Presionas una tecla mientras hablas | Control total, sin falsos positivos | Ocupas las manos, no es natural |
| **VAD** | Jarvis escucha siempre y detecta cuándo hablas | Manos libres, natural | Puede fallar en ambientes ruidosos |
| **Texto** | Escribes en la terminal | 100% preciso | Lento, incómodo |

El VAD es el **santo grial** de los asistentes de voz: la capacidad de tener una
conversación natural con una máquina, sin tener que apretar botones ni decir
palabras de activación.

---

## Cómo funciona Silero VAD por dentro

Silero VAD es un modelo de **deep learning** (red neuronal) entrenado para
distinguir entre voz humana y cualquier otro sonido. Fue desarrollado por el equipo
de Silero y es uno de los sistemas de VAD más precisos y rápidos que existen.

### La arquitectura

Silero VAD usa una arquitectura llamada **CRNN** (Convolutional Recurrent Neural Network):

```
Audio crudo (16kHz)
        │
        ▼
┌──────────────────────┐
│  Capas convolucionales│  ← Detectan patrones locales en el audio
│  (1D Conv + BatchNorm)│    (tonos, formantes, ritmo)
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Capas recurrentes    │  ← Analizan la evolución temporal
│  (GRU / LSTM)         │    (el habla tiene estructura en el tiempo)
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Capa fully connected │  ← Clasificación final
│  (Sigmoid)            │    0.0 = silencio, 1.0 = voz
└──────────────────────┘
        │
        ▼
   [0.87] ← probabilidad de voz
```

El modelo procesa fragmentos de audio de **32ms** (512 samples a 16kHz) y produce
un número entre 0 y 1 que representa la probabilidad de que haya voz en ese fragmento.

### ¿Qué lo hace especial?

Silero VAD es excepcional porque:

1. **Es tiny** — el modelo pesa ~1.7 MB, cabe en cualquier lado
2. **Es rápido** — procesa un fragmento en <1ms en CPU
3. **Es preciso** — distingue voz de ruido ambiente, golpes, respiraciones, toses
4. **Es resiliente** — funciona en condiciones de ruido de hasta 10dB SNR
5. **Es abierto** — licencia MIT, puedes usarlo en cualquier proyecto

---

## La implementación en Jarvis

### Módulo 5: VADManager

Creamos una clase `VADManager` en `jarvis.py` (Módulo 5) que encapsula toda la
lógica de detección de voz. Su diseño es simple por diseño:

```python
class VADManager:
    def __init__(self):
        self.model = None        # El modelo Silero VAD (se carga bajo demanda)
        self.running = False     # Flag para detener loops limpiamente

    def load(self):              # Carga el modelo (una vez)
    def is_speech(self, chunk):  # ¿Este fragmento tiene voz? (True/False)
    def listen_for_speech(stream):  # Bucle completo: espera → graba → silencio
```

### El pipeline de detección

El proceso tiene 3 fases:

#### Fase 1: Escucha pasiva (esperando)

```
Tiempo:  ────·────·────·────·────·────·────·────·────·────·───
Audio:   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
VAD:     0.03  0.02  0.01  0.04  0.02  0.03  0.01  0.02  0.04
         ↓  Umbral: 0.5
         Todos por debajo → no hay voz, seguir escuchando
```

Cada 32ms, tomamos un fragmento de audio del micrófono y lo evaluamos.
Si la probabilidad de voz está por debajo de 0.5 (configurable), no hay voz.

#### Fase 2: Disparo (speech detected)

```
Tiempo:  ────·────·────·────·────·────·────·────·────·────·───
Audio:   ░░░░████████████████████████████░░░░░░░░░░░░░░░░░░░░░░
VAD:     0.02  0.87  0.92  0.95  0.88  0.91  0.03  0.02  0.04
               ↑     ↑     ↑     ↑     ↑
               ¡Voz detectada! Acumulando en pre_buffer...
               
               ┌──────────────────────────────┐
               │ Cuando llevamos 6 chunks     │
               │ consecutivos con voz (~192ms)│
               │ → PASAMOS A FASE 3           │
               └──────────────────────────────┘
```

No queremos reaccionar a un solo fragmento con voz (podría ser un golpe, una tos,
un portazo). Esperamos a tener **~6 fragmentos consecutivos** (~192ms) de habla
para estar seguros de que la persona realmente está hablando.

**Lección aprendida**: El `pre_buffer` es crítico. Originalmente descartábamos estos
6 fragmentos y empezábamos a grabar desde cero. Esto significaba perder ~200ms del
inicio de cada frase. Whisper puede tolerarlo, pero no es ideal. Ahora guardamos
esos fragmentos en un `pre_buffer` y los incluimos en la grabación final.

#### Fase 3: Grabación activa

```
Tiempo:  ────·────·────·────·────·────·────·────·────·────·────·────·───
Audio:   ░░░░██████████████████████████████████████████████░░░░░░░░░░░░
         ^pre_buffer^     FRASES LARGAS...                   ^1.5s      ^
         |←  ~192ms  →|←        Grabando todo        →|   silencio   |
                                                          → CORTAR
```

Una vez activados, grabamos **todo** el audio que entre por el micrófono hasta que:

- Se detecten **~50 chunks consecutivos de silencio** (~1.6 segundos)
- O pasen **30 segundos** (timeout de seguridad)

Cuando se cumple alguna de estas condiciones, detenemos la grabación y enviamos
el audio a Whisper para transcribirlo.

### Por qué 512 samples por chunk

```
Condición de Silero VAD:  sr / samples > 31.25 → error
                          samples > sr / 31.25
                          samples > 16000 / 31.25
                          samples > 512
```

**Lección aprendida**: Pusimos `VAD_CHUNK_SIZE = 480` (30ms), pero el modelo
Silero VAD **exige** un mínimo de 512 samples (32ms). Esto nos costó un error
críptico:

```
builtins.ValueError: Input audio chunk is too short
```

La lección: **siempre verificar las constantes mínimas de las librerías que usas**.
Silero VAD documenta que soporta chunks de 512, 768, 1024, 1536 samples (múltiplos
de 256) a 16kHz. Usar 480 viola ese requisito.

---

## Aprendizajes clave del desarrollo

### 1. Los modelos tienen requisitos específicos de entrada

Cada modelo de IA espera los datos en un formato muy particular:

| Silero VAD espera | Lo que nosotros enviamos |
|---|---|
| Float32 normalizado [-1, 1] | Int16 → convertir dividiendo por 32768 |
| Mono (1 canal) | ✅ Mono |
| 16kHz sample rate | ✅ 16kHz |
| Mínimo 512 samples | ❌ Originalmente 480 |

### 2. No abrir/cerrar streams por cada utterance

Originalmente, `vad_loop()` abría un stream de PyAudio, ejecutaba
`listen_for_speech()`, cerraba el stream, y lo volvía a abrir en la siguiente
iteración.

**Problema**: Abrir un stream de audio tiene latencia (~50-200ms). Entre
iteraciones se perdía audio y había clics audibles.

**Solución**: Abrir el stream una sola vez al entrar a `vad_loop()` y cerrarlo
en el `finally` al salir. El stream vive durante toda la sesión VAD.

```python
# ❌ Antes: abría/cerraba en cada utterance
while self.running:
    stream = self.audio.p.open(...)
    frames = self.vad.listen_for_speech(stream)
    stream.stop_stream()
    stream.close()
    # procesar frames...

# ✅ Ahora: stream persistente durante toda la sesión
stream = self.audio.p.open(...)  # ← Una vez
try:
    while self.running:
        frames = self.vad.listen_for_speech(stream)
        # procesar frames...
finally:
    stream.stop_stream()  # ← Al final
    stream.close()
```

### 3. El flag `running` debe estar en todos lados

Cuando el usuario presiona Ctrl+C, queremos que todo se detenga inmediatamente.
Pero si `listen_for_speech()` está bloqueada en un `while self.running` y
`self.running` vive en el objeto equivocado, nunca se entera.

**Lección**: Tanto `Jarvis` como `VADManager` tienen su propio `self.running`.
Cuando uno se detiene, el otro también debe hacerlo:

```python
# En vad_loop():
except KeyboardInterrupt:
    self.running = False       # Detiene el while de Jarvis
    break
finally:
    self.vad.running = False   # Detiene el while de VADManager
    stream.stop_stream()
    stream.close()
```

### 4. No perder los chunks de activación

Cuando detectamos que alguien empezó a hablar, ya pasaron ~6 chunks (~192ms).
Originalmente descartábamos ese audio y empezábamos a grabar desde cero.

**Solución**: Buffer acumulativo (`pre_buffer`):

```python
pre_buffer = []

# Mientras esperamos a que se active:
if self.is_speech(data):
    speech_chunks += 1
    pre_buffer.append(data)      # ← Guardamos
    if speech_chunks >= VAD_MIN_SPEECH_CHUNKS:
        break
else:
    speech_chunks = 0
    pre_buffer.clear()           # ← Si deja de hablar, descartamos

# Cuando se activa, empezamos con el buffer:
frames = list(pre_buffer)        # ← Incluimos lo que ya teníamos
```

### 5. El umbral de sensibilidad es clave

`VAD_SPEECH_THRESHOLD = 0.5` es el valor por defecto de Silero. Pero en la
práctica:

- **0.5** — Balanceado. Funciona bien en entornos silenciosos.
- **0.6 - 0.7** — Menos sensible. Mejor para ambientes con ruido de fondo.
- **0.3 - 0.4** — Más sensible. Captura voz más baja pero puede tener falsos positivos.

El valor óptimo depende de tu micrófono y entorno. Recomendación: empezar con
0.5 y ajustar hacia arriba si hay falsos positivos, hacia abajo si no detecta
voz suave.

---

## Configuraciones de VAD en Jarvis

Todas las constantes están al inicio de `jarvis.py`:

```python
# ─── VAD (Voice Activity Detection) ──────────────────────────────────────────
# Silero VAD trabaja con chunks de 32ms (512 samples) a 16000 Hz

VAD_CHUNK_SIZE = 512       # 32ms a 16000 Hz (mínimo: 512 para Silero)
VAD_SPEECH_THRESHOLD = 0.5 # Probabilidad mínima para considerar voz (0.0-1.0)
VAD_MIN_SPEECH_CHUNKS = 6  # ~192ms de habla continua para activar grabación
VAD_SILENCE_CHUNKS = 50    # ~1.6s de silencio para detener grabación
VAD_TIMEOUT_SECS = 30      # Timeout total de grabación en segundos
```

### Guía de ajuste

| Situación | Qué ajustar |
|---|---|
| Falsos positivos (Jarvis se activa solo) | Subir `VAD_SPEECH_THRESHOLD` a 0.6-0.7 |
| No detecta voz suave | Bajar `VAD_SPEECH_THRESHOLD` a 0.3-0.4 |
| Corta las frases antes de tiempo | Subir `VAD_SILENCE_CHUNKS` a 60-70 |
| Tarda en cortar después de hablar | Bajar `VAD_SILENCE_CHUNKS` a 30-40 |
| Demasiado lento en activarse | Bajar `VAD_MIN_SPEECH_CHUNKS` a 3-4 |

---

## Comparativa con otros métodos de activación

### VAD vs Push-to-talk

```
Situación: estás comiendo y quieres preguntar algo

Push-to-talk: ❌ Necesitas soltar los cubiertos y presionar espacio
VAD:          ✅ Solo hablas "Jarvis, ¿qué clima hace?"
```

### VAD vs Wake Word (Oye Jarvis)

```
                    VAD puro          Wake Word + VAD
┌──────────────────────────────────────────────────────────┐
│ Activación    │  Cualquier habla  │  Solo después de     │
│               │                   │  "Oye Jarvis"        │
├──────────────────────────────────────────────────────────┤
│ Privacidad    │  Menor (siempre   │  Mayor (solo         │
│               │  escucha)         │  escucha palabra)    │
├──────────────────────────────────────────────────────────┤
│ Naturalidad   │  Máxima           │  Media (hay que      │
│               │                   │  decir el nombre)    │
├──────────────────────────────────────────────────────────┤
│ Falsos +      │  Puede activarse  │  Muy baja            │
│               │  con ruido/TV     │                      │
└──────────────────────────────────────────────────────────┘
```

El **VAD puro** es más natural (hablas y ya), pero consume más recursos porque
el modelo siempre está activo. La **Wake Word** primero detecta una palabra
específica y solo entonces activa el VAD completo.

---

## Flujo completo de datos en modo VAD

```
┌────────────┐    512 bytes      ┌────────────┐   probabilidad    ┌──────────┐
│ Micrófono  │ ────────────────▶ │  Silero    │ ────────────────▶ │  Lógica  │
│ (PyAudio)  │   PCM int16       │  VAD       │   0.0 - 1.0       │  Jarvis  │
└────────────┘    16kHz          └────────────┘                   └──────────┘
                                                                    │
                                                                    │ si prob > 0.5
                                                                    │ por 6 chunks
                                                                    ▼
                                                           ┌──────────────────┐
                                                           │  Acumular audio  │
                                                           │  en buffer       │
                                                           └──────────────────┘
                                                                    │
                                                                    │ silencio por
                                                                    │ 50 chunks
                                                                    ▼
                                                           ┌──────────────────┐
                                                           │  Guardar WAV     │
                                                           │  temporal        │
                                                           └──────────────────┘
                                                                    │
                                                                    ▼
                                                           ┌──────────────────┐
                                                           │  Whisper (STT)   │
                                                           │  → texto         │
                                                           └──────────────────┘
                                                                    │
                                                                    ▼
                                                           ┌──────────────────┐
                                                           │  Ollama (LLM)    │
                                                           │  → respuesta     │
                                                           └──────────────────┘
                                                                    │
                                                                    ▼
                                                           ┌──────────────────┐
                                                           │  Piper (TTS)     │
                                                           │  → audio 48kHz   │
                                                           └──────────────────┘
                                                                    │
                                                                    ▼
                                                           ┌──────────────────┐
                                                           │  Parlantes       │
                                                           │  🔊              │
                                                           └──────────────────┘
```

---

## Referencias

- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [Silero VAD en PyPI](https://pypi.org/project/silero-vad/)
- [Documentación de PyTorch Hub - Silero VAD](https://pytorch.org/hub/snakers4_silero-vad_vad/)
- [Comparativa: Silero vs WebRTC vs Cobra](https://picovoice.ai/blog/best-voice-activity-detection-vad/)
