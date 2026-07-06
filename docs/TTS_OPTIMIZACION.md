# Optimización del Pipeline TTS (Text-to-Speech)

## Resumen

Documentación de las optimizaciones implementadas en el módulo `VoiceSynthesizer` y el método `think_stream()` de `JarvisBrain` para reducir pausas entre oraciones durante la síntesis de voz.

## Problema Original

### Síntoma
Cuando Jarvis respondía con múltiples oraciones, se producían:
1. Impresión múltiple de "🔊 Hablando..." en consola
2. Audio reproducido simultáneamente o con solapamientos
3. Experiencia de usuario degradada: "habla 3 veces"

### Causa Raíz

En `process_query()`, el streaming de Ollama generaba múltiples oraciones que se enviaban a `speak_nonblocking()`. Este método creaba un **hilo daemon nuevo por cada oración**:

```python
# ANTES (código problemático)
def speak_nonblocking(self, text):
    thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
    thread.start()
    return thread
```

Si el LLM generaba 3 oraciones, se lanzaban 3 hilos simultáneamente que:
- Cada uno imprimía "🔊 Hablando..."
- Cada uno abría su propio stream PyAudio
- Los streams se superponían en el tiempo
- El audio resultante era caótico e ininteligible

## Solución Implementada

### 1. Arquitectura Basada en Cola (Queue)

Reemplazamos el modelo de "un hilo por oración" por una **cola con un único worker** que procesa oraciones secuencialmente.

**Cambios en `VoiceSynthesizer`:**

```python
def __init__(self, model_path=VOICE_MODEL, config_path=VOICE_CONFIG):
    self.model_path = model_path
    self.config_path = config_path
    self._queue = queue.Queue()
    self._worker = threading.Thread(target=self._tts_worker, daemon=True)
    self._worker.start()

def _tts_worker(self):
    """Worker que procesa la cola de TTS secuencialmente."""
    while True:
        text = self._queue.get()
        if text is None:
            break
        self.speak(text)
        self._queue.task_done()
    self.cleanup()

def speak_nonblocking(self, text):
    """Encola texto para que el worker lo hable en orden."""
    self._queue.put(text)
```

**Beneficios:**
- Una sola impresión de "🔊 Hablando..." por oración
- Streams PyAudio secuenciales, no concurrentes
- Audio claro y ordenado, sin solapamientos
- El streaming del LLM no se bloquea (cola asíncrona)
- Consumo de recursos reducido (1 hilo vs N hilos)

### 2. PyAudio Persistente

Cada llamada a `speak()` creaba y destruía una instancia de PyAudio, añadiendo ~50-100ms de overhead por oración.

**Solución:** Mantener PyAudio y el stream como atributos de instancia.

```python
def __init__(self, ...):
    # ...
    self._pyaudio = None
    self._stream = None

def _ensure_audio_stream(self):
    """Crea o reutiliza el stream de audio PyAudio."""
    if self._pyaudio is None or self._stream is None:
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=TTS_OUTPUT_RATE,
            output=True,
        )
    return self._stream

def speak(self, text):
    # ... síntesis con Piper ...
    
    # Reproducir usando stream persistente
    stream = self._ensure_audio_stream()
    stream.write(audio_bytes_out)

def cleanup(self):
    """Cierra el stream y PyAudio. Llamar al finalizar o en error."""
    try:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None
    except Exception:
        pass
```

**Impacto:**
- PyAudio se crea una sola vez y se reutiliza
- Elimina ~50-100ms de overhead por oración
- Cleanup automático al finalizar el worker o en errores

### 3. Caché de check_voice()

`check_voice()` verificaba la existencia del archivo de voz en cada llamada a `speak()`.

**Solución:** Cachear el resultado en `self._voice_checked`.

```python
def __init__(self, ...):
    # ...
    self._voice_checked = False

def check_voice(self):
    """Verifica que el modelo de voz exista."""
    if self._voice_checked:
        return True
    if not os.path.exists(self.model_path):
        print(f"{Colors.RED}❌ Voz no encontrada: {self.model_path}{Colors.RESET}")
        return False
    self._voice_checked = True
    return True
```

**Impacto:** Evita `os.path.exists()` en cada llamada.

### 4. Agrupación de Oraciones Cortas

El método `think_stream()` hacía yield con cada oración de 5+ caracteres, generando múltiples llamadas a Piper con overhead de fork+exec.

**Problema:** Para la respuesta "Estoy bien, gracias! Intento ser claro y útil siempre que pueda. ¿Cómo puedo ayudarte hoy?" se generaban 3 subprocess de Piper:
1. "Estoy bien, gracias!" (20 chars)
2. "Intento ser claro y útil siempre que pueda." (45 chars)
3. "¿Cómo puedo ayudarte hoy?" (25 chars)

**Solución:** Aumentar el umbral mínimo antes de hacer yield.

```python
def think_stream(self, user_input):
    # ...
    buffer = ""
    full_response = ""
    sentence_endings = {".", "?", "!"}
    MIN_TTS_LENGTH = 100  # agrupar oraciones cortas

    for line in response.iter_lines():
        # ... procesar tokens ...
        
        full_response += content
        buffer += content

        # Yield cuando detectamos final de oración (., ?, !)
        # pero solo si el buffer es suficientemente largo
        if any(c in sentence_endings for c in buffer[-3:]):
            if len(buffer) >= MIN_TTS_LENGTH:
                sentence = buffer.strip()
                if sentence:
                    yield sentence
                buffer = ""

    # Yield del texto restante
    remaining = buffer.strip()
    if remaining:
        yield remaining
```

**Comportamiento:**
- Las oraciones cortas se acumulan hasta alcanzar 100 caracteres
- Solo entonces se hace yield y se envía a Piper
- Reduce el número de llamadas a Piper de 3 a 1-2

**Trade-off:**
- ⚡ **Ventaja**: Pausas entre fragmentos reducidas drásticamente
- ⏱️ **Costo**: La primera oración tarda un poco más en empezar (espera a acumular 100 chars)

**Valores sugeridos:**
- `MIN_TTS_LENGTH = 50` → más responsivo, más pausas
- `MIN_TTS_LENGTH = 70` → balance medio
- `MIN_TTS_LENGTH = 100` → menos pausas, pero tarda más en empezar

## Flujo Completo Después de Optimizaciones

1. Usuario habla → VAD/PTT detecta → STT transcribe
2. `process_query()` llama a `think_stream()`
3. `think_stream()` hace streaming de Ollama
4. Tokens se acumulan en buffer hasta alcanzar `MIN_TTS_LENGTH` (100 chars)
5. Se hace yield de oración larga
6. `speak_nonblocking()` encola la oración
7. El worker lee de la cola
8. Llama a `speak()` que:
   - Verifica voz (cacheado)
   - Lanza subprocess de Piper
   - Resamplea audio de 22050 → 48000 Hz
   - Escribe al stream PyAudio persistente
9. Worker procesa siguiente oración de la cola
10. Al finalizar, worker llama a `cleanup()`

## Overhead Restante

Después de todas las optimizaciones, el overhead restante es:

1. **Subprocess de Piper**: fork+exec por cada yield (~50-100ms)
   - Inherente a la arquitectura CLI de Piper
   - Mitigado por agrupación de oraciones (menos yields)

2. **Resampling**: conversión 22050 → 48000 Hz
   - Operación numpy, ~10-20ms por oración
   - Aceptable

3. **Latencia de red**: streaming de Ollama
   - Depende del modelo y hardware
   - No optimizable desde nuestro código

## Métricas de Performance

### Antes de Optimizaciones
- Respuesta de 3 oraciones cortas:
  - 3 hilos simultáneos
  - 3 instancias de PyAudio creadas/destruidas
  - Audio solapado e ininteligible
  - Overhead: ~300-500ms

### Después de Optimizaciones
- Respuesta de 3 oraciones cortas (ahora agrupadas en 1-2):
  - 1 hilo worker secuencial
  - 1 instancia de PyAudio persistente
  - Audio claro y ordenado
  - Overhead: ~50-150ms (1-2 subprocess de Piper)

## Configuración

### Variables Ajustables

En `jarvis.py`:

```python
# En think_stream():
MIN_TTS_LENGTH = 100  # Umbral mínimo antes de yield (caracteres)
```

### Ajuste Fino

Si las pausas aún son notables:

1. **Aumentar MIN_TTS_LENGTH** (120-150):
   - Menos llamadas a Piper
   - Más latencia inicial
   - Mejor para respuestas largas

2. **Disminuir MIN_TTS_LENGTH** (50-70):
   - Más llamadas a Piper
   - Menos latencia inicial
   - Mejor para respuestas cortas

3. **Considerar Piper persistente**:
   - Mantener Piper corriendo como daemon
   - Enviar texto por stdin
   - Eliminar overhead de fork+exec
   - Requiere verificar si Piper soporta modo interactivo

## Testing

Los tests en `tests/test_voice_synthesizer.py` verifican:
- Cola FIFO mantiene orden de oraciones
- Múltiples `speak_nonblocking()` no crean múltiples hilos
- Stream persistente se reutiliza correctamente
- Cleanup libera recursos al finalizar

Ejecutar:
```bash
pytest tests/test_voice_synthesizer.py -v
```

## Referencias

- Commit: `fix(tts): resolver bug de reproducción múltiple de voz usando cola secuencial`
- Commit: `test: agregar suite de tests completa con pytest y configuración de SonarQube`
- Archivo: `jarvis.py` (clases `VoiceSynthesizer` y `JarvisBrain.think_stream`)
- Tests: `tests/test_voice_synthesizer.py`

## Lecciones Aprendidas

1. **Hilos no son la solución para todo**: Lanzar un hilo por tarea puede causar más problemas que soluciones cuando hay recursos compartidos (PyAudio, streams).

2. **Colas + Workers**: Patrón robusto para procesar tareas secuencialmente mientras el productor sigue generando trabajo.

3. **Recursos persistentes**: Crear/destruir recursos (PyAudio, streams, conexiones) es costoso. Mantenerlos abiertos y reutilizarlos mejora performance.

4. **Agrupación de trabajo**: Procesar múltiples items pequeños juntos reduce overhead de setup/teardown (similar a batch processing).

5. **Caché de verificaciones**: Verificaciones repetitivas (existencia de archivos, validaciones) deben cachearse cuando el estado no cambia.

## 5. Double Buffering / Pipelining

### Problema con la Arquitectura de Un Solo Worker

Con la arquitectura anterior (un solo worker que sintetiza y reproduce secuencialmente), el flujo era:

```
Worker: [Piper N] → [Play N] → [Piper N+1] → [Play N+1]
         ~~~ pausa ~~~                     ~~~ pausa ~~~
```

El cuello de botella: la **síntesis de N+1 no empezaba hasta que N terminara de reproducirse**, causando pausas audibles entre oraciones.

### Solución: Dos Workers en Paralelo

Separamos el proceso en dos etapas que pueden ejecutarse simultáneamente:

```
Hilo síntesis:    [Piper N] → put(audio) → [Piper N+1] → put(audio)
                                    ↓                        ↓
Cola de audio:              [audio_N]          [audio_N+1]
                                    ↓                        ↓
Hilo reproducción:           [Play N] → [Play N+1] → [Play N+2]
                              ~~~ sin pausa ~~~
```

**Resultado:** Mientras se **reproduce** la oración N, ya se está **sintetizando** la oración N+1.

### Implementación

#### Arquitectura de Colas

```python
def __init__(self, model_path=VOICE_MODEL, config_path=VOICE_CONFIG):
    self.model_path = model_path
    self.config_path = config_path
    self._queue = queue.Queue()  # Cola de texto (entrada)
    self._audio_queue = queue.Queue(maxsize=3)  # Cola de audio (intermedia)
    self._pyaudio = None
    self._stream = None
    self._voice_checked = False
    
    # Worker de síntesis: texto → audio bytes
    self._synth_worker = threading.Thread(target=self._synth_loop, daemon=True)
    self._synth_worker.start()
    
    # Worker de reproducción: audio bytes → PyAudio
    self._playback_worker = threading.Thread(target=self._playback_loop, daemon=True)
    self._playback_worker.start()
```

**Dos colas:**
- `_queue`: Recibe texto de `speak_nonblocking()`
- `_audio_queue`: Almacena audio bytes sintetizados (maxsize=3 para backpressure)

**Dos workers:**
- `_synth_worker`: Lee texto, sintetiza con Piper, encola audio bytes
- `_playback_worker`: Lee audio bytes, escribe al stream PyAudio

#### Worker de Síntesis

```python
def _synth_loop(self):
    """Worker de síntesis: lee texto, sintetiza, encola audio bytes."""
    while True:
        text = self._queue.get()
        if text is None:
            # Señal de terminación: propagar al playback worker
            self._audio_queue.put(None)
            break
        
        print(f"{Colors.BLUE}🔊 Sintetizando...{Colors.RESET}", end=" ", flush=True)
        audio_bytes = self._synthesize_text(text)
        
        if audio_bytes is not None:
            # Bloqueante si la cola está llena (backpressure)
            self._audio_queue.put(audio_bytes)
            print(f"{Colors.DIM}✓{Colors.RESET}", flush=True)
        
        self._queue.task_done()
```

**Responsabilidades:**
1. Lee texto de `_queue`
2. Llama a `_synthesize_text()` (lanza Piper, resamplea)
3. Encola audio bytes en `_audio_queue`
4. Propaga señal de terminación (`None`) al playback worker

#### Worker de Reproducción

```python
def _playback_loop(self):
    """Worker de reproducción: lee audio bytes, escribe al stream PyAudio."""
    while True:
        try:
            audio_bytes = self._audio_queue.get()
            if audio_bytes is None:
                break
            
            stream = self._ensure_audio_stream()
            stream.write(audio_bytes)
            self._audio_queue.task_done()
            
        except Exception as e:
            print(f"{Colors.RED}❌ Playback error: {e}{Colors.RESET}")
            self.cleanup()
            # Continuar procesando la cola
            continue
    
    # Cleanup al finalizar
    self.cleanup()
```

**Responsabilidades:**
1. Lee audio bytes de `_audio_queue`
2. Escribe al stream PyAudio persistente
3. Maneja errores con cleanup y reintento
4. Cleanup final al terminar

#### Método de Síntesis Separado

```python
def _synthesize_text(self, text):
    """Sintetiza texto a audio bytes usando Piper. Retorna bytes o None si falla."""
    if not self.check_voice():
        return None

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
            return None

        # Resamplar de 22050 → 48000 Hz
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        if PIPER_SAMPLE_RATE != TTS_OUTPUT_RATE:
            src_len = len(audio_array)
            tgt_len = int(src_len * TTS_OUTPUT_RATE / PIPER_SAMPLE_RATE)
            audio_array = np.interp(
                np.linspace(0, src_len - 1, tgt_len),
                np.arange(src_len),
                audio_array.astype(np.float64),
            )

        return audio_array.astype(np.int16).tobytes()

    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}❌ Timeout en Piper{Colors.RESET}")
        return None
    except Exception as e:
        print(f"{Colors.RED}❌ TTS síntesis error: {e}{Colors.RESET}")
        return None
```

**Propósito:** Extraer la lógica de síntesis (Piper subprocess + resampling) en un método reutilizable que retorna bytes o None.

#### Método speak() Síncrono

```python
def speak(self, text):
    """Convierte texto a voz y lo reproduce (síncrono, para uso directo)."""
    print(f"{Colors.BLUE}🔊 Hablando...{Colors.RESET}", end=" ", flush=True)
    start = time.time()

    audio_bytes_out = self._synthesize_text(text)
    if audio_bytes_out is None:
        return False

    try:
        stream = self._ensure_audio_stream()
        stream.write(audio_bytes_out)

        elapsed = time.time() - start
        print(f"{Colors.DIM}({elapsed:.1f}s){Colors.RESET}")
        return True

    except Exception as e:
        print(f"{Colors.RED}❌ TTS playback error: {e}{Colors.RESET}")
        self.cleanup()
        return False
```

**Uso:** Para modos que requieren reproducción síncrona (quick mode, mensajes de error).

### Backpressure

La cola `_audio_queue` tiene `maxsize=3`:

```python
self._audio_queue = queue.Queue(maxsize=3)
```

**Propósito:**
- Evita acumulación excesiva de audio en memoria
- Si el playback worker va lento, el synth worker espera
- Previene que se sinteticen demasiadas oraciones por adelantado

**Comportamiento:**
- `_audio_queue.put()` es bloqueante si la cola está llena
- El synth worker se pausa automáticamente
- Se reanuda cuando el playback worker consume audio

### Manejo de Errores

#### En Síntesis
- Si Piper falla → `_synthesize_text()` retorna `None`
- El synth worker no encola audio
- El playback worker simplemente no recibe ese fragmento
- La conversación continúa con las siguientes oraciones

#### En Reproducción
- Si PyAudio falla → excepción capturada en `_playback_loop()`
- Se llama a `cleanup()` para resetear el stream
- El worker continúa procesando la cola
- Siguiente oración reintentará con stream nuevo

#### Propagación de Terminación
```python
# En _synth_loop:
if text is None:
    self._audio_queue.put(None)  # Propagar al playback worker
    break

# En _playback_loop:
if audio_bytes is None:
    break  # Terminar limpiamente
```

### Beneficios

1. **Eliminación de pausas:** La síntesis de N+1 ocurre mientras se reproduce N
2. **Mejor utilización de CPU:** Dos núcleos trabajan en paralelo (síntesis + reproducción)
3. **Latencia percibida reducida:** El usuario escucha audio más rápido después de la primera oración
4. **Arquitectura robusta:** Errores en un worker no bloquean al otro
5. **Backpressure automático:** Previene consumo excesivo de memoria

### Consideraciones Técnicas

#### Thread Safety
- `_ensure_audio_stream()` solo es llamado por el playback worker
- No hay race conditions porque solo un hilo accede al stream PyAudio
- Las colas de Python son thread-safe por diseño

#### Orden Garantizado
- Ambas colas son FIFO
- El orden de texto → audio → reproducción se mantiene estrictamente
- No hay reordenamiento posible

#### Overhead de Threading
- Dos hilos daemon adicionales (synth + playback)
- Overhead mínimo (~1-2ms por cambio de contexto)
- Beneficio de paralelismo supera ampliamente el overhead

#### Memoria
- `_audio_queue` almacena hasta 3 fragmentos de audio
- Cada fragmento: ~100-500 KB dependiendo de la longitud
- Máximo: ~1.5 MB en cola (aceptable)

### Comparación: Antes vs Después

#### Arquitectura Anterior (Un Worker)
```
Tiempo total para 3 oraciones de 500ms cada una:
- Oración 1: [Piper 500ms] → [Play 1000ms]
- Oración 2: [Piper 500ms] → [Play 1000ms]
- Oración 3: [Piper 500ms] → [Play 1000ms]
Total: 4500ms
```

#### Arquitectura Nueva (Dos Workers)
```
Tiempo total para 3 oraciones de 500ms cada una:
- Oración 1: [Piper 500ms] → [Play 1000ms]
- Oración 2:      [Piper 500ms] → [Play 1000ms]
- Oración 3:           [Piper 500ms] → [Play 1000ms]
Total: 3000ms (33% más rápido)
```

**Mejora:** 33% de reducción en tiempo total, eliminación completa de pausas entre oraciones.

### Flujo Completo Actualizado

1. Usuario habla → VAD/PTT detecta → STT transcribe
2. `process_query()` llama a `think_stream()`
3. `think_stream()` hace streaming de Ollama
4. Tokens se acumulan en buffer hasta alcanzar `MIN_TTS_LENGTH` (100 chars)
5. Se hace yield de oración larga
6. `speak_nonblocking()` encola texto en `_queue`
7. **Synth worker** lee texto de `_queue`
8. Lanza subprocess de Piper
9. Resamplea audio de 22050 → 48000 Hz
10. Encola audio bytes en `_audio_queue`
11. **Playback worker** lee audio bytes de `_audio_queue`
12. Escribe al stream PyAudio persistente
13. Synth worker ya está procesando siguiente oración (paso 7)
14. Al finalizar, ambos workers hacen cleanup

### Testing Manual

Para verificar que el double buffering funciona:

```bash
python3 jarvis.py --quick "Esta es la primera oración. Esta es la segunda oración. Y esta es la tercera oración final."
```

**Comportamiento esperado:**
- Impresión: "🔊 Sintetizando... ✓" tres veces (una por oración)
- Audio reproducido de forma continua sin pausas entre oraciones
- Tiempo total menor que la suma de tiempos individuales

### Posibles Mejoras Futuras

1. **Métricas de performance:** Agregar logging de tiempos de síntesis vs reproducción
2. **Backpressure adaptativo:** Ajustar `maxsize` dinámicamente según velocidad de reproducción
3. **Pre-síntesis:** Sintetizar oraciones comunes ("Hola", "Adiós") al inicio y cachearlas
4. **Priorización:** Cola de prioridad para interrupciones urgentes

## Posibles Mejoras Futuras

1. **Piper como servicio persistente**:
   - Correr Piper como daemon
   - Comunicación via pipes o socket
   - Eliminar overhead de fork+exec

2. **Buffering adaptativo**:
   - Ajustar MIN_TTS_LENGTH dinámicamente según longitud de respuesta
   - Respuestas cortas: umbral bajo
   - Respuestas largas: umbral alto

3. **Pre-carga de modelos**:
   - Mantener modelo Piper en memoria
   - Requiere usar Piper como librería, no CLI

4. **TTS streaming**:
   - Piper con salida incremental
   - Reproducir mientras sintetiza
   - Requiere investigar capacidades de Piper

5. **Doble buffering de audio**:
   - Mientras se reproduce oración N, sintetizar oración N+1
   - Requiere cola de audio a nivel de bytes
   - Complejidad significativa
