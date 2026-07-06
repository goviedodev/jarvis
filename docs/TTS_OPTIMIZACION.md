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
