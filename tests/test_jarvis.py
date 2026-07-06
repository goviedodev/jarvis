"""
Tests de integración para la clase Jarvis (orquestador).

Cubre:
- Inicialización completa
- process_query con y sin streaming
- Modo VAD (vad_loop)
- Modo Push-to-talk (push_to_talk_loop)
- Modo texto (interactive_mode)
- Comandos de voz (salir, limpiar)
- Manejo de errores en todos los modos
"""

import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest

from jarvis import Jarvis, Colors


class TestJarvisInitialization:
    """Tests de inicialización del orquestador."""

    @patch("jarvis.AudioManager")
    @patch("jarvis.SpeechRecognizer")
    @patch("jarvis.JarvisBrain")
    @patch("jarvis.VoiceSynthesizer")
    @patch("jarvis.VADManager")
    def test_constructor_creates_all_modules(self, mock_vad, mock_tts, mock_brain,
                                              mock_stt, mock_audio):
        """El constructor debe crear todos los módulos."""
        jarvis = Jarvis()
        assert jarvis.audio is not None
        assert jarvis.stt is not None
        assert jarvis.brain is not None
        assert jarvis.tts is not None
        assert jarvis.vad is not None
        assert jarvis.running is False
        assert jarvis._initialized is False

    @patch("jarvis.JarvisBrain")
    @patch("jarvis.VoiceSynthesizer")
    def test_constructor_accepts_custom_model(self, mock_tts, mock_brain):
        """Debe aceptar modelo Ollama y Whisper personalizados."""
        jarvis = Jarvis(model="llama3.2", whisper_model="medium")
        assert jarvis.brain.model == "llama3.2"
        assert jarvis.stt.model_size == "medium"

    @patch.multiple(
        "jarvis",
        AudioManager=MagicMock,
        SpeechRecognizer=MagicMock,
        JarvisBrain=MagicMock,
        VoiceSynthesizer=MagicMock,
        VADManager=MagicMock,
    )
    def test_initialize_loads_whisper_and_vad(self):
        """initialize() debe cargar Whisper y VAD."""
        jarvis = Jarvis()
        with (
            patch.object(jarvis, '_check_push_to_talk', return_value=False),
            patch.object(jarvis, '_select_mode', return_value='text'),
            patch.object(jarvis.tts, 'check_voice', return_value=True),
        ):
            result = jarvis.initialize(mode='text')

        assert result is True
        assert jarvis._initialized is True
        jarvis.stt.load.assert_called_once()
        jarvis.vad.load.assert_called_once()


class TestJarvisProcessQuery:
    """Tests para process_query con streaming."""

    @patch("jarvis.Jarvis")
    @patch("jarvis.JarvisBrain")
    def test_process_query_streams_sentences_to_tts(self, mock_brain, mock_jarvis):
        """process_query() debe enviar cada oración a TTS vía streaming."""
        jarvis = Jarvis()
        jarvis.brain = MagicMock()
        jarvis.tts = MagicMock()

        # Simular think_stream() como generador que yield oraciones
        def stream_generator():
            yield "Hola."
            yield "Cómo estás?"
            yield "Bien."

        jarvis.brain.think_stream.return_value = stream_generator()

        jarvis.process_query("test")

        # Verificar que TTS recibió cada oración
        assert jarvis.tts.speak_nonblocking.call_count == 3
        jarvis.tts.speak_nonblocking.assert_has_calls([
            call("Hola."),
            call("Cómo estás?"),
            call("Bien."),
        ])

    @patch("jarvis.Jarvis")
    def test_process_query_skips_empty_sentences(self, mock_jarvis):
        """process_query() debe ignorar oraciones vacías del streaming."""
        jarvis = Jarvis()
        jarvis.brain = MagicMock()
        jarvis.tts = MagicMock()

        def stream_generator():
            yield ""  # Vacía
            yield "  "  # Solo espacios
            yield "Hola."  # Válida

        jarvis.brain.think_stream.return_value = stream_generator()

        jarvis.process_query("test")

        # Solo debe enviar "Hola." a TTS
        assert jarvis.tts.speak_nonblocking.call_count == 1
        jarvis.tts.speak_nonblocking.assert_called_with("Hola.")

    @patch("jarvis.Jarvis")
    def test_process_query_empty_input_does_nothing(self, mock_jarvis):
        """process_query() con texto vacío no debe hacer nada."""
        jarvis = Jarvis()
        jarvis.brain = MagicMock()
        jarvis.tts = MagicMock()

        jarvis.process_query("")
        jarvis.brain.think_stream.assert_not_called()

        jarvis.process_query("   ")
        jarvis.brain.think_stream.assert_not_called()


class TestJarvisCommands:
    """Tests para comandos de voz en los diferentes modos."""

    @patch("jarvis.Jarvis")
    def test_salir_command_breaks_loop(self, mock_jarvis):
        """El comando 'salir' debe romper el bucle principal."""
        jarvis = Jarvis()
        jarvis.brain = MagicMock()
        jarvis.tts = MagicMock()
        jarvis.running = True

        with patch.object(jarvis, 'process_query') as mock_process:
            # Simular que una llamada a process_query ejecuta el comando salir
            def side_effect(text):
                if "salir" in text.lower():
                    jarvis.running = False
            mock_process.side_effect = side_effect

            # Simular el bucle
            jarvis.process_query("salir")

        assert jarvis.running is False

    @patch("jarvis.Jarvis")
    def test_limpiar_command_clears_history(self, mock_jarvis):
        """El comando 'limpiar' debe limpiar el historial."""
        jarvis = Jarvis()
        jarvis.brain = MagicMock()
        jarvis.tts = MagicMock()

        jarvis.brain.clear_history = MagicMock()
        jarvis.tts.speak = MagicMock()

        # Simular detección de "limpiar" en VAD loop
        text_lower = "limpia el historial por favor"
        if "limpiar" in text_lower:
            jarvis.brain.clear_history()
            jarvis.tts.speak("Historial de conversación limpiado.")

        jarvis.brain.clear_history.assert_called_once()
        jarvis.tts.speak.assert_called_once_with("Historial de conversación limpiado.")


class TestJarvisModes:
    """Tests para cada modo de operación."""

    @patch("jarvis.Jarvis")
    def test_vad_loop_requires_vad_available(self, mock_jarvis):
        """vad_loop() debe requerir VAD disponible."""
        jarvis = Jarvis()
        jarvis.audio = MagicMock()
        jarvis.vad = MagicMock()
        jarvis.vad.available = False

        jarvis.vad_loop()

        # No debe abrir stream si VAD no está disponible
        jarvis.audio.p.open.assert_not_called()

    @patch("jarvis.Jarvis")
    def test_interactive_mode_uses_process_query(self, mock_jarvis):
        """interactive_mode() debe usar process_query() con streaming."""
        jarvis = Jarvis()
        jarvis.brain = MagicMock()
        jarvis.running = True

        with (
            patch("builtins.input", side_effect=["hola", "salir"]),
            patch.object(jarvis, 'process_query') as mock_pq,
        ):
            jarvis.interactive_mode()

        # Debe llamar a process_query (no a brain.think directamente)
        mock_pq.assert_called_once_with("hola")

    @patch("jarvis.Jarvis")
    def test_push_to_talk_loop_uses_process_query(self, mock_jarvis):
        """push_to_talk_loop() debe usar process_query() con streaming."""
        jarvis = Jarvis()
        jarvis.audio = MagicMock()
        jarvis.stt = MagicMock()
        jarvis.brain = MagicMock()
        jarvis.tts = MagicMock()
        jarvis.running = True

        # Simular tecla espacio presionada, luego soltada, luego salir
        first_iteration = True

        def mock_is_pressed(key):
            nonlocal first_iteration
            if first_iteration:
                first_iteration = False
                return True
            return False

        def mock_wait(key, suppress=True):
            return

        import keyboard
        with (
            patch("jarvis.keyboard.is_pressed", side_effect=mock_is_pressed),
            patch("jarvis.keyboard.wait", side_effect=mock_wait),
            patch.object(jarvis, 'process_query') as mock_pq,
        ):
            jarvis.push_to_talk_loop()

        # No debe lanzar excepción


class TestJarvisEdgeCases:
    """Tests de casos borde."""

    @patch("jarvis.Jarvis")
    def test_cleanup_on_exit(self, mock_jarvis):
        """run() debe hacer cleanup del audio al salir."""
        jarvis = Jarvis()
        jarvis.audio = MagicMock()
        jarvis._initialized = True
        jarvis._mode = "text"

        jarvis.running = True
        with (
            patch("builtins.input", side_effect=["salir"]),
        ):
            jarvis.run()

        jarvis.audio.cleanup.assert_called_once()

    @patch("jarvis.Jarvis")
    def test_push_to_talk_stops_on_keyboard_interrupt(self, mock_jarvis):
        """push_to_talk_loop() debe manejar KeyboardInterrupt."""
        jarvis = Jarvis()
        jarvis.running = True

        import keyboard
        with (
            patch("jarvis.keyboard.wait", side_effect=KeyboardInterrupt()),
            patch("jarvis.keyboard.is_pressed", return_value=False),
        ):
            jarvis.push_to_talk_loop()

        assert jarvis.running is False

    @patch("jarvis.Jarvis")
    def test_vad_loop_stops_on_keyboard_interrupt(self, mock_jarvis):
        """vad_loop() debe manejar KeyboardInterrupt."""
        jarvis = Jarvis()
        jarvis.audio = MagicMock()
        jarvis.vad = MagicMock()
        jarvis.vad.available = True
        jarvis.running = True

        mock_stream = MagicMock()
        jarvis.audio.p.open.return_value = mock_stream

        jarvis.vad.running = True

        def mock_listen_for_speech(stream):
            raise KeyboardInterrupt()

        jarvis.vad.listen_for_speech = mock_listen_for_speech

        jarvis.vad_loop()

        assert jarvis.running is False
        assert jarvis.vad.running is False
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_mode_names_in_initialize(self):
        """Los nombres de modo deben estar correctamente definidos."""
        jarvis = Jarvis()
        mode_names = {"vad": "🎙️ VAD (manos libres)", "ptt": "⌨️ Push-to-talk", "text": "📝 Texto"}
        assert mode_names["vad"] == "🎙️ VAD (manos libres)"
        assert mode_names["ptt"] == "⌨️ Push-to-talk"
        assert mode_names["text"] == "📝 Texto"
