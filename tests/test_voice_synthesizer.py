"""
Tests unitarios para VoiceSynthesizer (Piper TTS).

Cubre:
- Verificación de modelos de voz
- Síntesis de texto a voz (speak)
- Síntesis no bloqueante (speak_nonblocking)
- Resampling de audio 22050 → 48000 Hz
- Manejo de errores: modelo faltante, timeout de Piper
"""

import os
import subprocess
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest
import numpy as np

from jarvis import VoiceSynthesizer, PIPER_SAMPLE_RATE, TTS_OUTPUT_RATE


class TestVoiceSynthesizer:
    """Suite de tests para VoiceSynthesizer."""

    # ─── check_voice ──────────────────────────────────────────────────────

    def test_check_voice_returns_true_if_model_exists(self, mock_voice_files):
        """Debe retornar True si el modelo de voz existe."""
        synth = VoiceSynthesizer()
        assert synth.check_voice() is True

    def test_check_voice_returns_false_if_model_missing(self, mock_missing_voice):
        """Debe retornar False si el modelo de voz no existe."""
        synth = VoiceSynthesizer(model_path=str(mock_missing_voice))
        result = synth.check_voice()
        assert result is False

    def test_check_voice_reports_error_on_missing_model(self, mock_missing_voice, capsys):
        """Debe imprimir un mensaje de error si falta el modelo."""
        synth = VoiceSynthesizer()
        synth.check_voice()
        captured = capsys.readouterr()
        assert "Voz no encontrada" in captured.out

    # ─── speak - Basic synthesis ──────────────────────────────────────────

    @patch("jarvis.subprocess.Popen")
    def test_speak_calls_piper_subprocess(self, mock_popen, mock_voice_files):
        """speak() debe ejecutar Piper como subproceso."""
        # Configurar mock del subprocess
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (
            np.zeros(PIPER_SAMPLE_RATE, dtype=np.int16).tobytes(),
            b"",
        )
        mock_popen.return_value = mock_proc

        synth = VoiceSynthesizer()
        synth.speak("Hola mundo")

        # Verificar que se llamó a Piper
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        assert "piper" in args[0][0]
        assert "--model" in args[0]
        assert "--output-raw" in args[0]

    @patch("jarvis.subprocess.Popen")
    def test_speak_passes_text_via_stdin(self, mock_popen, mock_voice_files):
        """Debe enviar el texto a Piper vía stdin."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"audio_data", b"")
        mock_popen.return_value = mock_proc

        synth = VoiceSynthesizer()
        synth.speak("Hola mundo")

        # Verificar que el texto se envió por stdin
        args, kwargs = mock_proc.communicate.call_args
        assert kwargs["input"] == "Hola mundo".encode("utf-8")

    @patch("jarvis.subprocess.Popen")
    def test_speak_returns_true_on_success(self, mock_popen, mock_voice_files):
        """Debe retornar True si la síntesis fue exitosa."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"audio_data", b"")
        mock_popen.return_value = mock_proc

        synth = VoiceSynthesizer()
        assert synth.speak("Hola") is True

    # ─── speak - Audio resampling ─────────────────────────────────────────

    @patch("jarvis.subprocess.Popen")
    def test_speak_resamples_audio_to_output_rate(self, mock_popen, mock_voice_files):
        """Debe resamplear el audio de 22050 Hz a 48000 Hz."""
        # Proporcionar audio a 22050 Hz (1 segundo)
        audio_22050 = np.sin(np.linspace(0, 440 * np.pi, PIPER_SAMPLE_RATE)).astype(np.int16).tobytes()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (audio_22050, b"")
        mock_popen.return_value = mock_proc

        synth = VoiceSynthesizer()
        synth.speak("Hola")

        if PIPER_SAMPLE_RATE != TTS_OUTPUT_RATE:
            # Verificar que se hizo el resampling (se usó numpy.interp)
            import jarvis
            jarvis.np.interp.assert_called()

    # ─── speak - Error handling ───────────────────────────────────────────

    @patch("jarvis.subprocess.Popen")
    def test_speak_returns_false_on_piper_error(self, mock_popen, mock_voice_files):
        """Debe retornar False si Piper retorna código de error."""
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"Error de Piper")
        mock_popen.return_value = mock_proc

        synth = VoiceSynthesizer()
        assert synth.speak("Hola") is False

    @patch("jarvis.subprocess.Popen")
    def test_speak_returns_false_on_timeout(self, mock_popen, mock_voice_files):
        """Debe retornar False si Piper excede el timeout."""
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired("piper", 30)
        mock_popen.return_value = mock_proc

        synth = VoiceSynthesizer()
        assert synth.speak("Hola") is False

    @patch("jarvis.subprocess.Popen")
    def test_speak_returns_false_if_voice_missing(self, mock_popen, mock_missing_voice):
        """Debe retornar False si no hay modelo de voz."""
        synth = VoiceSynthesizer()
        assert synth.speak("Hola") is False
        mock_popen.assert_not_called()

    @patch("jarvis.subprocess.Popen")
    def test_speak_handles_empty_text(self, mock_popen, mock_voice_files):
        """Debe manejar texto vacío (enviar string vacío a Piper)."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"", b"")
        mock_popen.return_value = mock_proc

        synth = VoiceSynthesizer()
        result = synth.speak("")

        # Debería enviar string vacío a Piper
        args, kwargs = mock_proc.communicate.call_args
        assert kwargs["input"] == "".encode("utf-8")

    @patch("jarvis.subprocess.Popen")
    def test_speak_handles_stderr_from_piper(self, mock_popen, mock_voice_files, capsys):
        """Debe capturar y mostrar stderr de Piper en caso de error."""
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"ALSA error: device not found")
        mock_popen.return_value = mock_proc

        synth = VoiceSynthesizer()
        synth.speak("Hola")

        captured = capsys.readouterr()
        assert "ALSA error" in captured.out or "Piper error" in captured.out

    # ─── speak_nonblocking ────────────────────────────────────────────────

    @patch("jarvis.threading.Thread")
    def test_speak_nonblocking_creates_thread(self, mock_thread, mock_voice_files):
        """speak_nonblocking() debe crear un hilo para la síntesis."""
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        synth = VoiceSynthesizer()
        with patch.object(synth, 'speak', return_value=True) as mock_speak:
            result = synth.speak_nonblocking("Hola")

        mock_thread.assert_called_once()
        args, kwargs = mock_thread.call_args
        assert kwargs["target"] == mock_speak
        assert kwargs["args"] == ("Hola",)
        assert kwargs["daemon"] is True
        mock_thread_instance.start.assert_called_once()

    @patch("jarvis.threading.Thread")
    def test_speak_nonblocking_returns_thread(self, mock_thread, mock_voice_files):
        """Debe retornar la referencia al hilo creado."""
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        synth = VoiceSynthesizer()
        with patch.object(synth, 'speak', return_value=True):
            result = synth.speak_nonblocking("Hola")

        assert result == mock_thread_instance
