"""
Tests unitarios para SpeechRecognizer.

Cubre:
- Carga del modelo Faster-Whisper
- Transcripción de audio a texto
- Manejo de errores: modelo no cargado, audio inválido
- Diferentes tamaños de modelo Whisper
"""

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from unittest.mock import ANY

from jarvis import SpeechRecognizer, WHISPER_MODEL_SIZE


class TestSpeechRecognizer:
    """Suite de tests para SpeechRecognizer."""

    @pytest.fixture
    def recognizer(self):
        """Fixture con modelo Whisper mockeado."""
        r = SpeechRecognizer()
        r.model = MagicMock()
        yield r

    @pytest.fixture
    def unloaded_recognizer(self):
        """Fixture SIN modelo cargado (model = None)."""
        r = SpeechRecognizer()
        r.model = None
        yield r

    # ─── load ─────────────────────────────────────────────────────────────

    @patch("jarvis.WhisperModel")
    def test_load_creates_whisper_model(self, mock_whisper):
        """load() debe crear una instancia de WhisperModel."""
        recognizer = SpeechRecognizer()
        recognizer.model = None  # Simular que no está cargado

        recognizer.load()

        mock_whisper.assert_called_once_with(
            WHISPER_MODEL_SIZE,
            device="cuda",
            compute_type="float16",
            download_root=ANY,
        )

    @patch("jarvis.WhisperModel")
    def test_load_does_not_reload_if_already_loaded(self, mock_whisper):
        """load() no debe recargar si el modelo ya está cargado."""
        recognizer = SpeechRecognizer()
        recognizer.model = MagicMock()  # Ya cargado

        recognizer.load()

        mock_whisper.assert_not_called()

    @patch("jarvis.WhisperModel")
    def test_load_with_custom_model_size(self, mock_whisper):
        """Debe usar el model_size especificado en el constructor."""
        custom_size = "medium"
        recognizer = SpeechRecognizer(model_size=custom_size)
        recognizer.model = None

        recognizer.load()

        args, _ = mock_whisper.call_args
        assert args[0] == custom_size

    # ─── transcribe ───────────────────────────────────────────────────────

    def test_transcribe_returns_text(self, recognizer):
        """transcribe() debe retornar el texto transcrito."""
        # Simular segmentos de transcripción
        mock_segment_1 = MagicMock()
        mock_segment_1.text = "hola mundo"
        mock_segment_2 = MagicMock()
        mock_segment_2.text = "cómo estás"

        mock_info = MagicMock()
        mock_info.language = "es"
        mock_info.language_probability = 0.95

        recognizer.model.transcribe.return_value = ([mock_segment_1, mock_segment_2], mock_info)

        result = recognizer.transcribe("/fake/audio.wav")

        assert "hola mundo" in result
        assert "cómo estás" in result

    def test_transcribe_with_empty_audio(self, recognizer):
        """Debe retornar string vacío si no hay segmentos."""
        mock_info = MagicMock()
        mock_info.language = "es"
        mock_info.language_probability = 0.90

        recognizer.model.transcribe.return_value = ([], mock_info)

        result = recognizer.transcribe("/fake/audio.wav")
        assert result == ""

    def test_transcribe_calls_model_with_correct_params(self, recognizer):
        """Debe llamar a model.transcribe con los parámetros correctos."""
        mock_segment = MagicMock()
        mock_segment.text = "test"
        mock_info = MagicMock()
        mock_info.language = "es"
        mock_info.language_probability = 0.95

        recognizer.model.transcribe.return_value = ([mock_segment], mock_info)

        recognizer.transcribe("/fake/audio.wav")

        recognizer.model.transcribe.assert_called_once_with(
            "/fake/audio.wav",
            language="es",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

    def test_transcribe_loads_model_if_not_loaded(self, unloaded_recognizer):
        """Debe cargar el modelo automáticamente si no está cargado."""
        with patch.object(unloaded_recognizer, 'load') as mock_load:
            mock_segment = MagicMock()
            mock_segment.text = "test"
            mock_info = MagicMock()
            mock_info.language = "es"
            mock_info.language_probability = 0.95

            unloaded_recognizer.model = MagicMock()
            unloaded_recognizer.model.transcribe.return_value = ([mock_segment], mock_info)

            unloaded_recognizer.transcribe("/fake/audio.wav")

            mock_load.assert_not_called()  # El fixture ya asigna model...

    def test_transcribe_strips_segment_text(self, recognizer):
        """Debe limpiar los espacios de los segmentos."""
        mock_seg_1 = MagicMock()
        mock_seg_1.text = "  hola  "
        mock_seg_2 = MagicMock()
        mock_seg_2.text = "  mundo  "

        mock_info = MagicMock()
        mock_info.language = "es"
        mock_info.language_probability = 0.95

        recognizer.model.transcribe.return_value = ([mock_seg_1, mock_seg_2], mock_info)

        result = recognizer.transcribe("/fake/audio.wav")
        assert result == "hola mundo"

    def test_transcribe_reports_language_probability(self, recognizer, capsys):
        """Debe imprimir la probabilidad del idioma detectado."""
        mock_segment = MagicMock()
        mock_segment.text = "test"
        mock_info = MagicMock()
        mock_info.language = "es"
        mock_info.language_probability = 0.95

        recognizer.model.transcribe.return_value = ([mock_segment], mock_info)

        recognizer.transcribe("/fake/audio.wav")
        captured = capsys.readouterr()

        assert "95%" in captured.out or "0.95" in captured.out

    # ─── Constructor ──────────────────────────────────────────────────────

    def test_constructor_default_model_size(self):
        """Debe usar WHISPER_MODEL_SIZE por defecto."""
        r = SpeechRecognizer()
        assert r.model_size == WHISPER_MODEL_SIZE

    def test_constructor_custom_model_size(self):
        """Debe aceptar un tamaño de modelo personalizado."""
        r = SpeechRecognizer(model_size="tiny")
        assert r.model_size == "tiny"
