"""
Tests unitarios para JarvisBrain (Ollama LLM).

Cubre:
- think(): consulta no-streaming
- think_stream(): consulta streaming con yield de oraciones
- Manejo de errores: conexión, timeout, API errors
- Límite de historial de conversación
- Limpieza de historial
"""

import json
import time
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from jarvis import JarvisBrain, OLLAMA_HOST, OLLAMA_MODEL, JARVIS_SYSTEM_PROMPT


class MockResponse:
    """Simula una respuesta de requests.post."""

    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if not self.ok:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class TestJarvisBrain:
    """Suite de tests para JarvisBrain."""

    @pytest.fixture
    def brain(self):
        """Fixture básico de JarvisBrain."""
        b = JarvisBrain()
        b.conversation_history = []
        return b

    @pytest.fixture
    def brain_with_history(self, brain):
        """Fixture con historial de conversación."""
        brain.conversation_history = [
            {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "hola en qué puedo ayudarte"},
            {"role": "user", "content": "cuál es el clima"},
            {"role": "assistant", "content": "hoy hace sol"},
        ]
        return brain

    # ─── Constructor ──────────────────────────────────────────────────────

    def test_constructor_defaults(self):
        """Debe usar valores por defecto."""
        brain = JarvisBrain()
        assert brain.model == OLLAMA_MODEL
        assert brain.system_prompt == JARVIS_SYSTEM_PROMPT
        assert brain.conversation_history == []

    def test_constructor_custom_values(self):
        """Debe aceptar modelo y system_prompt personalizados."""
        brain = JarvisBrain(model="llama3.2", system_prompt="Eres un asistente.")
        assert brain.model == "llama3.2"
        assert brain.system_prompt == "Eres un asistente."

    # ─── think() - Non-streaming ──────────────────────────────────────────

    @patch("jarvis.requests.post")
    def test_think_returns_response(self, mock_post, brain):
        """think() debe retornar la respuesta del LLM."""
        mock_post.return_value = MockResponse(
            {"message": {"role": "assistant", "content": "El clima es soleado."}}
        )

        result = brain.think("cómo está el clima")

        assert result == "El clima es soleado."

    @patch("jarvis.requests.post")
    def test_think_sends_correct_payload(self, mock_post, brain):
        """Debe enviar el payload correcto a la API de Ollama."""
        mock_post.return_value = MockResponse(
            {"message": {"role": "assistant", "content": "ok"}}
        )

        brain.think("prueba")

        args, kwargs = mock_post.call_args
        assert args[0] == f"{OLLAMA_HOST}/api/chat"
        assert kwargs["json"]["model"] == OLLAMA_MODEL
        assert kwargs["json"]["stream"] is False
        assert kwargs["json"]["options"]["temperature"] == 0.7
        assert kwargs["json"]["options"]["num_predict"] == 512

    @patch("jarvis.requests.post")
    def test_think_includes_system_prompt(self, mock_post, brain):
        """Debe incluir el system prompt en los mensajes."""
        mock_post.return_value = MockResponse(
            {"message": {"role": "assistant", "content": "ok"}}
        )

        brain.think("prueba")

        args, kwargs = mock_post.call_args
        messages = kwargs["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == JARVIS_SYSTEM_PROMPT

    @patch("jarvis.requests.post")
    def test_think_updates_conversation_history(self, mock_post, brain):
        """Debe agregar la interacción al historial."""
        mock_post.return_value = MockResponse(
            {"message": {"role": "assistant", "content": "Respuesta de prueba."}}
        )

        brain.think("Mensaje de prueba")

        assert len(brain.conversation_history) == 2
        assert brain.conversation_history[-2]["role"] == "user"
        assert brain.conversation_history[-2]["content"] == "Mensaje de prueba"
        assert brain.conversation_history[-1]["role"] == "assistant"
        assert brain.conversation_history[-1]["content"] == "Respuesta de prueba."

    @patch("jarvis.requests.post")
    def test_think_connection_error_returns_message(self, mock_post, brain):
        """Debe retornar mensaje de error si no hay conexión a Ollama."""
        mock_post.side_effect = requests.exceptions.ConnectionError()

        result = brain.think("prueba")

        assert "no puedo conectar" in result.lower()

    @patch("jarvis.requests.post")
    def test_think_timeout_returns_error_message(self, mock_post, brain):
        """Debe manejar timeout de conexión."""
        mock_post.side_effect = requests.exceptions.Timeout()

        result = brain.think("prueba")

        assert "error" in result.lower() or "lo siento" in result.lower()

    @patch("jarvis.requests.post")
    def test_think_http_error_returns_error_message(self, mock_post, brain):
        """Debe manejar errores HTTP."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Bad Request")
        mock_post.return_value = mock_response

        result = brain.think("prueba")

        assert "error" in result.lower() or "lo siento" in result.lower()

    @patch("jarvis.requests.post")
    def test_think_includes_history_in_context(self, mock_post, brain_with_history):
        """Debe incluir el historial de conversación en el contexto."""
        mock_post.return_value = MockResponse(
            {"message": {"role": "assistant", "content": "Respuesta."}}
        )

        brain_with_history.think("siguiente pregunta")

        args, kwargs = mock_post.call_args
        messages = kwargs["json"]["messages"]
        # system + 4 historiales + 1 nuevo = 6
        assert len(messages) == 6
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hola"

    @patch("jarvis.requests.post")
    def test_think_maintains_10_message_limit(self, mock_post, brain):
        """Debe limitar el historial a 10 mensajes máximo."""
        mock_post.return_value = MockResponse(
            {"message": {"role": "assistant", "content": "resp."}}
        )

        # Llenar historial con más de 10 mensajes
        for i in range(8):  # 8 interacciones = 16 mensajes
            brain.think(f"mensaje {i}")

        assert len(brain.conversation_history) <= 10

    # ─── think_stream() - Streaming ──────────────────────────────────────

    def test_think_stream_yields_sentences_with_punctuation(self, brain):
        """
        think_stream() debe yield oraciones completas cuando detecta
        puntuación (. ! ?).
        """
        # Simular chunks de streaming con puntuación
        chunk_data = [
            b'{"message":{"role":"assistant","content":"Hola"},"done":false}',
            b'{"message":{"role":"assistant","content":". "},"done":false}',
            b'{"message":{"role":"assistant","content":"Cómo"},"done":false}',
            b'{"message":{"role":"assistant","content":" estás"},"done":false}',
            b'{"message":{"role":"assistant","content":"?"},"done":false}',
            b'{"message":{"role":"assistant","content":""},"done":true}',
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = chunk_data
        mock_response.raise_for_status = MagicMock()
        mock_response.ok = True

        with patch("jarvis.requests.post", return_value=mock_response):
            sentences = list(brain.think_stream("prueba"))

        assert len(sentences) >= 1
        assert "Hola" in sentences[0]

    def test_think_stream_yields_sentences_with_multiple_endings(self, brain):
        """Debe yield oraciones separadas por diferentes signos de puntuación."""
        chunk_data = [
            b'{"message":{"role":"assistant","content":"Primera oración. "},"done":false}',
            b'{"message":{"role":"assistant","content":"Segunda oración"},"done":false}',
            b'{"message":{"role":"assistant","content":"!"},"done":false}',
            b'{"message":{"role":"assistant","content":"Tercera"},"done":false}',
            b'{"message":{"role":"assistant","content":"?"},"done":false}',
            b'{"message":{"role":"assistant","content":""},"done":true}',
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = chunk_data
        mock_response.raise_for_status = MagicMock()
        mock_response.ok = True

        with patch("jarvis.requests.post", return_value=mock_response):
            sentences = list(brain.think_stream("prueba"))

        assert len(sentences) >= 2  # Al menos 2 oraciones yield

    def test_think_stream_updates_history_after_stream(self, brain):
        """Debe actualizar el historial después de completar el streaming."""
        chunk_data = [
            b'{"message":{"role":"assistant","content":"Hola. "},"done":false}',
            b'{"message":{"role":"assistant","content":"Mundo."},"done":false}',
            b'{"message":{"role":"assistant","content":""},"done":true}',
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = chunk_data
        mock_response.raise_for_status = MagicMock()
        mock_response.ok = True

        with patch("jarvis.requests.post", return_value=mock_response):
            list(brain.think_stream("mi consulta"))

        assert len(brain.conversation_history) == 2
        assert brain.conversation_history[0]["content"] == "mi consulta"
        assert brain.conversation_history[1]["role"] == "assistant"

    def test_think_stream_sends_stream_true(self, brain):
        """Debe enviar stream=True en el payload."""
        chunk_data = [
            b'{"message":{"role":"assistant","content":"ok"},"done":false}',
            b'{"message":{"role":"assistant","content":""},"done":true}',
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = chunk_data
        mock_response.raise_for_status = MagicMock()

        with patch("jarvis.requests.post", return_value=mock_response) as mock_post:
            list(brain.think_stream("prueba"))

        args, kwargs = mock_post.call_args
        assert kwargs["json"]["stream"] is True
        assert kwargs["stream"] is True

    def test_think_stream_connection_error(self, brain):
        """Debe yield mensaje de error si no hay conexión."""
        with patch("jarvis.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError()

            sentences = list(brain.think_stream("prueba"))

        assert len(sentences) >= 1
        assert "no puedo conectar" in sentences[0].lower()

    def test_think_stream_general_error(self, brain):
        """Debe yield mensaje de error genérico."""
        with patch("jarvis.requests.post") as mock_post:
            mock_post.side_effect = Exception("Error inesperado")

            sentences = list(brain.think_stream("prueba"))

        assert len(sentences) >= 1
        assert "error" in sentences[0].lower()

    # ─── clear_history ────────────────────────────────────────────────────

    def test_clear_history_empties_conversation(self, brain):
        """clear_history() debe vaciar el historial."""
        brain.conversation_history = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "respuesta"},
        ]

        brain.clear_history()

        assert brain.conversation_history == []

    @patch("jarvis.requests.post")
    def test_think_has_30_second_timeout(self, mock_post, brain):
        """Debe tener timeout de 30 segundos."""
        mock_post.return_value = MockResponse(
            {"message": {"role": "assistant", "content": "ok"}}
        )

        brain.think("prueba")

        args, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 30
