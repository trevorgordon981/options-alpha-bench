"""Regression tests for OP03: endpoint normalization, provider origin auth, and request shapes."""
import io
import json
import os
import sys
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import book
import llm_client
import run_model_alpha


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self):
        return self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestEndpointNormalizationAndAuth(unittest.TestCase):
    def test_normalize_chat_endpoint_prevents_double_v1(self):
        self.assertEqual(
            llm_client.normalize_chat_endpoint("https://openrouter.ai/api/v1"),
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertEqual(
            llm_client.normalize_chat_endpoint("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertEqual(
            llm_client.normalize_chat_endpoint("https://openrouter.ai/api/v1/chat/completions"),
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertEqual(
            llm_client.normalize_chat_endpoint("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertEqual(
            llm_client.normalize_chat_endpoint("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertEqual(
            llm_client.normalize_chat_endpoint("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions",
        )

    def test_normalize_rejects_malformed_urls(self):
        for bad in ("", "   ", "ftp://openrouter.ai/api/v1", "not-a-url"):
            with self.assertRaises(ValueError):
                llm_client.normalize_chat_endpoint(bad)

    def test_openrouter_auth_attached_for_both_construction_and_book(self):
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            return _FakeHTTPResponse({"choices": [{"message": {"content": '{"close": [], "open": []}'}}]})

        env = {"OPENROUTER_API_KEY": "test-openrouter-secret-value"}
        with mock.patch.dict(os.environ, env, clear=False), mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            run_model_alpha.call_model("hello", "https://openrouter.ai/api/v1", "provider/model-x")
            b = book.Book("https://openrouter.ai/api/v1", "provider/model-x", verbose=False)
            b._call("hello book")

        self.assertEqual(len(captured), 2)
        for req in captured:
            self.assertEqual(req.full_url, "https://openrouter.ai/api/v1/chat/completions")
            self.assertEqual(req.get_header("Authorization"), "Bearer test-openrouter-secret-value")
            self.assertEqual(req.get_header("Content-type"), "application/json")
            body = json.loads(req.data.decode("utf-8"))
            self.assertEqual(body["model"], "provider/model-x")
            self.assertEqual(body["reasoning"], {"enabled": False})

    def test_untrusted_url_substring_spoofing_does_not_leak_openrouter_key(self):
        spoofed_urls = [
            "https://evil.example/openrouter/v1",
            "https://openrouter.ai.evil.example/api/v1",
            "http://openrouter.ai/api/v1",  # plain HTTP must NOT receive bearer token
            "http://127.0.0.1:8000/openrouter",
        ]
        env = {"OPENROUTER_API_KEY": "do-not-leak"}
        with mock.patch.dict(os.environ, env, clear=False):
            for url in spoofed_urls:
                req = llm_client.build_chat_request(url, "test-model", "prompt")
                self.assertIsNone(
                    req.get_header("Authorization"),
                    f"Authorization header leaked to untrusted URL: {url}",
                )
                body = json.loads(req.data.decode("utf-8"))
                self.assertNotIn("reasoning", body)
                self.assertNotIn("provider", body)

    def test_local_endpoint_request_shape_and_explicit_api_key(self):
        req_local = llm_client.build_chat_request("http://127.0.0.1:8082", "local-m", "hi")
        self.assertEqual(req_local.full_url, "http://127.0.0.1:8082/v1/chat/completions")
        self.assertIsNone(req_local.get_header("Authorization"))

        req_explicit = llm_client.build_chat_request(
            "http://127.0.0.1:8082", "local-m", "hi", api_key="custom-local-token"
        )
        self.assertEqual(req_explicit.get_header("Authorization"), "Bearer custom-local-token")

    def test_null_message_content_raises_clean_value_error(self):
        def fake_urlopen(req, timeout=None):
            return _FakeHTTPResponse({"choices": [{"message": {"content": None}}]})

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(ValueError):
                llm_client.chat_completion("http://127.0.0.1:8000", "m", "hi")


if __name__ == "__main__":
    unittest.main()
