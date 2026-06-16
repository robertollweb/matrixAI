import io
import pathlib
import tempfile
import unittest
from http.client import HTTPMessage
from unittest.mock import patch

from matrixai.parser import parse_file
from matrixai.server import MatrixAIServerHandler, RateLimiter


class MockServer:
    def __init__(self, key, backend="stdlib"):
        self.matrixai_api_key = key
        self.matrixai_backend = backend
        self.matrixai_rate_limiter = RateLimiter(0)  # disabled — unit tests don't test rate limiting
        self.matrixai_cors_origins = ["*"]

        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("""
PROJECT EmptyDoc

VECTOR Email[1]
  val: Score
END

FUNCTION MockFunc
  C: Score = 0
END

GRAPH
  Email -> MockFunc
END
""")
            fname = f.name

        self.matrixai_program = parse_file(fname)
        pathlib.Path(fname).unlink()

        self.matrixai_parameter_set = None
        self.matrixai_metrics = {
            "uptime_start": 0,
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "requests_rate_limited": 0,
            "items_processed": 0,
            "last_request_ms": 0.0,
        }


class TestServerHandler(unittest.TestCase):
    def setUp(self):
        self.server = MockServer("secret-key")
        self.client_address = ("127.0.0.1", 50000)

    def _create_handler(self, method, path, headers=None, body=b""):
        handler = object.__new__(MatrixAIServerHandler)
        handler.server = self.server
        handler.client_address = self.client_address
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.path = path
        handler.command = method
        handler.request_version = "HTTP/1.1"
        handler.headers = HTTPMessage()
        if headers:
            for key, value in headers.items():
                handler.headers[key] = value
        handler.close_connection = True
        return handler

    @patch("matrixai.server.MatrixAIServerHandler._send_response")
    def test_health_check(self, mock_send):
        handler = self._create_handler("GET", "/health")

        handler.do_GET()
        mock_send.assert_called_once()
        args, _ = mock_send.call_args
        self.assertEqual(args[0], 200)
        self.assertIn("status", args[1])
        self.assertIn("metrics", args[1])

    @patch("matrixai.server.MatrixAIServerHandler._send_response")
    def test_openapi(self, mock_send):
        handler = self._create_handler("GET", "/openapi.json")

        handler.do_GET()
        mock_send.assert_called_once()
        args, _ = mock_send.call_args
        self.assertEqual(args[0], 200)
        self.assertIn("openapi", args[1])
        self.assertEqual(args[1]["info"]["title"], "MatrixAI Prediction API")
        schemas = args[1]["components"]["schemas"]
        self.assertIn("EmailInput", schemas)
        self.assertEqual(schemas["EmailInput"]["properties"]["val"]["minimum"], 0.0)
        self.assertEqual(schemas["EmailInput"]["properties"]["val"]["maximum"], 1.0)
        request_schema = args[1]["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        self.assertEqual(request_schema["oneOf"][0]["$ref"], "#/components/schemas/PredictionInput")
        self.assertEqual(request_schema["oneOf"][1]["$ref"], "#/components/schemas/PredictionBatch")

    @patch("matrixai.server.MatrixAIServerHandler._send_html")
    def test_docs(self, mock_html):
        handler = self._create_handler("GET", "/docs")

        handler.do_GET()
        mock_html.assert_called_once()
        args, _ = mock_html.call_args
        self.assertEqual(args[0], 200)
        self.assertIn("Swagger UI", args[1])

    @patch("matrixai.server.MatrixAIServerHandler._send_response")
    def test_predict_invalid_auth(self, mock_send):
        handler = self._create_handler("POST", "/predict", {"Authorization": "Bearer wrong-key"})

        handler.do_POST()
        mock_send.assert_called_once()
        args, _ = mock_send.call_args
        self.assertEqual(args[0], 401)
        self.assertEqual(self.server.matrixai_metrics["requests_total"], 1)
        self.assertEqual(self.server.matrixai_metrics["requests_failed"], 1)

    @patch("matrixai.server.MatrixAIServerHandler._send_response")
    def test_predict_missing_auth(self, mock_send):
        handler = self._create_handler("POST", "/predict", {})

        handler.do_POST()
        mock_send.assert_called_once()
        args, _ = mock_send.call_args
        self.assertEqual(args[0], 401)
        self.assertEqual(self.server.matrixai_metrics["requests_total"], 1)
        self.assertEqual(self.server.matrixai_metrics["requests_failed"], 1)

    @patch("matrixai.server.MatrixAIServerHandler._send_response")
    def test_predict_success(self, mock_send):
        body = b"{}"
        headers = {
            "Authorization": "Bearer secret-key",
            "Content-Length": str(len(body))
        }
        handler = self._create_handler("POST", "/predict", headers, body)

        handler.do_POST()
        mock_send.assert_called_once()
        args, _ = mock_send.call_args
        self.assertEqual(args[0], 200)
        self.assertIn("state", args[1])
        self.assertEqual(self.server.matrixai_metrics["requests_total"], 1)
        self.assertEqual(self.server.matrixai_metrics["requests_successful"], 1)
        self.assertEqual(self.server.matrixai_metrics["items_processed"], 1)
        self.assertGreaterEqual(self.server.matrixai_metrics["last_request_ms"], 0.0)

    @patch("matrixai.server.MatrixAIServerHandler._send_response")
    def test_predict_batch(self, mock_send):
        body = b"[{},{}]"
        headers = {
            "Authorization": "Bearer secret-key",
            "Content-Length": str(len(body))
        }
        handler = self._create_handler("POST", "/predict", headers, body)

        handler.do_POST()
        mock_send.assert_called_once()
        args, _ = mock_send.call_args
        self.assertEqual(args[0], 200)
        self.assertTrue(isinstance(args[1], list))
        self.assertEqual(len(args[1]), 2)
        self.assertIn("state", args[1][0])
        self.assertIn("state", args[1][1])
        self.assertEqual(self.server.matrixai_metrics["requests_total"], 1)
        self.assertEqual(self.server.matrixai_metrics["requests_successful"], 1)
        self.assertEqual(self.server.matrixai_metrics["items_processed"], 2)

    @patch("matrixai.server.MatrixAIServerHandler._send_response")
    def test_predict_invalid_json_counts_failed_request(self, mock_send):
        body = b"{"
        headers = {
            "Authorization": "Bearer secret-key",
            "Content-Length": str(len(body)),
        }
        handler = self._create_handler("POST", "/predict", headers, body)

        handler.do_POST()

        args, _ = mock_send.call_args
        self.assertEqual(args[0], 400)
        self.assertEqual(self.server.matrixai_metrics["requests_total"], 1)
        self.assertEqual(self.server.matrixai_metrics["requests_failed"], 1)
        self.assertEqual(self.server.matrixai_metrics["items_processed"], 0)

    @patch("matrixai.server.MatrixAIServerHandler._send_response")
    def test_predict_empty_payload_counts_failed_request(self, mock_send):
        headers = {
            "Authorization": "Bearer secret-key",
            "Content-Length": "0",
        }
        handler = self._create_handler("POST", "/predict", headers, b"")

        handler.do_POST()

        args, _ = mock_send.call_args
        self.assertEqual(args[0], 400)
        self.assertEqual(self.server.matrixai_metrics["requests_total"], 1)
        self.assertEqual(self.server.matrixai_metrics["requests_failed"], 1)
        self.assertEqual(self.server.matrixai_metrics["items_processed"], 0)

if __name__ == "__main__":
    unittest.main()
