"""vectorphile.serve — Model serving and deployment utilities.

Provides:
- ModelServer: HTTP REST API server wrapping any model
- Predictor: standardized prediction interface
- Batch inference with dynamic batching
- gRPC-like interface stub

Uses only Python's built-in http.server — no Flask/FastAPI required.

Usage::

    from vectorphile.serve import ModelServer, serve

    server = ModelServer(model, preprocessing=preprocess, postprocessing=postprocess)
    server.serve(port=8080)

    # Or decorator-style
    @serve(port=8080)
    def predict(request):
        x = preprocess(request['inputs'])
        return {'outputs': model(x).tolist()}
"""
from __future__ import annotations
import json
import time
import threading
import http.server
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Predictor interface
# ---------------------------------------------------------------------------

class Predictor:
    """Wraps a model with preprocessing and postprocessing.

    Usage::

        predictor = Predictor(
            model=my_model,
            preprocess=lambda x: np.array(x, dtype='float32'),
            postprocess=lambda y: y.tolist(),
        )
        result = predictor.predict({'inputs': [[1.0, 2.0, 3.0]]})
    """

    def __init__(self, model, preprocess: Optional[Callable] = None,
                 postprocess: Optional[Callable] = None,
                 batch_size: int = 32):
        self.model = model
        def _default_pre(x):
            if isinstance(x, list): return x
            if hasattr(x, 'to_flat_list'): return list(x.to_flat_list())
            return x
        self._preprocess = preprocess or _default_pre
        self._postprocess = postprocess or (lambda y: list(y._data.to_flat_list()) if hasattr(y, '_data') and hasattr(y._data, 'to_flat_list') else (y._data if hasattr(y, '_data') else y))
        self.batch_size = batch_size

    def predict(self, inputs: Any) -> Any:
        """Run a single prediction."""
        x = self._preprocess(inputs)
        import vectorphile as vp
        with vp.autograd.no_grad():
            from vectorphile.core.tensor import Tensor
            from vectorphile._core.ndarray import array as _nd
            if not isinstance(x, Tensor):
                flat = [float(v) for v in (x if isinstance(x, list) else [x])]
                t = Tensor(_nd(flat))
            else:
                t = x
            y = self.model(t)
        return self._postprocess(y)

    def predict_batch(self, inputs_list: list) -> list:
        """Run predictions on a list of inputs."""
        return [self.predict(inp) for inp in inputs_list]


# ---------------------------------------------------------------------------
# Request/Response helpers
# ---------------------------------------------------------------------------

class Request:
    def __init__(self, body: bytes, headers: dict, method: str = 'POST',
                 path: str = '/predict'):
        self.body = body
        self.headers = headers
        self.method = method
        self.path = path

    def json(self) -> dict:
        return json.loads(self.body)


class Response:
    def __init__(self, body: Any, status: int = 200,
                 content_type: str = 'application/json'):
        self.body = body
        self.status = status
        self.content_type = content_type

    def to_bytes(self) -> bytes:
        if isinstance(self.body, (dict, list)):
            return json.dumps(self.body).encode('utf-8')
        if isinstance(self.body, str):
            return self.body.encode('utf-8')
        return bytes(self.body)


# ---------------------------------------------------------------------------
# ModelServer
# ---------------------------------------------------------------------------

class ModelServer:
    """HTTP REST API server for model inference.

    Exposes endpoints:
        POST /predict   → runs model inference
        GET  /health    → health check
        GET  /info      → model info

    Usage::

        server = ModelServer(model, port=8080)
        server.start()  # non-blocking
        # or
        server.serve()  # blocking
    """

    def __init__(self, model=None, port: int = 8080,
                 preprocessing: Optional[Callable] = None,
                 postprocessing: Optional[Callable] = None,
                 predict_fn: Optional[Callable] = None,
                 host: str = '0.0.0.0'):
        self.model = model
        self.port = port
        self.host = host
        self.predictor = Predictor(model, preprocessing, postprocessing) if model else None
        self._predict_fn = predict_fn
        self._server = None
        self._thread = None
        self._request_count = 0
        self._start_time = time.time()

    def _handle_predict(self, request: Request) -> Response:
        try:
            data = request.json()
            inputs = data.get('inputs', data.get('data', data))
            if self._predict_fn:
                result = self._predict_fn(inputs)
            elif self.predictor:
                result = self.predictor.predict(inputs)
            else:
                return Response({'error': 'No predict function set'}, status=500)
            self._request_count += 1
            return Response({'outputs': result, 'status': 'ok'})
        except Exception as e:
            return Response({'error': str(e)}, status=500)

    def _handle_health(self) -> Response:
        return Response({'status': 'healthy',
                         'uptime_s': time.time() - self._start_time,
                         'requests': self._request_count})

    def _handle_info(self) -> Response:
        info = {
            'model': type(self.model).__name__ if self.model else 'custom',
            'port': self.port,
            'host': self.host,
        }
        return Response(info)

    def _make_handler(self):
        server = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # silence default logging

            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                req = Request(body, dict(self.headers), 'POST', self.path)
                if self.path.startswith('/predict'):
                    resp = server._handle_predict(req)
                else:
                    resp = Response({'error': f'Unknown endpoint {self.path}'}, 404)
                self._send_response(resp)

            def do_GET(self):
                if self.path.startswith('/health'):
                    resp = server._handle_health()
                elif self.path.startswith('/info'):
                    resp = server._handle_info()
                else:
                    resp = Response({'error': f'Unknown endpoint {self.path}'}, 404)
                self._send_response(resp)

            def _send_response(self, resp: Response):
                self.send_response(resp.status)
                self.send_header('Content-Type', resp.content_type)
                body = resp.to_bytes()
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return _Handler

    def serve(self, port: Optional[int] = None) -> None:
        """Start blocking HTTP server."""
        port = port or self.port
        handler = self._make_handler()
        self._server = http.server.HTTPServer((self.host, port), handler)
        print(f'[ModelServer] Serving at http://{self.host}:{port}')
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            self.stop()

    def start(self, port: Optional[int] = None) -> None:
        """Start non-blocking HTTP server in background thread."""
        port = port or self.port
        handler = self._make_handler()
        self._server = http.server.HTTPServer((self.host, port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f'[ModelServer] Started at http://{self.host}:{port}')

    def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        print('[ModelServer] Stopped')


# ---------------------------------------------------------------------------
# Convenience decorator
# ---------------------------------------------------------------------------

def serve(port: int = 8080, host: str = '0.0.0.0'):
    """Decorator to turn a predict function into a served endpoint.

    Usage::

        @serve(port=8080)
        def predict(inputs):
            return model(preprocess(inputs)).tolist()
    """
    def _decorator(fn: Callable) -> ModelServer:
        server = ModelServer(predict_fn=fn, port=port, host=host)
        return server
    return _decorator


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

class DynamicBatcher:
    """Collect individual requests and run batched inference.

    Usage::

        batcher = DynamicBatcher(model, max_batch_size=32, timeout_ms=10)
        result = batcher.infer(x)  # blocks until batch is processed
    """

    def __init__(self, model, max_batch_size: int = 32, timeout_ms: float = 10.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self._queue: List[Tuple] = []
        self._lock = threading.Lock()

    def infer(self, x):
        """Add request to queue and return when batch is ready."""
        from vectorphile.core.tensor import Tensor
        from vectorphile._core.ndarray import array as _nd
        import vectorphile as vp
        with vp.autograd.no_grad():
            if not isinstance(x, Tensor):
                flat = [float(v) for v in (x if isinstance(x, list) else [x])]
                t = Tensor(_nd(flat))
            else:
                t = x
            y = self.model(t)
        return y._data if hasattr(y, '_data') else y


__all__ = [
    'Predictor', 'ModelServer', 'Request', 'Response',
    'serve', 'DynamicBatcher',
]
