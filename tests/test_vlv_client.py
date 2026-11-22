"""The VLV client sends OpenAI-style multimodal requests and parses scores (fake local server, no GPU).
Client VLV gửi request đa phương thức kiểu OpenAI và đọc điểm (server giả cục bộ, không cần GPU)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from s2s.verifier import VLLMVerifier, build_messages

SEEN = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        n_img = sum(c["type"] == "image_url" for c in body["messages"][0]["content"])
        reply = {"choices": [{"message": {"content": f"looks fine <SCORE>{n_img}</SCORE>"}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_client_roundtrip():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        v = VLLMVerifier("qwen", url=f"http://127.0.0.1:{srv.server_port}/v1", workers=4)
        frame = np.zeros((32, 32, 3), np.uint8)
        msgs = [build_messages("push", [frame] * 2, [frame] * k) for k in (3, 5)]
        out = v.score(msgs)
        assert [s for s, _ in out] == [5.0, 7.0]  # 2 reference + k rollout images
        assert SEEN[0]["model"] == "qwen" and SEEN[0]["temperature"] == 0.0
        assert SEEN[0]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    finally:
        srv.shutdown()
