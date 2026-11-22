"""Two-stage verification (paper Sec. III-B): rule-based success gate, then a frozen VLM (VLV) scores
execution quality 0-10 from (task instruction, rollout video, reference video from the seeds).
Kiểm định hai tầng (paper mục III-B): cổng thành công theo luật, rồi VLM đóng băng (VLV) chấm chất lượng
0-10 từ (câu lệnh, video rollout, video tham chiếu từ seed).

The threshold gamma and the rubric below are fixed BEFORE any run and never tuned on results,
as the paper requires. / Ngưỡng gamma và rubric dưới đây được cố định TRƯỚC mọi lần chạy và không bao
giờ chỉnh theo kết quả, đúng như paper yêu cầu.
"""
from __future__ import annotations

import base64
import io
import re

import numpy as np
from PIL import Image

RUBRIC = """You are a strict judge of robot manipulation quality.
Task: "{task}"
The FIRST {n_ref} images are a reference demonstration of good execution, in time order.
The NEXT {n_roll} images are the attempt to evaluate, in time order. The attempt already reached the goal;
judge HOW WELL it was executed, comparing with the reference.

Score 0-10 with this rubric:
10: direct, purposeful pushes; no wasted motion; T-block slides smoothly onto the target and stays aligned.
8-9: efficient with minor hesitation or one small correction.
6-7: succeeds but with noticeable detours, repeated re-pushes, or jittery back-and-forth motion.
3-5: mostly aimless or chaotic motion; success looks accidental.
0-2: erratic, collides wildly, or the images do not show a coherent attempt.

Reply with a one-sentence reason, then the score as <SCORE>N</SCORE>."""

_SCORE_RE = re.compile(r"<SCORE>\s*(\d+(?:\.\d+)?)\s*</SCORE>")


def parse_score(text: str) -> float | None:
    """Last <SCORE>N</SCORE> clipped to [0,10]; None if missing.
    Lấy <SCORE>N</SCORE> cuối cùng, kẹp vào [0,10]; None nếu không có."""
    m = _SCORE_RE.findall(text)
    return min(10.0, max(0.0, float(m[-1]))) if m else None


def sample_frames(frames: np.ndarray, k: int) -> list[np.ndarray]:
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return [frames[i] for i in idx]


def _data_url(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_messages(task: str, ref_frames: list[np.ndarray], roll_frames: list[np.ndarray]) -> list[dict]:
    content = [{"type": "text", "text": RUBRIC.format(task=task, n_ref=len(ref_frames), n_roll=len(roll_frames))}]
    content += [{"type": "image_url", "image_url": {"url": _data_url(f)}} for f in ref_frames + roll_frames]
    return [{"role": "user", "content": content}]


class VLLMVerifier:
    """Client for a vLLM OpenAI-compatible server running the VLV (started by s2s/pipeline.py in its own
    virtualenv, because vLLM and LeRobot pin different torch versions). Requests run concurrently.
    Client gọi server vLLM (tương thích OpenAI) chạy VLV (s2s/pipeline.py khởi động nó trong virtualenv
    riêng, vì vLLM và LeRobot ghim các phiên bản torch khác nhau). Các request chạy song song."""

    def __init__(self, model: str, url: str = "http://127.0.0.1:8000/v1", workers: int = 16, timeout: int = 600):
        self.model, self.url, self.workers, self.timeout = model, url.rstrip("/"), workers, timeout

    def _one(self, messages: list[dict]) -> tuple[float | None, str]:
        import requests
        for attempt in range(3):
            try:
                r = requests.post(f"{self.url}/chat/completions", timeout=self.timeout, json={
                    "model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": 256, "seed": 0})
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"] or ""
                return parse_score(text), text
            except Exception as e:  # noqa: BLE001 - retry any transport error / thử lại mọi lỗi kết nối
                err = repr(e)
        return None, f"ERROR: {err}"

    def score(self, batch: list[list[dict]]) -> list[tuple[float | None, str]]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, batch))


class MockVerifier:
    """Deterministic stand-in for CPU tests: scores smoothness of the agent path (no GPU).
    Bản thay thế tất định cho test trên CPU: chấm theo độ mượt đường đi (không cần GPU)."""

    def score_traj(self, traj: dict) -> float:
        a = traj["actions"]
        jitter = np.abs(np.diff(a, 2, axis=0)).mean() if len(a) > 2 else 0.0
        return float(np.clip(10 - jitter / 2, 0, 10))


def verify(trajs: list[dict], gate: bool, use_vlv: bool, threshold: float, task: str,
           ref_frames: list[np.ndarray] | None, verifier=None, n_frames: int = 8, log=print) -> list[dict]:
    """Annotate each trajectory with `gate_pass`, `vlv_score`, `vlv_text`, `accepted`.
    Gắn cho mỗi quỹ đạo các trường `gate_pass`, `vlv_score`, `vlv_text`, `accepted`.

    arms / các nhánh: gate=False,use_vlv=False -> keep everything (no filtering)
                      gate=True, use_vlv=False -> SuperTiny^- (success gate only)
                      gate=True, use_vlv=True  -> Seed2Scale (gate + VLV)"""
    from s2s.rollout import replay

    for t in trajs:
        t["gate_pass"] = bool(t["success"])
        t["vlv_score"], t["vlv_text"] = None, ""
        t["accepted"] = t["gate_pass"] if gate else True
    if not use_vlv:
        return trajs
    cand = [t for t in trajs if t["gate_pass"]]
    if not cand:
        return trajs
    if isinstance(verifier, MockVerifier):
        for t in cand:
            t["vlv_score"] = verifier.score_traj(t)
    else:
        # Re-render the successful trajectories at VLV resolution by exact replay (cheaper than rendering all).
        # Render lại các quỹ đạo thành công ở độ phân giải cho VLV bằng cách chạy lại chính xác.
        batch = []
        for t in cand:
            frames = replay(t, t["world"], render_every=max(1, len(t["actions"]) // (4 * n_frames)))["frames"]
            batch.append(build_messages(task, ref_frames, sample_frames(frames, n_frames)))
        for t, (s, text) in zip(cand, verifier.score(batch)):
            t["vlv_score"], t["vlv_text"] = s, text
    for t in cand:
        t["accepted"] = t["vlv_score"] is not None and t["vlv_score"] >= threshold
    scores = [t["vlv_score"] for t in cand]
    log(f"    VLV scored {len(cand)} successes: mean={np.mean([s for s in scores if s is not None] or [0]):.2f}, "
        f"accepted={sum(t['accepted'] for t in cand)}, unparsable={sum(s is None for s in scores)}")
    return trajs
