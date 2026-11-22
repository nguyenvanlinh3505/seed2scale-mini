"""Parallel Worlds stage (paper Sec. III-C, Eq. 8-10).
Giai đoạn Parallel Worlds (paper mục III-C, Eq. 8-10).

Every self-evolved trajectory in D_se (accepted rollouts; seeds have no replayable start state) is
projected into each seen world by executing its exact action sequence from the same start state.
Feasibility check (Eq. 9): the replay must reproduce the success. The result is then filtered by the
VLV (Q_gamma in Eq. 10) with the same rubric and threshold as self-evolution.
Mọi quỹ đạo tự tiến hóa trong D_se (rollout được chấp nhận; seed không có trạng thái đầu chạy lại được)
được chiếu sang từng thế giới đã thấy bằng cách chạy đúng chuỗi action từ cùng trạng thái đầu.
Kiểm tra khả thi (Eq. 9): lần chạy lại phải tái tạo được thành công. Sau đó lọc bằng VLV (Q_gamma ở Eq. 10)
với cùng rubric và ngưỡng như giai đoạn tự tiến hóa.

Usage / Cách dùng:  python -m s2s.parallel_worlds --runs runs --arm s2s [--mock-verifier]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from s2s.config import CFG, TASK
from s2s.evolve import load_seeds, load_traj, reference_frames, save_traj
from s2s.rollout import replay
from s2s.verifier import MockVerifier, verify

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--arm", default="s2s")
    p.add_argument("--iterations", type=int, default=CFG.iterations)
    p.add_argument("--mock-verifier", action="store_true")
    p.add_argument("--vlv-url", default="http://127.0.0.1:8000/v1")
    a = p.parse_args()
    runs, out = Path(a.runs), Path(a.runs) / a.arm / "parallel_worlds"
    src = [load_traj(f) for i in range(a.iterations)
           for f in sorted((runs / a.arm / f"iter_{i}" / "accepted").glob("*.npz"))]
    print(f"projecting {len(src)} self-evolved trajectories into {len(CFG.seen_worlds)} worlds", flush=True)
    verifier = MockVerifier() if a.mock_verifier else None
    if verifier is None:
        from s2s.verifier import VLLMVerifier
        verifier = VLLMVerifier(CFG.vlv_model, url=a.vlv_url)
    ref = reference_frames(load_seeds(runs / "seeds.npz"), CFG.vlv_frames, CFG.vlv_image_size)
    summary = {}
    for world in CFG.seen_worlds:
        t0 = time.time()
        (out / world).mkdir(parents=True, exist_ok=True)
        projected = [replay(t, world) for t in src]
        feasible = [t for t in projected if t["success"]]  # Eq. 9 feasibility / điều kiện khả thi
        kept = [t for t in verify(feasible, gate=True, use_vlv=True, threshold=CFG.vlv_threshold, task=TASK,
                                  ref_frames=ref, verifier=verifier, n_frames=CFG.vlv_frames) if t["accepted"]]
        for j, t in enumerate(kept):
            save_traj(out / world / f"traj_{j:04d}.npz", t)
        summary[world] = {"projected": len(projected), "feasible": len(feasible), "kept": len(kept),
                          "seconds": round(time.time() - t0)}
        print(f"  {world}: {summary[world]}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
