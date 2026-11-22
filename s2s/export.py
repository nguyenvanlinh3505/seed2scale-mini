"""Build a local LeRobotDataset from trajectories (D(i) for an arm and iteration, or D_final with worlds).
Tạo LeRobotDataset cục bộ từ các quỹ đạo (D(i) của một nhánh ở một vòng, hoặc D_final có thêm các thế giới).

Usage / Cách dùng:
  python -m s2s.export --runs runs --arm s2s --iteration 4 --dst datasets/s2s_iter4
  python -m s2s.export --runs runs --arm s2s --iteration 8 --with-worlds --dst datasets/s2s_full
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from s2s.config import FPS, TASK

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FEATURES = {
    "observation.image": {"dtype": "image", "shape": (96, 96, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
}


def gather(runs: Path, arm: str, iteration: int, with_worlds: bool) -> list[dict]:
    """Seeds + accepted trajectories of iterations 0..iteration-1, i.e. D(iteration) (Eq. 7),
    plus projected parallel-world trajectories if requested (Eq. 10).
    Seed + quỹ đạo được chấp nhận ở các vòng 0..iteration-1, tức D(iteration) (Eq. 7),
    cộng thêm quỹ đạo chiếu sang các thế giới song song nếu được yêu cầu (Eq. 10)."""
    from s2s.evolve import load_seeds, load_traj
    trajs = list(load_seeds(runs / "seeds.npz"))
    for i in range(iteration):
        d = runs / arm / f"iter_{i}"
        if not (d / "summary.json").exists():
            raise SystemExit(f"{d} is not finished / chưa xong")
        trajs += [load_traj(f) for f in sorted((d / "accepted").glob("*.npz"))]
    if with_worlds:
        pw = runs / arm / "parallel_worlds"
        if not (pw / "summary.json").exists():
            raise SystemExit(f"{pw} is not finished / chưa xong")
        trajs += [load_traj(f) for f in sorted(pw.glob("*/*.npz"))]
    return trajs


def build(trajs: list[dict], dst: Path, repo_id: str) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if dst.exists():
        shutil.rmtree(dst)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=FEATURES, root=dst, use_videos=False,
                               image_writer_threads=4)
    for t in trajs:
        for img, st, act in zip(t["images"], t["states"], t["actions"]):
            ds.add_frame({"observation.image": img, "observation.state": np.asarray(st, np.float32),
                          "action": np.asarray(act, np.float32), "task": TASK})
        ds.save_episode()
    ds.finalize()
    print(f"exported {len(trajs)} episodes / {sum(len(t['actions']) for t in trajs)} frames -> {dst}")
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--arm", required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--with-worlds", action="store_true")
    p.add_argument("--dst", required=True)
    a = p.parse_args()
    build(gather(Path(a.runs), a.arm, a.iteration, a.with_worlds), Path(a.dst), f"local/{Path(a.dst).name}")


if __name__ == "__main__":
    main()
