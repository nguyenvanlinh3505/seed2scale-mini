"""End-to-end on CPU: accepted trajectory -> Parallel Worlds projection -> LeRobotDataset export.
Đầu-cuối trên CPU: quỹ đạo được chấp nhận -> chiếu Parallel Worlds -> xuất LeRobotDataset."""
import json
import sys

import numpy as np

from s2s import parallel_worlds
from s2s.config import CFG, TASK
from s2s.evolve import save_traj
from s2s.export import build, gather
from s2s.rollout import replay
from tests.test_core import SUCCESS_STATE, _traj


def _fake_runs(tmp_path):
    seed = {"init_state": np.full(5, np.nan), "images": np.zeros((6, 96, 96, 3), np.uint8),
            "states": np.zeros((6, 2), np.float32), "actions": np.zeros((6, 2), np.float32),
            "coverage": np.zeros(6, np.float32), "success": True, "world": "source", "reset_seed": -1}
    np.savez_compressed(tmp_path / "seeds.npz", seeds=np.array([seed], dtype=object))
    it = tmp_path / "s2s" / "iter_0"
    (it / "accepted").mkdir(parents=True)
    t = replay(_traj(SUCCESS_STATE, [[100, 450], [110, 450]]), "source")
    t.pop("frames")
    assert t["success"]
    save_traj(it / "accepted" / "traj_0000.npz", t)
    (it / "summary.json").write_text("{}")
    return tmp_path


def test_parallel_worlds_then_export(tmp_path, monkeypatch):
    runs = _fake_runs(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pw", "--runs", str(runs), "--arm", "s2s", "--iterations", "1",
                                      "--mock-verifier"])
    parallel_worlds.main()
    summary = json.loads((runs / "s2s" / "parallel_worlds" / "summary.json").read_text())
    assert set(summary) == set(CFG.seen_worlds)
    assert all(v["projected"] == v["feasible"] == 1 for v in summary.values())  # physics preserved

    trajs = gather(runs, "s2s", 1, with_worlds=True)
    worlds = sorted({t["world"] for t in trajs})
    assert len(trajs) == 1 + 1 + sum(v["kept"] for v in summary.values())
    assert "source" in worlds and set(worlds) - {"source"} <= set(CFG.seen_worlds)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dst = build(trajs, tmp_path / "ds", "local/ds")
    ds = LeRobotDataset("local/ds", root=dst)
    assert ds.num_episodes == len(trajs)
    item = ds[0]
    assert tuple(item["observation.image"].shape) == (3, 96, 96) and item["task"] == TASK
