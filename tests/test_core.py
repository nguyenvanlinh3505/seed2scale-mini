"""CPU tests for the pieces that must be exactly right before spending GPU hours.
Test trên CPU cho các phần phải đúng tuyệt đối trước khi tốn giờ GPU.  Run / Chạy: pytest -q"""
import numpy as np
import pytest
import torch

from s2s.metrics import smoothness
from s2s.rollout import replay
from s2s.seeds import CORNERS_96, block_centroid, pick_corner_episodes
from s2s.supertiny import ChunkDataset, SuperTiny, TemporalEnsembler, from_unit, n_params, to_unit
from s2s.verifier import MockVerifier, parse_score, verify
from s2s.worlds import WORLDS, make_env

# A start state whose block already covers the goal (found by grid search): success on the first step.
# Trạng thái đầu có khối T đã phủ vùng đích (tìm bằng dò lưới): thành công ngay bước đầu.
SUCCESS_STATE = np.array([100, 450, 226, 241, np.pi / 4])


def _traj(init_state, actions, world="source"):
    return {"init_state": np.asarray(init_state, float), "actions": np.asarray(actions, np.float32),
            "world": world, "reset_seed": 0}


# ---------- worlds / các thế giới ----------
def test_replay_is_identical_across_worlds():
    rng = np.random.default_rng(0)
    src = make_env("source")
    src.reset(seed=11)
    t = _traj(src.init_state, rng.uniform(100, 400, (60, 2)))
    ref = replay(t, "source")
    for w in WORLDS:
        r = replay(t, w)
        np.testing.assert_allclose(r["coverage"], ref["coverage"])
        np.testing.assert_allclose(r["states"], ref["states"])
        if w != "source":
            assert np.abs(r["images"][0].astype(int) - ref["images"][0]).mean() > 5  # looks different


def test_same_seed_same_start_and_success_state():
    a, b = make_env(), make_env()
    a.reset(seed=5), b.reset(seed=5)
    np.testing.assert_array_equal(a.init_state, b.init_state)
    r = replay(_traj(SUCCESS_STATE, [[100, 450]]), "w3")
    assert r["success"]


# ---------- seeds ----------
def test_pick_corner_episodes():
    frames = {}
    for ep, (cx, cy) in enumerate(list(CORNERS_96) + [(48, 48)]):
        img = np.full((96, 96, 3), 255, np.uint8)
        img[int(cy) - 3:int(cy) + 3, int(cx) - 3:int(cx) + 3] = (119, 136, 153)
        frames[ep] = img
    assert pick_corner_episodes(frames) == [0, 1, 2, 3]
    assert block_centroid(np.full((96, 96, 3), 255, np.uint8)) is None


# ---------- SuperTiny ----------
def test_supertiny_shapes_and_scaling():
    m = SuperTiny(chunk_size=30, pretrained=False).eval()
    y = m(torch.rand(3, 3, 96, 96), torch.zeros(3, 2))
    assert y.shape == (3, 30, 2) and y.abs().max() <= 1
    x = torch.tensor([0.0, 256.0, 512.0])
    torch.testing.assert_close(from_unit(to_unit(x)), x)


def test_supertiny_size_matches_paper():
    total, trainable = n_params(SuperTiny(task="push"))
    assert 44e6 < total < 55e6  # paper: 48M


def test_temporal_ensembling_eq3():
    te = TemporalEnsembler(1, 3, lam=0.5)
    te(0, np.array([[0.0], [1.0], [2.0]]))
    out = te(0, np.array([[10.0], [11.0], [12.0]]))
    w = np.exp(-0.5 * np.arange(2))
    np.testing.assert_allclose(out, (w[0] * 10 + w[1] * 1) / w.sum())


def test_chunk_dataset_pads_and_masks_episode_end():
    T = 5
    tr = {"images": np.zeros((T, 96, 96, 3), np.uint8), "states": np.zeros((T, 2), np.float32),
          "actions": np.arange(T * 2, dtype=np.float32).reshape(T, 2)}
    _, _, act, mask = ChunkDataset([tr], chunk_size=4)[3]
    assert mask.tolist() == [1, 1, 0, 0]
    torch.testing.assert_close(act[2], act[1])  # holds the last action / giữ action cuối


# ---------- verifier ----------
def test_parse_score():
    assert parse_score("good <SCORE>8</SCORE>") == 8
    assert parse_score("<SCORE>3</SCORE> then <SCORE>12</SCORE>") == 10
    assert parse_score("no score") is None


@pytest.mark.parametrize("gate,vlv,expected", [(False, False, [True, True, True]),
                                               (True, False, [True, True, False]),
                                               (True, True, [True, False, False])])
def test_verify_arms(gate, vlv, expected):
    smooth = np.linspace([100, 100], [300, 300], 50)
    jittery = smooth + np.random.default_rng(0).normal(0, 40, smooth.shape)
    trajs = [dict(_traj(SUCCESS_STATE, smooth), success=True),
             dict(_traj(SUCCESS_STATE, jittery), success=True),
             dict(_traj(SUCCESS_STATE, smooth), success=False)]
    out = verify(trajs, gate, vlv, threshold=7, task="t", ref_frames=None, verifier=MockVerifier(), log=lambda s: None)
    assert [t["accepted"] for t in out] == expected


def test_smoothness_metrics_rank_jitter_higher():
    smooth = np.linspace([100, 100], [300, 300], 100)
    jittery = smooth + np.random.default_rng(0).normal(0, 10, smooth.shape)
    s, j = smoothness(smooth), smoothness(jittery)
    assert j["tv"] > s["tv"] and j["jerk"] > s["jerk"] and j["hf_ratio"] > s["hf_ratio"]


# ---------- run-2 fixes: more seeds, auto seed count / sửa lỗi lần chạy 2 ----------
def test_extra_seeds_spread_out_after_corners():
    rng = np.random.default_rng(0)
    frames = {}
    pts = list(CORNERS_96) + [(48, 48), (49, 49), (20, 50)] + [tuple(x) for x in rng.uniform(15, 80, (10, 2))]
    for ep, (cx, cy) in enumerate(pts):
        img = np.full((96, 96, 3), 255, np.uint8)
        img[int(cy) - 3:int(cy) + 3, int(cx) - 3:int(cx) + 3] = (119, 136, 153)
        frames[ep] = img
    chosen = pick_corner_episodes(frames, 6)
    assert chosen[:4] == [0, 1, 2, 3] and len(set(chosen)) == 6
    assert not ({4, 5} <= set(chosen))  # two near-duplicates are never both picked / không lấy cả hai điểm gần trùng


def test_pick_n_seeds_and_layout(tmp_path):
    import json
    from s2s.pipeline import Work, layout, pick_n_seeds
    w = Work(tmp_path, None)
    (tmp_path / "state" / "calibrate.done").write_text(json.dumps({"4": 0.0, "8": 0.01, "16": 0.05, "32": 0.2}))
    assert pick_n_seeds(w, 0.03) == 16
    assert layout(4) == ("runs", "") and layout(16) == ("runs_n16", "_n16")


def test_research_registry_and_metric():
    from s2s.experiments import EXPERIMENTS, evolve_args, late_gain_pp
    assert abs(late_gain_pp([0.06, 0.05, 0.10, 0.12]) - 5.0) < 1e-9
    a = evolve_args(EXPERIMENTS["E3"])
    assert a[a.index("--train-mode") + 1] == "scratch" and a[a.index("--vlv-threshold") + 1] == "6"
    assert all(c["arm"] in ("s2s", "gate_only") for c in EXPERIMENTS.values())
