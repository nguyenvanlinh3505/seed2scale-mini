"""Batched rollouts of a policy in many PushT envs, plus exact replay of a trajectory in any world.
Rollout theo batch của policy trên nhiều env PushT, và chạy lại chính xác một quỹ đạo ở bất kỳ thế giới nào.

A trajectory is a dict / Một quỹ đạo là dict:
  init_state (5,)  exact start state, replayable / trạng thái đầu chính xác, chạy lại được
  images (T,96,96,3) uint8, states (T,2), actions (T,2)   obs BEFORE each action / obs TRƯỚC mỗi action
  coverage (T,)    coverage after each step / độ phủ sau mỗi bước
  success bool     env reported is_success / env báo thành công
  world str, reset_seed int
"""
from __future__ import annotations

import numpy as np
import torch

from s2s.supertiny import TemporalEnsembler, from_unit, to_unit
from s2s.worlds import make_env


@torch.no_grad()
def collect(model, n: int, seeds: list[int], device: str, chunk_size: int, lam: float,
            max_steps: int = 300, noise: float = 0.0, world: str = "source", log=print) -> list[dict]:
    """Run `n` episodes in parallel with the collector. Envs that finish stop stepping.
    Chạy song song `n` episode bằng collector. Env nào xong thì dừng."""
    envs = [make_env(world) for _ in range(n)]
    obs = [e.reset(seed=int(s))[0] for e, s in zip(envs, seeds)]
    trajs = [{"init_state": e.init_state.copy(), "images": [], "states": [], "actions": [], "coverage": [],
              "success": False, "world": world, "reset_seed": int(s)} for e, s in zip(envs, seeds)]
    ens = TemporalEnsembler(n, chunk_size, lam)
    active = np.ones(n, dtype=bool)
    model.to(device).eval()
    rng = np.random.default_rng(seeds[0])
    for step in range(max_steps):
        idx = np.nonzero(active)[0]
        if len(idx) == 0:
            break
        img = torch.from_numpy(np.stack([obs[i]["pixels"] for i in idx])).permute(0, 3, 1, 2).float().div(255)
        st = to_unit(torch.from_numpy(np.stack([obs[i]["agent_pos"] for i in idx]).astype(np.float32)))
        chunks = from_unit(model(img.to(device), st.to(device))).cpu().numpy()  # (b, K, 2) in px
        for j, i in enumerate(idx):
            a = ens(i, chunks[j])
            if noise > 0:
                a = a + rng.normal(0, noise, size=2)
            a = np.clip(a, 0, 512).astype(np.float32)
            tr = trajs[i]
            tr["images"].append(obs[i]["pixels"])
            tr["states"].append(obs[i]["agent_pos"].astype(np.float32))
            tr["actions"].append(a)
            obs[i], _, term, trunc, info = envs[i].step(a)
            tr["coverage"].append(float(info["coverage"]))
            if info.get("is_success"):
                tr["success"] = True
            if term or trunc or tr["success"]:
                active[i] = False
        if step % 50 == 0:
            log(f"    rollout step {step}: active={active.sum()}/{n}, "
                f"successes so far={sum(t['success'] for t in trajs)}")
    for e in envs:
        e.close()
    return [_finalize(t) for t in trajs]


def _finalize(t: dict) -> dict:
    t["images"] = np.stack(t["images"]).astype(np.uint8)
    t["states"] = np.stack(t["states"]).astype(np.float32)
    t["actions"] = np.stack(t["actions"]).astype(np.float32)
    t["coverage"] = np.asarray(t["coverage"], dtype=np.float32)
    return t


def replay(traj: dict, world: str, render_every: int = 0, vlv_size: int = 224) -> dict:
    """Execute a trajectory's exact action sequence from its init_state in `world`. Physics is identical
    across worlds, so this is the paper's projection M_m (Eq. 9) with a trivially satisfied feasibility check;
    we still verify it by checking the replay reaches the same success/coverage.
    Chạy lại đúng chuỗi action từ init_state trong `world`. Vật lý giống nhau giữa các thế giới, nên đây
    là phép chiếu M_m (Eq. 9) với điều kiện khả thi luôn thỏa; vẫn kiểm tra lại success/coverage."""
    env = make_env(world, vlv_size=vlv_size)
    obs, _ = env.reset(options={"reset_to_state": traj["init_state"]})
    out = {"init_state": traj["init_state"].copy(), "images": [], "states": [], "actions": [], "coverage": [],
           "success": False, "world": world, "reset_seed": traj.get("reset_seed", -1), "frames": []}
    for t, a in enumerate(traj["actions"]):
        if render_every and t % render_every == 0:
            out["frames"].append(env.render())
        out["images"].append(obs["pixels"])
        out["states"].append(obs["agent_pos"].astype(np.float32))
        out["actions"].append(a)
        obs, _, term, trunc, info = env.step(a)
        out["coverage"].append(float(info["coverage"]))
        out["success"] = out["success"] or bool(info.get("is_success"))
    if render_every:
        out["frames"].append(env.render())
    env.close()
    frames = out.pop("frames")
    out = _finalize(out)
    out["frames"] = np.stack(frames) if frames else None
    return out
