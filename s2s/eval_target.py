"""Evaluate a trained LeRobot policy in one or more worlds with fixed seeds, using LeRobot's own rollout loop.
Đánh giá một policy LeRobot đã train trên một hoặc nhiều thế giới với seed cố định, dùng chính vòng rollout của LeRobot.

Usage / Cách dùng:
  python -m s2s.eval_target --ckpt targets/s2s_iter4/checkpoints/last/pretrained_model --worlds source u1 u2 u3 \
      --out results/eval_s2s_iter4.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import einops
import gymnasium as gym
import numpy as np
import torch

from s2s.config import CFG

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ENV_ID = "s2s/WorldPushT-v0"
if ENV_ID not in gym.registry:
    gym.register(ENV_ID, entry_point="s2s.worlds:WorldPushT", max_episode_steps=300)


def make_vec_env(world: str, n: int) -> gym.vector.SyncVectorEnv:
    return gym.vector.SyncVectorEnv([lambda: gym.make(ENV_ID, world=world, render_mode="rgb_array")
                                     for _ in range(n)])


def load_policy(ckpt: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import PushtEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(ckpt),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=PushtEnvConfig(), policy_cfg=cfg)
    return policy, pre, post, env_pre, env_post


def evaluate(ckpt: Path, worlds: list[str], n_episodes: int, batch: int, seed0: int, device: str) -> dict:
    from lerobot.scripts.lerobot_eval import rollout
    policy, pre, post, env_pre, env_post = load_policy(ckpt, device)
    results = {}
    for world in worlds:
        t0, successes, coverages = time.time(), [], []
        for start in range(0, n_episodes, batch):
            n = min(batch, n_episodes - start)
            env = make_vec_env(world, n)
            with torch.no_grad():
                data = rollout(env, policy, env_pre, env_post, pre, post,
                               seeds=list(range(seed0 + start, seed0 + start + n)))
            env.close()
            # Same masking as lerobot_eval: ignore steps after an env's first `done` (auto-reset).
            # Cùng cách che như lerobot_eval: bỏ các bước sau lần `done` đầu tiên (tự reset).
            steps = data["done"].shape[1]
            done_idx = torch.argmax(data["done"].to(int), dim=1)
            mask = (torch.arange(steps) <= einops.repeat(done_idx + 1, "b -> b s", s=steps)).int()
            successes += einops.reduce(data["success"] * mask, "b n -> b", "any").tolist()
            coverages += (data["reward"] * mask).max(dim=1).values.tolist()  # PushT reward = coverage
        sr = float(np.mean(successes))
        # Wilson 95% interval for a proportion / khoảng tin cậy Wilson 95% cho tỉ lệ
        z, n = 1.96, len(successes)
        c = (sr + z * z / (2 * n)) / (1 + z * z / n)
        h = z * np.sqrt(sr * (1 - sr) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        results[world] = {"success_rate": sr, "ci95": [c - h, c + h], "n": n,
                          "mean_max_coverage": float(np.mean(coverages)), "seconds": round(time.time() - t0)}
        print(f"  {world}: success={sr:.3f} [{c - h:.3f}, {c + h:.3f}] n={n} "
              f"coverage={results[world]['mean_max_coverage']:.3f}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--worlds", nargs="+", default=["source"])
    p.add_argument("--episodes", type=int, default=CFG.eval_episodes)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = evaluate(Path(a.ckpt), a.worlds, a.episodes, a.batch, CFG.eval_seed, device)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "results": res}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
