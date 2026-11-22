"""Test models on N random variants (random start state + random colour palette), with GIFs and a table.
Thử model trên N biến thể ngẫu nhiên (vị trí đầu ngẫu nhiên + bảng màu ngẫu nhiên), kèm GIF và bảng.

Each variant gets a fresh palette (background, goal, T-block, agent) never used in training, and a start state
from reset seeds 900000+ (disjoint from training rollouts 1000000+ and the target eval 100000+). Both models see
exactly the same variants.
Mỗi biến thể có bảng màu mới chưa từng dùng khi train, và trạng thái đầu từ seed 900000+ (tách biệt với rollout
khi train 1000000+ và bước đánh giá model mục tiêu 100000+). Cả hai model thấy đúng cùng các biến thể.

Usage / Cách dùng:
  python -m s2s.variants_eval --collector runs_n32/gate_only/iter_7/collector.pt \
      --smolvla targets/seed_only_n32/checkpoints/last/pretrained_model --n 10 --out results/variants
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from s2s.config import CFG, TASK

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

GIF_SIZE = 224


def random_palette(rng: np.random.Generator, min_dist: float = 90.0) -> tuple:
    """Four mutually distinct RGB colours (background, goal, block, agent).
    Bốn màu RGB khác biệt rõ với nhau (nền, đích, khối T, tác tử)."""
    while True:
        cols = rng.integers(0, 256, size=(4, 3))
        d = np.linalg.norm(cols[:, None].astype(float) - cols[None], axis=-1)
        if d[np.triu_indices(4, 1)].min() >= min_dist:
            return tuple(tuple(int(v) for v in c) for c in cols)


def make_variants(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{"id": k, "reset_seed": 900_000 + seed * 1000 + k, "palette": random_palette(rng)} for k in range(n)]


def save_gif(frames: list[np.ndarray], path: Path, every: int = 2):
    imgs = [Image.fromarray(f).resize((GIF_SIZE, GIF_SIZE)) for f in frames[::every]] or [Image.new("RGB", (8, 8))]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100 * every, loop=0)


@torch.no_grad()
def run_collector(ckpt: Path, variants: list[dict], out: Path, device: str) -> list[dict]:
    from s2s.supertiny import SuperTiny, TemporalEnsembler, from_unit, to_unit
    from s2s.worlds import make_env
    model = SuperTiny(CFG.chunk_size, CFG.d_model, CFG.n_decoder_layers, task=TASK)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.to(device).eval()
    rows = []
    for v in variants:
        env = make_env(vlv_size=GIF_SIZE, palette=v["palette"])
        obs, _ = env.reset(seed=v["reset_seed"])
        ens, frames, cov, success = TemporalEnsembler(1, CFG.chunk_size, CFG.ensemble_lambda), [env.render()], [], False
        for _ in range(CFG.max_steps):
            img = torch.from_numpy(obs["pixels"]).permute(2, 0, 1)[None].float().div(255).to(device)
            st = to_unit(torch.from_numpy(obs["agent_pos"].astype(np.float32)))[None].to(device)
            a = np.clip(ens(0, from_unit(model(img, st)).cpu().numpy()[0]), 0, 512).astype(np.float32)
            obs, _, term, trunc, info = env.step(a)
            frames.append(env.render())
            cov.append(float(info["coverage"]))
            success = success or bool(info.get("is_success"))
            if term or trunc or success:
                break
        save_gif(frames, out / f"collector_v{v['id']}.gif")
        rows.append({"model": "collector", "variant": v["id"], "success": success,
                     "max_coverage": max(cov), "steps": len(cov)})
        print(f"  collector v{v['id']}: success={success} max_coverage={max(cov):.2f} steps={len(cov)}", flush=True)
    return rows


def run_smolvla(ckpt: Path, variants: list[dict], out: Path, device: str) -> list[dict]:
    import gymnasium as gym
    from lerobot.scripts.lerobot_eval import rollout

    from s2s.eval_target import ENV_ID, load_policy
    policy, pre, post, env_pre, env_post = load_policy(ckpt, device)
    rows = []
    for v in variants:
        env = gym.vector.SyncVectorEnv([lambda v=v: gym.make(ENV_ID, palette=v["palette"], render_mode="rgb_array",
                                                             visualization_width=GIF_SIZE,
                                                             visualization_height=GIF_SIZE)])
        frames: list[np.ndarray] = []
        with torch.no_grad():
            data = rollout(env, policy, env_pre, env_post, pre, post, seeds=[v["reset_seed"]],
                           render_callback=lambda e: frames.append(e.envs[0].render()))
        env.close()
        done = data["done"][0].to(int)
        last = int(torch.argmax(done)) if done.any() else len(done) - 1
        success = bool(data["success"][0][: last + 1].any())
        max_cov = float(data["reward"][0][: last + 1].max())  # PushT reward = coverage / phần thưởng = độ phủ
        save_gif(frames[: last + 2], out / f"smolvla_v{v['id']}.gif")
        rows.append({"model": "smolvla", "variant": v["id"], "success": success,
                     "max_coverage": max_cov, "steps": last + 1})
        print(f"  smolvla v{v['id']}: success={success} max_coverage={max_cov:.2f} steps={last + 1}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collector", default=None, help="SuperTiny collector.pt")
    p.add_argument("--smolvla", default=None, help="LeRobot pretrained_model directory")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/variants")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variants = make_variants(a.n, a.seed)
    t0, rows = time.time(), []
    if a.collector:
        print(f"collector: {a.collector}", flush=True)
        rows += run_collector(Path(a.collector), variants, out, device)
    if a.smolvla:
        print(f"smolvla: {a.smolvla}", flush=True)
        rows += run_smolvla(Path(a.smolvla), variants, out, device)
    summary = {m: {"success_rate": float(np.mean([r["success"] for r in rows if r["model"] == m])),
                   "mean_max_coverage": float(np.mean([r["max_coverage"] for r in rows if r["model"] == m]))}
               for m in sorted({r["model"] for r in rows})}
    (out / "results.json").write_text(json.dumps({"variants": variants, "rows": rows, "summary": summary,
                                                  "seconds": round(time.time() - t0)}, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
