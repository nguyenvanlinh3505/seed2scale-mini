"""Self-evolution stage (paper Sec. III-C, Eq. 5-7) for one arm, with resume after interruption.
Giai đoạn tự tiến hóa (paper mục III-C, Eq. 5-7) cho một nhánh, chạy tiếp được sau khi bị ngắt.

  D(0) = D_seed
  for i in 0..I-1:  train collector pi(i) on D(i); roll out N episodes; filter; D(i+1) = D(i) U D_silver(i)

Arms / Các nhánh:
  s2s        success gate + VLV (Seed2Scale*)                       / cổng thành công + VLV
  gate_only  success gate only (the paper's "SuperTiny^-" ablation) / chỉ cổng thành công
  no_filter  keep every rollout (tests the collapse claim; not in the paper) / giữ mọi rollout

Layout / Cấu trúc thư mục:  <out>/<arm>/iter_<i>/{collector.pt, meta.jsonl, summary.json, accepted/*.npz}
Usage / Cách dùng:  python -m s2s.evolve --arm s2s --out runs [--mock-verifier] [--tiny]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from s2s.config import CFG, TASK
from s2s.metrics import smoothness
from s2s.rollout import collect
from s2s.supertiny import SuperTiny, train_collector
from s2s.verifier import MockVerifier, sample_frames, verify

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ARMS = {"s2s": (True, True), "gate_only": (True, False), "no_filter": (False, False)}
TRAJ_KEYS = ("init_state", "images", "states", "actions", "coverage")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_traj(path: Path, t: dict):
    np.savez_compressed(path, **{k: t[k] for k in TRAJ_KEYS}, success=t["success"], world=t["world"],
                        reset_seed=t["reset_seed"])


def load_traj(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    t = {k: z[k] for k in TRAJ_KEYS}
    t.update(success=bool(z["success"]), world=str(z["world"]), reset_seed=int(z["reset_seed"]))
    return t


def load_seeds(cache: Path) -> list[dict]:
    """Seed episodes as trajectories (cached). Seeds come from the dataset, so they have no replayable
    init_state (NaN) and are never projected into other worlds.
    Các episode seed ở dạng quỹ đạo (có cache). Seed lấy từ dataset nên không có init_state chạy lại
    được (NaN) và không được chiếu sang thế giới khác."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["seeds"])
    from s2s.seeds import load_seed_episodes
    seeds = [{"init_state": np.full(5, np.nan), "images": e["images"], "states": e["states"],
              "actions": e["actions"], "coverage": np.full(len(e["actions"]), np.nan, dtype=np.float32),
              "success": True, "world": "source", "reset_seed": -1, "episode_index": e["episode_index"]}
             for e in load_seed_episodes(CFG.seed_dataset)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, seeds=np.array(seeds, dtype=object))
    return seeds


def reference_frames(seeds: list[dict], n: int, size: int) -> list[np.ndarray]:
    """Reference video for the VLV: one seed demo, upsampled to the VLV frame size.
    Video tham chiếu cho VLV: một demo seed, phóng to về kích thước khung của VLV."""
    from PIL import Image
    return [np.asarray(Image.fromarray(f).resize((size, size), Image.BICUBIC))
            for f in sample_frames(seeds[0]["images"], n)]


def rollout_seeds(iteration: int, n: int) -> list[int]:
    """Same start states for every arm at a given iteration (fair comparison).
    Cùng trạng thái đầu cho mọi nhánh ở cùng một vòng (so sánh công bằng)."""
    return [1_000_000 + iteration * 10_000 + j for j in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=list(ARMS), required=True)
    p.add_argument("--out", default="runs")
    p.add_argument("--iterations", type=int, default=CFG.iterations)
    p.add_argument("--n-rollouts", type=int, default=CFG.n_rollouts)
    p.add_argument("--mock-verifier", action="store_true", help="CPU test without a VLM / test CPU không cần VLM")
    p.add_argument("--tiny", action="store_true", help="smoke test: tiny budgets / chạy thử với ngân sách rất nhỏ")
    p.add_argument("--vlv-url", default="http://127.0.0.1:8000/v1")
    args = p.parse_args()

    cfg = CFG
    if args.tiny:
        cfg = replace(CFG, collector_steps_first=20, collector_steps=10, max_steps=40, collector_batch=16)
    gate, use_vlv = ARMS[args.arm]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out) / args.arm
    out.mkdir(parents=True, exist_ok=True)
    log(f"arm={args.arm} gate={gate} vlv={use_vlv} device={device} iterations={args.iterations} "
        f"rollouts={args.n_rollouts}")

    seeds = load_seeds(Path(args.out) / "seeds.npz")
    data = list(seeds)
    model = SuperTiny(cfg.chunk_size, cfg.d_model, cfg.n_decoder_layers, task=TASK)

    # Resume: reload everything accepted by finished iterations / Chạy tiếp: nạp lại mọi thứ đã chấp nhận
    start = 0
    while (out / f"iter_{start}" / "summary.json").exists():
        data += [load_traj(f) for f in sorted((out / f"iter_{start}" / "accepted").glob("*.npz"))]
        start += 1
    if start > 0:
        model.load_state_dict(torch.load(out / f"iter_{start - 1}" / "collector.pt", map_location="cpu"),
                              strict=False)
        log(f"resuming at iteration {start} with |D|={len(data)} trajectories")

    verifier = None
    if use_vlv and start < args.iterations:
        verifier = MockVerifier() if args.mock_verifier else None
        if verifier is None:
            from s2s.verifier import VLLMVerifier
            verifier = VLLMVerifier(cfg.vlv_model, url=args.vlv_url)
    ref = reference_frames(seeds, cfg.vlv_frames, cfg.vlv_image_size)

    for i in range(start, args.iterations):
        t0 = time.time()
        d = out / f"iter_{i}"
        (d / "accepted").mkdir(parents=True, exist_ok=True)
        steps = cfg.collector_steps_first if i == 0 else cfg.collector_steps
        log(f"iteration {i}: training collector on |D({i})|={len(data)} trajectories for {steps} steps")
        losses = train_collector(model, data, steps, cfg.collector_lr, cfg.collector_batch, device, log=log)
        torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("t5.")}, d / "collector.pt")

        log(f"iteration {i}: {args.n_rollouts} rollouts")
        trajs = collect(model, args.n_rollouts, rollout_seeds(i, args.n_rollouts), device, cfg.chunk_size,
                        cfg.ensemble_lambda, cfg.max_steps, cfg.exploration_noise, log=log)
        trajs = verify(trajs, gate, use_vlv, cfg.vlv_threshold, TASK, ref, verifier, cfg.vlv_frames, log=log)

        accepted = [t for t in trajs if t["accepted"]]
        for j, t in enumerate(accepted):
            save_traj(d / "accepted" / f"traj_{j:04d}.npz", t)
        with open(d / "meta.jsonl", "w", encoding="utf-8") as f:
            for t in trajs:
                f.write(json.dumps({
                    "reset_seed": t["reset_seed"], "success": t["success"], "len": len(t["actions"]),
                    "max_coverage": float(t["coverage"].max()), "final_coverage": float(t["coverage"][-1]),
                    "gate_pass": t["gate_pass"], "vlv_score": t["vlv_score"], "vlv_text": t["vlv_text"],
                    "accepted": t["accepted"], **smoothness(t["actions"])}, ensure_ascii=False) + "\n")
        data += accepted
        summary = {"iteration": i, "arm": args.arm, "collector_success_rate": float(np.mean([t["success"] for t in trajs])),
                   "n_rollouts": len(trajs), "n_success": int(sum(t["success"] for t in trajs)),
                   "n_accepted": len(accepted), "dataset_trajs_after": len(data),
                   "final_collector_loss": losses[-1] if losses else None, "seconds": round(time.time() - t0)}
        (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")  # written last = done
        log(f"iteration {i} done: collector success={summary['collector_success_rate']:.3f} "
            f"accepted={len(accepted)} |D({i + 1})|={len(data)} ({summary['seconds']}s)")


if __name__ == "__main__":
    main()
