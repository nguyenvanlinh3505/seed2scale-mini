"""Train the target model with `lerobot-train` (paper Sec. III-D, IV-A).
Train model mục tiêu bằng `lerobot-train` (paper mục III-D, IV-A).

"Train SmolVLA from scratch" (paper Sec. IV-D) is implemented as: SmolVLM2 backbone with its pretrained
weights (`load_vlm_weights=true`), a freshly initialised action expert, and a new run for every dataset
(never continuing from `lerobot/smolvla_base` or from an earlier iteration). With `load_vlm_weights=false`
LeRobot would initialise the whole VLM randomly AND keep it frozen, which LeRobot itself warns is
"unlikely to yield good results".
"Train SmolVLA from scratch" được hiện thực là: backbone SmolVLM2 với trọng số pretrained
(`load_vlm_weights=true`), action expert khởi tạo mới, và mỗi dataset một lần train mới (không tiếp tục
từ `lerobot/smolvla_base` hay từ vòng trước). Với `load_vlm_weights=false`, LeRobot khởi tạo ngẫu nhiên
cả VLM VÀ đóng băng nó, chính LeRobot cảnh báo là "unlikely to yield good results".

Usage / Cách dùng:
  python -m s2s.train_target --dataset datasets/s2s_iter4 --out targets/s2s_iter4 [--steps 20000] [--policy act]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from s2s.config import CFG


def command(dataset: Path, out: Path, steps: int, policy: str, batch: int, lr: float, workers: int) -> list[str]:
    cmd = [str(Path(sys.executable).with_name("lerobot-train" + (".exe" if sys.platform == "win32" else ""))),
           f"--dataset.repo_id=local/{dataset.name}", f"--dataset.root={dataset}",
           f"--policy.type={policy}", "--policy.push_to_hub=false",
           f"--output_dir={out}", f"--steps={steps}", f"--batch_size={batch}",
           f"--save_freq={max(1, steps // 4)}", f"--log_freq={max(1, min(200, steps // 20))}",
           f"--num_workers={workers}", "--seed=0"]
    if policy == "smolvla":
        cmd += ["--policy.load_vlm_weights=true",      # pretrained SmolVLM2 backbone / backbone pretrained
                f"--policy.optimizer_lr={lr}",           # PAPER: AdamW lr 3e-4
                f"--policy.scheduler_decay_steps={steps}"]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=CFG.target_steps)
    p.add_argument("--policy", default="smolvla", help="smolvla (paper) or act (fast CPU format test)")
    p.add_argument("--batch", type=int, default=CFG.target_batch)
    p.add_argument("--lr", type=float, default=CFG.target_lr)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    cmd = command(Path(a.dataset).resolve(), Path(a.out).resolve(), a.steps, a.policy, a.batch, a.lr, a.workers)
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
