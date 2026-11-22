"""All experiment settings in one place, each tagged PAPER (taken from arXiv:2603.08260) or
DEVIATION (changed for PushT / one-GPU budget, with the reason).
Toàn bộ cấu hình ở một chỗ; mỗi mục gắn nhãn PAPER (lấy từ arXiv:2603.08260) hoặc
DEVIATION (đổi cho PushT / ngân sách một GPU, kèm lý do).
"""
from __future__ import annotations

from dataclasses import dataclass, field

TASK = "Push the T-shaped block onto the T-shaped target."  # task string of lerobot/pusht / câu lệnh gốc của lerobot/pusht
FPS = 10                                                     # lerobot/pusht fps


@dataclass
class Config:
    # --- Seeds / Dữ liệu seed ---
    n_seeds: int = 4                 # PAPER: four seed demos at the four workspace corners / 4 demo ở 4 góc
    seed_dataset: str = "lerobot/pusht"

    # --- SuperTiny collector / Bộ thu thập SuperTiny ---
    chunk_size: int = 30             # PAPER: K = 30
    ensemble_lambda: float = 0.1     # PAPER uses exponential temporal ensembling; lambda not reported / paper không nêu lambda
    d_model: int = 256               # DEVIATION: decoder width not reported; with ResNet-18 + T5-small encoder
    n_decoder_layers: int = 4        # the total is ~48M params like the paper / tổng ~48M tham số như paper
    collector_lr: float = 3e-4
    collector_steps_first: int = 8000  # iteration 0 trains from scratch / vòng 0 train từ đầu
    collector_steps: int = 3000        # later iterations warm-start from the previous collector
                                       # các vòng sau tiếp tục từ collector vòng trước
    collector_batch: int = 64

    # --- Self-evolution / Tự tiến hóa ---
    iterations: int = 8              # PAPER: 8 iterations in the scaling study (Fig. 4)
    n_rollouts: int = 256            # DEVIATION: N_rollout not reported; PushT success is rare early on
    max_steps: int = 300             # gym_pusht default episode length / độ dài mặc định
    exploration_noise: float = 0.0   # PAPER mentions no injected noise; diversity comes from random starts

    # --- Verifier (VLV) ---
    vlv_model: str = "Qwen/Qwen3-VL-32B-Instruct"  # PAPER: Qwen3-VL-32B
    vlv_threshold: float = 7.0       # PAPER: gamma fixed a priori from a 0-10 rubric (value not reported)
    vlv_frames: int = 8              # frames sampled per video / số khung hình lấy từ mỗi video
    vlv_image_size: int = 224        # render size for verifier videos / kích thước ảnh cho verifier

    # --- Target model (SmolVLA) / Model mục tiêu ---
    target_steps: int = 20_000       # DEVIATION: paper trains 100k steps; 5x fewer to fit ~6 trainings
    target_lr: float = 3e-4          # PAPER: AdamW lr 3e-4
    target_batch: int = 16           # PAPER: batch size 16
    target_iters: tuple[int, ...] = (0, 2, 4, 8)  # DEVIATION: paper trains at all 8 iterations (Fig. 4)

    # --- Parallel worlds ---
    seen_worlds: tuple[str, ...] = ("w1", "w2", "w3", "w4", "w5")  # see s2s/worlds.py
    unseen_worlds: tuple[str, ...] = ("u1", "u2", "u3")            # PAPER: 3 disjoint unseen scenes

    # --- Evaluation / Đánh giá ---
    eval_episodes: int = 100
    eval_seed: int = 100_000         # fixed seeds 100000..100099 / seed cố định

    extra: dict = field(default_factory=dict)


CFG = Config()
