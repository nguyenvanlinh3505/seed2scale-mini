"""Registry of research experiments (auto-research/s2s-bootstrap/contract.md).
Danh sách các thí nghiệm nghiên cứu (auto-research/s2s-bootstrap/contract.md).

All use 32 seeds and 6 self-evolution iterations, with the same rollout start states as runs 1-2.
Run with the pipeline stage `research=E1+E3+E4` (one VLV server for the whole batch).
Tất cả dùng 32 seed và 6 vòng tự tiến hóa, cùng trạng thái đầu rollout như lần chạy 1-2.
"""
from __future__ import annotations

N_SEEDS = 32
ITERATIONS = 6

EXPERIMENTS: dict[str, dict] = {
    # H1: new collector each iteration, fixed epoch budget (removes cumulative over-training)
    "E1": {"arm": "s2s", "train": "scratch", "gamma": 7},
    # H2: gamma 6 accepts the VLV's frequent "6" scores (post-hoc choice: flagged)
    "E2": {"arm": "s2s", "train": "warm", "gamma": 6},
    # H3: both
    "E3": {"arm": "s2s", "train": "scratch", "gamma": 6},
    # H5: gate-only control with H1 training
    "E4": {"arm": "gate_only", "train": "scratch"},
    # H4: twice the rollouts per iteration
    "E5": {"arm": "s2s", "train": "scratch", "gamma": 6, "rollouts": 512},
    "E6": {"arm": "gate_only", "train": "scratch", "rollouts": 512},
    # H6: exploration noise on executed actions (px)
    "E7": {"arm": "s2s", "train": "scratch", "gamma": 6, "noise": 8.0},
    "E8": {"arm": "gate_only", "train": "scratch", "noise": 8.0},
}


def evolve_args(cfg: dict) -> list[str]:
    """CLI arguments for s2s.evolve for one experiment config.
    Tham số dòng lệnh cho s2s.evolve của một cấu hình thí nghiệm."""
    args = ["--arm", cfg["arm"], "--n-seeds", str(N_SEEDS), "--iterations", str(cfg.get("iterations", ITERATIONS)),
            "--n-rollouts", str(cfg.get("rollouts", 256)), "--train-mode", cfg["train"]]
    if "gamma" in cfg:
        args += ["--vlv-threshold", str(cfg["gamma"])]
    if "noise" in cfg:
        args += ["--noise", str(cfg["noise"])]
    return args


def late_gain_pp(rates: list[float]) -> float:
    """Primary metric: mean success of the last two iterations minus iteration 0, in percentage points.
    Chỉ số chính: trung bình success hai vòng cuối trừ vòng 0, tính bằng điểm phần trăm."""
    return 100.0 * (sum(rates[-2:]) / len(rates[-2:]) - rates[0])
