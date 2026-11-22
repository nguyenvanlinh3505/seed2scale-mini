"""One-command, resumable orchestration of the whole study on a single GPU box (built for molab).
Điều phối toàn bộ nghiên cứu bằng một lệnh, chạy tiếp được, trên một máy một GPU (làm cho molab).

Stages (each leaves a marker in <work>/state; rerunning skips finished work):
Các giai đoạn (mỗi giai đoạn để lại dấu ở <work>/state; chạy lại sẽ bỏ qua phần đã xong):
  setup      separate virtualenv with vLLM (vLLM and LeRobot pin different torch versions)
  smoke      tiny end-to-end run + VLV health check + SmolVLA speed calibration
  evolve     self-evolution for arms s2s (with VLV server), gate_only, no_filter
  worlds     Parallel Worlds projection of the s2s arm (with VLV server)
  targets    export -> train SmolVLA -> evaluate, in priority order (TARGETS)
  analyze    figures and tables
After every stage, results are pushed to a private Hugging Face repo if HF_TOKEN and --hf-repo are set,
and pulled back at start, so a killed session loses at most the stage in progress.
Sau mỗi giai đoạn, kết quả được đẩy lên repo Hugging Face riêng tư nếu có HF_TOKEN và --hf-repo, và được
kéo về khi bắt đầu, nên phiên bị tắt chỉ mất phần đang làm dở.

Usage / Cách dùng:
  python -m s2s.pipeline --work /root/work --hf-repo <user>/seed2scale-mini [--stages smoke] [--target-steps 20000]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from s2s.config import CFG

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

PY = sys.executable
STAGES = ["setup", "smoke", "evolve", "worlds", "targets", "analyze"]
# (name, arm, iteration, with parallel worlds), most important first.
# (tên, nhánh, vòng, có parallel worlds), quan trọng nhất trước.
TARGETS = [
    ("seed_only", "s2s", 0, False),     # SeedOnly baseline (Table I)
    ("s2s_i8", "s2s", 8, False),         # Seed2Scale* after 8 iterations
    ("gate_only_i8", "gate_only", 8, False),  # SuperTiny^- ablation: value of the VLV (Sec. IV-G)
    ("s2s_full_i8", "s2s", 8, True),     # full Seed2Scale with Parallel Worlds (Table I, unseen scenes)
    ("no_filter_i8", "no_filter", 8, False),  # no filtering at all: collapse test (not in the paper)
    ("s2s_i4", "s2s", 4, False),         # scaling curve points (Fig. 4)
    ("s2s_i2", "s2s", 2, False),
]
EVAL_WORLDS = ["source", *CFG.unseen_worlds]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], logfile: Path, env: dict | None = None, echo=("[", "iteration", "success", "Error",
        "Traceback", "step:", "exported", "projecting", "  w", "  u", "  source")) -> int:
    """Run a command, full output to `logfile`, selected progress lines to stdout.
    Chạy lệnh, toàn bộ output vào `logfile`, chỉ in các dòng tiến độ chọn lọc."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    log("$ " + " ".join(cmd))
    with open(logfile, "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                             env={**os.environ, **(env or {})})
        for line in p.stdout:
            f.write(line)
            last = line.rstrip("\n").split("\r")[-1]
            if any(k in last for k in echo):
                print("   " + last[:300], flush=True)
        return p.wait()


class Work:
    def __init__(self, root: Path, hf_repo: str | None):
        self.root, self.hf_repo = root, hf_repo
        for d in ("state", "logs", "runs", "datasets", "targets", "results"):
            (root / d).mkdir(parents=True, exist_ok=True)
        self.token = os.environ.get("HF_TOKEN")
        if hf_repo and not self.token:
            log("WARNING: --hf-repo given but HF_TOKEN is not set; results stay local only")

    def done(self, name: str) -> bool:
        return (self.root / "state" / f"{name}.done").exists()

    def mark(self, name: str, info: dict | None = None):
        (self.root / "state" / f"{name}.done").write_text(json.dumps(info or {}), encoding="utf-8")
        self.push(name)

    def push(self, label: str):
        if not (self.hf_repo and self.token):
            return
        from huggingface_hub import HfApi
        api = HfApi(token=self.token)
        api.create_repo(self.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=self.hf_repo, repo_type="dataset", folder_path=str(self.root),
                          commit_message=f"stage {label}",
                          allow_patterns=["state/*", "logs/*", "runs/**", "results/**",
                                          "targets/*/checkpoints/last/pretrained_model/*"],
                          ignore_patterns=["**/.cache/**"])
        log(f"pushed results to hf://{self.hf_repo} ({label})")

    def pull(self):
        if not (self.hf_repo and self.token):
            return
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError
        try:
            snapshot_download(self.hf_repo, repo_type="dataset", local_dir=str(self.root), token=self.token)
            log(f"restored previous progress from hf://{self.hf_repo}")
        except RepositoryNotFoundError:
            log("no previous progress on the Hub; starting fresh")


class VLVServer:
    """vLLM OpenAI-compatible server for Qwen3-VL in its own virtualenv. / Server vLLM cho Qwen3-VL."""

    def __init__(self, work: Work, vllm_python: Path, gpu_mem: float):
        self.work, self.py, self.gpu_mem, self.proc = work, vllm_python, gpu_mem, None

    def __enter__(self):
        cmd = [str(self.py), "-m", "vllm.entrypoints.openai.api_server", "--model", CFG.vlv_model,
               "--port", "8000", "--max-model-len", "16384", "--gpu-memory-utilization", str(self.gpu_mem),
               "--limit-mm-per-prompt", json.dumps({"image": 2 * CFG.vlv_frames}), "--seed", "0"]
        log("starting VLV server: " + " ".join(cmd))
        self.logf = open(self.work.root / "logs" / "vlv_server.log", "a", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT)
        import requests
        t0 = time.time()
        while time.time() - t0 < 3600:  # first start downloads ~65 GB / lần đầu tải ~65 GB
            if self.proc.poll() is not None:
                raise RuntimeError("VLV server exited; see logs/vlv_server.log")
            try:
                if requests.get("http://127.0.0.1:8000/v1/models", timeout=5).ok:
                    log(f"VLV server ready after {time.time() - t0:.0f}s")
                    return self
            except Exception:  # noqa: BLE001 - not up yet / chưa lên
                pass
            time.sleep(15)
        raise TimeoutError("VLV server did not start within 1 h")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.logf.close()
        log("VLV server stopped")


def stage_setup(w: Work, vllm_python: Path):
    if w.done("setup"):
        return
    venv = vllm_python.parent.parent
    run(["uv", "venv", "--python", "3.12", str(venv)], w.root / "logs" / "setup.log")
    rc = run(["uv", "pip", "install", "--python", str(vllm_python), "vllm"], w.root / "logs" / "setup.log")
    if rc != 0:
        raise SystemExit("vLLM install failed; see logs/setup.log")
    w.mark("setup")


def stage_smoke(w: Work, vllm_python: Path, gpu_mem: float):
    if w.done("smoke"):
        return
    L = w.root / "logs" / "smoke.log"
    smoke = w.root / "smoke"
    if run([PY, "-m", "s2s.evolve", "--arm", "gate_only", "--out", str(smoke), "--tiny", "--iterations", "1",
            "--n-rollouts", "8"], L) != 0:
        raise SystemExit("smoke evolve failed")
    with VLVServer(w, vllm_python, gpu_mem):
        import numpy as np

        from s2s.verifier import VLLMVerifier, build_messages
        frame = np.full((224, 224, 3), 255, np.uint8)
        score, text = VLLMVerifier(CFG.vlv_model).score([build_messages("push", [frame] * 8, [frame] * 8)])[0]
        log(f"VLV health check: score={score} text={text[:120]!r}")
        if score is None:
            raise SystemExit("VLV returned no parsable score; check the rubric/model")
    run([PY, "-m", "s2s.export", "--runs", str(smoke), "--arm", "gate_only", "--iteration", "0",
         "--dst", str(w.root / "datasets" / "smoke_seeds")], L)
    t0 = time.time()
    rc = run([PY, "-m", "s2s.train_target", "--dataset", str(w.root / "datasets" / "smoke_seeds"),
              "--out", str(smoke / "smolvla"), "--steps", "100"], L)
    sec_per_step = (time.time() - t0) / 100
    if rc != 0:
        raise SystemExit("SmolVLA smoke training failed; see logs/smoke.log")
    log(f"SmolVLA calibration: ~{sec_per_step:.2f}s/step incl. startup -> "
        f"{CFG.target_steps} steps ~ {sec_per_step * CFG.target_steps / 3600:.1f} h per target")
    w.mark("smoke", {"smolvla_sec_per_step_upper_bound": sec_per_step})


def stage_evolve(w: Work, vllm_python: Path, gpu_mem: float, n_rollouts: int):
    runs = str(w.root / "runs")
    if not w.done("evolve_s2s"):
        with VLVServer(w, vllm_python, gpu_mem):
            if run([PY, "-m", "s2s.evolve", "--arm", "s2s", "--out", runs, "--n-rollouts", str(n_rollouts)],
                   w.root / "logs" / "evolve_s2s.log") != 0:
                raise SystemExit("evolve s2s failed")
            w.mark("evolve_s2s")
            if not w.done("worlds"):  # reuse the running server / dùng lại server đang chạy
                if run([PY, "-m", "s2s.parallel_worlds", "--runs", runs, "--arm", "s2s"],
                       w.root / "logs" / "worlds.log") != 0:
                    raise SystemExit("parallel worlds failed")
                w.mark("worlds")
    for arm in ("gate_only", "no_filter"):
        if not w.done(f"evolve_{arm}"):
            if run([PY, "-m", "s2s.evolve", "--arm", arm, "--out", runs, "--n-rollouts", str(n_rollouts)],
                   w.root / "logs" / f"evolve_{arm}.log") != 0:
                raise SystemExit(f"evolve {arm} failed")
            w.mark(f"evolve_{arm}")


def stage_worlds(w: Work, vllm_python: Path, gpu_mem: float):
    if w.done("worlds"):
        return
    with VLVServer(w, vllm_python, gpu_mem):
        if run([PY, "-m", "s2s.parallel_worlds", "--runs", str(w.root / "runs"), "--arm", "s2s"],
               w.root / "logs" / "worlds.log") != 0:
            raise SystemExit("parallel worlds failed")
    w.mark("worlds")


def stage_targets(w: Work, steps: int, only: list[str] | None):
    for name, arm, it, worlds in TARGETS:
        if (only and name not in only) or w.done(f"target_{name}"):
            continue
        L = w.root / "logs" / f"target_{name}.log"
        ds, out = w.root / "datasets" / name, w.root / "targets" / name
        cmd = [PY, "-m", "s2s.export", "--runs", str(w.root / "runs"), "--arm", arm, "--iteration", str(it),
               "--dst", str(ds)] + (["--with-worlds"] if worlds else [])
        if run(cmd, L) != 0:
            log(f"skip {name}: export failed (stage not finished?)")
            continue
        if not (out / "checkpoints" / "last").exists():
            if run([PY, "-m", "s2s.train_target", "--dataset", str(ds), "--out", str(out), "--steps", str(steps)], L) != 0:
                raise SystemExit(f"training {name} failed")
        ckpt = out / "checkpoints" / "last" / "pretrained_model"
        if run([PY, "-m", "s2s.eval_target", "--ckpt", str(ckpt), "--worlds", *EVAL_WORLDS,
                "--out", str(w.root / "results" / f"eval_{name}.json")], L) != 0:
            raise SystemExit(f"eval {name} failed")
        w.mark(f"target_{name}")


def main():
    p = argparse.ArgumentParser(description="Seed2Scale-mini pipeline")
    p.add_argument("--work", default="work")
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--targets", default=None, help="comma-separated subset of TARGETS names")
    p.add_argument("--target-steps", type=int, default=CFG.target_steps)
    p.add_argument("--n-rollouts", type=int, default=CFG.n_rollouts)
    p.add_argument("--vlv-gpu-mem", type=float, default=0.75)
    a = p.parse_args()
    w = Work(Path(a.work).resolve(), a.hf_repo)
    w.pull()
    vllm_python = w.root / ".venv-vllm" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    stages = a.stages.split(",")
    t0 = time.time()
    if "setup" in stages:
        stage_setup(w, vllm_python)
    if "smoke" in stages:
        stage_smoke(w, vllm_python, a.vlv_gpu_mem)
    if "evolve" in stages:
        stage_evolve(w, vllm_python, a.vlv_gpu_mem, a.n_rollouts)
    if "worlds" in stages:
        stage_worlds(w, vllm_python, a.vlv_gpu_mem)
    if "targets" in stages:
        stage_targets(w, a.target_steps, a.targets.split(",") if a.targets else None)
    if "analyze" in stages:
        run([PY, "-m", "s2s.analyze", "--work", str(w.root)], w.root / "logs" / "analyze.log")
        w.push("analyze")
    log(f"pipeline finished requested stages in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
