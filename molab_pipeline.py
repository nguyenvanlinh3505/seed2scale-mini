import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Seed2Scale-mini on molab

    Runs the whole study as **one background process** (`python -m s2s.pipeline`), so it keeps going while
    cells re-run, and **resumes** from where it stopped if the session dies (progress is mirrored to a
    private Hugging Face repo after every stage).

    Chạy toàn bộ nghiên cứu như **một tiến trình nền**, nên vẫn chạy khi các cell chạy lại, và **chạy tiếp**
    từ chỗ dừng nếu phiên bị tắt (tiến độ được sao lưu lên repo Hugging Face riêng tư sau mỗi giai đoạn).

    1. Fill in the settings below / Điền cấu hình bên dưới.
    2. Click **Install** once, then **Start** / Bấm **Install** một lần, rồi **Start**.
    3. Keep this tab open; the progress panel refreshes every 30 s / Giữ tab mở; bảng tiến độ tự làm mới mỗi 30 giây.
    """
    )
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(value="https://github.com/nguyenvanlinh3505/seed2scale-mini.git", label="Code repo URL", full_width=True)
    hf_repo = mo.ui.text(value="", label="HF results repo, optional, e.g. <user>/seed2scale-mini-runs (private, auto-created)", full_width=True)
    hf_token = mo.ui.text(kind="password", label="HF token (write), optional: without it results stay on molab only", full_width=True)
    stages = mo.ui.text(value="setup,smoke", label="Stages (start with setup,smoke; then evolve,worlds,targets,analyze)", full_width=True)
    target_steps = mo.ui.number(value=20000, start=100, stop=100000, step=100, label="SmolVLA steps per target")
    mo.vstack([repo_url, hf_repo, hf_token, stages, target_steps])
    return hf_repo, hf_token, repo_url, stages, target_steps


@app.cell
def _():
    import os
    import signal
    import subprocess
    from pathlib import Path

    HOME = Path.home()
    CODE = HOME / "seed2scale-mini"
    WORK = HOME / "s2s-work"
    VENV_PY = CODE / ".venv" / "bin" / "python"
    LOG = WORK / "pipeline.log"
    PIDFILE = WORK / "pipeline.pid"
    WORK.mkdir(parents=True, exist_ok=True)

    def sh(cmd: str) -> str:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return (r.stdout + r.stderr)[-3000:]

    def running() -> bool:
        if not PIDFILE.exists():
            return False
        try:
            _pid = int(PIDFILE.read_text())
            # Reap the child if it already exited, otherwise a zombie looks "alive".
            # Thu dọn tiến trình con nếu đã thoát, nếu không zombie trông như vẫn "sống".
            try:
                os.waitpid(_pid, os.WNOHANG)
            except ChildProcessError:
                pass
            os.kill(_pid, 0)
            return True
        except (OSError, ValueError):
            return False
    return CODE, LOG, PIDFILE, VENV_PY, WORK, os, running, sh, signal, subprocess


@app.cell
def _(mo):
    install = mo.ui.run_button(label="Install / update code + deps")
    start = mo.ui.run_button(label="Start pipeline")
    stop = mo.ui.run_button(label="Stop pipeline", kind="danger")
    mo.hstack([install, start, stop], justify="start")
    return install, start, stop


@app.cell
def _(CODE, install, mo, repo_url, sh):
    mo.stop(not install.value, mo.md("_Install not run yet._"))
    out = sh(f"command -v uv || pip install -q uv; nvidia-smi --query-gpu=name,memory.total --format=csv")
    out += sh(f"if [ -d {CODE}/.git ]; then git -C {CODE} pull; else git clone {repo_url.value} {CODE}; fi")
    out += sh(f"cd {CODE} && uv venv -q --allow-existing --python 3.12 .venv && uv pip install -q --python .venv/bin/python -e . "
              f"&& .venv/bin/python -m pytest -q 2>&1 | tail -3")
    mo.md(f"```\n{out}\n```")
    return


@app.cell
def _(CODE, LOG, PIDFILE, VENV_PY, WORK, hf_repo, hf_token, mo, os, running, stages, start, subprocess, target_steps):
    mo.stop(not start.value)
    if running():
        _msg = "Already running / Đang chạy rồi."
    else:
        _env = {**os.environ, "HF_TOKEN": hf_token.value, "PYTHONUNBUFFERED": "1"}
        _cmd = [str(VENV_PY), "-m", "s2s.pipeline", "--work", str(WORK),
                "--stages", stages.value, "--target-steps", str(int(target_steps.value))]
        if hf_repo.value.strip():
            _cmd += ["--hf-repo", hf_repo.value.strip()]
        _p = subprocess.Popen(_cmd, cwd=CODE, env=_env, stdout=open(LOG, "a"), stderr=subprocess.STDOUT,
                              start_new_session=True)
        PIDFILE.write_text(str(_p.pid))
        _msg = f"Started pid {_p.pid}: `{' '.join(_cmd)}`"
    mo.md(_msg)
    return


@app.cell
def _(PIDFILE, mo, os, running, signal, stop):
    mo.stop(not stop.value)
    if running():
        os.killpg(int(PIDFILE.read_text()), signal.SIGTERM)
        _m = "Sent SIGTERM; finished stages are kept / Đã gửi lệnh dừng; các giai đoạn đã xong được giữ lại."
    else:
        _m = "Not running / Không chạy."
    mo.md(_m)
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(default_interval="30s")
    refresh
    return (refresh,)


@app.cell
def _(LOG, WORK, mo, refresh, running):
    refresh
    _state = sorted(p.stem for p in (WORK / "state").glob("*.done")) if (WORK / "state").exists() else []
    _tail = "".join(open(LOG, encoding="utf-8", errors="replace").readlines()[-40:]) if LOG.exists() else ""
    _items = [mo.md(f"**Running:** {running()}  |  **Finished stages:** {', '.join(_state) or '-'}"),
              mo.md(f"```\n{_tail}\n```")]
    for _f in ("fig_collector.png", "fig_scaling.png"):
        if (WORK / "results" / _f).exists():
            _items.append(mo.image(str(WORK / "results" / _f), width=520))
    if (WORK / "results" / "summary.md").exists():
        _items.append(mo.md((WORK / "results" / "summary.md").read_text(encoding="utf-8")))
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
