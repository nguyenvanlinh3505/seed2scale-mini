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
    # Seed2Scale-mini: test models on random variants
    # Thử model trên các biến thể ngẫu nhiên

    Runs the SuperTiny collector and the SmolVLA target on the **same N random variants**: a random start
    position AND a random colour palette never seen in training. Shows a GIF per episode and a summary table.
    Can run in a second molab notebook in parallel with `molab_pipeline.py`.

    Chạy collector SuperTiny và model mục tiêu SmolVLA trên **cùng N biến thể ngẫu nhiên**: vị trí xuất phát ngẫu
    nhiên VÀ bảng màu ngẫu nhiên chưa từng thấy khi train. Hiển thị GIF từng ván và bảng tổng kết.
    Có thể chạy trong notebook molab thứ hai, song song với `molab_pipeline.py`.

    Checkpoints are read from `~/s2s-work` if this notebook shares the machine with the pipeline; otherwise they
    are downloaded from the private HF results repo. / Checkpoint được đọc từ `~/s2s-work` nếu cùng máy với
    pipeline; nếu không thì tải từ repo HF riêng tư.
    """
    )
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(value="https://github.com/nguyenvanlinh3505/seed2scale-mini.git", label="Code repo URL", full_width=True)
    hf_repo = mo.ui.text(value="", label="HF results repo (only needed on another machine)", full_width=True)
    hf_token = mo.ui.text(kind="password", label="HF token (read is enough)", full_width=True)
    collector_rel = mo.ui.text(value="runs_n32/gate_only/iter_7/collector.pt", label="Collector checkpoint (relative to work dir)", full_width=True)
    smolvla_rel = mo.ui.text(value="targets/seed_only_n32/checkpoints/020000/pretrained_model", label="SmolVLA checkpoint dir (relative to work dir)", full_width=True)
    n_variants = mo.ui.number(value=10, start=1, stop=50, step=1, label="Number of variants")
    seed = mo.ui.number(value=0, start=0, stop=10000, step=1, label="Variant seed")
    mo.vstack([repo_url, hf_repo, hf_token, collector_rel, smolvla_rel, mo.hstack([n_variants, seed], justify="start")])
    return collector_rel, hf_repo, hf_token, n_variants, repo_url, seed, smolvla_rel


@app.cell
def _():
    import json
    import os
    import subprocess
    from pathlib import Path

    HOME = Path.home()
    CODE = HOME / "seed2scale-mini"
    WORK = HOME / "s2s-work"
    VENV_PY = CODE / ".venv" / "bin" / "python"
    OUT = HOME / "s2s-variants"

    def sh(cmd: str, env: dict | None = None) -> str:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ, **(env or {})})
        return (r.stdout + r.stderr)[-3000:]
    return CODE, OUT, Path, VENV_PY, WORK, json, os, sh


@app.cell
def _(mo):
    install = mo.ui.run_button(label="1. Install / update code + deps")
    run = mo.ui.run_button(label="2. Fetch checkpoints + run test")
    mo.hstack([install, run], justify="start")
    return install, run


@app.cell
def _(CODE, install, mo, repo_url, sh):
    mo.stop(not install.value, mo.md("_Install not run yet (skip if this machine already has ~/seed2scale-mini/.venv)._"))
    _out = sh("command -v uv || pip install -q uv; nvidia-smi --query-gpu=name,memory.total --format=csv")
    _out += sh(f"if [ -d {CODE}/.git ]; then git -C {CODE} pull; else git clone {repo_url.value} {CODE}; fi")
    _out += sh(f"cd {CODE} && uv venv -q --allow-existing --python 3.12 .venv && "
               f"uv pip install -q --python .venv/bin/python -e . && .venv/bin/python -m pytest -q 2>&1 | tail -3")
    mo.md(f"```\n{_out}\n```")
    return


@app.cell
def _(CODE, OUT, VENV_PY, WORK, collector_rel, hf_repo, hf_token, mo, n_variants, run, seed, sh, smolvla_rel):
    mo.stop(not run.value, mo.md("_Test not run yet._"))
    _log = ""
    _need = [p for p in (collector_rel.value, smolvla_rel.value) if p.strip() and not (WORK / p).exists()]
    if _need:
        if not (hf_repo.value.strip() and hf_token.value):
            mo.stop(True, mo.md(f"Missing locally and no HF repo/token given: `{_need}`"))
        _pats = repr([p.rstrip("/") + ("/*" if not p.endswith(".pt") else "") for p in _need])
        _log += sh(f"{VENV_PY} -c \"from huggingface_hub import snapshot_download as s; "
                   f"s('{hf_repo.value.strip()}', repo_type='dataset', local_dir='{WORK}', allow_patterns={_pats})\"",
                   env={"HF_TOKEN": hf_token.value})
    _args = f"--n {int(n_variants.value)} --seed {int(seed.value)} --out {OUT}"
    if collector_rel.value.strip():
        _args += f" --collector {WORK / collector_rel.value.strip()}"
    if smolvla_rel.value.strip():
        _args += f" --smolvla {WORK / smolvla_rel.value.strip()}"
    with mo.status.spinner(title="Running variants (a few minutes) / Đang chạy các biến thể (vài phút)..."):
        _log += sh(f"cd {CODE} && {VENV_PY} -m s2s.variants_eval {_args} 2>&1 | grep -E 'collector|smolvla|SUMMARY|Error|Traceback'")
    mo.md(f"```\n{_log}\n```")
    return


@app.cell
def _(OUT, json, mo, run):
    mo.stop(not run.value or not (OUT / "results.json").exists())
    _res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    _rows = {(r["model"], r["variant"]): r for r in _res["rows"]}
    _models = sorted({r["model"] for r in _res["rows"]})
    _summary = mo.md("**Summary / Tổng kết:** " + "; ".join(
        f"{m}: success {v['success_rate']:.0%}, mean max coverage {v['mean_max_coverage']:.2f}"
        for m, v in _res["summary"].items()))
    _table = mo.ui.table([{"variant": v["id"], "start seed": v["reset_seed"],
                           **{f"{m} success": _rows[(m, v["id"])]["success"] for m in _models if (m, v["id"]) in _rows},
                           **{f"{m} max coverage": round(_rows[(m, v["id"])]["max_coverage"], 2)
                              for m in _models if (m, v["id"]) in _rows}}
                          for v in _res["variants"]], selection=None)
    _grid = []
    for _v in _res["variants"]:
        _cells = [mo.md(f"**v{_v['id']}**")]
        for _m in _models:
            _g = OUT / f"{_m}_v{_v['id']}.gif"
            if _g.exists():
                _r = _rows[(_m, _v["id"])]
                _cells.append(mo.vstack([mo.image(str(_g), width=200),
                                         mo.md(f"{_m}: {'✅' if _r['success'] else '❌'} cov {_r['max_coverage']:.2f}")]))
        _grid.append(mo.hstack(_cells, justify="start"))
    mo.vstack([_summary, _table, *_grid])
    return


if __name__ == "__main__":
    app.run()
