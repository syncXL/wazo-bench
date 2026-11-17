# Wazobia Benchmark CLI

This repository runs the selected Ekko-Bench ASR reports and compiles their metrics into the Wazo-Bench leaderboard format. The Fire CLI is the entry point. The two Gradio frontends are maintained separately in `Lab/spaces/wazo-bench` and `Lab/spaces/ekko-runs`.

```powershell
uv run python main.py bench_model --model_id="openai/whisper-small" --repo_id="nolimitsxl/wazo-bench" --device="cuda"
```

Run artifacts are written locally under `runs/<org>/<model>/run_<timestamp>/` and uploaded to the dataset repo under `ekko_runs/`. Compiled leaderboard records are written under `compiled/<org>/<model>.yaml` and uploaded under `compiled/`.

To recompile all registered models from local runs:

```powershell
uv run python main.py compile_all --repo_id="nolimitsxl/wazo-bench" --push_to_hub=True
```

Set `push_to_hub=False` when you only want local artifacts.
