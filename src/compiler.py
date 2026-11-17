import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from ekko_bench.utils import hf_helper

from . import models


def read_json_file(file_path: str | Path) -> Any:
    with Path(file_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_yaml_file(file_path: str | Path, items: list[dict[str, Any]]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(items, file, sort_keys=False, allow_unicode=True)


SOURCE_URL = "https://huggingface.co/spaces/nolimitsxl/Wazo-Bench"
DATASET_ID = "nolimitsxl/wazobia-asr"


class EvalCompiler:
    """Compile Ekko run metrics into the Wazo-Bench eval.yaml-shaped records."""

    def _run_dirs(self, root: Path, model_id: str) -> list[Path]:
        model_root = root / model_id
        if not model_root.is_dir():
            return []
        return sorted(
            (path for path in model_root.glob("run_*") if (path / "metrics.json").is_file()),
            key=lambda path: path.name,
        )

    def _task_values(self, runs: list[Path]) -> dict[str, float]:
        values: dict[str, float] = {}
        for run in runs:
            metrics = read_json_file(run / "metrics.json")
            values.update(metrics.get("tasks", {}))
            # Reports preserve the corpus/language split needed by the leaderboard.
            for report_id, report_values in metrics.get("reports", {}).items():
                if not isinstance(report_values, dict):
                    continue
                if report_id == "performance":
                    metric, suffix = "rtfx", "fleurs"
                elif report_id in {"wer_pcm_by_corpus", "ter_yor_by_corpus", "ter_ibo_by_corpus"}:
                    metric, suffix = report_id.split("_", 1)[0], ""
                else:
                    metric, _, suffix = report_id.partition("_")
                for key, value in report_values.items():
                    if report_id == "performance":
                        task_id = f"{metric}_{suffix}_{key}"
                    elif report_id.endswith("_by_corpus"):
                        task_id = f"{metric}_{key}_{report_id.split('_')[1]}"
                    else:
                        task_id = f"{metric}_{suffix}_{key}"
                    try:
                        values[task_id] = float(value)
                    except (TypeError, ValueError):
                        continue
        return values

    def compile_metrics(
        self,
        model_id: str,
        repo_id: str | None = None,
        run_dir: str = "runs",
        output_dir: str = "compiled",
        push_to_hub: bool = False,
        path_in_repo: str = "compiled",
    ) -> Path:
        if model_id not in models.MODELS:
            raise KeyError(f"Unknown model id: {model_id}")
        selected_model = models.MODELS[model_id]
        values = self._task_values(self._run_dirs(Path(run_dir), model_id))
        date = datetime.now(timezone.utc).date().isoformat()
        rows = [
            {
                "dataset": {"id": DATASET_ID, "task_id": task_id},
                "value": value,
                "date": date,
                "source": {"url": SOURCE_URL, "name": "Wazo-Bench", "user": "nolimitsxl"},
                "notes": f"Evaluated with ekko-bench. Model: {selected_model.id}",
            }
            for task_id, value in sorted(values.items())
        ]
        destination = Path(output_dir) / f"{model_id}.yaml"
        save_yaml_file(destination, rows)
        if push_to_hub:
            if not repo_id:
                raise ValueError("repo_id is required when push_to_hub is enabled")
            hf_helper.upload_dir(str(Path(output_dir)), repo_id, path_in_repo, repo_type="dataset")
        return destination

    def compile_all(
        self,
        repo_id: str | None = None,
        run_dir: str = "runs",
        output_dir: str = "compiled",
        push_to_hub: bool = False,
        path_in_repo: str = "compiled",
    ) -> list[Path]:
        outputs = [
            self.compile_metrics(model_id, run_dir=run_dir, output_dir=output_dir)
            for model_id in models.MODELS
        ]
        if push_to_hub:
            if not repo_id:
                raise ValueError("repo_id is required when push_to_hub is enabled")
            hf_helper.upload_dir(output_dir, repo_id, path_in_repo, repo_type="dataset")
        return outputs
