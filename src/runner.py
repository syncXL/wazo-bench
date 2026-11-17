from ekko_bench.main import evaluate

from . import models
from .compiler import EvalCompiler


class EvalRunner:
    def __init__(self):
        self.reports = [
            "wer_africanvoices",
            "wer_naijavoices",
            "wer_fleurs",
            "wer_openbible",
            "wer_afrispeech",
            "wer_pcm_by_corpus",
            "ter_yor_by_corpus",
            "ter_ibo_by_corpus",
            "performance",
        ]

    def bench_model(
        self,
        model_id: str,
        repo_id: str,
        device: str,
        output_dir: str = "runs",
        push_to_hub: bool = True,
    ):
        model_obj = models.MODELS[model_id].caller(device=device)
        evaluate(
            model_obj,
            "automatic-speech-recognition",
            "report",
            output_dir,
            32,
            repo_id if push_to_hub else None,
            "ekko_runs",
            *self.reports,
        )
        EvalCompiler().compile_metrics(
            model_id,
            repo_id=repo_id,
            run_dir=output_dir,
            output_dir="compiled",
            push_to_hub=push_to_hub,
        )
        print(f"Evaluation completed for {model_id}")

    def bench_all_models(
        self,
        repo_id: str,
        device: str,
        output_dir: str = "runs",
        push_to_hub: bool = True,
    ):
        for key in models.MODELS:
            self.bench_model(key, repo_id, device, output_dir, push_to_hub)

    def compile_all(
        self,
        repo_id: str | None = None,
        run_dir: str = "runs",
        output_dir: str = "compiled",
        push_to_hub: bool = False,
    ):
        return EvalCompiler().compile_all(repo_id, run_dir, output_dir, push_to_hub)
            
