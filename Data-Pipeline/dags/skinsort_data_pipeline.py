"""SkinSort data pipeline DAG.

    [ingest_isic, ingest_ham10000, ingest_pad_ufes_20, ingest_fitzpatrick17k]   (parallel)
      -> ingest_report -> version_raw_data
                       -> harmonize -> image_stats -> quality_filter -> dedupe
                          -> transform -> split -> version_processed_data

Every task retries on failure and sends a Slack/email alert when it finally fails.
Each preprocessing task is a thin wrapper around ``scripts.preprocess.<module>.run``
so the same code also runs from the command line and from ``dvc repro``.
"""

import importlib
import json
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

from scripts.common.alerts import airflow_failure_callback

DEFAULT_ARGS = {
    "owner": "skinsort-team18",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "on_failure_callback": airflow_failure_callback,
}


def run_step(module: str, **kwargs) -> dict:
    """Import scripts.<module> and call its run(params)."""
    from scripts.common.config import load_params
    return importlib.import_module(f"scripts.{module}").run(load_params(), **kwargs)


@dag(
    dag_id="skinsort_data_pipeline",
    description="Download, validate, preprocess and version SkinSort datasets",
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    schedule=None,  # triggered manually or by new-data events, not on a timer
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["skinsort", "data"],
)
def skinsort_data_pipeline():

    # --- acquisition --------------------------------------------------------------
    @task(execution_timeout=timedelta(hours=3))
    def ingest_isic() -> dict:
        return run_step("ingest.isic")

    @task(execution_timeout=timedelta(hours=2))
    def ingest_ham10000() -> dict:
        return run_step("ingest.ham10000")

    @task(execution_timeout=timedelta(hours=2))
    def ingest_pad_ufes_20() -> dict:
        return run_step("ingest.pad_ufes")

    @task(execution_timeout=timedelta(hours=2))
    def ingest_fitzpatrick17k() -> dict:
        return run_step("ingest.fitzpatrick17k")

    @task
    def ingest_report(summaries: list) -> dict:
        """Write one JSON report of what every download step produced."""
        from scripts.common.config import load_params, resolve_path
        from scripts.common.logging_utils import get_logger

        report = {s["dataset"]: s for s in summaries}
        path = resolve_path(load_params(), "logs") / "ingest_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str))
        get_logger("dag").info("Ingest report written to %s: %s", path, report)
        return report

    @task(retries=1)
    def version_raw_data() -> dict:
        from scripts.versioning import version_data
        return version_data("data/raw")

    # --- preprocessing ------------------------------------------------------------
    @task
    def harmonize() -> dict:
        return run_step("preprocess.harmonize")

    @task(execution_timeout=timedelta(hours=1))
    def image_stats() -> dict:
        return run_step("preprocess.image_stats")

    @task
    def quality_filter() -> dict:
        return run_step("preprocess.quality")

    @task
    def dedupe() -> dict:
        return run_step("preprocess.dedupe")

    @task(execution_timeout=timedelta(hours=1))
    def transform() -> dict:
        return run_step("preprocess.transform")

    @task
    def split() -> dict:
        return run_step("preprocess.split")

    @task(retries=1)
    def version_processed_data() -> dict:
        from scripts.versioning import commit_pipeline_outputs
        return commit_pipeline_outputs()

    report = ingest_report(
        [ingest_isic(), ingest_ham10000(), ingest_pad_ufes_20(), ingest_fitzpatrick17k()]
    )
    report >> version_raw_data()
    (report >> harmonize() >> image_stats() >> quality_filter() >> dedupe()
     >> transform() >> split() >> version_processed_data())


skinsort_data_pipeline()
