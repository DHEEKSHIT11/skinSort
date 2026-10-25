"""Download scripts, one module per dataset.

Every module exposes ``run(params) -> dict`` (used by the Airflow DAG) and a
``main()`` command-line entry point. Downloads are idempotent: files that are
already present and recorded in the manifest are skipped on re-runs.
"""
