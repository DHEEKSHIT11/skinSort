"""Preprocessing steps, run in this order by the DAG:

harmonize -> image_stats -> quality -> dedupe -> transform -> split

Each module exposes ``run(params) -> dict`` and a command-line ``main()``.
The logic lives in small pure functions (DataFrame in, DataFrame out) so it
can be unit-tested without real images.
"""
