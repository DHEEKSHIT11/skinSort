# Airflow image with the SkinSort pipeline dependencies baked in, so they are
# not re-installed every time a container starts.
FROM apache/airflow:3.3.2-python3.11

USER root
# git is needed by DVC to read the repository it versions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt
