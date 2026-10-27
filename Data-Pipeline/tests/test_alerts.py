"""Tests for Slack/email alerts (scripts/common/alerts.py). No real messages are sent."""

import pytest
import requests

from scripts.common import alerts

WEBHOOK = "https://hooks.slack.com/services/T000/B000/test"


class FakeResponse:
    def __init__(self, status=200, text="ok"):
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")


@pytest.fixture
def no_channels(monkeypatch):
    for name in ["SLACK_WEBHOOK_URL", "ALERT_EMAIL_TO", "SMTP_HOST"]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def slack(monkeypatch, no_channels):
    """Point alerts at a fake webhook and record what would be posted."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", WEBHOOK)
    calls = []

    def install(response):
        def fake_post(url, json, timeout, allow_redirects):
            calls.append({"url": url, "json": json, "allow_redirects": allow_redirects})
            return response
        monkeypatch.setattr(alerts.requests, "post", fake_post)
        return calls
    return install


def test_no_channel_configured_only_logs(no_channels):
    assert alerts.send_alert("Title", "Body") == {}


def test_slack_ok_reply_counts_as_delivered(slack):
    calls = slack(FakeResponse(200, "ok"))
    assert alerts.send_alert("Data anomaly", "brightness drifted") == {"slack": True}
    assert calls[0]["url"] == WEBHOOK
    assert "Data anomaly" in calls[0]["json"]["text"]
    assert calls[0]["allow_redirects"] is False


def test_bad_webhook_redirect_page_is_reported_as_failure(slack):
    """A malformed URL makes Slack answer with its docs page, not "ok"."""
    slack(FakeResponse(200, "<!doctype html><html>Slack developer docs</html>"))
    assert alerts.send_alert("Title", "Body") == {"slack": False}


def test_redirect_status_is_reported_as_failure(slack):
    slack(FakeResponse(302, ""))
    assert alerts.send_alert("Title", "Body") == {"slack": False}


def test_revoked_webhook_is_reported_as_failure(slack):
    slack(FakeResponse(403, "invalid_token"))
    assert alerts.send_alert("Title", "Body") == {"slack": False}


def test_network_error_never_raises(slack, monkeypatch):
    def broken_post(*args, **kwargs):
        raise requests.ConnectionError("no network")
    monkeypatch.setattr(alerts.requests, "post", broken_post)
    assert alerts.send_alert("Title", "Body") == {"slack": False}


def test_airflow_failure_callback_includes_task_and_error(slack):
    calls = slack(FakeResponse(200, "ok"))

    class TaskInstance:
        dag_id, task_id, try_number = "skinsort_data_pipeline", "detect_anomalies", 1
        log_url = "http://localhost:8080/log"

    alerts.airflow_failure_callback({"task_instance": TaskInstance(), "run_id": "manual_1",
                                     "exception": ValueError("critical drift")})
    text = calls[0]["json"]["text"]
    assert "skinsort_data_pipeline.detect_anomalies" in text
    assert "critical drift" in text and "manual_1" in text
