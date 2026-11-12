"""Tests for src/notify.py and the command list in src/training/pipeline.py."""
import json

import requests

from src.notify import build_message, send
from src.training.dataset import load_params
from src.training.pipeline import STEPS, commands


def test_message_includes_steps_and_bias_alert(tmp_path):
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"steps": [{"step": "bias", "exit_code": 0, "seconds": 3}],
                                   "selected": {"model": "fusion_oof", "bias_result": "FAIL", "reason": "best"}}))
    title, body = build_message("success", "model pipeline", summary)
    assert title == "SkinSort model pipeline: success" and "- bias: exit 0" in body and "ALERT" in body


CHANNEL_VARS = ("SLACK_WEBHOOK_URL", "ALERT_EMAIL_TO", "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD")


class FakeResponse:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_send_without_channels_only_logs(monkeypatch, capsys):
    for key in CHANNEL_VARS:
        monkeypatch.delenv(key, raising=False)
    assert send("t", "b", "failure") == {}
    assert "no channel configured" in capsys.readouterr().out


def test_slack_delivered_only_when_slack_says_ok(monkeypatch):
    for key in CHANNEL_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T/B/X")
    posted = {}

    def fake_post(url, json, timeout, allow_redirects):
        posted.update(json)
        return FakeResponse("ok")
    monkeypatch.setattr(requests, "post", fake_post)
    assert send("SkinSort model pipeline: failure", "body", "failure") == {"slack": True}
    assert posted["text"].startswith(":rotating_light: *SkinSort model pipeline: failure*")

    # a malformed webhook is redirected to Slack's docs page with HTTP 200: not delivered
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse("<html>docs</html>"))
    assert send("t", "b", "success") == {"slack": False}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse("no_service", 404))
    assert send("t", "b", "success") == {"slack": False}


def test_email_copy_when_configured(monkeypatch):
    import src.notify as notify
    for key in CHANNEL_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALERT_EMAIL_TO", "a@example.com, b@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    sent = {}
    monkeypatch.setattr(notify, "_send_email", lambda to, title, body: sent.update(to=to, title=title))
    assert send("t", "b", "success") == {"email": True}
    assert sent["to"] == ["a@example.com", "b@example.com"]


def test_pipeline_commands_cover_all_steps():
    cmds = commands(load_params(), "efficientnet_b3", "fusion_oof_rw", ["train"], fail_if_rejected=True,
                    reweight=["sex"], mitigate=["sex"])
    assert list(cmds) == STEPS
    assert "--fail-if-rejected" in cmds["registry"] and "--include-val" in cmds["bias"]
    assert cmds["fit_final"][-2:] == ["--reweight", "sex"] and "--mitigate" in cmds["bias"]


def test_collect_figures_copies_and_lists(tmp_path):
    from src.training.collect_figures import FIGURES, collect
    out = tmp_path / "outputs"
    first = FIGURES["selection"][0][0]
    (out / first).parent.mkdir(parents=True)
    (out / first).write_bytes(b"png")
    missing = collect(out=out, dest=tmp_path / "figs")
    assert first not in missing and len(missing) > 0
    assert (tmp_path / "figs" / "selection" / first.replace("/", "_")).exists()
    assert first.replace("/", "_") in (tmp_path / "figs" / "README.md").read_text()
