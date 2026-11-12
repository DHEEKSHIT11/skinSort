"""Tests for src/notify.py and the command list in src/training/pipeline.py."""
import json

from src.notify import build_message, send
from src.training.dataset import load_params
from src.training.pipeline import STEPS, commands


def test_message_includes_steps_and_bias_alert(tmp_path):
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"steps": [{"step": "bias", "exit_code": 0, "seconds": 3}],
                                   "selected": {"model": "fusion_oof", "bias_result": "FAIL", "reason": "best"}}))
    msg = build_message("success", "model pipeline", summary)
    body = msg.get_content()
    assert "[SkinSort]" in msg["Subject"] and "- bias: exit 0" in body and "ALERT" in body


def test_send_without_settings_only_prints(monkeypatch, capsys):
    for key in ("SMTP_USER", "SMTP_PASSWORD", "NOTIFY_TO"):
        monkeypatch.delenv(key, raising=False)
    assert send(build_message("failure", "x", None)) is False
    assert "printing instead" in capsys.readouterr().out


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
