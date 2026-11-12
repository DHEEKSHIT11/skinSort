"""Model pipeline notifications to Slack and/or email — the same channels, settings and rules as
the data pipeline's alerts (Data-Pipeline/scripts/common/alerts.py), so data and model alerts
arrive in the same place (#pipeline-alerts). Used by the CI/CD workflows.

    python -m src.notify --status success --title "model pipeline" [--summary outputs/pipeline_summary.json]

Channels are configured with environment variables (GitHub secrets in CI), never stored in Git:
    SLACK_WEBHOOK_URL   Slack incoming-webhook URL (main channel)
    ALERT_EMAIL_TO      comma-separated recipients (optional email copy)
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM

If no channel is configured the message is still printed to the log, so a missing webhook never
hides a result, and a failed delivery never fails the pipeline step that called it.
"""
import argparse
import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

ICONS = {"success": ":white_check_mark:", "failure": ":rotating_light:", "bias-alert": ":warning:"}


def build_message(status: str, title: str, summary_path: Optional[Path], details: str = "") -> Tuple[str, str]:
    """(title, body) for one pipeline result."""
    lines = []
    if details:
        lines += [details, ""]
    if summary_path and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for step in summary.get("steps", []):
            lines.append(f"- {step['step']}: exit {step['exit_code']} ({step['seconds']} s)")
        sel = summary.get("selected")
        if sel:
            lines += ["", f"Selected model: {sel.get('model')} (bias check: {sel.get('bias_result')})",
                      f"Why: {sel.get('reason')}"]
            if sel.get("bias_result") == "FAIL":
                lines.append("ALERT: the selected model has a bias gap above the limit; see outputs/bias.")
    run_url = os.environ.get("RUN_URL")
    if run_url:
        lines += ["", f"Run: {run_url}"]
    return f"SkinSort {title}: {status}", "\n".join(lines) or "(no details)"


def _send_slack(webhook: str, title: str, body: str, status: str) -> None:
    icon = ICONS.get(status, ":information_source:")
    response = requests.post(webhook, json={"text": f"{icon} *{title}*\n{body}"},
                             timeout=15, allow_redirects=False)
    response.raise_for_status()
    # Slack answers "ok" when the message is posted; a malformed webhook URL is redirected to
    # Slack's docs page (also HTTP 200), so the status code alone is not proof (same rule as
    # the data pipeline).
    if response.text.strip() != "ok":
        raise requests.RequestException(
            f"Slack did not post the message (HTTP {response.status_code}): {response.text.strip()[:80]!r}")


def _send_email(recipients: List[str], title: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = f"[SkinSort model pipeline] {title}"
    msg["From"] = os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", "skinsort@localhost"))
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", 587)), timeout=15) as smtp:
        smtp.starttls()
        if os.environ.get("SMTP_USER"):
            smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)


def send(title: str, body: str, status: str = "info") -> Dict[str, bool]:
    """Send to every configured channel. Never raises; returns delivery status per channel."""
    print(f"NOTIFY: {title}\n{body}\n")
    delivered: Dict[str, bool] = {}
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        try:
            _send_slack(webhook, title, body, status)
            delivered["slack"] = True
        except requests.RequestException as exc:
            print(f"notify: Slack message failed: {exc}")
            delivered["slack"] = False
    recipients = [r.strip() for r in os.environ.get("ALERT_EMAIL_TO", "").split(",") if r.strip()]
    if recipients and os.environ.get("SMTP_HOST"):
        try:
            _send_email(recipients, title, body)
            delivered["email"] = True
        except (OSError, smtplib.SMTPException) as exc:
            print(f"notify: email failed: {exc}")
            delivered["email"] = False
    if not delivered:
        print("notify: no channel configured (set SLACK_WEBHOOK_URL or SMTP_*); message only logged")
    return delivered


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a model pipeline notification (Slack / email)")
    parser.add_argument("--status", required=True, help="success / failure / bias-alert")
    parser.add_argument("--title", default="model pipeline")
    parser.add_argument("--summary", type=Path, default=Path("outputs/pipeline_summary.json"))
    parser.add_argument("--details", default="")
    args = parser.parse_args()
    title, body = build_message(args.status, args.title, args.summary, args.details)
    send(title, body, args.status)


if __name__ == "__main__":
    main()
