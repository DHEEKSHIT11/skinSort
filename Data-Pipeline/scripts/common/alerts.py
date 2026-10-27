"""Send pipeline alerts to Slack and/or email.

Channels are configured with environment variables so no secret is stored in Git:
    SLACK_WEBHOOK_URL   Slack incoming-webhook URL
    ALERT_EMAIL_TO      comma-separated recipients
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM

If no channel is configured the alert is still written to the log, so a missing
webhook never hides a failure.
"""

import os
import smtplib
from email.message import EmailMessage
from typing import Any, Dict, List

import requests

from scripts.common.logging_utils import get_logger

logger = get_logger("alerts")


def _send_slack(webhook: str, title: str, body: str) -> None:
    response = requests.post(webhook, json={"text": f":rotating_light: *{title}*\n{body}"},
                             timeout=15, allow_redirects=False)
    response.raise_for_status()
    # Slack replies "ok" when the message is posted. A malformed webhook URL is
    # redirected to Slack's docs page (also HTTP 200), so status alone is not proof.
    if response.text.strip() != "ok":
        raise requests.RequestException(
            f"Slack did not post the alert (HTTP {response.status_code}): "
            f"{response.text.strip()[:80]!r}"
        )


def _send_email(recipients: List[str], title: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = f"[SkinSort pipeline] {title}"
    msg["From"] = os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", "skinsort@localhost"))
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", 587)),
                      timeout=15) as smtp:
        smtp.starttls()
        if os.environ.get("SMTP_USER"):
            smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)


def send_alert(title: str, body: str) -> Dict[str, bool]:
    """Send an alert to every configured channel. Never raises; returns delivery status."""
    logger.error("ALERT: %s | %s", title, body)
    status = {}
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        try:
            _send_slack(webhook, title, body)
            status["slack"] = True
        except requests.RequestException as exc:
            logger.warning("Slack alert failed: %s", exc)
            status["slack"] = False
    recipients = [r.strip() for r in os.environ.get("ALERT_EMAIL_TO", "").split(",") if r.strip()]
    if recipients and os.environ.get("SMTP_HOST"):
        try:
            _send_email(recipients, title, body)
            status["email"] = True
        except (OSError, smtplib.SMTPException) as exc:
            logger.warning("Email alert failed: %s", exc)
            status["email"] = False
    if not status:
        logger.warning("No alert channel configured (set SLACK_WEBHOOK_URL or SMTP_*)")
    return status


def airflow_failure_callback(context: Dict[str, Any]) -> None:
    """Airflow on_failure_callback: alert with the DAG, task, error and log link."""
    ti = context.get("task_instance")
    dag_id = getattr(ti, "dag_id", "unknown")
    task_id = getattr(ti, "task_id", "unknown")
    body = (
        f"DAG: {dag_id}\nTask: {task_id}\n"
        f"Run: {context.get('run_id')}\n"
        f"Try: {getattr(ti, 'try_number', '?')}\n"
        f"Error: {context.get('exception')}\n"
        f"Log: {getattr(ti, 'log_url', 'n/a')}"
    )
    send_alert(f"Task failed: {dag_id}.{task_id}", body)
