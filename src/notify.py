"""Email notification for the model pipeline (same approach as the course Airflow lab:
smtplib + Gmail SMTP). Used by the CI workflow on success and on failure.

    python -m src.notify --status success --title "model pipeline" [--summary outputs/pipeline_summary.json]

Settings come from environment variables (GitHub secrets in CI):
    SMTP_HOST (default smtp.gmail.com), SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD,
    NOTIFY_TO (comma-separated)
If SMTP_USER / SMTP_PASSWORD / NOTIFY_TO are not set, the message is only printed, so
local runs and forks without secrets do not fail.
"""
import argparse
import json
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path
from typing import Optional


def build_message(status: str, title: str, summary_path: Optional[Path], details: str = "") -> EmailMessage:
    lines = [f"SkinSort {title}: {status.upper()}", ""]
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
    msg = EmailMessage()
    msg["Subject"] = f"[SkinSort] {title}: {status}"
    msg.set_content("\n".join(lines))
    return msg


def send(msg: EmailMessage) -> bool:
    user, password, to = (os.environ.get(k) for k in ("SMTP_USER", "SMTP_PASSWORD", "NOTIFY_TO"))
    if not (user and password and to):
        print("notify: SMTP settings not set, printing instead of sending\n")
        print(msg)
        return False
    msg["From"], msg["To"] = user, to
    with smtplib.SMTP(os.environ.get("SMTP_HOST", "smtp.gmail.com"), int(os.environ.get("SMTP_PORT", "587"))) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
    print(f"notify: email sent to {to}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a pipeline notification email")
    parser.add_argument("--status", required=True, help="success / failure / bias-alert")
    parser.add_argument("--title", default="model pipeline")
    parser.add_argument("--summary", type=Path, default=Path("outputs/pipeline_summary.json"))
    parser.add_argument("--details", default="")
    args = parser.parse_args()
    try:
        send(build_message(args.status, args.title, args.summary, args.details))
    except (smtplib.SMTPException, OSError) as exc:     # a failed email must not hide the real result
        print(f"notify: could not send email: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
