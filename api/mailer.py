"""SMTP email delivery for analysis and password-reset messages.

Set SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM, and
SMTP_USE_TLS in the environment. When SMTP_HOST is not configured, the
application keeps the report in SQLite and returns a clear not-configured
status instead of pretending that an email was sent.
"""

import os
import smtplib
from email.message import EmailMessage
from typing import Dict


def send_email(recipient: str, subject: str, body: str) -> Dict[str, str]:
    host = os.getenv("SMTP_HOST")
    if not host:
        return {"status": "not_configured", "message": "SMTP_HOST is not configured; report was saved but not emailed."}

    username = os.getenv("SMTP_USERNAME", "")
    password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("SMTP_FROM", username or "noreply@tumorganet.local")
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)

    configured_port = int(os.getenv("SMTP_PORT", "587"))
    # Port 587 is STARTTLS and port 465 is implicit SSL. Prefer the standard
    # port semantics so a stale SMTP_USE_SSL value cannot put SSL on port 587.
    configured_ssl = configured_port == 465
    attempts = [(configured_port, configured_ssl)]
    # Gmail commonly supports both STARTTLS/587 and implicit SSL/465. If the
    # configured transport is closed by the provider, retry the other secure mode.
    alternate = (465, True) if configured_port == 587 else (587, False)
    if alternate not in attempts:
        attempts.append(alternate)

    failures = []
    for port, use_ssl in attempts:
        try:
            smtp_class = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
            with smtp_class(host, port, timeout=30) as smtp:
                if not use_ssl and os.getenv("SMTP_USE_TLS", "true").lower() != "false":
                    smtp.ehlo()
                    smtp.starttls()
                    smtp.ehlo()
                if username:
                    smtp.login(username, password)
                smtp.send_message(message)
            return {"status": "sent", "message": f"Report emailed to {recipient} via port {port}."}
        except smtplib.SMTPAuthenticationError:
            raise
        except (smtplib.SMTPException, OSError) as error:
            failures.append(f"port {port}: {type(error).__name__}: {error}")

    raise ConnectionError("; ".join(failures))
