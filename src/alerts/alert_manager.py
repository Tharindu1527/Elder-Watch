"""
Alert Manager
Dispatches fall/inactivity alerts via:
  - Twilio SMS
  - SMTP Email
  - Telegram Bot
Includes cooldown logic to prevent alert flooding.
"""

import logging
import time
import os
import threading
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alert dispatch with cooldown and snapshot capture.
    All network calls happen in a background thread to avoid blocking inference.
    """

    def __init__(self, config: dict):
        self.config           = config
        self.enabled          = config.get("enabled", True)
        self.cooldown_secs    = config.get("alert_cooldown_seconds", 30)
        self.capture_snapshot = config.get("capture_snapshot", True)
        self.snapshot_res     = tuple(config.get("snapshot_resolution", [320, 240]))

        self._last_alert_time: float = 0.0
        self._alert_count: int       = 0
        os.makedirs("logs/snapshots", exist_ok=True)

        logger.info(f"AlertManager initialized. Channels: "
                    f"SMS={config.get('twilio_enabled',False)}, "
                    f"Email={config.get('email_enabled',False)}, "
                    f"Telegram={config.get('telegram_enabled',False)}")

    # ──────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────

    def trigger(self, event_type: str, snapshot: Optional[np.ndarray] = None):
        """
        Trigger an alert (respects cooldown).
        event_type: 'FALL_DETECTED' or 'INACTIVE_ALERT'
        """
        if not self.enabled:
            return

        now = time.time()
        if now - self._last_alert_time < self.cooldown_secs:
            remaining = self.cooldown_secs - (now - self._last_alert_time)
            logger.debug(f"Alert cooldown active ({remaining:.0f}s remaining)")
            return

        self._last_alert_time = now
        self._alert_count += 1

        ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg     = self._build_message(event_type, ts)
        img_path = None

        # Save snapshot
        if self.capture_snapshot and snapshot is not None:
            img_path = self._save_snapshot(snapshot, ts, event_type)

        # Dispatch in background thread (non-blocking)
        t = threading.Thread(
            target=self._dispatch_all,
            args=(msg, img_path, event_type),
            daemon=True
        )
        t.start()
        logger.warning(f"ALERT #{self._alert_count} triggered: {event_type} @ {ts}")

    # ──────────────────────────────────────────────────────────────────
    # Message building
    # ──────────────────────────────────────────────────────────────────

    def _build_message(self, event_type: str, timestamp: str) -> str:
        labels = {
            "FALL_DETECTED":  "⚠️ FALL DETECTED",
            "INACTIVE_ALERT": "⏰ PROLONGED INACTIVITY DETECTED",
        }
        label = labels.get(event_type, f"⚠️ ALERT: {event_type}")
        return (
            f"{label}\n"
            f"Time: {timestamp}\n"
            f"System: Elder Watch (University of Ruhuna)\n"
            f"Action required: Please check on your elder immediately.\n"
            f"Alert #{self._alert_count}"
        )

    def _save_snapshot(self, frame: np.ndarray, ts: str, event_type: str) -> str:
        safe_ts = ts.replace(":", "-").replace(" ", "_")
        fname   = f"logs/snapshots/{event_type}_{safe_ts}.jpg"
        small   = cv2.resize(frame, self.snapshot_res)
        # Overlay event label on snapshot
        cv2.putText(small, event_type, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(fname, small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        logger.info(f"Snapshot saved: {fname}")
        return fname

    # ──────────────────────────────────────────────────────────────────
    # Dispatch
    # ──────────────────────────────────────────────────────────────────

    def _dispatch_all(self, message: str, img_path: Optional[str], event_type: str):
        if self.config.get("twilio_enabled", False):
            self._send_sms(message)

        if self.config.get("email_enabled", False):
            self._send_email(message, img_path, event_type)

        if self.config.get("telegram_enabled", False):
            self._send_telegram(message, img_path)

    def _send_sms(self, message: str):
        try:
            from twilio.rest import Client
            client = Client(
                self.config["twilio_account_sid"],
                self.config["twilio_auth_token"]
            )
            client.messages.create(
                body=message,
                from_=self.config["twilio_from"],
                to=self.config["twilio_to"]
            )
            logger.info("SMS alert sent via Twilio.")
        except Exception as e:
            logger.error(f"Twilio SMS failed: {e}")

    def _send_email(self, message: str, img_path: Optional[str], subject_suffix: str):
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.mime.image import MIMEImage

            msg = MIMEMultipart()
            msg["From"]    = self.config["smtp_user"]
            msg["To"]      = self.config["recipient_email"]
            msg["Subject"] = f"Elder Watch Alert: {subject_suffix}"
            msg.attach(MIMEText(message, "plain"))

            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    img_data = f.read()
                image = MIMEImage(img_data, name=os.path.basename(img_path))
                msg.attach(image)

            with smtplib.SMTP(self.config["smtp_host"], self.config["smtp_port"]) as s:
                s.starttls()
                s.login(self.config["smtp_user"], self.config["smtp_password"])
                s.send_message(msg)
            logger.info("Email alert sent.")
        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    def _send_telegram(self, message: str, img_path: Optional[str]):
        try:
            import requests
            token   = self.config["telegram_bot_token"]
            chat_id = self.config["telegram_chat_id"]
            base    = f"https://api.telegram.org/bot{token}"

            # Send text
            requests.post(f"{base}/sendMessage",
                          data={"chat_id": chat_id, "text": message},
                          timeout=10)

            # Send photo if available
            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    requests.post(f"{base}/sendPhoto",
                                  data={"chat_id": chat_id},
                                  files={"photo": f},
                                  timeout=15)
            logger.info("Telegram alert sent.")
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")
