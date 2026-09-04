# Discrepancy engine — decides whether a frame is worth harvesting.
#
# Harvest condition:
#   delta > tau   (teacher and student disagree significantly)
#   OR
#   entropy > entropy_tau  (student is uncertain regardless of delta)
#
# Tau is kept stable by a simple PID controller that targets a 5% harvest
# rate.  Without this, a scene change would flood the harvest queue or
# starve it depending on which direction the statistics shift.

import asyncio
import os
import smtplib
import time
from collections import deque
from email.mime.text import MIMEText

import httpx
import numpy as np
from loguru import logger

from config import PI_BASE_URL, TAU_INITIAL
from database import Database, Event

# PID gains — empirically tuned for 10 Hz, 100-frame window
_Kp = 0.30
_Ki = 0.02
_Kd = 0.05

TARGET_HARVEST_RATE = 0.05   # aim for 5% of frames harvested
TAU_MIN             = 0.05
TAU_MAX             = 0.80
ENTROPY_TAU_INITIAL = 0.85   # flag frames where the student is very uncertain

# Email alert config — loaded from .env, silently skipped if not set
_ALERT_EMAIL    = os.getenv("ALERT_EMAIL", "")
_ALERT_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
_EMAIL_COOLDOWN = 60.0       # seconds between emails (avoid inbox flooding)


def _send_harvest_email(frame_id: str, delta: float, scenario: str, entropy: float):
    if not _ALERT_EMAIL or not _ALERT_PASSWORD:
        return
    try:
        msg = MIMEText(
            f"A harvest event was triggered.\n\n"
            f"Frame:    {frame_id}\n"
            f"Delta:    {delta:.4f}\n"
            f"Entropy:  {entropy:.4f}\n"
            f"Scenario: {scenario}"
        )
        msg["Subject"] = f"AECS-SDC harvest alert — {scenario}"
        msg["From"]    = _ALERT_EMAIL
        msg["To"]      = _ALERT_EMAIL
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(_ALERT_EMAIL, _ALERT_PASSWORD)
            server.send_message(msg)
        logger.info(f"Alert email sent for {frame_id}")
    except Exception as e:
        logger.warning(f"Email failed: {e}")


class DiscrepancyEngine:
    def __init__(self, db: Database):
        self.tau              = TAU_INITIAL
        self.entropy_tau      = ENTROPY_TAU_INITIAL
        self.db               = db
        self.harvest_count    = 0
        self.total_frames     = 0
        self.current_scenario = "unknown"

        self._harvest_window  = deque(maxlen=100)
        self._pid_integral    = 0.0
        self._pid_prev_error  = 0.0
        self._last_email_time = 0.0

    def set_scenario(self, name: str):
        self.current_scenario = name
        logger.info(f"Scenario set to: {name}")

    def set_tau(self, tau: float):
        logger.info(f"Tau: {self.tau:.3f} → {tau:.3f}")
        self.tau = float(tau)

    def _step_pid(self):
        # Wait for at least 20 frames before the rate estimate is meaningful
        if len(self._harvest_window) < 20:
            return

        rate  = sum(self._harvest_window) / len(self._harvest_window)
        error = rate - TARGET_HARVEST_RATE

        self._pid_integral += error
        deriv               = error - self._pid_prev_error
        self._pid_prev_error = error

        # Positive error means we're harvesting too much → raise tau
        adjustment = _Kp * error + _Ki * self._pid_integral + _Kd * deriv
        self.tau   = float(np.clip(self.tau + adjustment, TAU_MIN, TAU_MAX))

    async def evaluate(
        self,
        frame_id:   str,
        c_teacher:  float,
        c_student:  float,
        entropy:    float = 0.0,
        sonar_m:    float = 0.0,
        accel_mag:  float = 0.0,
        detections: list  = None,
        record:     bool  = True,
    ) -> dict:
        import json as _json
        delta = abs(c_teacher - c_student)
        self.total_frames += 1

        harvested = delta > self.tau

        self._harvest_window.append(int(harvested))
        self._step_pid()

        if record:
            self.db.log_event(Event(
                frame_id       = frame_id,
                timestamp      = time.time(),
                c_teacher      = c_teacher,
                c_student      = c_student,
                delta          = delta,
                harvested      = harvested,
                scenario       = self.current_scenario,
                entropy        = entropy,
                sonar_m        = sonar_m,
                accel_mag      = accel_mag,
                teacher_labels = _json.dumps(detections or []),
            ))

            if harvested:
                self.harvest_count += 1
                asyncio.create_task(self._trigger_harvest(frame_id))
                now = time.time()
                if now - self._last_email_time > _EMAIL_COOLDOWN:
                    self._last_email_time = now
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(
                        None, _send_harvest_email,
                        frame_id, delta, self.current_scenario, entropy
                    )

        return {
            "frame_id":      frame_id,
            "delta":         round(delta, 4),
            "tau":           round(self.tau, 4),
            "entropy":       round(entropy, 4),
            "entropy_tau":   round(self.entropy_tau, 4),
            "sonar_m":       round(sonar_m, 4),
            "accel_mag":     round(accel_mag, 4),
            "harvested":     harvested,
            "harvest_count": self.harvest_count,
            "harvest_rate":  round(self.harvest_count / max(self.total_frames, 1) * 100, 2),
        }

    async def _trigger_harvest(self, frame_id: str):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{PI_BASE_URL}/harvest",
                    params={"frame_id": frame_id},
                    timeout=5.0,
                )
            if resp.status_code == 200:
                logger.success(f"Harvest triggered: {frame_id}")
            else:
                logger.warning(f"Harvest HTTP {resp.status_code}: {frame_id}")
        except Exception as e:
            logger.error(f"Harvest trigger failed: {e}")
