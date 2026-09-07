"""Watches for the failures that produce silence rather than errors."""
from __future__ import annotations

import time

NO_FRAMES_S = 30.0


class Health:
    """Alerting systems fail by going quiet, which looks like a safe site.

    A camera that dropped its stream, a queue that stopped draining, and
    a model loaded with the wrong classes all produce zero alerts, and
    zero alerts is indistinguishable from good news.
    """

    def check(self, last_frame_ts, alerts_24h, detections_1h, now=None):
        now = time.time() if now is None else now
        problems = []
        if now - last_frame_ts > NO_FRAMES_S:
            problems.append("no frames for 30 s, camera or decoder down")
        if detections_1h == 0:
            problems.append("no detections in an hour, model or input broken")
        if alerts_24h == 0:
            problems.append("no alerts in 24 h, verify with a test walk")
        return problems
