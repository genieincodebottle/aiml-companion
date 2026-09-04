"""Reads RTSP with a bounded queue that drops old frames rather than falling
behind."""
from __future__ import annotations

import queue
import threading


class Camera:
    """Bounded queue of 1. A full queue drops the OLD frame, not the new one.

    An unbounded queue on a live camera grows forever once the pipeline
    slips behind, and the system then processes frames that are minutes
    stale while looking perfectly healthy.
    """

    def __init__(self, url: str, camera_id: str):
        import cv2  # imported here so the module loads without OpenCV

        self.cap = cv2.VideoCapture(url)
        self.camera_id = camera_id
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.dropped = 0
        self._stop = threading.Event()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                continue
            self._offer(frame)

    def _offer(self, frame):
        """Separated from the read loop so it is testable without a camera."""
        if self.q.full():
            try:
                self.q.get_nowait()      # discard the stale frame
                self.dropped += 1
            except queue.Empty:
                pass
        self.q.put((frame, self.camera_id))

    def stop(self):
        self._stop.set()


class ReplayCamera:
    """Same interface over a list of frames, for tests and the demo.

    It uses the identical bounded-queue drop policy, so the behaviour
    under back pressure is the thing being tested rather than a
    simplified stand-in for it.
    """

    def __init__(self, frames, camera_id: str):
        self.camera_id = camera_id
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.dropped = 0
        self._frames = list(frames)

    _offer = Camera._offer

    def feed_all(self):
        for f in self._frames:
            self._offer(f)
