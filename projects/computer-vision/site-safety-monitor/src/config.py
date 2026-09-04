"""One typed settings object, including the budget and the two thresholds
that shape the alert stream."""
from __future__ import annotations

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pydantic v1 keeps BaseSettings in the main package
    from pydantic import BaseSettings

    SettingsConfigDict = None


class Settings(BaseSettings):
    cameras: int = 2
    fps: int = 15
    imgsz: int = 640

    detect_every: int = 2          # tracker carries the frames between
    conf_threshold: float = 0.45   # precision 0.89, recall 0.88
    iou_threshold: float = 0.50
    max_detections: int = 300      # NMS is quadratic in survivors

    dwell_seconds: float = 3.0     # below this it is a person walking past
    class_vote_window: int = 15    # one second of evidence at 15 fps
    alert_cooldown_s: float = 300.0
    clip_seconds: int = 7
    track_lost_max: int = 30

    shift_hours: float = 10.0
    engine_path: str = "artifacts/yolo.engine"

    # 1000 / (2 x 15). Every stage is measured against this.
    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / (self.cameras * self.fps)

    @property
    def frames_per_shift(self) -> int:
        return int(self.cameras * self.fps * 3600 * self.shift_hours)

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    else:                                     # pragma: no cover - pydantic v1
        class Config:
            env_file = ".env"
            extra = "ignore"


settings = Settings()
