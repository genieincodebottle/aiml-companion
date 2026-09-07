"""One typed settings object. The root dependency of every other module."""
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_FILE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def _defaults() -> dict:
    if _CONFIG_FILE.exists():
        return yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8")) or {}
    return {}


class Settings(BaseSettings):
    """Values come from configs/base.yaml, overridden by the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # model
    backbone: str = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    image_size: int = 224
    embed_dim: int = 768
    num_classes: int = 7

    # The two numbers the whole gate depends on. Widening accept_above from 0.98
    # to 0.99 sends more images to a human, which is a cost and quality trade a
    # plant manager makes rather than an engineering constant.
    accept_above: float = 0.98
    reject_below: float = 0.02

    # Fitted on validation by src/calibrate.py and written back.
    temperature: float = 1.0

    index_path: str = "artifacts/index.faiss"
    top_k: int = 8

    data_root: str = "data/"
    device: str = "cpu"


settings = Settings(**_defaults())
