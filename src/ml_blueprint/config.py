from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class SplitConfig(BaseModel):
    train_size: float = Field(0.6, ge=0.0, le=1.0)
    val_size: float = Field(0.2, ge=0.0, le=1.0)
    test_size: float = Field(0.2, ge=0.0, le=1.0)
    random_state: int = 42

    @model_validator(mode="after")
    def _sizes_sum_to_one(self) -> "SplitConfig":
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test sizes must sum to 1.0, got {total:.6f}")
        return self


class DataConfig(BaseModel):
    source_path: Path = Path("./data/playground-series-s4e12")
    target_path: Path = Path("./data")
    db_path: Optional[Path] = None
    split: SplitConfig = SplitConfig()


class XGBConfig(BaseModel):
    n_estimators: int = Field(1000, ge=1)
    learning_rate: float = Field(0.05, gt=0, le=1)
    max_depth: int = Field(10, ge=1)
    subsample: float = Field(0.7, gt=0, le=1)
    colsample_bytree: float = Field(0.8, gt=0, le=1)
    reg_lambda: float = Field(1.0, ge=0)
    reg_alpha: float = Field(0.0, ge=0)
    objective: str = "reg:squarederror"
    booster: str = "dart"
    random_state: Optional[int] = None
    n_jobs: int = -1
    tree_method: str = "auto"
    verbosity: int = 2


class CatBoostConfig(BaseModel):
    iterations: int = Field(500, ge=1)
    learning_rate: float = Field(0.1, gt=0, le=1)
    depth: int = Field(6, ge=1)
    l2_leaf_reg: float = Field(3.0, ge=0)
    early_stopping_rounds: Optional[int] = Field(None, ge=1)
    verbose: int = 100
    random_state: Optional[int] = None


class ModelConfig(BaseModel):
    random_state: int = 42
    xgb: Optional[XGBConfig] = None
    catboost: Optional[CatBoostConfig] = None


class TrainingConfig(BaseModel):
    target_column: str = "Premium Amount"
    drop_columns: list[str] = []


class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AppConfig":
        """Load and validate config from a YAML file.

        Accepts top-level keys either directly or under a namespaced
        key `ml_blueprint` or `hotel_cancellation` for convenience.
        """
        yaml_path = Path(path)
        with yaml_path.open("r", encoding="utf-8") as f:
            data: Any = yaml.safe_load(f) or {}

        # Handle namespaces
        if isinstance(data, dict):
            if "ml_blueprint" in data:
                data = data["ml_blueprint"]
            elif "hotel_cancellation" in data:
                data = data["hotel_cancellation"]

        return cls.model_validate(data)


def load_config(path: Path | str = "config/config.yaml") -> AppConfig:
    """Convenience loader with sensible default path.

    Raises `ValidationError` if the YAML doesn't match the schema.
    """
    return AppConfig.from_yaml(path)
