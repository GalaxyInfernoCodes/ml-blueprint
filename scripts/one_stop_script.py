from __future__ import annotations

import argparse
from pathlib import Path

from ml_blueprint.data import DataHandler
from ml_blueprint.model import ModelTrainer
from ml_blueprint.config import load_config


def run_pipeline(config_path: Path) -> None:
    """Run the end-to-end pipeline using the provided config path."""
    cfg = load_config(config_path)

    data_handler = DataHandler(cfg.data)
    data_handler.prepare_data()
    train_df, validation_df, test_df = data_handler.get_data()

    model_trainer = ModelTrainer(
        train_df, validation_df, test_df, training_cfg=cfg.training
    )
    model_trainer.separate_columns()
    model = model_trainer.build_pipeline(cfg.model)
    model.fit(model_trainer.X_train, model_trainer.y_train)
    model_trainer.evaluate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end ML pipeline.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
