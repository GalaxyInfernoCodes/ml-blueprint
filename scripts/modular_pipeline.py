"""
Modular pipeline script for Hotel Cancellation prediction.
Uses the ml_blueprint package for data handling and model training.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
from mlflow.models import infer_signature

from ml_blueprint.config import load_config
from ml_blueprint.data import DuckDBDataHandler
from ml_blueprint.model import CatBoostTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(config_path: Path) -> None:
    """Run the end-to-end pipeline using package modules."""
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

    # Data handling with DuckDB (from package)
    logger.info("Initializing DuckDBDataHandler from package...")
    data_handler = DuckDBDataHandler(config)
    data_handler.prepare_data()
    train_df, val_df, test_df = data_handler.get_data()
    
    # MLflow tracking
    mlflow.set_experiment("Hotel Cancellation Experiment")
    with mlflow.start_run(run_name="Modular Pipeline Run"):
        # Model training (from package)
        logger.info("Initializing CatBoostTrainer from package...")
        trainer = CatBoostTrainer(config, train_df, val_df, test_df)
        trainer.prepare_features()
        model = trainer.train()
        metrics = trainer.evaluate()
        
        # Log to MLflow
        mlflow.log_param("target_column", config.training.target_column)
        mlflow.log_param("model_type", "CatBoostClassifier")
        mlflow.log_param("pipeline_type", "modular")
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("val_size", len(val_df))
        mlflow.log_param("test_size", len(test_df))
        
        # Log all CatBoost parameters (including defaults)
        mlflow.log_params(model.get_all_params())
        
        mlflow.log_metrics(metrics)
        
        # Log model with signature and input example
        input_example = trainer.X_val.head()
        signature = infer_signature(trainer.X_val, trainer.y_val)
        
        mlflow.catboost.log_model(
            model, 
            "model",
            signature=signature,
            input_example=input_example
        )
        
        logger.info("✅ Modular pipeline completed successfully!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run modular Hotel Cancellation pipeline."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
