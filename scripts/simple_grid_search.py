"""
Simple grid search script for CatBoost parameters.
"""
from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
import copy
import yaml

import mlflow
import pandas as pd

from ml_blueprint.config import load_config
from ml_blueprint.data import DuckDBDataHandler
from ml_blueprint.model import CatBoostTrainer
from catboost import CatBoostClassifier, Pool

class GridSearchCatBoostTrainer(CatBoostTrainer):
    """CatBoostTrainer with parameter override support."""
    
    def __init__(self, config, train_df, val_df, test_df, override_params=None):
        super().__init__(config, train_df, val_df, test_df)
        self.override_params = override_params or {}

    def train(self) -> CatBoostClassifier:
        """Train CatBoost classifier with overrides."""
        if not self.config.model.catboost:
            raise ValueError("CatBoost configuration missing")

        # Get CatBoost parameters from config
        cb_params = self.config.model.catboost.model_dump(exclude_none=True)
        if self.config.model.random_state is not None:
            cb_params['random_state'] = self.config.model.random_state
        
        # Apply overrides
        cb_params.update(self.override_params)
        
        # Create CatBoost pools for efficient training
        train_pool = Pool(
            data=self.X_train,
            label=self.y_train,
            cat_features=self.cat_features
        )
        
        val_pool = Pool(
            data=self.X_val,
            label=self.y_val,
            cat_features=self.cat_features
        )
        
        logger.info(f"Training CatBoost model with params: {cb_params}")
        self.model = CatBoostClassifier(**cb_params)
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            plot=False
        )
        
        return self.model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_grid_search(config_path: Path, metric: str = 'accuracy') -> None:
    """Run grid search for CatBoost parameters."""
    # Load configuration
    try:
        base_config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

    # Data handling with DuckDB (load once)
    logger.info("Initializing DuckDBDataHandler...")
    data_handler = DuckDBDataHandler(base_config)
    data_handler.prepare_data()
    train_df, val_df, test_df = data_handler.get_data()
    
    # Define parameter grid from config
    # We read the YAML directly to bypass Pydantic model strictness if grid_search is not in AppConfig
    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f) or {}
            
        # Handle namespaces similar to AppConfig
        if "ml_blueprint" in raw_config:
            raw_config = raw_config["ml_blueprint"]
        elif "hotel_cancellation" in raw_config:
            raw_config = raw_config["hotel_cancellation"]
            
        grid_search_config = raw_config.get("grid_search", {})
    except Exception as e:
        logger.warning(f"Failed to read raw config for grid_search: {e}")
        grid_search_config = {}

    if not grid_search_config:
        logger.warning("No grid_search configuration found in YAML. Using default parameters.")
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5]
        }
    else:
        param_grid = grid_search_config

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"Starting grid search with {len(combinations)} combinations...")
    
    best_score = 0
    best_params = {}
    
    # MLflow tracking
    mlflow.set_experiment("Hotel Cancellation Grid Search")
    
    for i, params in enumerate(combinations):
        logger.info(f"Running combination {i+1}/{len(combinations)}: {params}")
        
        # Create a config copy with updated parameters
        # We need to be careful to update the pydantic model correctly
        # Since config.model.catboost is a Pydantic model, we can't just assign dict values directly if we want to be safe,
        # but here we are modifying the config object which will be passed to the trainer.
        
        # Deep copy the config to avoid side effects (though we are not modifying it anymore)
        current_config = copy.deepcopy(base_config)
        
        with mlflow.start_run(run_name=f"Grid Search Run {i+1}"):
            # Model training
            # Use our subclass to inject parameters
            trainer = GridSearchCatBoostTrainer(current_config, train_df, val_df, test_df, override_params=params)
            trainer.prepare_features()
            model = trainer.train()
            metrics = trainer.evaluate()
            
            # Log params
            mlflow.log_params(params)
            mlflow.log_param("iterations", current_config.model.catboost.iterations)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Track best
            current_score = metrics.get(metric, 0)
            if current_score > best_score:
                best_score = current_score
                best_params = params
                logger.info(f"New best score: {best_score:.4f} with params {best_params}")
                
    logger.info("="*50)
    logger.info("GRID SEARCH COMPLETE")
    logger.info("="*50)
    logger.info(f"Best {metric.capitalize()}: {best_score:.4f}")
    logger.info(f"Best Parameters: {best_params}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run grid search for Hotel Cancellation pipeline."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="accuracy",
        help="Metric to optimize (default: accuracy).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    run_grid_search(args.config, args.metric)


if __name__ == "__main__":
    main()
