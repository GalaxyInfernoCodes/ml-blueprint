"""
One-stop script for Hotel Cancellation prediction.
Demonstrates:
- DuckDB for data loading and splitting
- CatBoost for classification
- Modular code structure for ML tutorials
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import duckdb
import pandas as pd
import mlflow
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score

from ml_blueprint.config import load_config, AppConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DuckDBDataHandler:
    """Handle data loading and splitting using DuckDB."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.db_path = config.data.db_path
        self.source_path = config.data.source_path
        self.train_size = config.data.split.train_size
        self.val_size = config.data.split.val_size
        self.test_size = config.data.split.test_size
        self.random_state = config.data.split.random_state
        
        if not self.db_path:
            raise ValueError("db_path must be set in configuration")
    
    def prepare_data(self) -> None:
        """Load CSV into DuckDB and split into train/val/test tables."""
        csv_path = self.source_path / "train.csv"
        logger.info(f"Loading data from {csv_path}...")
        
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute(f"""
                CREATE OR REPLACE TABLE full_data AS 
                SELECT * FROM read_csv_auto('{csv_path}')
            """)
            
            # Get total count
            total_rows = conn.execute("SELECT COUNT(*) FROM full_data").fetchone()[0]
            logger.info(f"Total rows: {total_rows:,}")
            
            # Create splits using percentage-based sampling
            train_pct = self.train_size * 100
            val_pct = self.val_size * 100
            
            logger.info("Creating train/validation/test splits...")
            
            # Add row number for deterministic splitting
            conn.execute(f"""
                CREATE OR REPLACE TABLE full_data_indexed AS
                SELECT *, 
                       hash(hotel || reservation_status_date || {self.random_state}) as row_hash
                FROM full_data
            """)
            
            # Create train table (first portion)
            conn.execute(f"""
                CREATE OR REPLACE TABLE train AS
                SELECT * EXCLUDE (row_hash)
                FROM full_data_indexed
                WHERE (row_hash % 100) < {int(train_pct)}
            """)
            
            # Create validation table (middle portion)
            conn.execute(f"""
                CREATE OR REPLACE TABLE validation AS
                SELECT * EXCLUDE (row_hash)
                FROM full_data_indexed
                WHERE (row_hash % 100) >= {int(train_pct)} 
                  AND (row_hash % 100) < {int(train_pct + val_pct)}
            """)
            
            # Create test table (remaining portion)
            conn.execute(f"""
                CREATE OR REPLACE TABLE test AS
                SELECT * EXCLUDE (row_hash)
                FROM full_data_indexed
                WHERE (row_hash % 100) >= {int(train_pct + val_pct)}
            """)
            
            # Display split sizes
            train_size = conn.execute("SELECT COUNT(*) FROM train").fetchone()[0]
            val_size = conn.execute("SELECT COUNT(*) FROM validation").fetchone()[0]
            test_size = conn.execute("SELECT COUNT(*) FROM test").fetchone()[0]
            
            logger.info(f"Train set: {train_size:,} rows ({train_size/total_rows*100:.1f}%)")
            logger.info(f"Validation set: {val_size:,} rows ({val_size/total_rows*100:.1f}%)")
            logger.info(f"Test set: {test_size:,} rows ({test_size/total_rows*100:.1f}%)")
    
    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Retrieve train/val/test DataFrames from DuckDB."""
        with duckdb.connect(str(self.db_path)) as conn:
            train_df = conn.execute("SELECT * FROM train").fetchdf()
            val_df = conn.execute("SELECT * FROM validation").fetchdf()
            test_df = conn.execute("SELECT * FROM test").fetchdf()
        
        return train_df, val_df, test_df


class CatBoostTrainer:
    """Train and evaluate CatBoost classifier."""
    
    def __init__(self, config: AppConfig, train_df: pd.DataFrame, 
                 val_df: pd.DataFrame, test_df: pd.DataFrame):
        self.config = config
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        self.target_column = config.training.target_column
        self.drop_columns = config.training.drop_columns
        
    def prepare_features(self) -> None:
        """Separate features and target, identify categorical columns."""
        # Prepare training data
        self.y_train = self.train_df[self.target_column]
        self.X_train = self.train_df.drop(columns=[self.target_column] + self.drop_columns)
        
        # Prepare validation data
        self.y_val = self.val_df[self.target_column]
        self.X_val = self.val_df.drop(columns=[self.target_column] + self.drop_columns)
        
        # Prepare test data
        self.y_test = self.test_df[self.target_column]
        self.X_test = self.test_df.drop(columns=[self.target_column] + self.drop_columns)
        
        # Identify categorical features (CatBoost handles them natively)
        self.cat_features = self.X_train.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values
        # For numeric columns: fill with median
        numeric_cols = self.X_train.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            median_val = self.X_train[col].median()
            self.X_train[col] = self.X_train[col].fillna(median_val)
            self.X_val[col] = self.X_val[col].fillna(median_val)
            self.X_test[col] = self.X_test[col].fillna(median_val)
        
        # For categorical columns: fill with 'missing'
        for col in self.cat_features:
            self.X_train[col] = self.X_train[col].fillna('missing')
            self.X_val[col] = self.X_val[col].fillna('missing')
            self.X_test[col] = self.X_test[col].fillna('missing')
        
        logger.info(f"Target column: {self.target_column}")
        logger.info(f"Dropped columns: {self.drop_columns}")
        logger.info(f"Categorical features ({len(self.cat_features)}): {self.cat_features[:5]}...")
        logger.info(f"Total features: {len(self.X_train.columns)}")
        logger.info(f"Class distribution in training:\n{self.y_train.value_counts(normalize=True)}")
    
    def train(self) -> CatBoostClassifier:
        """Train CatBoost classifier."""
        if not self.config.model.catboost:
            raise ValueError("CatBoost configuration missing")

        # Get CatBoost parameters from config
        cb_params = self.config.model.catboost.model_dump(exclude_none=True)
        if self.config.model.random_state is not None:
            cb_params['random_state'] = self.config.model.random_state
        
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
        
        logger.info("Training CatBoost model...")
        self.model = CatBoostClassifier(**cb_params)
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            plot=False
        )
        
        return self.model
    
    def evaluate(self) -> dict:
        """Evaluate model on validation set."""
        # Predictions
        y_pred = self.model.predict(self.X_val)
        y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_val, y_pred)
        roc_auc = roc_auc_score(self.y_val, y_pred_proba)
        f1_macro = f1_score(self.y_val, y_pred, average='macro')
        
        logger.info("="*50)
        logger.info("VALIDATION RESULTS")
        logger.info("="*50)
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"F1 Macro: {f1_macro:.4f}")
        logger.info(f"Classification Report:\n{classification_report(self.y_val, y_pred)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 Most Important Features:\n{feature_importance.head(10).to_string(index=False)}")
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1_macro': f1_macro
        }


def run_pipeline(config_path: Path) -> None:
    """Run the end-to-end pipeline."""
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

    # Data handling with DuckDB
    data_handler = DuckDBDataHandler(config)
    data_handler.prepare_data()
    train_df, val_df, test_df = data_handler.get_data()
    
    # MLflow tracking
    mlflow.set_experiment("Hotel Cancellation Experiment")
    with mlflow.start_run(run_name="CatBoost Baseline"):
        # Model training
        trainer = CatBoostTrainer(config, train_df, val_df, test_df)
        trainer.prepare_features()
        model = trainer.train()
        metrics = trainer.evaluate()
        
        # Log to MLflow
        mlflow.log_param("target_column", config.training.target_column)
        mlflow.log_param("model_type", "CatBoostClassifier")
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("val_size", len(val_df))
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.catboost.log_model(model, "model")
        
        logger.info("✅ Pipeline completed successfully!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Hotel Cancellation prediction pipeline with DuckDB and CatBoost."
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
