from __future__ import annotations

import logging
from typing import Tuple

import duckdb
import pandas as pd

from ml_blueprint.config import AppConfig

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
