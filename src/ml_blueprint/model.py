import numpy as np
import pandas as pd
from typing import Optional
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from ml_blueprint.config import ModelConfig, TrainingConfig


class ModelTrainer:
    def __init__(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        training_cfg: TrainingConfig,
    ):
        self.train_df = train_df
        self.validation_df = validation_df
        self.test_df = test_df
        self.training_cfg = training_cfg

    def separate_columns(self) -> None:
        # Separate features/target
        self.y_train = self.train_df[self.training_cfg.target_column].astype(float)
        self.X_train = self.train_df.drop(
            columns=[self.training_cfg.target_column] + self.training_cfg.drop_columns
        )

        # Infer column types
        self.cat_cols = self.X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.num_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()

        print("Categorical cols:", self.cat_cols)
        print("Numeric cols:", self.num_cols)

    def build_pipeline(self, model_cfg: Optional[ModelConfig] = None) -> Pipeline:
        # Numeric: median impute (XGBoost doesn't require scaling)
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        # Categorical: impute missing with constant and OHE with 'ignore' for unseen
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )

        preprocess = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.num_cols),
                ("cat", cat_pipe, self.cat_cols),
            ]
        )

        # Configure XGBoost from config (fallback to defaults)
        params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 7,
            "subsample": 0.4,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "objective": "reg:squarederror",
            "booster": "dart",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "auto",
            "verbosity": 2,
        }

        if model_cfg is not None:
            # Override with xgb-specific settings
            xgb_overrides = model_cfg.xgb.model_dump(exclude_none=True)
            params.update(xgb_overrides)
            # Top-level random_state takes precedence if provided
            if model_cfg.random_state is not None:
                params["random_state"] = model_cfg.random_state

        xgb = XGBRegressor(**params)
        print("xgb.get_params():", xgb.get_params())

        self.model = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("xgb", xgb),
            ]
        )

        return self.model

    def evaluate(self) -> float:
        X_val = self.validation_df.drop(
            columns=[self.training_cfg.target_column] + self.training_cfg.drop_columns
        )
        y_val = self.validation_df[self.training_cfg.target_column].astype(float)

        pred_valid = self.model.predict(X_val)
        mse = mean_squared_error(y_val, pred_valid)
        print(f"Validation MSE: {mse:,.3f}")
        return mse
