from __future__ import annotations

import logging

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score

from ml_blueprint.config import AppConfig

logger = logging.getLogger(__name__)


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
