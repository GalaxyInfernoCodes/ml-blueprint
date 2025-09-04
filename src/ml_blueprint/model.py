import numpy as np
import pandas as pd

class ModelTrainer:
    def __init__(self, train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame):
        self.train_df = train_df
        self.validation_df = validation_df
        self.test_df = test_df

    def separate_columns(self, target_col: str):
        # Separate features/target
        self.y_train = self.train_df[target_col].astype(float)
        self.X_train = self.train_df.drop(columns=[target_col])

        # Infer column types
        self.cat_cols = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()

        print("Categorical cols:", self.cat_cols)
        print("Numeric cols:", self.num_cols)
