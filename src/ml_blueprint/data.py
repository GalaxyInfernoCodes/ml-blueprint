import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, data_source_path: Path, data_target_path: Path):
        self.data_source_path = data_source_path
        self.data_target_path = data_target_path
        self.train_df = None

    def prepare_data(self):
        df = pd.read_csv(self.data_source_path / "train.csv", index_col=0)
        # Split into train, validation, and test sets, # 60% train, 40% temp
        self.train_df, temp = train_test_split(df, test_size=0.4, random_state=42)
        # 20% validation, 20% test
        self.validation_df, self.test_df = train_test_split(
            temp, test_size=0.5, random_state=42
        )

        # Display the sizes of the splits
        print("Length of train set:", len(self.train_df))
        print("Length of validation set:", len(self.validation_df))
        print("Length of test set:", len(self.test_df))

        self.train_df.to_csv(self.data_target_path / "train.csv", index=True)
        self.validation_df.to_csv(self.data_target_path / "validation.csv", index=True)
        self.test_df.to_csv(self.data_target_path / "test.csv", index=True)

    def get_data(self):
        if self.train_df is None:
            self.prepare_data()
        train_df = pd.read_csv(self.data_target_path / "train.csv", index_col=0)
        validation_df = pd.read_csv(
            self.data_target_path / "validation.csv", index_col=0
        )
        test_df = pd.read_csv(self.data_target_path / "test.csv", index_col=0)
        return train_df, validation_df, test_df
