import pandas as pd
from sklearn.model_selection import train_test_split
from ml_blueprint.config import DataConfig


class DataHandler:
    def __init__(self, data_config: DataConfig):
        self.cfg = data_config
        # Local copies for convenience
        self.train_size = float(self.cfg.split.train_size)
        self.val_size = float(self.cfg.split.val_size)
        self.test_size = float(self.cfg.split.test_size)
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test sizes must sum to 1.0, got {total:.6f}")
        self.random_state = int(self.cfg.split.random_state)
        self.train_df = None

    def prepare_data(self):
        df = pd.read_csv(self.cfg.source_path / "train.csv", index_col=0)
        # Split into train, validation, and test sets per config
        temp_size = 1.0 - self.train_size
        self.train_df, temp = train_test_split(
            df, test_size=temp_size, random_state=self.random_state
        )
        # Proportion of test within temp chunk
        denom = self.val_size + self.test_size
        test_share_in_temp = self.test_size / denom if denom > 0 else 0.5
        self.validation_df, self.test_df = train_test_split(
            temp, test_size=test_share_in_temp, random_state=self.random_state
        )

        # Display the sizes of the splits
        print("Length of train set:", len(self.train_df))
        print("Length of validation set:", len(self.validation_df))
        print("Length of test set:", len(self.test_df))

        self.train_df.to_csv(self.cfg.target_path / "train.csv", index=True)
        self.validation_df.to_csv(self.cfg.target_path / "validation.csv", index=True)
        self.test_df.to_csv(self.cfg.target_path / "test.csv", index=True)

    def get_data(self):
        if self.train_df is None:
            self.prepare_data()
        train_df = pd.read_csv(self.cfg.target_path / "train.csv", index_col=0)
        validation_df = pd.read_csv(
            self.cfg.target_path / "validation.csv", index_col=0
        )
        test_df = pd.read_csv(self.cfg.target_path / "test.csv", index_col=0)
        return train_df, validation_df, test_df
