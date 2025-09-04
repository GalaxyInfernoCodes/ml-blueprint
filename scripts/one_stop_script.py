from ml_blueprint.data import DataHandler
from ml_blueprint.model import ModelTrainer
from pathlib import Path


data_source = Path("./data/playground-series-s4e12")
data_target = Path("./data")

data_handler = DataHandler(data_source_path=data_source, data_target_path=data_target)
data_handler.prepare_data()
train_df, validation_df, test_df = data_handler.get_data()

model_trainer = ModelTrainer(train_df, validation_df, test_df)
model_trainer.separate_columns(target_col="Premium Amount")
