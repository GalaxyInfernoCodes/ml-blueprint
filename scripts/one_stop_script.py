from ml_blueprint.data import DataHandler
from ml_blueprint.model import ModelTrainer
from ml_blueprint.config import load_config


cfg = load_config("config/config.yaml")

data_handler = DataHandler(cfg.data)
data_handler.prepare_data()
train_df, validation_df, test_df = data_handler.get_data()

model_trainer = ModelTrainer(
    train_df, validation_df, test_df, training_cfg=cfg.training
)
model_trainer.separate_columns()
model = model_trainer.build_pipeline(cfg.model)
model.fit(model_trainer.X_train, model_trainer.y_train)
model_trainer.evaluate()
