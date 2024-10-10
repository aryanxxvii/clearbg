from clearbg.constants import *
from clearbg.utils.common import read_yaml, create_directories
from clearbg.entity.config_entity import DataIngestionConfig, TrainingConfig, EvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        root_dir = (PROJECT_ROOT / config.root_dir).resolve()
        train_local_zipped_path = (PROJECT_ROOT / config.train_local_zipped_path).resolve()
        test_local_zipped_path = (PROJECT_ROOT / config.test_local_zipped_path).resolve()
        
        create_directories([root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=root_dir,
            train_source_url=config.train_source_url,
            test_source_url=config.test_source_url,
            train_local_zipped_path=train_local_zipped_path,
            test_local_zipped_path=test_local_zipped_path
        )
        return data_ingestion_config
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        params = self.params.training
        print(config)
        # Resolve the root directory for training artifacts
        root_dir = (PROJECT_ROOT / config.root_dir).resolve()

        training_config = TrainingConfig(
            root_dir=root_dir,  # Include root_dir in the training config
            epochs=params.epochs,
            image_size=params.image_size,
            learning_rate=params.learning_rate,
            batch_size=params.batch_size
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        # Here we gather configurations needed for the model evaluation
        return EvaluationConfig(
            model_path=Path(self.config['model_evaluation']['model_path']),
            test_data_dir=Path(self.config['model_evaluation']['test_data_dir']),
            prediction_dir=Path(self.config['model_evaluation']['prediction_dir']),
            params=self.params  
        )