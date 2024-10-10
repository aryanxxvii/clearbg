from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    train_source_url: str
    test_source_url: str
    train_local_zipped_path: Path
    test_local_zipped_path: Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path  # Added to specify the training artifacts directory
    epochs: int
    image_size: int
    learning_rate: float
    batch_size: int

@dataclass(frozen=True)
class EvaluationConfig:
    model_path: Path
    test_data_dir: Path
    prediction_dir: Path
    params: dict