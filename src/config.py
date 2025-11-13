from dataclasses import dataclass
from pathlib import Path

@dataclass
class TwitterPathConfig:
    BASE_DIR: Path = Path().resolve()
    DATASET_DIR: Path = BASE_DIR / "data" / "TwitterSentimentAnalysisDataset" / "preprocessed"
    DATASET_FILE_PATH: Path = DATASET_DIR / "twitter_training.csv"

@dataclass
class LSTMModelConfig:
    embedding_dim: int = 100
    lstm_units: int = 64
    dense_units: int = 64
    dropout_rate: float = 0.3
    maxlen: int = 100
    batch_size: int = 64
    epochs: int = 15
    test_size: float = 0.4
    random_state: int = 42
