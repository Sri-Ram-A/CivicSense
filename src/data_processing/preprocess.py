import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from loguru import logger

from src.config import TwitterPathConfig, LSTMModelConfig

def load_data(path_config: TwitterPathConfig):
    logger.info(f"Loading dataset from {path_config.DATASET_FILE_PATH}")
    df = pd.read_csv(path_config.DATASET_FILE_PATH, usecols=["sentiment", "cleaned_text"])
    df['cleaned_text'] = df['cleaned_text'].astype(str)
    logger.info(f"Loaded {len(df)} samples")
    return df

def tokenize_and_pad(df: pd.DataFrame, model_config:LSTMModelConfig):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['cleaned_text'])
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
    padded_sequences = pad_sequences(sequences, maxlen=model_config.maxlen, padding='pre')
    vocab_size = len(tokenizer.word_index) + 1
    logger.info(f"Total unique tokens: {vocab_size}")
    return padded_sequences, vocab_size

def encode_labels(df: pd.DataFrame):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['sentiment'])
    y = to_categorical(y_encoded)
    logger.info(f"Encoded {len(label_encoder.classes_)} sentiment classes.")
    return y, label_encoder
