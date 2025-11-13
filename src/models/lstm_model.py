from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from loguru import logger
from src import config 

def build_lstm_model(vocab_size: int, num_classes: int, config: config.LSTMModelConfig):
    model = Sequential([
        Embedding(vocab_size, config.embedding_dim, input_length=config.maxlen),
        LSTM(config.lstm_units, return_sequences=False),
        Dropout(config.dropout_rate),
        Dense(config.dense_units, activation='relu'),
        Dropout(config.dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    logger.info("LSTM model built and compiled successfully.")
    return model

def train_model(model, X_train, y_train, X_val, y_val, config: config.LSTMModelConfig):
    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1
    )
    logger.info("Training complete.")
    return history
