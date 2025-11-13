from loguru import logger
from sklearn.model_selection import train_test_split
import numpy as np
from src import config
from src.data_processing.preprocess import load_data, tokenize_and_pad, encode_labels
from src.models.lstm_model import build_lstm_model, train_model
from src.utils.visualization import plot_training_history, plot_confusion_matrix, show_classification_report

def LSTMTrain():
    path_config = config.TwitterPathConfig()
    model_config = config.LSTMModelConfig()

    # 1️⃣ Load and preprocess data
    df = load_data(path_config)
    X, vocab_size = tokenize_and_pad(df, model_config)
    y, label_encoder = encode_labels(df)

    # 2️⃣ Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=model_config.test_size, random_state=model_config.random_state
    )

    # 3️⃣ Build and train model
    model = build_lstm_model(vocab_size, y.shape[1], model_config)
    history = train_model(model, X_train, y_train, X_val, y_val, model_config)

    # 4️⃣ Visualizations and metrics
    plot_training_history(history)
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)
    plot_confusion_matrix(y_true, y_pred, label_encoder)
    show_classification_report(y_true, y_pred, label_encoder)

if __name__ == "__main__":
    logger.add("training_log.log", rotation="1 MB", level="INFO")
    LSTMTrain()
