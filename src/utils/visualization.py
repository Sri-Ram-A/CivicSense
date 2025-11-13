import numpy as np
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, label_encoder):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Purples', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

def show_classification_report(y_true, y_pred, label_encoder):
    logger.info("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
