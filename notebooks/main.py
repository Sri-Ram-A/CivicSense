# %%
"""
<!-- https://www.kaggle.com/code/iakhtar0/63-next-word-predictor-lstm-campusx -->
"""

# %%
%matplotlib inline

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# %%
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# %%
# Deep Learning library
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D
from tensorflow.keras import backend as K
print(tf.__version__)

# %%
from pathlib import Path
NOTEBOOK_DIR  = Path().resolve()
BASE_DIR = NOTEBOOK_DIR.parent
DATASET_DIR = BASE_DIR /  "data" / 'TwitterSentimentAnalysisDataset' / "preprocessed"
DATASET_FILE_PATH = DATASET_DIR  / 'twitter_training.csv'
DATASET_FILE_PATH.exists()

# %%
df = pd.read_csv(DATASET_FILE_PATH,usecols=["sentiment","cleaned_text"])
df.tail(3)

# %%
df["word_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))
max_words = df["word_count"].max()
avg_words = df["word_count"].mean()

print("Max words in a sentence:", max_words)
print("Average words in a sentence:", avg_words)

# %%
from tensorflow.keras.preprocessing.text import Tokenizer

# %%
df['cleaned_text'] = df['cleaned_text'].astype(str)
df.dtypes

# %%
tokenizer = Tokenizer()

# %%
# Gives ID to each word in the sentences (entire column)
tokenizer.fit_on_texts(df['cleaned_text'])
vocab_size  = len(tokenizer.word_index)
print("Total unique words : ", vocab_size )
tokenizer.word_index

# %%
tokenizer.get_config()

# %%
sequences  = tokenizer.texts_to_sequences(df['cleaned_text'])
sequences[:4]

# %%
print("Length of Sequence",len(sequences) )# = total rows
print("Length of 1st sequence",len(sequences[0])) # = 1st row word count

# %%
# Get lengths of each sentence
sentence_lengths = [len(seq) for seq in sequences]

# Plot histogram
plt.figure(figsize=(8,5))
sns.histplot(sentence_lengths, bins=50, kde=True, color='purple')
plt.title('Distribution of Sentence Lengths (in tokens)')
plt.xlabel('Sentence Length')
plt.ylabel('Number of Sentences')
plt.show()

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences
maxlen = 100  
padded_sentence_sequences = pad_sequences(sequences, maxlen = maxlen, padding='pre')
padded_sentence_sequences

# %%
padded_sentence_sequences.shape

# %%
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['sentiment'])
y = to_categorical(y_encoded)  # one-hot encode for multi-class

# %%
y_encoded

# %%
y

# %%
tokenizer.get_config()

# %%
! ipynb-py-convert 2-rnn.ipynb main.py

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GRU

embedding_dim = 100  # dimension of word vectors
num_classes = y.shape[1]
# vocab_size - total no of unique words in the vacabulary 
# embedding_dim - output vector after embedding - 100
# input_length - each row of X contains 100 (maxlen) words 
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# %%
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(padded_sentence_sequences, y, test_size=0.4, random_state=42)

# %%

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=15,
                    batch_size=64,
                    verbose=1)


# %%
import matplotlib.pyplot as plt

# Create a 1-row, 2-column subplot
fig, axes = plt.subplots(1, 2, figsize=(18, 3))

# ---- Accuracy subplot ----
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# ---- Loss subplot ----
axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.show()


# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1️⃣ Make predictions on the validation (test) data
y_pred = model.predict(X_val)

# 2️⃣ Convert probabilities → class labels
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# 3️⃣ Generate confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# 4️⃣ Display it visually
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Sentiment Classification")
plt.show()

# 5️⃣ Optional: detailed classification metrics
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))


# %%
