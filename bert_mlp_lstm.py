import os
import inspect
import logging

app_path = inspect.getfile(inspect.currentframe())
sub_dir = os.path.realpath(os.path.dirname(app_path))
main_dir = os.path.dirname(sub_dir)

import pandas as pd

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

tf.gfile = tf.io.gfile
logging.basicConfig(level=logging.INFO)
from sklearn.metrics import classification_report
from keras.preprocessing.sequence import pad_sequences
from transformers import XLNetModel, XLNetTokenizer, BertTokenizer
from tensorflow.keras.models import Sequential, Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dataf = pd.read_csv("dataset/data.csv", encoding='latin1')
dataf = dataf.iloc[:, [3, -1]]
dataf.head()

dataf.columns = ["text", "label"]
dataf = dataf.sample(frac=0.5)

dataf.describe()

dataf.dropna(inplace=True)
dataf.isnull().sum()

sns.countplot(x="label", data=dataf)
plt.show()

dataf['text'] = dataf['text'].apply(str)

sentences = dataf.text.values

data = [sentence + " [SEP] [CLS]" for sentence in sentences]
labels = dataf['label']

MAX_LEN = 200

"""##BERT TOKENIZATION"""
texts = dataf.text.values
bert_texts = ["[CLS]" + sentence + "[SEP]" for sentence in texts]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(sent) for sent in bert_texts]
print("Tokenize the first sentence:")
print(tokenized_texts[0])

# Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
bert_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# Pad our input tokens
bert_input_ids = pad_sequences(bert_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
bert_attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in bert_input_ids:
    seq_mask = [float(i > 0) for i in seq]
    bert_attention_masks.append(seq_mask)

all_inputs = bert_input_ids

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(all_inputs, labels,
                                                                                    random_state=2018, test_size=0.2)

# Fit MLP
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_inputs, train_labels, epochs=100, batch_size=200, shuffle=True)
scores = model.evaluate(validation_inputs, validation_labels)

print("results =", scores[1])

scores = model.predict(validation_inputs)

out = np.argmax(scores.round(), axis=1)
print(classification_report(validation_labels, out))

validation_labels.value_counts()

train_set = np.expand_dims(train_inputs, axis=-1)
validation_set = np.expand_dims(validation_inputs, axis=-1)

train_labels1 = to_categorical(train_labels)
validation_labels1 = to_categorical(validation_labels)

# Creating LSTM model
max_words = 150000
max_len = 400
inputs = Input(name='inputs', shape=[max_len])
layer = Embedding(max_words, 50, input_length=max_len)(inputs)
layer = LSTM(64)(layer)
layer = Dense(256, name='FC1')(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(1, name='out_layer')(layer)
layer = Activation('sigmoid')(layer)
model = Model(inputs=inputs, outputs=layer)

model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

model.fit(train_inputs, train_labels, batch_size=128, epochs=10, validation_split=0.2)

scores = model.predict(validation_inputs)

# Y_test = np.argmax(validation_labels, axis=1) # Convert one-hot to index
out = np.argmax(scores.round(), axis=1)
print(classification_report(validation_labels, out))