import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from flask import Flask, request, jsonify
import librosa
import os
from sentence_transformers import SentenceTransformer
from custom_metrics import RSquared, WeightedAverageInaccuracy, AverageInaccuracy

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Whether to restore model weights from the epoch with the best value of the monitored quantity
    verbose=1           # Whether to output a message when training is stopped
)



dataset = pd.read_csv("data/descriptions_to_car_prices.csv")
dataset['Price'] = pd.to_numeric(dataset['Price'], errors='coerce')
dataset = dataset.dropna(subset=['Price'])
dataset['Price'] = (dataset['Price'] / 500).round() * 500
# Drop all the rows in the dataset where the description column contains a particular substring
# Escaping special characters in the substring for regular expression
dataset = dataset[~dataset['Description'].str.contains(r'\*\*', regex=True, case=False)]
dataset = dataset[~dataset['Description'].str.contains(r'%', regex=True, case=False)]
dataset = dataset[~dataset['Description'].str.contains(r'S S S ', regex=True, case=False)]
dataset = dataset[dataset['Price'] <= 120000]
dataset = dataset[dataset['Price'] >= 500]


# Embedding Model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings
embeddings = dataset['Description'].apply(lambda x: model.encode(x))

# Convert embeddings to a suitable format
embeddings = np.stack(embeddings.values)


def build_transformer_model(embedding_dim, num_heads, ff_dim):
    inputs = tf.keras.Input(shape=(embedding_dim,))
    x = tf.keras.layers.Reshape((1, embedding_dim))(inputs)

    x = transformer_block(x, num_heads, ff_dim)

    x = tf.keras.layers.Flatten()(x)  # Flatten the output
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def transformer_block(inputs, num_heads, ff_dim, rate=0.15):
    # Multi-head self attention
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attention_output = tf.keras.layers.Dropout(rate)(attention_output)
    attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed-forward layer
    ff_output = tf.keras.layers.Dense(ff_dim, activation="relu")(attention_output)
    ff_output = tf.keras.layers.Dense(inputs.shape[-1])(ff_output)
    ff_output = tf.keras.layers.Dropout(rate)(ff_output)
    ff_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + ff_output)
    return ff_output

# Prepare labels
labels = dataset['Price'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Define the model
embedding_dim = (X_train.shape[1]) # Make sure input_shape matches the shape of embeddings
model = build_transformer_model(embedding_dim, num_heads=8, ff_dim=32)


# Compile the model
# optimizer =  tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
# loss_func = tf.keras.losses.Huber(delta=1.0)
mse = tf.keras.losses.MeanSquaredError()
mae = MeanAbsoluteError()
rmse = RootMeanSquaredError()
rsq = RSquared()
weighted_inacc = WeightedAverageInaccuracy()  # Custom metric class
average_inacc = AverageInaccuracy()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[mse, mae, rmse, rsq, weighted_inacc, average_inacc])

history = model.fit(X_train, y_train, epochs=500, batch_size=128, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print("Test Loss: ", test_loss)


model.save('production_model')
