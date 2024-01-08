import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import pprint as pp
import re
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Concatenate, Embedding, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.metrics import Metric




CATEGORICAL_COLUMNS = ['Brand', 'Year', 'Model', 'Car/Suv', 'UsedOrNew', 'Transmission', 'Engine', 'DriveType', 'FuelType', 'ColourExtInt', 'CylindersinEngine', 'BodyType', 'Doors', 'Seats']


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attn_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    attn_output = Dropout(dropout)(attn_output)
    x = Add()([x, attn_output])

    # Feed Forward Part
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x



class PercentageAccuracy(Metric):
    def __init__(self, threshold=0.5, name='percentage_accuracy', **kwargs):  # Increased threshold
        super(PercentageAccuracy, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)  # Ensure float32 for division
        y_pred = tf.cast(y_pred, tf.float32)
        error = tf.abs((y_true - y_pred) / tf.clip_by_value(y_true, 1e-6, tf.float32.max))
        accuracy = tf.cast(tf.less(error, self.threshold), tf.float32)
        self.total.assign_add(tf.reduce_sum(accuracy))
        self.count.assign_add(tf.cast(tf.size(accuracy), tf.float32))
        
        # # For debugging:
        # tf.print("Actuals:", y_true[:5], summarize=-1)
        # tf.print("Predictions:", y_pred[:5], summarize=-1)
        # tf.print("Errors:", error[:5], summarize=-1)
        # tf.print("Accurate Predictions:", tf.reduce_sum(accuracy), "out of", tf.size(accuracy))

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)




# Sample DataFrame loading and preprocessing
df = pd.read_csv('Australian Vehicle Prices.csv')
df = df.drop(['Title', 'Location', 'FuelConsumption'], axis=1)
df = df.dropna()

# Integer encode categorical columns
label_encoders = {}
for col in CATEGORICAL_COLUMNS:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Convert non-numeric entries to NaN
df['Kilometres'] = pd.to_numeric(df['Kilometres'], errors='coerce')

# Optional: Handle missing values (NaNs)
# You can choose a strategy like 'mean', 'median', etc.
imputer = SimpleImputer(strategy='mean')
df['Kilometres'] = imputer.fit_transform(df[['Kilometres']])

# Use Min-Max Scaler
min_max_scaler = MinMaxScaler()
numerical_features = df[['Kilometres']]
numerical_scaled = min_max_scaler.fit_transform(numerical_features)

# Splitting the data
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


y_train = pd.to_numeric(y_train, errors='coerce')
y_test = pd.to_numeric(y_test, errors='coerce')
y_train = y_train.fillna(y_train.mean())
y_test = y_test.fillna(y_test.mean())

# Define the inputs for the model
input_layers = []
concat_layers = []

# Embedding for each categorical column
for col in CATEGORICAL_COLUMNS:
    input_layer = Input(shape=(1,), name=f'{col}_input')
    embedding = Embedding(
        input_dim=len(label_encoders[col].classes_),
        output_dim=int(len(label_encoders[col].classes_)**0.5),
        name=f'{col}_embedding'
    )(input_layer)
    flatten = Flatten(name=f'{col}_flatten')(embedding)
    input_layers.append(input_layer)
    concat_layers.append(flatten)

# Input for numerical data
numerical_input = Input(shape=(1,), name='numerical_input')
input_layers.append(numerical_input)
concat_layers.append(numerical_input)

# Concatenate all layers
concatenated = Concatenate(name='concatenate')(concat_layers)

# Assuming 'concatenated' is the concatenated embeddings of categorical features and numerical features
num_features = concatenated.shape[-1]  # Number of features after concatenation

# Transformer encoder layers expect sequences, so we need to add a dimension to treat each example as a sequence
transformer_input = tf.expand_dims(concatenated, axis=1)

# Apply the transformer encoder
transformer_block = transformer_encoder(
    transformer_input, head_size=256, num_heads=6, ff_dim=6, dropout=0.25
)

# Flatten the output to remove the sequence dimension
transformer_output = Flatten()(transformer_block)

# Output layer
output = Dense(1, name='output')(transformer_output)

# Create the model
model = Model(inputs=input_layers, outputs=output)

# Compile the model
optimizer = Adam(learning_rate=0.01, clipvalue=1.0)  # Use the initial learning rate from the step_decay function
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=[PercentageAccuracy(threshold=0.2), 'mae', 'mse'])


# Prepare the data for model input
# Assuming you have 14 categorical features and 1 numerical feature
train_data = [X_train[col].values for col in CATEGORICAL_COLUMNS]  # Replace categorical_columns with your actual categorical columns
train_data.append(X_train['Kilometres'].values.reshape(-1, 1))  # Adding numerical feature

test_data = [X_test[col].values for col in CATEGORICAL_COLUMNS]
test_data.append(X_test['Kilometres'].values.reshape(-1, 1))

# Train the model
history = model.fit(
    train_data,
    y_train,
    epochs=100000,
    batch_size=64,
    validation_split=0.25,
    verbose=1)

# Evaluate the model
model.evaluate(test_data, y_test, verbose=2)

# Make a prediction
prediction = model.predict([np.array([test_data[i][0]]) for i in range(len(test_data))])
print(prediction)
print(y_test[:1])


# After training, to inspect the predictions vs. actuals
predictions = model.predict(test_data)
predictions = predictions.flatten()
threshold = 0.5  # Or any other threshold you deem appropriate
accurate_predictions = np.abs((y_test - predictions) / y_test) < threshold
accuracy = np.mean(accurate_predictions)
print(f"Manual accuracy calculation: {accuracy:.2%}")

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
