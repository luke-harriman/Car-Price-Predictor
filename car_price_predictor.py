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



CATEGORICAL_COLUMNS = ['Brand', 'Year', 'Model', 'Car/Suv', 'UsedOrNew', 'Transmission', 'Engine', 'DriveType', 'FuelType', 'ColourExtInt', 'CylindersinEngine', 'BodyType', 'Doors', 'Seats']



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
    embedding = Embedding(input_dim=len(label_encoders[col].classes_), output_dim=int(len(label_encoders[col].classes_)**0.5), name=f'{col}_embedding')(input_layer)
    flatten = Flatten(name=f'{col}_flatten')(embedding)
    input_layers.append(input_layer)
    concat_layers.append(flatten)

# Input for numerical data
numerical_input = Input(shape=(1,), name='numerical_input')
input_layers.append(numerical_input)
concat_layers.append(numerical_input)

# Concatenate all layers
concatenated = Concatenate(name='concatenate')(concat_layers)

# Dense layers
dense1 = Dense(128, activation='relu', name='dense_1')(concatenated)
dropout1 = Dropout(0.3, name='dropout_1')(dense1)
dense2 = Dense(64, activation='relu', name='dense_2')(dropout1)
dropout2 = Dropout(0.3, name='dropout_2')(dense2)
output = Dense(1, name='output')(dropout2)

# Create the model
model = Model(inputs=input_layers, outputs=output)

# Compile the model
optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

# Prepare the data for model input
# Assuming you have 14 categorical features and 1 numerical feature
train_data = [X_train[col].values for col in CATEGORICAL_COLUMNS]  # Replace categorical_columns with your actual categorical columns
train_data.append(X_train['Kilometres'].values.reshape(-1, 1))  # Adding numerical feature

test_data = [X_test[col].values for col in CATEGORICAL_COLUMNS]
test_data.append(X_test['Kilometres'].values.reshape(-1, 1))

# Train the model
history = model.fit(train_data, y_train, epochs=250, batch_size=64, validation_split=0.2)

# Evaluate the model
model.evaluate(test_data, y_test, verbose=2)

# Make a prediction

prediction = model.predict([np.array([test_data[i][0]]) for i in range(len(test_data))])
print(prediction)
print(y_test[0:1])

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
