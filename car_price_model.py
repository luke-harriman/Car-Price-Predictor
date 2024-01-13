import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import pprint as pp
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, LayerNormalization, Add, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Metric
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import joblib
import json



# Constants
CATEGORICAL_COLUMNS = ['Brand', 'Year', 'Model', 'Car/Suv', 'UsedOrNew', 'Transmission', 'Engine', 'DriveType', 'FuelType', 'ColourExtInt', 'CylindersinEngine', 'BodyType', 'Doors', 'Seats']
NUMERICAL_COLUMNS = ['Kilometers']
with open('columns_info.json', 'w') as file:
    json.dump({'categorical_columns': CATEGORICAL_COLUMNS, 'numerical_columns': NUMERICAL_COLUMNS}, file)


# Transformer Encoder Function
def transformer_block(x, head_size, num_heads, ff_dim, dropout=0.1, reg=l2(1e-6)):
    # Multi-Head Attention
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    attn_output = Dropout(dropout)(attn_output)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-Forward Network
    ff_output = Dense(ff_dim, activation="relu", kernel_regularizer=reg)(x)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(x.shape[-1], kernel_regularizer=reg)(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    return x

def transformer_encoder(inputs, head_size, num_heads, ff_dim, num_layers, dropout=0, reg=l2(1e-6)):
    x = inputs
    for _ in range(num_layers):  # Stacking multiple layers
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout, reg)
    return x

# Custom Metric Class
class PercentageAccuracy(Metric):
    def __init__(self, threshold=0.5, name='percentage_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        error = tf.abs((y_true - y_pred) / tf.clip_by_value(y_true, 1e-6, tf.float32.max))
        accuracy = tf.cast(tf.less(error, self.threshold), tf.float32)
        self.total.assign_add(tf.reduce_sum(accuracy))
        self.count.assign_add(tf.cast(tf.size(accuracy), tf.float32))

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0)
        self.count.assign(0)

class MeanPercentageError(Metric):
    def __init__(self, name='mean_percentage_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Avoid division by zero
        y_true = tf.where(tf.equal(y_true, 0), 1e-6, y_true)
        percentage_error = (y_pred - y_true) / y_true
        self.total_error.assign_add(tf.reduce_sum(percentage_error))
        self.count.assign_add(tf.cast(tf.size(percentage_error), tf.float32))

    def result(self):
        return self.total_error / self.count

    def reset_state(self):
        self.total_error.assign(0)
        self.count.assign(0)


# Data Preprocessing Function
def preprocess_data(filename, categorical_columns):
    df = pd.read_csv(filename)
    df.loc[df['Year'].isna()]
    df.drop(2391, axis=0, inplace=True)
    df = df.drop(df[df['Kilometres'] == '-'].index)
    df = df.drop(df[df['Kilometres'] == '- / -'].index)
    df = df.drop(df[df['Price'] == 'POA'].index)
    df.loc[df['Price'].isna()]
    df.drop([10156, 11039], axis=0, inplace=True)
    df = df.drop(['Title', 'Location', 'FuelConsumption'], axis=1).dropna()

    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        df[col] = label_encoders[col].fit_transform(df[col])

    df['Kilometres'] = pd.to_numeric(df['Kilometres'], errors='coerce')
    imputer = SimpleImputer(strategy='mean')
    df['Kilometres'] = imputer.fit_transform(df[['Kilometres']])
    min_max_scaler = MinMaxScaler()
    df['Kilometres'] = min_max_scaler.fit_transform(df[['Kilometres']])

    X = df.drop('Price', axis=1)
    y = pd.to_numeric(df['Price'], errors='coerce').fillna(method='bfill')

    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(min_max_scaler, 'min_max_scaler.pkl')

    return train_test_split(X, y, test_size=0.2, random_state=42), label_encoders, min_max_scaler

# Model Creation Function
def create_model(categorical_columns, label_encoders):
    input_layers = []
    concat_layers = []

    for col in categorical_columns:
        input_layer = Input(shape=(1,), name=f'{col}_input')
        embedding = Embedding(input_dim=len(label_encoders[col].classes_), output_dim=int(len(label_encoders[col].classes_)**0.5), name=f'{col}_embedding')(input_layer)
        flatten = Flatten(name=f'{col}_flatten')(embedding)
        input_layers.append(input_layer)
        concat_layers.append(flatten)

    numerical_input = Input(shape=(1,), name='numerical_input')
    input_layers.append(numerical_input)
    concat_layers.append(numerical_input)

    concatenated = Concatenate(name='concatenate')(concat_layers)
    transformer_input = tf.expand_dims(concatenated, axis=1)
    transformer_block = transformer_encoder(
        transformer_input, 
        head_size=256, 
        num_heads=8, 
        ff_dim=512, 
        num_layers=3,  # Number of stacked transformer encoder layers
        dropout=0.1, 
        reg=l2(1e-4)
    )
    transformer_output = Flatten()(transformer_block)
    output = Dense(1, activation='linear', kernel_regularizer=l2(1e-4), name='output')(transformer_output)  # Added L2 regularization

    model = Model(inputs=input_layers, outputs=output)
    optimizer = Adam(learning_rate=0.01, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[MeanPercentageError(), 'mae', 'mse'])
    return model

# Adjusted Learning Rate Scheduler
def adjusted_scheduler(epoch, lr):
    if epoch < 5:
        return 0.01  # Initial learning rate
    else:
        return 0.005  # Reduce after 5 epochs
    # else:
    #     return 0.001  # Reduce further after 10 epochs


# Function to make predictions on new data
def make_predictions(model, data):
    # Return the prediction, the data used to make the prediction
    prediction = model.predict(data)
    return prediction, data

def display_predictions_with_details(predictions, test_data, label_encoders, actual_prices):
    for i, prediction in enumerate(predictions):
        print('\n')
        # Decoding categorical features
        car_details = {}
        for idx, col in enumerate(CATEGORICAL_COLUMNS):
            encoded_value = test_data[idx][i]
            original_value = label_encoders[col].inverse_transform([encoded_value])[0]
            car_details[col] = original_value

        # Printing car details and the actual price
        print(f"Actual Price: ${actual_prices[i]:.2f}")
        print(f"Predicted Price: ${prediction[0]:.2f}")
        print(f'Difference in Price: ${abs(actual_prices[i] - prediction[0]):.2f}')
        print(f'Percentage Difference in Price: {abs(actual_prices[i] - prediction[0]) / actual_prices[i] * 100:.2f}%')
        print("Car Details:")
        for col, value in car_details.items():
            print(f"{col}: {value}")

def dowload_to_csv(predictions, test_data, label_encoders, actual_prices):
    results = []  # List to store all prediction details
    
    for i, prediction in enumerate(predictions):
        # Decoding categorical features
        car_details = {}
        for idx, col in enumerate(CATEGORICAL_COLUMNS):
            encoded_value = test_data[idx][i]
            original_value = label_encoders[col].inverse_transform([encoded_value])[0]
            car_details[col] = original_value

        # Adding price details
        car_details['Actual Price'] = actual_prices[i]
        car_details['Predicted Price'] = prediction[0]
        car_details['Difference in Price'] = abs(actual_prices[i] - prediction[0])
        car_details['Percentage Difference in Price'] = abs(actual_prices[i] - prediction[0]) / actual_prices[i] * 100

        # Append the car details to the results list
        results.append(car_details)
    
    # Create a DataFrame from the results and save it as a CSV file
    df = pd.DataFrame(results)
    df.to_csv('car_price_predictions.csv', index=False)


# Main Function
def main():
    (X_train, X_test, y_train, y_test), label_encoders, min_max_scaler = preprocess_data('data/Australian Vehicle Prices.csv', CATEGORICAL_COLUMNS)
    model = create_model(CATEGORICAL_COLUMNS, label_encoders)

    train_data = [X_train[col].values for col in CATEGORICAL_COLUMNS]
    train_data.append(X_train['Kilometres'].values.reshape(-1, 1))
    test_data = [X_test[col].values for col in CATEGORICAL_COLUMNS]
    test_data.append(X_test['Kilometres'].values.reshape(-1, 1))

    callback_lr = LearningRateScheduler(adjusted_scheduler)
    callback_es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        train_data, y_train, 
        epochs=100, 
        batch_size=64,  # Adjusted batch size
        validation_split=0.15, 
        verbose=1, 
        callbacks=[callback_lr, callback_es]
    )
    model.evaluate(test_data, y_test, verbose=2)
    model.save('saved_model')
    # Predictions
    predictions, data = make_predictions(model, test_data)
    # display_predictions_with_details(predictions, test_data, label_encoders, y_test.values)

    # Download decoded training data to compare with the predicted data
    # download_decoded_training_data(X_train, y_train, label_encoders)

    # Plot data
    fig, ax1 = plt.subplots()

    # Plotting Training and Validation Loss on the left y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    line1, = ax1.plot(history.history['loss'], label='Training Loss', color='tab:red')
    line2, = ax1.plot(history.history['val_loss'], label='Validation Loss', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')

    # Setting up the right y-axis for Mean Percentage Error
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean Percentage Error', color='tab:blue')
    line3, = ax2.plot(history.history['mean_percentage_error'], label='Training MPE', color='tab:blue')
    line4, = ax2.plot(history.history['val_mean_percentage_error'], label='Validation MPE', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    # Title of the plot
    plt.title('Training and Validation Loss, and Mean Percentage Error')
    plt.show()


if __name__ == '__main__':
    main()
