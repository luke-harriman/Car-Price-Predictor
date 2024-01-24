import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import pandas as pd 

# Define a function to convert text data into input features
def convert_examples_to_tf_dataset(examples, labels, tokenizer, max_length=512):
    input_ids, attention_masks, labels_out = [], [], []

    for example, label in zip(examples, labels):
        bert_input = tokenizer.encode_plus(
            example,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
        )
        
        input_ids.append(bert_input['input_ids'])
        attention_masks.append(bert_input['attention_mask'])
        labels_out.append([label])

    # Convert to numpy arrays
    input_ids = np.array(input_ids)
    attention_masks = np.array(attention_masks)
    labels_out = np.array(labels_out)

    # Print shapes for debugging
    # print(f"Input IDs shape: {input_ids.shape}")
    # print(f"Attention Masks shape: {attention_masks.shape}")
    # print(f"Labels shape: {labels_out.shape}")

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(((input_ids, attention_masks), labels_out))

    return dataset

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Prepare the dataset

data = pd.read_csv('data/descriptions_to_car_prices.csv')
data['Prices'] = pd.to_numeric(data['Price'], errors='coerce')
data = data.dropna()

descriptions = data['Description']
prices = data['Prices']

# Split the dataset into training and validation sets
train_desc, val_desc, train_prices, val_prices = train_test_split(descriptions, prices, test_size=0.1)

# Convert the data into TensorFlow dataset
# Convert the data into TensorFlow dataset
train_data = convert_examples_to_tf_dataset(train_desc, train_prices, tokenizer)
val_data = convert_examples_to_tf_dataset(val_desc, val_prices, tokenizer)

# Batch the data
batch_size = 254
train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

# Load pre-trained DistilBERT model


model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)

# Compile the model
optimizer = Adam(learning_rate=0.001)
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optimizer, loss=loss)

# Train the model
model.fit(train_data, epochs=10, validation_data=val_data)

# Save Model
model.save_pretrained('production_models')
tokenizer.save_pretrained('production_models')

