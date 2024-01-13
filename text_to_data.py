import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam

# Set the mixed precision policy
# Currently not compatible with my hardware (i.e M1)
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}:")
        if 'loss' in logs:
            print(f"    loss: {logs['loss']}")
        if 'val_loss' in logs:
            print(f"    validation loss: {logs['val_loss']}")

def tokenize_data(df, tokenizer):
    input_encodings = tokenizer(df['input_text'].tolist(), padding=True, truncation=True, max_length=512)
    target_encodings = tokenizer(df['target_text'].tolist(), padding=True, truncation=True, max_length=512)
    
    # Convert target_encodings['input_ids'] to a NumPy array
    target_input_ids = np.array(target_encodings['input_ids'])

    # Prepare decoder_input_ids
    decoder_input_ids = np.zeros_like(target_input_ids)
    decoder_input_ids[:,1:] = target_input_ids[:,:-1]
    decoder_input_ids[:,0] = tokenizer.pad_token_id

    return input_encodings, target_encodings, decoder_input_ids

# Create TensorFlow dataset
def create_tf_dataset(input_encodings, target_encodings, decoder_input_ids):
    return tf.data.Dataset.from_tensor_slices((
        {"input_ids": input_encodings['input_ids'], "attention_mask": input_encodings['attention_mask'], "decoder_input_ids": decoder_input_ids},
        target_encodings['input_ids']
    ))

# Function to generate structured output and convert to numpy array
def generate_structured_output(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='tf', max_length=512, truncation=True)
    output_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_str

def convert_to_numpy_array(structured_string):
    array_elements = structured_string.split(',')
    return np.array(array_elements)

# Load and prepare the dataset
file_path = 'data/structured_data_descriptions.csv'
df = pd.read_csv(file_path)
df['input_text'] = df['Generated_Description']
df['target_text'] = df.apply(lambda x: ','.join([str(x[col]) for col in df.columns[:-1]]), axis=1)
train_df, test_df = train_test_split(df, test_size=0.1)

# Tokenize the data
tokenizer = T5Tokenizer.from_pretrained('t5-small')

train_input, train_target, train_decoder_input_ids = tokenize_data(train_df, tokenizer)
test_input, test_target, test_decoder_input_ids = tokenize_data(test_df, tokenizer)

train_dataset = create_tf_dataset(train_input, train_target, train_decoder_input_ids).shuffle(len(train_df)).batch(32)
test_dataset = create_tf_dataset(test_input, test_target, test_decoder_input_ids).batch(32)


# Load and fine-tune the T5 model
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
optimizer = Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)
model.fit(train_dataset, epochs=2, validation_data=test_dataset, callbacks=[CustomCallback()])
model.save("text_to_data")
# Example usage
sample_description = "I am looking at a 2022 MG MG3, used, black, 1.5L 4 cyl engine, front-wheel drive, automatic, 5 doors, 5 seats."
structured_output = generate_structured_output(sample_description)
numpy_array_output = convert_to_numpy_array(structured_output)

# Print the numpy array
print(numpy_array_output)
