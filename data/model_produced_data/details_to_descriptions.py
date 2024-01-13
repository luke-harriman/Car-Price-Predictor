import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AdamW

# Load the dataset
file_path = 'details_to_descriptions.csv'
df = pd.read_csv(file_path)

# Combine the input columns into a single text input and create input-output pairs
df['input_text'] = df[['Brand', 'Year', 'Model', 'Car/Suv', 'Title', 'UsedOrNew', 'Transmission', 'Engine', 'DriveType', 'FuelType', 'Kilometres', 'ColourExtInt', 'CylindersinEngine', 'BodyType', 'Doors', 'Seats']].agg(' '.join, axis=1)
df['target_text'] = df['Description']

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1)

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def prepare_data(df):
    tokenized_inputs = tokenizer(df['input_text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokenized_targets = tokenizer(df['target_text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    return tokenized_inputs.input_ids, tokenized_inputs.attention_mask, tokenized_targets.input_ids

train_input_ids, train_attention_mask, train_target_ids = prepare_data(train_df)
val_input_ids, val_attention_mask, val_target_ids = prepare_data(val_df)

# Create data loaders
train_data = TensorDataset(train_input_ids, train_attention_mask, train_target_ids)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=8)

val_data = TensorDataset(val_input_ids, val_attention_mask, val_target_ids)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=8)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=3e-4)

# Training loop
model.train()
for epoch in range(150):  # Number of training epochs
    total_loss = 0
    num_batches = 0

    for batch in train_dataloader:
        b_input_ids, b_attention_mask, b_target_ids = batch
        optimizer.zero_grad()
        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_target_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches
    print(f"Epoch {epoch}: Average Loss = {average_loss}")
# Evaluate the model
model.eval()
# Add evaluation code here if needed

model_save_path = 'fine_tuned_t5_details_to_description'
model.save_pretrained(model_save_path)

