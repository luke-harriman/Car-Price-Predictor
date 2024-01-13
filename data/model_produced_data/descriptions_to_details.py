from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pandas as pd

# Load the fine-tuned model and tokenizer
model_path = 'fine_tuned_t5_details_to_description'  # Adjust the path if necessary
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Set the model to evaluation mode
model.eval()

# Load the dataset
file_path = 'completed_dataset.csv'  # Adjust the path if necessary
df = pd.read_csv(file_path)

# Combine the input columns into a single text input
df['input_text'] = df[['Brand', 'Year', 'Model', 'Car/Suv', 'Title', 'UsedOrNew', 'Transmission', 'Engine', 'DriveType', 'FuelType', 'Kilometres', 'ColourExtInt', 'CylindersinEngine', 'BodyType', 'Doors', 'Seats']].astype(str).agg(' '.join, axis=1)

def generate_description(input_text):
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description

# Generate descriptions for each row and track progress
print("Starting the description generation process...")
for i, row in enumerate(df.itertuples(), 1):
    df.at[row.Index, 'Generated_Description'] = generate_description(row.input_text)
    if i % 50 == 0:  # Adjust the frequency of messages as needed
        print(f"Processed {i} rows...")

# Save the dataframe with generated descriptions
output_file_path = 'generated_descriptions.csv'
df.to_csv(output_file_path, index=False)
print("Description generation complete. File saved.")



