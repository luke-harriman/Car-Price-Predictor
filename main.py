from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
import torch
import os 
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification


def load_audio_model(model_name):
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        processor = WhisperProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return None, None
    return processor, model

def speech_to_text(processor, model, audio_file):
    try:
        waveform, sampling_rate = librosa.load(audio_file, sr=16000)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return ""
    inputs = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription[0]

def load_prediction_model(model_path):
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    return model, tokenizer 

def tokenize_description(text, tokenizer, max_length=512):
    bert_input = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
    )
    input_ids = np.array([bert_input['input_ids']])
    attention_masks = np.array([bert_input['attention_mask']])
    return input_ids, attention_masks

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # Processing
            processor, audio_model = load_audio_model('openai/whisper-base')
            description = speech_to_text(processor, audio_model, file)
            prediction_model, tokenizer = load_prediction_model('production_models/description_to_price')
            input_ids, attention_masks = tokenize_description(description, tokenizer)
            prediction = prediction_model.predict([input_ids, attention_masks])

            return jsonify({'prediction': str(prediction.logits[0][0])})

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)
