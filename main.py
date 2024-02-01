from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
import torch
import os 
from transformers import WhisperProcessor, WhisperForConditionalGeneration, DistilBertTokenizer, TFDistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from custom_metrics import RSquared, WeightedAverageInaccuracy, AverageInaccuracy
# gcloud builds submit --tag gcr.io/car-price-predictor-2/my_flask_app:archv3 .;gcloud run deploy my_flask_app --image gcr.io/car-price-predictor-2/my_flask_app:archv3 --platform managed

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

embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define custom objects
custom_objects = {
    "MeanSquaredError": tf.keras.losses.MeanSquaredError,
    'RSquared': RSquared, 
    'WeightedAverageInaccuracy': WeightedAverageInaccuracy, 
    'AverageInaccuracy': AverageInaccuracy
}

# Load the model with custom objects
prediction_model = tf.keras.models.load_model('production_model', custom_objects=custom_objects)


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
            
            input_ids = embedding_model.encode(description)
            prediction = prediction_model.predict(np.array([input_ids]))

            return jsonify({'prediction': str(prediction[0][0])})

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)
