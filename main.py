import tensorflow as tf 
import numpy as np 
from flask import Flask, request, jsonify 
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch

#  Transform audio to text

def load_asr_model(model_name):
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


# Function: Transform text to structured data 

# Function: Infer car price based on structured data. Return as Json file to use in application. 

# Create App 

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST'
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'Error: "'})
        
        try: 
            # audio --> text
            # text --> numpy
            # numpy --> prediction
            # return prediction as jsonify dictionary
            pass 
        except Exception as e:
            return jsonify({"Error": str(e)})
 
# if __name__ == '__main__':
#     app.run(debug=True)

# if __name__ == "__main__":
#     audio_file = "data/car_description.wav"  # Update this to the path of your audio file
#     model_name = "openai/whisper-base"  # Update model name if necessary

#     processor, model = load_asr_model(model_name)
#     if processor is not None and model is not None:
#         transcription = speech_to_text(processor, model, audio_file)
#         print(f'Transcription: {transcription}')