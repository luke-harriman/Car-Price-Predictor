# iOS Car Pricing App

## Overview

This repository contains the source code for an iOS app that interfaces with a Flask application hosted on Google Cloud Platform (GCP). The Flask app is designed to process verbal descriptions of cars and predict their prices.

## Functionality
**Speech to Text**: The app takes verbal descriptions of cars (e.g., Brand, Model, Year, Mileage, etc.) and converts them into text using OpenAI's Whisper speech-to-text model.

**Text Processing**: The textual description is then tokenized using a sentence embedding model - called *paraphrase-MiniLM-L6-v2* - from (Sentence Transformers)[https://huggingface.co/sentence-transformers] available on Hugging Face. The Sentence Transformer model produces sentence embeddings that increased accuracy over the sentence encodings from the DistilBERT-Uncased model - doing so with a vector half the size. 

**Model Training and Prediction**: The embedding vector is passed to a custom transformer model trained on GCP. This model was trained specifically for this purpose, outperforming a fine-tuned DistilBERT-Uncased model in prediction accuracy.

## Data Source and Preparation
**Data Origin**: The dataset was sourced from (Nidula Elgiriyewithana)[https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices] on Kaggle and was initially in tabular form with individual columns for Brand, Model, etc.

**Data Transformation**: Due to challenges in processing tabular data directly from speech, the data was converted to textual descriptions. ChatGPT was used to generate 150 descriptions from the tabular data. This sample was then used to fine-tune a model to convert the remaining 16,500 rows of tabular data into descriptions and their corresponding price.

**Data Cleaning and Training**: Post-generation, the descriptions underwent a cleaning process. The final dataset comprising ~16,000 car descriptions and their associated prices was used to train the transformer model.

## Deployment
**Flask App on GCP**: The Flask app, once the pipeline was established and the model trained, was straightforwardly deployed to GCP.

**Endpoint Functionality**: The Flask app endpoint accepts ``.wav`` audio files and returns a price prediction for the described car.

## iOS App
**App Code**: The ``ios_app`` directory contains the iOS app code (written in *swift*).

**App Store Availability**: The app is also available for download from the App Store, allowing users to get price estimates for their cars. The training data is limited and quite old so the accuracy is not so great - working on a web scraper in a future repository. 

**Usage Instructions**: Users can clone this repository and test the app using the provided Swift code in Xcode. The user must specify their GCP endpoint in the ``NetworkManager.swift`` file to send Iphone audio recodings to the Flask application.

**Repository Cloning Instructions**
```bash
git clone https://github.com/luke-harriman/car_price_predictor.git
cd car_price_predictor
```

Feel free to deploy the model and test out the app using these instructions.