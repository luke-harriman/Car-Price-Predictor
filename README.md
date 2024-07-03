# iOS Car Pricing App

## Overview

The following repository is a very simple iOS app that provides the user with a price prediction on their car. The app uses AI to trascribe and process the audio from the iPhone's microphone to extract relevant features that help predict what the car might be worth. 

## Functionality
**Speech to Text**: The app takes verbal descriptions of cars (e.g., Brand, Model, Year, Mileage, etc.) and converts them into text using [OpenAI's Whisper speech-to-text model](https://huggingface.co/openai/whisper-base).

**Text Processing**: The textual description is then tokenized using a sentence embedding model - called *paraphrase-MiniLM-L6-v2* - from [Sentence Transformers](https://huggingface.co/sentence-transformers) available on Hugging Face. The Sentence Transformer model produces sentence embeddings that are better than a DistilBERT-Uncased model with a vector that is half the size. 

**Model Training and Prediction**: The embedding vector is then passed to a fine-tuned DETR model running on GCP where it's processed used to create a prediction for the price of the car. 

## Data Source and Preparation
**Data Origin**: The dataset was sourced from [Nidula Elgiriyewithana](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices) on Kaggle and was initially in tabular form with individual columns for Brand, Model, etc.

**Data Transformation**: Due to challenges in processing tabular data directly from speech, the data was converted to textual descriptions. ChatGPT was used to generate 150 descriptions from the tabular data. This sample was then used to fine-tune a smaller model to convert the remaining 16,500 rows of tabular data into descriptions and their corresponding price.

**Data Cleaning and Training**: Post-generation, the descriptions underwent a cleaning process where odd descriptions containing special characters and repeating words were removed. The final dataset comprised of ~16,000 car descriptions and their associated prices.

## Deployment
**Flask App on GCP**: The Flask app was deployed to GCP's Cloud Run service.

**Endpoint Functionality**: The GCP endpoint accepts ``.wav`` audio files and returns a JSON response containing the price of the car.

## iOS App
**App Code**: The ``ios_app`` directory contains the iOS app code (written in *swift*).

**App Store Availability**: The app is also available for download from the App Store, allowing users to get price estimates for their cars. The training data is limited and quite old so the accuracy is not so great - working on a web scraper in a future repository. 

**Usage Instructions**: Users can clone this repository and test the app using the provided Swift code in Xcode (Apple's iOS app development IDE). The GCP endpoint must to inserted into the ``NetworkManager.swift`` file to send Iphone audio recodings to the Flask application.
