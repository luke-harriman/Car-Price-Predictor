# car_price_predictor
An iOS app that deploys an transformer model to accurately price your car.


Number plate look up 
https://docs.autograb.com.au/api


Display listings


We need to clean up csv data files. We should have 3 files:
Original: Australian Vehicle Prices.csv
Cleaned: training_data.csv
Descriptions: generated_descriptions.csv 


gcloud builds submit --tag gcr.io/car-price-predictor-2/predict
gcloud run deploy --image gcr.io/car-price-predictor-2/predict --platform managed
