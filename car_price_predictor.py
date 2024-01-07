import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv('Australian Vehicle Prices.csv')
dataset = np.array(df)

car_details = []
car_prices = []


for i in dataset:
    try:
        car_prices.append(float(i[-1]))
        i = i[:-1]
        i = [str(x) for x in i]
        car_details.append(i)
    except:
        pass

# Turn this cleaned dataset into a tensor
    
tensor_dataset = tf.data.Dataset.from_tensor_slices((car_details, car_prices))
tensor_dataset = tensor_dataset.shuffle(len(car_details))

# Create a vocabulary for each feature

categorical_columns = ['make', 'model', 'body', 'engine', 'transmission', 
                       'fuel', 'drive', 'seats', 'doors', 'wheelbase', 
                       'length', 'width', 'height', 'weight', 
                       'fuel_capacity', 'fuel_efficiency', 'power', 
                       'torque', 'acceleration']

vocabularies = {}

for column in categorical_columns:
    vocabularies[column] = set(df[column].dropna().unique())