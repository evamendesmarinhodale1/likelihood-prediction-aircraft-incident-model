import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_fatality_prediction_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='softplus')
    ])
    
    return model

# Load the trained model
model = create_fatality_prediction_model(input_dim=YOUR_INPUT_DIM)  # Replace YOUR_INPUT_DIM with the actual number of features
model.load_weights('path_to_your_saved_weights.h5')

# Load the scaler
scaler = StandardScaler()
# You need to fit this scaler with your training data
# scaler.fit(X_train)

def predict_fatalities(features):
    # Ensure features is a 2D array
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    # Scale the features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    # Return the prediction
    return prediction[0][0]
