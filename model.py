# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def prepare_data(data):
    # Prepare features (X) and target variable (y)
    categorical_cols = [
        'base_type', 'Aircraft_damage', 'gpe', 'Phase', 'Nature', 
        'season', 'navpoint', 'operator', 'manufacturer', 'weekday'
    ]
    X = pd.get_dummies(data[categorical_cols])  # Convert categorical to dummy variables

    # Add numeric columns with proper handling of missing values
    numeric_cols = ['Year_of_manufacture', 'year']
    numeric_imputer = SimpleImputer(strategy='median')
    numeric_data = pd.DataFrame(
        numeric_imputer.fit_transform(data[numeric_cols]),
        columns=numeric_cols,
        index=data.index
    )

    X = pd.concat([X, numeric_data], axis=1)

    # Target variable
    y = data['fat.'].fillna(0)

    return X, y

def create_fatality_prediction_model(input_dim):
    # Create a neural network model
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

# Load dataset
data = pd.read_csv('data/cleaned_data.csv')

# Prepare data
X, y = prepare_data(data)

# Find the number of features
input_dim = X.shape[1]
print(f"Number of features: {input_dim}")  # Displays the total number of features in the dataset

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=(y > 0).astype(int)
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and compile the model
model = create_fatality_prediction_model(input_dim)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X_train_scaled, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=1
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

