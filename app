from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_fatalities
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract features from the request
    seat_row = int(data['seatNumber'][:-1])
    seat_col = ord(data['seatNumber'][-1]) - ord('A')
    aircraft_type = data['aircraftType']
    
    # Convert aircraft_type to a numerical value
    aircraft_types = ['Boeing 737', 'Airbus A320', 'Boeing 787', 'Airbus A350', 'Embraer E175']
    aircraft_type_num = aircraft_types.index(aircraft_type)
    
    # Prepare input for the model
    # You may need to adjust this based on your actual model's input requirements
    input_data = np.array([[seat_row, seat_col, aircraft_type_num]])
    
    # Make prediction
    fatalities = predict_fatalities(input_data)
    
    # Convert prediction to a survival probability
    # This is a simplistic conversion and may need to be adjusted
    max_fatalities = 300  # Assume maximum fatalities for a typical commercial aircraft
    survival_probability = max(0, (1 - (fatalities / max_fatalities))) * 100
    
    return jsonify({'survivalProbability': float(survival_probability)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
