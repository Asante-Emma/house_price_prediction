from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = load_model('../models/house_price_prediction_model.keras')
scaler = joblib.load('../models/standard_scaler.pkl')

# Required columns
required_columns = ['Carpet Area', 'Bathroom', 'Balcony', 'Super Area', 'Current_Floor', 'Total_Floors',
                    'Floor_Ratio', 'Transaction_Other', 'Transaction_Rent/Lease', 'Transaction_Resale',
                    'Furnishing_Semi-Furnished', 'Furnishing_Unfurnished', 'facing_North',
                    'facing_North - East', 'facing_North - West', 'facing_South', 'facing_South - East',
                    'facing_South -West', 'facing_West', 'overlooking_Garden/Park, Main Road',
                    'overlooking_Garden/Park, Main Road, Pool', 'overlooking_Garden/Park, Not Available',
                    'overlooking_Garden/Park, Pool', 'overlooking_Garden/Park, Pool, Main Road',
                    'overlooking_Garden/Park, Pool, Main Road, Not Available', 'overlooking_Main Road',
                    'overlooking_Main Road, Garden/Park', 'overlooking_Main Road, Garden/Park, Pool',
                    'overlooking_Main Road, Not Available', 'overlooking_Main Road, Pool',
                    'overlooking_Main Road, Pool, Garden/Park', 'overlooking_Not Specified',
                    'overlooking_Pool', 'overlooking_Pool, Garden/Park', 'overlooking_Pool, Garden/Park, Main Road',
                    'overlooking_Pool, Main Road', 'overlooking_Pool, Main Road, Garden/Park',
                    'overlooking_Pool, Main Road, Not Available', 'Ownership_Freehold', 'Ownership_Leasehold',
                    'Ownership_Power Of Attorney', 'location_freq', 'Society_freq', 'Car_Parking_freq']

# Preprocess function
def preprocess_input(data, scaler, required_columns):
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, list):
        data = pd.DataFrame(data)
    
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0
    
    data = data[required_columns]
    scaled_data = scaler.transform(data)
    return scaled_data

# Prediction function
def predict_price(data, model, scaler, required_columns):
    processed_data = preprocess_input(data, scaler, required_columns)
    predictions = model.predict(processed_data)
    return np.expm1(predictions)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.json
        
        # Make predictions
        prediction = predict_price(input_data, model, scaler, required_columns)
        
        # Return the prediction as JSON
        return jsonify({'predicted_price': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
