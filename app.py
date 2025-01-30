from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression  # Assuming you're using a linear regression model
import pickle  # If you want to use a pre-trained model

# Initialize the Flask application
app = Flask(__name__)


class TaxiFarePredictor:
    def __init__(self):
        # Just for demo purposes, we use a basic Linear Regression model.
        self.model = LinearRegression()
        # Sample data for model training  
        data = {
            'Trip_Distance_km': [2, 5, 10, 15, 3],
            'Passenger_Count': [1, 2, 3, 4, 1],
            'Time_of_Day': [0, 1, 2, 3, 0],  # 0: Morning, 1: Afternoon, 2: Evening, 3: Night
            'Day_of_Week': [0, 1, 2, 3, 4],  # 0: Monday, 1: Tuesday, 2: Wednesday, etc.
            'Traffic_Conditions': [0, 1, 2, 1, 0],  # 0: Low, 1: Moderate, 2: High
            'Weather': [0, 1, 2, 0, 1],  # 0: Clear, 1: Rainy, 2: Stormy
            'Base_Fare': [5, 7, 10, 12, 5],
            'Per_Km_Rate': [2, 3, 4, 5, 2],
            'Per_Minute_Rate': [0.5, 1, 1.5, 2, 0.5],
            'Trip_Duration_Minutes': [10, 20, 30, 40, 15],
        }
        
        df = pd.DataFrame(data)
        
        # Prepare the features (X) and target (y)
        X = df.drop('Trip_Duration_Minutes', axis=1)
        y = df['Trip_Duration_Minutes']
        
        # Train the model (This is just an example, you will likely have a pre-trained model)
        self.model.fit(X, y)
    
    def predict(self, features):
        # Make prediction using the trained model
        return self.model.predict([features])

# Instantiate the model
taxi_fare_predictor = TaxiFarePredictor()

@app.route('/')
def home():
    return "Welcome to the Taxi Fare Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get data from the frontend
        
        # Extract input features from the request
        trip_distance_km = float(data['Trip_Distance_km'])
        time_of_day = int(data['Time_of_Day'])
        day_of_week = int(data['Day_of_Week'])
        passenger_count = int(data['Passenger_Count'])
        traffic_conditions = int(data['Traffic_Conditions'])
        weather = int(data['Weather'])
        base_fare = float(data['Base_Fare'])
        per_km_rate = float(data['Per_Km_Rate'])
        per_minute_rate = float(data['Per_Minute_Rate'])
        trip_duration_minutes = float(data['Trip_Duration_Minutes'])

        # Prepare the features for prediction (this is the order expected by the model)
        features = [
            trip_distance_km,
            time_of_day,
            day_of_week,
            passenger_count,
            traffic_conditions,
            weather,
            base_fare,
            per_km_rate,
            per_minute_rate,
        ]
        
        # Call the model to predict the fare
        predicted_fare = taxi_fare_predictor.predict(features)

        # Return the predicted fare
        return jsonify({
            'Predicted_Trip_Price': predicted_fare[0]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
