# app.py

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model, scaler, and feature names
model_path = os.path.join('model', 'wine_cultivar_model.pkl')
scaler_path = os.path.join('model', 'scaler.pkl')
features_path = os.path.join('model', 'selected_features.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
selected_features = joblib.load(features_path)

# Cultivar names
cultivar_names = {
    0: "Cultivar 1",
    1: "Cultivar 2",
    2: "Cultivar 3"
}

@app.route('/')
def home():
    """Render the home page with input form"""
    return render_template('index.html', features=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input values from the form
        input_values = []
        for feature in selected_features:
            value = float(request.form.get(feature, 0))
            input_values.append(value)
        
        # Convert to numpy array and reshape
        input_array = np.array(input_values).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Get cultivar name and confidence
        cultivar_name = cultivar_names[prediction]
        confidence = prediction_proba[prediction] * 100
        
        return render_template(
            'index.html',
            features=selected_features,
            prediction=cultivar_name,
            confidence=f"{confidence:.2f}",
            input_values=dict(zip(selected_features, input_values))
        )
    
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template(
            'index.html',
            features=selected_features,
            error=error_message
        )

if __name__ == '__main__':
    # Use PORT environment variable for Render deployment
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=False)
