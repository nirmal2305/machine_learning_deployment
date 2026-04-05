import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Export the scaler and feature columns
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X.columns.tolist(), 'model_columns.joblib')

# Initialize Flask app
app = Flask(__name__)

# Load the model, scaler, and columns
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')
model_columns = joblib.load('model_columns.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        query_df = pd.DataFrame(json_)

        # Ensure the order of columns matches the training data
        query_df = query_df[model_columns]

        # Scale numerical features
        numerical_features_to_scale = [col for col in model_columns if col in current_numerical_features] # Use current_numerical_features from notebook state
        query_df[numerical_features_to_scale] = scaler.transform(query_df[numerical_features_to_scale])

        prediction = model.predict(query_df)
        return jsonify({'prediction': list(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
  print("starting the app")
    app.run(debug=True)
  
