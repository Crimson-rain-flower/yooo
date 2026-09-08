from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route('/', methods=['GET'])
def home():
    return "ML Model Deployment API is Live! Send POST requests to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not found. Train the model first."}), 500
    try:
        data = request.get_json()
        features = data['features']
        prediction = model.predict([features])
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)