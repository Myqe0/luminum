from flask import Flask, request, jsonify
from model import TradingModel
import torch

app = Flask(__name__)

# Load your AI model
model = TradingModel()
model.load_state_dict(torch.load('model.pth'))  # Pre-trained model file
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data.get('message')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # Get prediction from the model
    result = model.predict(message)
    return jsonify({'is_trade': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
