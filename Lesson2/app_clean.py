from flask import Flask, request, jsonify
import base64
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load TensorFlow Lite model
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TensorFlow Lite model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    interpreter = None

def decode_base64_image(base64_string):
    """Decode base64 encoded image string"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        return image_data
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise

def preprocess_image(image_data, target_size=(224, 224)):
    """Preprocess image for model inference"""
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def predict_with_tflite(preprocessed_image):
    """Perform inference using TensorFlow Lite model"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = {
            'predictions': output_data.tolist(),
            'predicted_class': int(np.argmax(output_data)),
            'confidence': float(np.max(output_data)),
            'all_confidences': [float(x) for x in output_data[0]]
        }
        return predictions
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

@app.route('/')
def home():
    return """
    <h1>Enhanced Flask App with TensorFlow Lite</h1>
    <p>Endpoints available:</p>
    <ul>
        <li><a href="/health">/health</a> - Health check</li>
        <li>/predict - POST endpoint for image prediction</li>
        <li>/process-image - POST endpoint for image processing</li>
    </ul>
    """

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': interpreter is not None,
        'endpoints': ['/health', '/predict', '/process-image']
    })

@app.route('/process-image', methods=['POST'])
def process_image():
    """Endpoint for image processing without prediction"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        image_data = decode_base64_image(data['image'])
        processed_image = preprocess_image(image_data)
        return jsonify({
            'status': 'success',
            'message': 'Image processed successfully',
            'image_shape': processed_image.shape,
            'data_range': {
                'min': float(processed_image.min()),
                'max': float(processed_image.max())
            }
        })
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for image prediction"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        image_data = decode_base64_image(data['image'])
        processed_image = preprocess_image(image_data)
        predictions = predict_with_tflite(processed_image)
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
