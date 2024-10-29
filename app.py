from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the model
model = tf.keras.models.load_model('best_model_dropout_0.5.keras')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return  render_template("index.html")


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read image file and preprocess
        class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        img = Image.open(io.BytesIO(file.read())).resize((224, 224))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        # Convert to a TensorFlow tensor and add batch dimension
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = model.predict(img_tensor)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Return prediction
        return jsonify({'predicted_class': class_names[int(predicted_class)], 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)