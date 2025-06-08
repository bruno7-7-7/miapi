from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import logging
import os
import mediapipe as mp

# Inicializa la app Flask
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# MediaPipe para detectar la mano
mp_hands = mp.solutions.hands

# Texto acumulado
spelling = []
MAX_LENGTH = 20

# Cache de modelos cargados
loaded_models = {}

def load_model_and_labels(model_name):
    """Carga modelo y etiquetas desde carpeta si no están en caché."""
    if model_name in loaded_models:
        return loaded_models[model_name]

    model_dir = os.path.join("models", model_name)
    model_path = os.path.join(model_dir, "model.savedmodel")
    labels_path = os.path.join(model_dir, "labels.txt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo '{model_path}'")

    model = tf.saved_model.load(model_path)

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"No se encontró labels.txt para '{model_name}'")

    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    loaded_models[model_name] = (model, labels)
    return model, labels


def preprocess_image(image_bytes):
    """Preprocesa la imagen y recorta la mano si se detecta."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(image)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(img_np)

        if results.multi_hand_landmarks:
            h, w, _ = img_np.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in results.multi_hand_landmarks[0].landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            hand_crop = img_np[y_min:y_max, x_min:x_max]
            image = Image.fromarray(hand_crop).resize((224, 224))
        else:
            logging.warning("No se detectó ninguna mano.")
            image = image.resize((224, 224))

    img_array = np.asarray(image) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)


def get_prediction(img_tensor, model, labels):
    """Realiza la predicción y retorna la clase y confianza."""
    infer = model.signatures["serving_default"]
    input_tensor = tf.convert_to_tensor(img_tensor)
    output = infer(input_tensor)

    predictions = list(output.values())[0].numpy()
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    predicted_index = np.argmax(probabilities)
    confidence = float(probabilities[predicted_index])
    predicted_class = labels[predicted_index]

    return predicted_class, confidence


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        model_name = request.form.get('model_name', 'model1')  # Por defecto usa model1

        # Carga dinámica del modelo y etiquetas
        model, labels = load_model_and_labels(model_name)

        file = request.files['file']
        image_bytes = file.read()
        img_tensor = preprocess_image(image_bytes)
        predicted_class, confidence = get_prediction(img_tensor, model, labels)

        if len(spelling) == 0 or (spelling[-1] != predicted_class and confidence > 0.9):
            spelling.append(predicted_class)
            if len(spelling) > MAX_LENGTH:
                spelling.pop(0)

        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 4),
            'spelling': ''.join(spelling)
        })

    except Exception as e:
        logging.exception("Error durante la predicción:")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_spelling():
    spelling.clear()
    logging.info("Texto acumulado reiniciado.")
    return jsonify({'message': 'Texto acumulado reiniciado'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
