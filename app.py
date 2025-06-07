# ======== FLASK SERVER ========
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import logging
import mediapipe as mp

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Carga el modelo desde SavedModel
model = tf.saved_model.load("model/model.savedmodel")

# Carga las etiquetas
with open("model/labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands

# Texto acumulado
spelling = []
MAX_LENGTH = 20


def preprocess_image(image_bytes):
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

            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            hand_crop = img_np[y_min:y_max, x_min:x_max]
            image = Image.fromarray(hand_crop).resize((224, 224))
        else:
            image = image.resize((224, 224))

    img_array = np.asarray(image) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image_bytes = file.read()
        img_tensor = preprocess_image(image_bytes)

        infer = model.signatures["serving_default"]
        input_tensor = tf.convert_to_tensor(img_tensor)
        output = infer(input_tensor)

        predictions = list(output.values())[0].numpy()
        predicted_index = np.argmax(predictions)
        confidence = float(predictions[0][predicted_index])
        predicted_class = class_names[predicted_index]

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
        logging.error(f"Error en la predicción: {e}")
        return jsonify({'error': 'Prediction failed'}), 500


@app.route('/reset', methods=['POST'])
def reset():
    spelling.clear()
    return jsonify({'message': 'Texto acumulado reiniciado'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
