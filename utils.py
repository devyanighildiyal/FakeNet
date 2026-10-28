import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# File name must match the repo exactly: Streamlit Cloud runs on Linux, which is case-sensitive.
MODEL_PATH = "Hybrid_deepfake.h5"

# compile=False skips the optimizer/loss, which inference doesn't need and which
# can cause Keras version errors when loading an older .h5 file.
model = load_model(MODEL_PATH, compile=False)

IMG_SIZE = (224, 224)

def preprocess_image(image):
    # Convert to RGB so PNGs with an alpha channel or grayscale images don't break the model input shape.
    img = image.convert("RGB").resize(IMG_SIZE)
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_deepfake(image):
    processed = preprocess_image(image)
    prediction = float(model.predict(processed, verbose=0)[0][0])
    label = "FAKE" if prediction > 0.5 else "REAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, round(confidence * 100, 2)
