from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = FastAPI()

# Load the trained model
model = load_model('path/to/model.h5')

# Define the target image size expected by the model
target_size = (224, 224)


def preprocess_image(image):
    img = image.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    image = Image.open(io.BytesIO(await file.read()))

    # Preprocess the image
    img = preprocess_image(image)

    # Perform the prediction
    pred = model.predict(img)
    class_indices = {0: "WHITESPOT", 1: "BACTERIALFIN", 2: "REDSPOT", 3: "HEALTHY"}
    predicted_class = class_indices[np.argmax(pred)]
    confidence = np.max(pred) * 100

    return {"predicted_class": predicted_class, "confidence": confidence}
