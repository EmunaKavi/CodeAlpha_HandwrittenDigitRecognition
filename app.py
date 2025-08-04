import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Load the trained model
model = tf.keras.models.load_model('model.h5')

app = FastAPI()

# Serve the index HTML file
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale

    # Resize the image to 28x28
    image = image.resize((28, 28))

    # Convert the image to a numpy array and normalize
    image_array = np.array(image) / 255.0

    # Reshape to match the model input shape (1, 28, 28, 1)
    image_array = np.reshape(image_array, (1, 28, 28, 1))

    # Predict the digit
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    # Convert numpy.int64 to a regular Python int
    return {"digit": int(predicted_digit)}
