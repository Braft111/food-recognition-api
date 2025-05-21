
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()
model = load_model("food101_mobilenetv2_15epochs.h5")

# Примерный список классов (можно заменить на полный список из Food-101)
classes = ['apple_pie', 'bibimbap', 'caesar_salad', 'cheesecake', 'chicken_curry', 
           'donuts', 'edamame', 'falafel', 'french_fries', 'hamburger', 
           'hot_and_sour_soup', 'lasagna', 'omelette', 'pad_thai', 'pizza', 
           'ramen', 'spaghetti_bolognese', 'spring_rolls', 'steak', 'sushi']

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224)).convert("RGB")
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess(image_bytes)
    prediction = model.predict(img)[0]
    class_idx = int(np.argmax(prediction))
    return {
        "label": classes[class_idx],
        "confidence": float(prediction[class_idx])
    }
