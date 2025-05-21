from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI(docs_url="/docs", redoc_url="/redoc")

model = load_model("food101_mobilenetv2_15epochs.h5")

classes = [
    "apple_pie", "bibimbap", "caesar_salad", "chicken_curry",
    "dumplings", "french_fries", "hot_dog", "lasagna",
    "omelette", "pizza", "ramen", "samosa", "steak", "sushi"
]

def preprocess(image_bytes):
    image = Image.open(BytesIO(image_bytes)).resize((224, 224)).convert("RGB")
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess(image_bytes)
    prediction = model.predict(image)[0]
    class_idx = int(np.argmax(prediction))
    return {
        "label": classes[class_idx],
        "confidence": float(prediction[class_idx])
    }

@app.get("/")
def read_root():
    return {"message": "Food recognition API is running"}
