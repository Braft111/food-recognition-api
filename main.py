from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO
import os

app = FastAPI()

# Загружаем модель
model = load_model("food101_mobilenetv2_15epochs.h5")

# Примерный список классов (можно заменить на полный список из Food-101)
classes = [
    "apple_pie", "bibimbap", "caesar_salad", "chicken_curry",
    "dumplings", "french_fries", "hot_dog", "lasagna",
    "omelette", "pizza", "ramen", "samosa", "steak", "sushi"
]

# Предобработка изображения
def preprocess(image_bytes):
    image = Image.open(BytesIO(image_bytes)).resize((224, 224)).convert("RGB")
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Роут для предсказания
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

# Корневой роут для проверки, что API запущен
@app.get("/")
def read_root():
    return {"message": "Food recognition API is running"}

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

