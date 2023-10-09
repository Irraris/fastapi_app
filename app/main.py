from fastapi import FastAPI, UploadFile
from PIL import Image
from ml.model_func import load_model, predict_image, load_classes

app = FastAPI()

model = load_model()
classes = load_classes()

@app.get("/")
def index():
    return {"text": "Sentiment Analysis"}

@app.post('/classify')
def classify(file: UploadFile):
    image = Image.open(file.file)
    prediction = predict_image(model, image)
    class_label = classes[prediction]
    return {"class_label": class_label}