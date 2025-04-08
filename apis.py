from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from validation import ImageClassifier
app = FastAPI()

# Define your class names here
class_names = ['Excavatedpit', 'Land', 'boundary', 'chimmney', 'concretecol', 'concretefoot', 'coworker',
               'earthmover', 'powerlines', 'residential', 'solarpanel', 'staircase', 'towercrane',
               'trees', 'watertanks']

# Load model once
classifier = ImageClassifier(model_path='model.pth', class_names=class_names)

# Request body model
class ImageRequest(BaseModel):
    image_base64: str

@app.post("/predict")
def predict_image(req: ImageRequest):
    try:
        predicted_class = classifier.predict(req.image_base64)
        return {
            "predicted_class": predicted_class,
            "uploaded_image_base64": req.image_base64
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))