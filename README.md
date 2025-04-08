# 🏗️ Image Classification API

This project builds a deep learning-based image classification model to detect features from images such as **Excavated Pit**, **Land**, **Boundary Wall**, **Solar Panel**, etc. It uses a fine-tuned **ResNet-18** model trained using PyTorch and exposes predictions via a FastAPI-powered REST API.

---

## 📌 Features

- Fine-tuned **ResNet-18**
- Simple training loop with GPU support
- FastAPI REST API for image prediction
- Accepts **Base64 encoded images**
- Includes 15 construction site classes

---

## 🧠 Classes

The model classifies images into one of the following classes:

- Excavatedpit  
- Land  
- Boundary  
- Chimmney  
- Concretecol  
- Concretefoot  
- Coworker  
- Earthmover  
- Powerlines  
- Residential  
- Solarpanel  
- Staircase  
- Towercrane  
- Trees  
- Watertanks

---

## 🗂️ Project Structure

```
.
├── data/                        # Your image data structured by class
├── model.pth                   # Saved trained model
├── training.py                 # Training script
├── validation.py               # ImageClassifier class for inference
├── api.py                      # FastAPI app
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Ensure python version 3.10.12 or above 3.10

### 1. Clone the repository

```bash
git clone https://github.com/itsmethahseer/Image-Classifier.git
cd Image-Classifier
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is not available, create one using:

```txt
torch
torchvision
fastapi
pydantic
uvicorn
pillow
```

Install using:

```bash
pip install torch torchvision fastapi pydantic uvicorn pillow
```

---

## 🏋️‍♂️ Train the Model

Make sure your images are stored inside subfolders of `data/`, with each subfolder named after its class.

```bash
python training.py
```

This will train the model and save the weights to `model.pth`.

---

## 🧪 Start the API

```bash
uvicorn api:app --reload
```

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to use the Swagger UI for testing.

---

## 🧾 API Usage

### 🔸 Endpoint

**POST** `/predict`

### 🔸 Request Body

```json
{
  "image_base64": "your_base64_encoded_image_string_here"
}
```

You can convert an image to base64 in Python:

```python
import base64

with open("your_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')
```

### 🔸 Response

```json
{
  "predicted_class": "residential",
  "uploaded_image_base64": "..."  // echoed base64 string
}
```

---

## 🛠️ To Do

- [ ] Add confidence scores
- [ ] Convert to multi-label classification (if needed)
- [ ] Dockerize the API
- [ ] Add frontend for uploading and displaying predictions

---

## 🤝 Acknowledgments

Built using:
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
