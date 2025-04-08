import torch
from torchvision import models, transforms
from PIL import Image
import io
import base64

class ImageClassifier:
    def __init__(self, model_path: str, class_names: list):
        self.class_names = class_names
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_base64: str) -> str:
        image = self._decode_image(image_base64)
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            return self.class_names[predicted.item()]

    def _decode_image(self, image_base64: str) -> Image.Image:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image