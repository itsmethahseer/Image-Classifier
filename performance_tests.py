from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torchvision.transforms as transforms
from torchvision import datasets, models
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pandas as pd

# Setup
all_preds = []
all_labels = []
data_dir = '/home/thahseer/Desktop/image_classification/data'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # use shuffle=False for consistent label order
model = models.resnet18(pretrained=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = dataset.classes
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Report
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

# Save report to CSV
report_df.to_csv("classification_report.csv", index=True)