import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np


# Define the MNIST-like Fully Connected Model
class FCModel(nn.Module):
    def __init__(self, num_classes):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ImageClassifier class encapsulating model loading, preprocessing, and prediction
class ImageClassifier:
    def __init__(self, weights_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model = self._load_model(weights_path)
        self.model.to(self.device)
        self.model.eval()
        self.transform = self._get_transform()

    def _load_model(self, weights_path):
        """Load the trained model with weights."""
        num_classes = len(self.class_names)
        model = FCModel(num_classes)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        print("Model loaded successfully.")
        return model

    def _get_transform(self):
        """Define the preprocessing transformations."""
        return transforms.Compose([
            transforms.ToPILImage(),          # Convert tensor to PIL Image
            transforms.Grayscale(),           # Convert to grayscale
            transforms.Resize((28, 28)),      # Resize to 28x28
            transforms.ToTensor(),            # Convert to tensor
        ])

    def preprocess_image(self, image):
        """Preprocess the input OpenCV image."""
        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)

    def predict(self, image):
        """Predict the class of the given image."""
        image_tensor = self.preprocess_image(image)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = self.class_names[predicted.item()]
        return predicted_class