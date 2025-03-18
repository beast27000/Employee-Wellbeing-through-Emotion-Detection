import cv2
import tkinter as tk
from PIL import Image, ImageTk
import random
import os
import torch
import torch.nn as nn  # Added this
import torchvision.models as models
import torchvision.transforms as transforms

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
base_path = "C:\\Employee Welbeing through Emotion Detection\\The Solution"
test_path = f"{base_path}\\grok\\data\\test"
model_path = f"{base_path}\\saved_emotion_model\\Variables\\resnet_best.pth"

# Load model
resnet = models.resnet50(weights=None)  # Updated to suppress warning
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Grayscale
resnet.fc = nn.Linear(resnet.fc.in_features, 7)
resnet.load_state_dict(torch.load(model_path, weights_only=True))  # Updated for safety
resnet.eval().to(device)

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Emotions
emotions = sorted(os.listdir(test_path))

# Select random test images
test_images = []
actual_labels = []
valid_extensions = ('.jpg', '.png', '.jpeg')
for emotion in emotions:
    emotion_path = os.path.join(test_path, emotion)
    imgs = [os.path.join(emotion_path, img) for img in os.listdir(emotion_path) 
            if img.lower().endswith(valid_extensions)]
    if imgs:
        test_images.extend(random.sample(imgs, 1))  # 1 per emotion for demo
        actual_labels.extend([emotion] * 1)
    else:
        print(f"Warning: No valid images found in {emotion_path}")

# Batch inference
images_tensor = torch.stack([transform(Image.open(img)) for img in test_images]).to(device)
with torch.no_grad():
    outputs = resnet(images_tensor)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
predicted_emotions = [emotions[p] for p in preds]

# GUI setup
root = tk.Tk()
root.title("Emotion Detection Results")

for img_path, actual, pred in zip(test_images, actual_labels, predicted_emotions):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_pil = Image.fromarray(img).resize((200, 200))
    img_tk = ImageTk.PhotoImage(img_pil)
    panel = tk.Label(root, image=img_tk)
    panel.image = img_tk  # Keep reference
    panel.pack()
    label = tk.Label(root, text=f"Actual: {actual} | Predicted: {pred}")
    label.pack()

# Accuracy
correct = sum(1 for a, p in zip(actual_labels, predicted_emotions) if a == p)
accuracy = correct / len(test_images)
tk.Label(root, text=f"Accuracy: {accuracy:.2f}").pack()

root.mainloop()