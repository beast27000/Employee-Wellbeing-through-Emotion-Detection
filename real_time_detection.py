import cv2
import time
import os
import shutil
from datetime import datetime
import psycopg
import tkinter as tk
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model_path = "C:\\Employee Welbeing through Emotion Detection\\The Solution\\saved_emotion_model\\Variables\\resnet_best.pth"
resnet = models.resnet50(weights=None)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 7)
try:
    resnet.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Please ensure training completes first.")
    exit()
resnet.eval().to(device)

# Transform (grayscale for FER)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Emotion mapping
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Database connection
try:
    conn = psycopg.connect(dbname="emotion_detection", user="postgres", password="Calcite*1234", host="localhost")
    cur = conn.cursor()
    print("Database connected.")
except psycopg.Error as e:
    print(f"Database connection failed: {e}")
    exit()

# Show emotion mapping
print("\nEmotion Number Mapping:")
for i, emotion in enumerate(emotions):
    print(f"{i}: {emotion}")
print("Tip: Big smile = happy (3), blank face = neutral (4), wide eyes/mouth = surprise (6)")
print("Note: Fear (2) and Anger (0) detection reduced to favor others.")

# GUI for user input
root = tk.Tk()
root.title("User Input")
tk.Label(root, text="User ID").pack()
user_id_entry = tk.Entry(root)
user_id_entry.pack()
tk.Label(root, text="Department").pack()
dept_entry = tk.Entry(root)
dept_entry.pack()
tk.Label(root, text="Consent (yes/no)").pack()
consent_entry = tk.Entry(root)
consent_entry.pack()
tk.Button(root, text="Start", command=root.quit).pack()
root.mainloop()

user_id, dept, consent = user_id_entry.get(), dept_entry.get(), consent_entry.get().lower()
if consent != "yes":
    print("Consent not given. Exiting.")
    cur.close()
    conn.close()
    exit()

# Record 2-minute video with live feedback and penalty
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    cur.close()
    conn.close()
    exit()

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("temp_video.avi", fourcc, 24.0, (640, 480))
start_time = time.time()
print("Recording 2-minute video...")
frame_counter = 0
while time.time() - start_time < 120:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Live prediction (every 10th frame)
    if frame_counter % 10 == 0:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = resnet(tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
            # Penalize fear (2) and anger (0)
            prob[0] *= 0.5  # Anger reduced by 50%
            prob[2] *= 0.5  # Fear reduced by 50%
            emotion_idx = np.argmax(prob)
            emotion = emotions[emotion_idx]
            confidence = prob[emotion_idx]
        cv2.putText(frame, f"{emotion_idx}: {emotion} ({confidence:.2%})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Recording", frame)
    out.write(frame)
    frame_counter += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
out.release()
cv2.destroyAllWindows()

# Process frames with penalty and sampling
frame_dir = "temp_frames"
sample_dir = "sample_frames"
os.makedirs(frame_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)
vid = cv2.VideoCapture("temp_video.avi")
frames, timestamps = [], []
frame_count = 0

print("Extracting frames...")
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break
    frame_path = f"{frame_dir}\\frame_{frame_count}.jpg"
    cv2.imwrite(frame_path, frame)
    frames.append(frame_path)
    timestamps.append(datetime.now())
    frame_count += 1
vid.release()

# Batch inference with penalty
batch_size = 32
saved_emotions = {}
priority_emotions = [3, 4, 6]  # happy, neutral, surprise
sample_limit = 15
for i in tqdm(range(0, len(frames), batch_size), desc="Processing frames"):
    batch_frames = frames[i:i + batch_size]
    batch_tensors = torch.stack([transform(Image.open(f)) for f in batch_frames]).to(device)
    with torch.no_grad():
        outputs = resnet(batch_tensors)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        # Penalize fear and anger
        probs[:, 0] *= 0.5  # Anger
        probs[:, 2] *= 0.5  # Fear
        emotion_idxs = np.argmax(probs, axis=1)
    
    for j, (frame_path, emotion_idx, prob) in enumerate(zip(batch_frames, emotion_idxs, probs.max(axis=1))):
        emotion = emotions[emotion_idx]
        print(f"Frame {i+j}: Predicted {emotion_idx} - {emotion} (Confidence: {prob:.2%})")
        if len(saved_emotions) < sample_limit and (emotion_idx in priority_emotions or emotion_idx not in saved_emotions):
            sample_path = f"{sample_dir}\\sample_{len(saved_emotions)}_{emotion_idx}_{emotion}.jpg"
            shutil.copy(frame_path, sample_path)
            saved_emotions[emotion_idx] = True
    
    # Store in database
    for frame_path, emotion_idx, ts in zip(batch_frames, emotion_idxs, timestamps[i:i + batch_size]):
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H:%M:%S")
        try:
            cur.execute(
                "INSERT INTO emotions (id, emotion, department, date_stamp, time_stamp) VALUES (%s, %s, %s, %s, %s)",
                (user_id, int(emotion_idx), dept, date_str, time_str)
            )
            conn.commit()
        except psycopg.Error as e:
            print(f"Database insert failed: {e}")
            conn.rollback()

# Cleanup
print("Cleaning up temporary files...")
shutil.rmtree(frame_dir)
cur.close()
conn.close()
print(f"Processed {frame_count} frames and cleaned up temporary files.")
print(f"Saved {len(saved_emotions)} sample frames in {sample_dir}, prioritizing happy (3), neutral (4), surprise (6).")