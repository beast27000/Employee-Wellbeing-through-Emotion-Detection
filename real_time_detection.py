

import cv2
import time
import os
import shutil
from datetime import datetime, timedelta
import psycopg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
BASE_DIR = "C:\\Employee Welbeing through Emotion Detection\\The Solution"
MODEL_PATH = f"{BASE_DIR}\\saved_emotion_model\\Variables\\ensemble_model.pth"
TEST_DATA_DIR = f"{BASE_DIR}\\grok\\data\\test"
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_COLORS = {
    'angry': (0, 0, 255),
    'disgust': (0, 140, 255),
    'fear': (0, 69, 255),
    'happy': (0, 255, 0),
    'neutral': (255, 255, 0),
    'sad': (255, 0, 0),
    'surprise': (255, 0, 255)
}
VALID_DEPTS = ['IT', 'Accounting', 'Marketing', 'All']
MAX_FRAMES_TO_PROCESS = 100
WINDOW_SIZE = (1280, 720)
DETECTION_CONFIDENCE_THRESHOLD = 0.4
CSV_FILE_PATH = os.path.join(BASE_DIR, "Emotions_Data.csv")

# Database connection parameters
DB_PARAMS = {
    "dbname": "emotion_detection",
    "user": "postgres",
    "password": "Calcite*1234",
    "host": "localhost"
}

# Custom ResNet for grayscale images
class ModifiedResNet(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(ModifiedResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# Enhanced CNN model
class EnhancedCNN(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(EnhancedCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout2d(0.2)
        )
        self.feature_size = 128 * (224 // 8) * (224 // 8)
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.feature_size)
        x = self.fc_layers(x)
        return x

# Ensemble model
class EnsembleModel:
    def __init__(self, model1, model2, weight1=0.6, weight2=0.4):
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        self.weight2 = weight2
    
    def predict(self, x):
        self.model1.eval()
        self.model2.eval()
        with torch.no_grad():
            output1 = self.model1(x)
            output2 = self.model2(x)
            weighted_sum = self.weight1 * torch.softmax(output1, dim=1) + self.weight2 * torch.softmax(output2, dim=1)
            return weighted_sum

# Transform for image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load models
def load_model():
    resnet = ModifiedResNet().to(device)
    cnn = EnhancedCNN().to(device)
    try:
        checkpoint = torch.load(MODEL_PATH)
        resnet.load_state_dict(checkpoint['resnet_state_dict'])
        cnn.load_state_dict(checkpoint['cnn_state_dict'])
        weight1 = checkpoint.get('weight1', 0.6)
        weight2 = checkpoint.get('weight2', 0.4)
        ensemble = EnsembleModel(resnet, cnn, weight1, weight2)
        print("Ensemble model loaded successfully.")
        return ensemble
    except FileNotFoundError:
        try:
            individual_model_path = f"{BASE_DIR}\\saved_emotion_model\\Variables\\resnet18_best.pth"
            resnet.load_state_dict(torch.load(individual_model_path))
            print("ResNet model loaded as fallback.")
            return EnsembleModel(resnet, resnet, 1.0, 0.0)
        except FileNotFoundError:
            print("No models found. Please ensure training completes first.")
            exit()

# Database connection
def connect_db():
    try:
        conn = psycopg.connect(**DB_PARAMS)
        print("Database connected.")
        return conn
    except psycopg.Error as e:
        print(f"Database connection failed: {e}")
        exit()

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_rois = []
    face_coords = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_rois.append(face_img)
        face_coords.append((x, y, w, h))
    return face_rois, face_coords

# Emotion history
class EmotionHistory:
    def __init__(self, history_size=5):
        self.history_size = history_size
        self.emotion_history = []
    
    def add_emotion(self, emotion, confidence):
        self.emotion_history.append((emotion, confidence))
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
    
    def get_dominant_emotion(self):
        if not self.emotion_history:
            return None, 0.0
        emotion_counts = {}
        emotion_confidences = {}
        for emotion, confidence in self.emotion_history:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                emotion_confidences[emotion] = []
            emotion_counts[emotion] += 1
            emotion_confidences[emotion].append(confidence)
        max_count = 0
        dominant_emotion = None
        for emotion, count in emotion_counts.items():
            if count > max_count:
                max_count = count
                dominant_emotion = emotion
        avg_confidence = sum(emotion_confidences[dominant_emotion]) / len(emotion_confidences[dominant_emotion])
        return dominant_emotion, avg_confidence

# GUI Application
class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection System")
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        
        self.model = load_model()
        self.conn = connect_db()
        self.user_id = None
        self.department = None
        self.emotion_history = EmotionHistory(history_size=7)
        self.emotions_detected = {emotion: 0 for emotion in EMOTIONS}
        
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        
        try:
            self.test_dataset = ImageFolder(TEST_DATA_DIR, transform=transform)
            print(f"Loaded test dataset from {TEST_DATA_DIR}")
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            self.test_dataset = None
        
        self.main_container = ttk.Frame(self.root, padding=10)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self.create_login_screen()
    
    def create_login_screen(self):
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        login_frame = ttk.Frame(self.main_container, padding=20)
        login_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(login_frame, text="Employee Wellbeing Emotion Detection", font=("Arial", 18, "bold"))
        title_label.pack(pady=20)
        
        id_frame = ttk.Frame(login_frame)
        id_frame.pack(fill=tk.X, pady=10)
        
        id_label = ttk.Label(id_frame, text="Employee ID:", font=("Arial", 12))
        id_label.pack(side=tk.LEFT, padx=5)
        
        self.id_entry = ttk.Entry(id_frame, width=30, font=("Arial", 12))
        self.id_entry.pack(side=tk.LEFT, padx=5)
        
        dept_frame = ttk.Frame(login_frame)
        dept_frame.pack(fill=tk.X, pady=10)
        
        dept_label = ttk.Label(dept_frame, text="Department:", font=("Arial", 12))
        dept_label.pack(side=tk.LEFT, padx=5)
        
        self.dept_var = tk.StringVar()
        self.dept_dropdown = ttk.Combobox(dept_frame, textvariable=self.dept_var, state="readonly", width=28, font=("Arial", 12))
        self.dept_dropdown['values'] = VALID_DEPTS
        self.dept_dropdown.current(0)
        self.dept_dropdown.pack(side=tk.LEFT, padx=5)
        
        login_button = ttk.Button(login_frame, text="Login", command=self.login)
        login_button.pack(pady=20)
        
        admin_button = ttk.Button(login_frame, text="Admin Dashboard", command=self.show_admin_dashboard)
        admin_button.pack(pady=5)
    
    def login(self):
        employee_id = self.id_entry.get().strip()
        department = self.dept_var.get()
        
        if not employee_id:
            messagebox.showerror("Error", "Please enter your Employee ID")
            return
        
        if department not in VALID_DEPTS:
            messagebox.showerror("Error", "Please select a valid department")
            return
        
        self.user_id = employee_id
        self.department = department
        
        with self.conn.cursor() as cur:
            cur.execute("SELECT employee_id FROM employees WHERE employee_id = %s", (self.user_id,))
            result = cur.fetchone()
            if not result:
                try:
                    cur.execute(
                        "INSERT INTO employees (employee_id, department) VALUES (%s, %s)",
                        (self.user_id, self.department)
                    )
                    self.conn.commit()
                except psycopg.Error as e:
                    print(f"Database error: {e}")
        
        self.create_main_screen()
    
    def show_admin_dashboard(self):
        admin_password = tk.simpledialog.askstring("Admin Access", "Enter Admin Password:", show='*')
        if admin_password != "admin123":
            messagebox.showerror("Access Denied", "Incorrect password")
            return
        self.create_admin_dashboard()
    
    def create_main_screen(self):
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        top_frame = ttk.Frame(self.main_container)
        top_frame.pack(fill=tk.X, pady=10)
        
        video_frame = ttk.Frame(self.main_container)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        bottom_frame = ttk.Frame(self.main_container)
        bottom_frame.pack(fill=tk.X, pady=10)
        
        user_info_label = ttk.Label(top_frame, text=f"Employee ID: {self.user_id} | Department: {self.department}", font=("Arial", 12))
        user_info_label.pack(side=tk.LEFT, padx=10)
        
        logout_button = ttk.Button(top_frame, text="Logout", command=self.logout_with_csv_prompt)
        logout_button.pack(side=tk.RIGHT, padx=10)
        
        self.video_canvas = tk.Canvas(video_frame, bg="black", width=640, height=480)
        self.video_canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        stats_frame = ttk.LabelFrame(video_frame, text="Real-time Emotion Stats", padding=10)
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.pie_chart = FigureCanvasTkAgg(self.fig, stats_frame)
        self.pie_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.emotion_label = ttk.Label(stats_frame, text="No emotion detected yet", font=("Arial", 14, "bold"))
        self.emotion_label.pack(pady=10)
        
        self.session_info = ttk.Label(stats_frame, text="Session time: 00:00:00")
        self.session_info.pack(pady=5)
        
        buttons_frame = ttk.Frame(bottom_frame)  # Fixed typo: should be bottom_frame
        buttons_frame.pack(fill=tk.X)
        
        start_button = ttk.Button(buttons_frame, text="Start Camera", command=self.start_camera)
        start_button.pack(side=tk.LEFT, padx=5)
        
        stop_button = ttk.Button(buttons_frame, text="Stop Camera", command=self.stop_camera)
        stop_button.pack(side=tk.LEFT, padx=5)
        
        upload_button = ttk.Button(buttons_frame, text="Upload Video", command=self.upload_video)
        upload_button.pack(side=tk.LEFT, padx=5)
        
        test_button = ttk.Button(buttons_frame, text="Test Random Images", command=self.test_random_images)
        test_button.pack(side=tk.LEFT, padx=5)
        
        self.cap = None
        self.is_running = False
        self.start_time = None
        self.frame_count = 0
        self.update_pie_chart()
    
    def start_camera(self):
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Camera failed to open")
            messagebox.showerror("Error", "Failed to open camera. Please check your camera connection.")
            return
        
        print("Camera opened successfully")
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.emotions_detected = {emotion: 0 for emotion in EMOTIONS}
        self.emotion_history = EmotionHistory(history_size=7)
        self.update_video()
    
    def stop_camera(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.record_session_data()
        
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            320, 240, text="Camera Off", fill="white", font=("Arial", 20)
        )
    
    def update_video(self):
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            self.stop_camera()
            messagebox.showerror("Error", "Failed to capture frame. Camera disconnected?")
            return
        
        print("Frame captured successfully")
        self.process_frame(frame)
        self.root.after(33, self.update_video)
    
    def process_frame(self, frame):
        print("Processing frame...")
        
        try:
            face_rois, face_coords = detect_faces(frame)
        except Exception as e:
            print(f"Face detection error: {e}")
            face_rois, face_coords = [], []
        
        for i, (face_roi, (x, y, w, h)) in enumerate(zip(face_rois, face_coords)):
            try:
                pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                img_tensor = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = self.model.predict(img_tensor)
                    probs = prediction[0].cpu().numpy()
                    emotion_idx = np.argmax(probs)
                    confidence = probs[emotion_idx]
                    predicted_emotion = EMOTIONS[emotion_idx]
                
                if confidence >= DETECTION_CONFIDENCE_THRESHOLD:
                    self.emotion_history.add_emotion(predicted_emotion, confidence)
                    dominant_emotion, avg_confidence = self.emotion_history.get_dominant_emotion()
                    
                    if dominant_emotion:
                        self.emotions_detected[dominant_emotion] += 1
                        color = EMOTION_COLORS.get(dominant_emotion, (255, 255, 255))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        emotion_text = f"{dominant_emotion}: {avg_confidence:.2f}"
                        cv2.putText(
                            frame, emotion_text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                        )
                        self.emotion_label.config(text=f"Detected emotion: {dominant_emotion.capitalize()}")
            except Exception as e:
                print(f"Emotion detection error: {e}")
        
        self.frame_count += 1
        session_duration = time.time() - self.start_time
        hours, remainder = divmod(int(session_duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        self.session_info.config(text=f"Session time: {hours:02}:{minutes:02}:{seconds:02}")
        
        if self.frame_count % 30 == 0:
            self.update_pie_chart()
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            self.photo = photo
            self.video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            print("Frame rendered on canvas")
        except Exception as e:
            print(f"Rendering error: {e}")
    
    def update_pie_chart(self):
        self.ax.clear()
        labels = []
        sizes = []
        colors = []
        total = sum(self.emotions_detected.values())
        
        if total == 0:
            self.ax.text(0.5, 0.5, "No emotions detected yet", ha='center', va='center', fontsize=12)
            self.ax.axis('off')
        else:
            for emotion, count in self.emotions_detected.items():
                if count > 0:
                    labels.append(emotion.capitalize())
                    sizes.append(count)
                    bgr_color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                    rgb_color = (bgr_color[2]/255, bgr_color[1]/255, bgr_color[0]/255)
                    colors.append(rgb_color)
            wedges, texts, autotexts = self.ax.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90
            )
            self.ax.axis('equal')
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
        
        self.ax.set_title("Emotion Distribution")
        self.pie_chart.draw()
    
    def record_session_data(self):
        if self.start_time is None or self.frame_count == 0:
            return
        
        end_time = time.time()
        session_duration = end_time - self.start_time
        dominant_emotion = max(self.emotions_detected.items(), key=lambda x: x[1])[0] if any(self.emotions_detected.values()) else "unknown"
        session_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions 
                    (employee_id, duration_seconds, dominant_emotion, session_date) 
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (self.user_id, session_duration, dominant_emotion, session_date)
                )
                session_id = cur.fetchone()[0]
                for emotion, count in self.emotions_detected.items():
                    if count > 0:
                        percentage = (count / self.frame_count) * 100
                        cur.execute(
                            """
                            INSERT INTO emotion_details 
                            (session_id, emotion, count, percentage) 
                            VALUES (%s, %s, %s, %s)
                            """,
                            (session_id, emotion, count, percentage)
                        )
                self.conn.commit()
                print(f"Session data recorded successfully. Session ID: {session_id}")
        except psycopg.Error as e:
            print(f"Database error recording session: {e}")
            self.conn.rollback()
    
    def upload_video(self):
        if self.is_running:
            self.stop_camera()
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov"), ("All files", "*.*")]
        )
        if not file_path:
            return
        self.process_uploaded_video(file_path)
    
    def process_uploaded_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_to_process = min(total_frames, MAX_FRAMES_TO_PROCESS)
        
        self.emotions_detected = {emotion: 0 for emotion in EMOTIONS}
        self.emotion_history = EmotionHistory(history_size=7)
        
        progress = ttk.Progressbar(self.main_container, orient="horizontal", length=300, mode="determinate")
        progress.pack(pady=10)
        progress["maximum"] = frames_to_process
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened() and frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            face_rois, _ = detect_faces(frame)
            for face_roi in face_rois:
                pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                img_tensor = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = self.model.predict(img_tensor)
                    probs = prediction[0].cpu().numpy()
                    emotion_idx = np.argmax(probs)
                    confidence = probs[emotion_idx]
                    predicted_emotion = EMOTIONS[emotion_idx]
                
                if confidence >= DETECTION_CONFIDENCE_THRESHOLD:
                    self.emotion_history.add_emotion(predicted_emotion, confidence)
                    dominant_emotion, _ = self.emotion_history.get_dominant_emotion()
                    if dominant_emotion:
                        self.emotions_detected[dominant_emotion] += 1
            
            frame_count += 1
            progress["value"] = frame_count
            self.root.update()
            if frames_to_process < total_frames:
                skip_frames = max(1, int(total_frames / frames_to_process) - 1)
                for _ in range(skip_frames):
                    cap.read()
        
        cap.release()
        progress.destroy()
        self.update_pie_chart()
        self.emotion_label.config(text="Video analysis complete")
        
        session_duration = time.time() - start_time
        self.start_time = start_time
        self.frame_count = frame_count
        self.record_session_data()
        
        result_message = "Video Analysis Results:\n"
        total = sum(self.emotions_detected.values())
        if total > 0:
            for emotion, count in sorted(self.emotions_detected.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                result_message += f"{emotion.capitalize()}: {percentage:.1f}%\n"
        else:
            result_message += "No emotions detected in the video."
        messagebox.showinfo("Analysis Complete", result_message)

    def test_random_images(self):
        if not self.test_dataset:
            messagebox.showerror("Error", "Test dataset not available")
            return
        
        num_samples = 10
        sample_indices = random.sample(range(len(self.test_dataset)), min(num_samples, len(self.test_dataset)))
        self.emotions_detected = {emotion: 0 for emotion in EMOTIONS}
        images = []
        emotions = []
        confidences = []
        
        # Classify 10 random images
        for idx in sample_indices:
            img_tensor, _ = self.test_dataset[idx]  # Ignore ground truth label
            img_pil = transforms.ToPILImage()(img_tensor * 0.5 + 0.5)  # Denormalize for display
            img_tensor = img_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = self.model.predict(img_tensor)
                probs = prediction[0].cpu().numpy()
                emotion_idx = np.argmax(probs)
                confidence = probs[emotion_idx]
                predicted_emotion = EMOTIONS[emotion_idx]
            self.emotions_detected[predicted_emotion] += 1
            images.append(img_pil)
            emotions.append(predicted_emotion.capitalize())
            confidences.append(f"{confidence:.2f}")
        
        # Create comic strip
        fig, axes = plt.subplots(1, 10, figsize=(20, 4), constrained_layout=True)
        for ax, img, emotion, conf in zip(axes, images, emotions, confidences):
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f"{emotion}\nConf: {conf}", fontsize=10, pad=5)
        plt.suptitle("Random Test Image Classifications", fontsize=14)
        
        # Display in a new Tkinter window
        strip_window = tk.Toplevel(self.root)
        strip_window.title("Test Image Strip")
        strip_window.geometry("1000x300")
        canvas = FigureCanvasTkAgg(fig, master=strip_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update pie chart and show results
        self.update_pie_chart()
        self.emotion_label.config(text=f"Tested {num_samples} random images")
        
        result_message = "Random Image Test Results:\n"
        for emotion, count in sorted(self.emotions_detected.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / num_samples) * 100
            result_message += f"{emotion.capitalize()}: {percentage:.1f}%\n"
        messagebox.showinfo("Test Complete", result_message)
    def create_admin_dashboard(self):
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        admin_frame = ttk.Frame(self.main_container, padding=10)
        admin_frame.pack(fill=tk.BOTH, expand=True)
        
        header_frame = ttk.Frame(admin_frame)
        header_frame.pack(fill=tk.X, pady=10)
        
        title_label = ttk.Label(header_frame, text="Admin Dashboard", font=("Arial", 18, "bold"))
        title_label.pack(side=tk.LEFT, padx=10)
        
        back_button = ttk.Button(header_frame, text="Back to Login", command=self.create_login_screen)
        back_button.pack(side=tk.RIGHT, padx=10)
        
        notebook = ttk.Notebook(admin_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Employee Statistics Tab
        employee_tab = ttk.Frame(notebook, padding=10)
        notebook.add(employee_tab, text="Employee Statistics")
        
        filter_frame = ttk.LabelFrame(employee_tab, text="Filters", padding=10)
        filter_frame.pack(fill=tk.X, pady=10)
        
        dept_label = ttk.Label(filter_frame, text="Department:")
        dept_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.admin_dept_var = tk.StringVar()
        admin_dept_dropdown = ttk.Combobox(filter_frame, textvariable=self.admin_dept_var, state="readonly", width=15)
        admin_dept_dropdown['values'] = VALID_DEPTS
        admin_dept_dropdown.current(0)
        admin_dept_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        date_label = ttk.Label(filter_frame, text="Date Range:")
        date_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        date_options = ["Last 7 Days", "Last 30 Days", "All Time"]
        self.admin_date_var = tk.StringVar()
        admin_date_dropdown = ttk.Combobox(filter_frame, textvariable=self.admin_date_var, state="readonly", width=15)
        admin_date_dropdown['values'] = date_options
        admin_date_dropdown.current(0)
        admin_date_dropdown.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        apply_button = ttk.Button(filter_frame, text="Apply Filters", command=self.update_admin_stats)
        apply_button.grid(row=0, column=4, padx=20, pady=5, sticky=tk.E)
        
        stats_frame = ttk.LabelFrame(employee_tab, text="Department Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.admin_fig, (self.admin_ax1, self.admin_ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.admin_canvas = FigureCanvasTkAgg(self.admin_fig, master=stats_frame)
        self.admin_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        download_stats_button = ttk.Button(stats_frame, text="Download Stats", command=self.download_admin_stats)
        download_stats_button.pack(pady=5)
        
        details_frame = ttk.LabelFrame(employee_tab, text="Employee Details", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        columns = ("employee_id", "department", "sessions", "avg_duration", "dominant_emotion")
        self.employee_tree = ttk.Treeview(details_frame, columns=columns, show="headings")
        self.employee_tree.heading("employee_id", text="Employee ID")
        self.employee_tree.heading("department", text="Department")
        self.employee_tree.heading("sessions", text="Sessions")
        self.employee_tree.heading("avg_duration", text="Avg Duration (min)")
        self.employee_tree.heading("dominant_emotion", text="Dominant Emotion")
        self.employee_tree.column("employee_id", width=100)
        self.employee_tree.column("department", width=100)
        self.employee_tree.column("sessions", width=70)
        self.employee_tree.column("avg_duration", width=120)
        self.employee_tree.column("dominant_emotion", width=120)
        
        tree_scroll = ttk.Scrollbar(details_frame, orient="vertical", command=self.employee_tree.yview)
        self.employee_tree.configure(yscrollcommand=tree_scroll.set)
        self.employee_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Department Overview Tab
        dept_tab = ttk.Frame(notebook, padding=10)
        notebook.add(dept_tab, text="Department Overview")
        
        dept_chart_frame = ttk.LabelFrame(dept_tab, text="Department Comparison", padding=10)
        dept_chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.dept_fig, self.dept_ax = plt.subplots(figsize=(10, 6))
        self.dept_canvas = FigureCanvasTkAgg(self.dept_fig, master=dept_chart_frame)
        self.dept_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        download_dept_button = ttk.Button(dept_chart_frame, text="Download Dept Chart", command=self.download_dept_chart)
        download_dept_button.pack(pady=5)
        
        export_frame = ttk.Frame(admin_frame)
        export_frame.pack(fill=tk.X, pady=10)
        
        export_button = ttk.Button(export_frame, text="Export Data", command=self.export_data)
        export_button.pack(side=tk.RIGHT, padx=10)
        
        self.update_admin_stats()
    
    def update_admin_stats(self):
        department = self.admin_dept_var.get()
        date_range = self.admin_date_var.get()
        
        today = datetime.now().date()
        if date_range == "Last 7 Days":
            start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        elif date_range == "Last 30 Days":
            start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        else:
            start_date = "2000-01-01"
        
        dept_condition = "" if department == "All" else f"AND e.department = '{department}'"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT ed.emotion, SUM(ed.count) AS total_count
                    FROM emotion_details ed
                    JOIN sessions s ON ed.session_id = s.id
                    JOIN employees e ON s.employee_id = e.employee_id
                    WHERE s.session_date >= '{start_date}'
                    {dept_condition}
                    GROUP BY ed.emotion
                    ORDER BY total_count DESC
                """)
                emotion_stats = cur.fetchall()
                
                cur.execute(f"""
                    SELECT e.department, COUNT(s.id) AS session_count,
                           AVG(s.duration_seconds) AS avg_duration
                    FROM sessions s
                    JOIN employees e ON s.employee_id = e.employee_id
                    WHERE s.session_date >= '{start_date}'
                    GROUP BY e.department
                    ORDER BY session_count DESC
                """)
                dept_stats = cur.fetchall()
                
                cur.execute(f"""
                    SELECT e.employee_id, e.department, 
                           COUNT(s.id) AS session_count,
                           AVG(s.duration_seconds) AS avg_duration,
                           MODE() WITHIN GROUP (ORDER BY s.dominant_emotion) AS dominant_emotion
                    FROM employees e
                    LEFT JOIN sessions s ON e.employee_id = s.employee_id
                    WHERE s.session_date >= '{start_date}'
                    {dept_condition}
                    GROUP BY e.employee_id, e.department
                    ORDER BY session_count DESC
                """)
                employee_stats = cur.fetchall()
        
        except psycopg.Error as e:
            print(f"Database query error: {e}")
            return
        
        self.admin_ax1.clear()
        if emotion_stats:
            emotions = [item[0].capitalize() for item in emotion_stats]
            counts = [item[1] for item in emotion_stats]
            colors = [EMOTION_COLORS.get(emotion.lower(), (255, 255, 255)) for emotion in emotions]
            colors = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]
            self.admin_ax1.pie(counts, labels=emotions, autopct='%1.1f%%', colors=colors, startangle=90)
            self.admin_ax1.axis('equal')
            self.admin_ax1.set_title("Emotion Distribution")
        else:
            self.admin_ax1.text(0.5, 0.5, "No emotion data available", ha='center', va='center')
            self.admin_ax1.axis('off')
        
        self.admin_ax2.clear()
        if dept_stats:
            departments = [item[0] for item in dept_stats]
            session_counts = [item[1] for item in dept_stats]
            avg_durations = [item[2]/60 if item[2] else 0 for item in dept_stats]
            x = np.arange(len(departments))
            width = 0.35
            self.admin_ax2.bar(x - width/2, session_counts, width, label='Sessions')
            self.admin_ax2.bar(x + width/2, avg_durations, width, label='Avg. Duration (min)')
            self.admin_ax2.set_xticks(x)
            self.admin_ax2.set_xticklabels(departments)
            self.admin_ax2.legend()
            self.admin_ax2.set_title("Department Activity")
        else:
            self.admin_ax2.text(0.5, 0.5, "No department data available", ha='center', va='center')
            self.admin_ax2.axis('off')
        
        self.admin_fig.tight_layout()
        self.admin_canvas.draw()
        
        self.dept_ax.clear()
        if dept_stats and emotion_stats:
            top_emotions = [item[0] for item in emotion_stats[:3]] if len(emotion_stats) >= 3 else [item[0] for item in emotion_stats]
            try:
                with self.conn.cursor() as cur:
                    dept_emotion_data = {}
                    for dept in [d[0] for d in dept_stats]:
                        cur.execute(f"""
                            SELECT ed.emotion, SUM(ed.count) AS total_count
                            FROM emotion_details ed
                            JOIN sessions s ON ed.session_id = s.id
                            JOIN employees e ON s.employee_id = e.employee_id
                            WHERE s.session_date >= '{start_date}'
                            AND e.department = '{dept}'
                            AND ed.emotion IN ('{"', '".join(top_emotions)}')
                            GROUP BY ed.emotion
                        """)
                        dept_emotions = cur.fetchall()
                        dept_emotion_dict = {emotion: 0 for emotion in top_emotions}
                        for emotion, count in dept_emotions:
                            dept_emotion_dict[emotion] = count
                        dept_emotion_data[dept] = dept_emotion_dict
                    
                    departments = list(dept_emotion_data.keys())
                    x = np.arange(len(departments))
                    width = 0.8 / len(top_emotions)
                    
                    for i, emotion in enumerate(top_emotions):
                        values = [dept_emotion_data[dept][emotion] for dept in departments]
                        emotion_color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                        rgb_color = (emotion_color[2]/255, emotion_color[1]/255, emotion_color[0]/255)
                        self.dept_ax.bar(
                            x + (i - len(top_emotions)/2 + 0.5) * width,
                            values,
                            width,
                            label=emotion.capitalize(),
                            color=rgb_color
                        )
                    self.dept_ax.set_xticks(x)
                    self.dept_ax.set_xticklabels(departments)
                    self.dept_ax.legend()
                    self.dept_ax.set_title("Top Emotions by Department")
                    self.dept_ax.set_ylabel("Count")
            except psycopg.Error as e:
                print(f"Department emotion query error: {e}")
                self.dept_ax.text(0.5, 0.5, "Error loading department emotion data", ha='center', va='center')
                self.dept_ax.axis('off')
        else:
            self.dept_ax.text(0.5, 0.5, "No data available for department comparison", ha='center', va='center')
            self.dept_ax.axis('off')
        
        self.dept_fig.tight_layout()
        self.dept_canvas.draw()
        
        for item in self.employee_tree.get_children():
            self.employee_tree.delete(item)
        
        for emp_id, dept, session_count, avg_duration, dominant_emotion in employee_stats:
            avg_min = round(avg_duration / 60, 1) if avg_duration else 0
            emotion_display = dominant_emotion.capitalize() if dominant_emotion else "N/A"
            self.employee_tree.insert("", "end", values=(emp_id, dept, session_count, avg_min, emotion_display))
    
    def download_admin_stats(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Department Statistics",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=f"department_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if file_path:
            self.admin_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Download Complete", f"Department statistics saved to {file_path}")
    
    def download_dept_chart(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Department Comparison",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=f"dept_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if file_path:
            self.dept_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Download Complete", f"Department comparison saved to {file_path}")
    
    def export_data(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return
        
        try:
            department = self.admin_dept_var.get()
            date_range = self.admin_date_var.get()
            today = datetime.now().date()
            if date_range == "Last 7 Days":
                start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
            elif date_range == "Last 30 Days":
                start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
            else:
                start_date = "2000-01-01"
            
            dept_condition = "" if department == "All" else f"AND e.department = '{department}'"
            
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT 
                        e.employee_id,
                        e.department,
                        s.session_date,
                        s.duration_seconds,
                        s.dominant_emotion,
                        ed.emotion,
                        ed.percentage
                    FROM 
                        employees e
                    JOIN 
                        sessions s ON e.employee_id = s.employee_id
                    JOIN 
                        emotion_details ed ON s.id = ed.session_id
                    WHERE 
                        s.session_date >= '{start_date}'
                        {dept_condition}
                    ORDER BY 
                        e.employee_id, s.session_date
                """)
                results = cur.fetchall()
            
            with open(file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([
                    "Employee ID", "Department", "Session Date", 
                    "Duration (seconds)", "Dominant Emotion",
                    "Emotion", "Percentage"
                ])
                for row in results:
                    csvwriter.writerow(row)
            messagebox.showinfo("Export Complete", f"Data exported successfully to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")
    
    def logout_with_csv_prompt(self):
        if self.is_running:
            self.stop_camera()
        
        # Prompt user to generate CSV
        generate_csv = messagebox.askyesno(
            "Generate CSV", 
            "Would you like to update Emotions_Data.csv with this session's data?"
        )
        
        if generate_csv:
            self.update_csv_for_session()
        
        self.create_login_screen()
    
    def update_csv_for_session(self):
        if not self.user_id:
            return
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        e.employee_id,
                        e.department,
                        s.id AS session_id,
                        ed.emotion,
                        ed.count,
                        ed.percentage,
                        s.session_date,
                        EXTRACT(HOUR FROM s.created_at)::TEXT || ':' || 
                        EXTRACT(MINUTE FROM s.created_at)::TEXT || ':' || 
                        EXTRACT(SECOND FROM s.created_at)::TEXT AS time_stamp
                    FROM 
                        employees e
                    JOIN 
                        sessions s ON e.employee_id = s.employee_id
                    JOIN 
                        emotion_details ed ON s.id = ed.session_id
                    WHERE 
                        e.employee_id = %s
                        AND s.created_at >= %s
                    ORDER BY 
                        s.session_date DESC, s.created_at DESC
                """, (self.user_id, datetime.fromtimestamp(self.start_time) if self.start_time else datetime.now()))
                new_data = cur.fetchall()
            
            # Check if file exists and read existing data
            existing_data = []
            file_exists = os.path.exists(CSV_FILE_PATH)
            if file_exists:
                with open(CSV_FILE_PATH, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)  # Skip header
                    existing_data = list(reader)
            
            # Combine existing data with new data
            all_data = existing_data + [list(row) for row in new_data]
            
            # Write all data back to the CSV
            with open(CSV_FILE_PATH, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([
                    "Employee ID", "Department", "Session ID", "Emotion", 
                    "Count", "Percentage", "Session Date", "Time Stamp"
                ])
                csvwriter.writerows(all_data)
            
            print(f"Updated {CSV_FILE_PATH} with session data")
            messagebox.showinfo("CSV Updated", f"Session data added to {CSV_FILE_PATH}")
        except Exception as e:
            print(f"Error updating CSV: {e}")
            messagebox.showerror("CSV Error", f"Failed to update CSV: {e}")

# ImageFolder class
class ImageFolder:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Database initialization
def init_database(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'employees'
                )
            """)
            tables_exist = cur.fetchone()[0]
            if not tables_exist:
                print("Initializing database tables...")
                cur.execute("""
                    CREATE TABLE employees (
                        employee_id VARCHAR(50) PRIMARY KEY,
                        department VARCHAR(50) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("""
                    CREATE TABLE sessions (
                        id SERIAL PRIMARY KEY,
                        employee_id VARCHAR(50) REFERENCES employees(employee_id),
                        duration_seconds FLOAT NOT NULL,
                        dominant_emotion VARCHAR(20),
                        session_date DATE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("""
                    CREATE TABLE emotion_details (
                        id SERIAL PRIMARY KEY,
                        session_id INTEGER REFERENCES sessions(id),
                        emotion VARCHAR(20) NOT NULL,
                        count INTEGER NOT NULL,
                        percentage FLOAT NOT NULL
                    )
                """)
                conn.commit()
                print("Database tables created successfully")
                add_sample_data(conn)
            else:
                print("Database tables already exist")
                
            # Create the consolidated_emotion_data table with employee_id as primary key
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'consolidated_emotion_data'
                )
            """)
            consolidated_exists = cur.fetchone()[0]
            if not consolidated_exists:
                print("Creating consolidated_emotion_data table...")
                cur|execute("""
                    CREATE TABLE consolidated_emotion_data (
                        employee_id VARCHAR(50) PRIMARY KEY,
                        department VARCHAR(50) NOT NULL,
                        session_id INTEGER REFERENCES sessions(id),
                        emotion VARCHAR(20) NOT NULL,
                        count INTEGER NOT NULL,
                        percentage FLOAT NOT NULL,
                        session_date DATE NOT NULL,
                        time_stamp TIME NOT NULL
                    )
                """)
                conn.commit()
                print("Consolidated table created successfully")
                
    except psycopg.Error as e:
        print(f"Database initialization error: {e}")
        conn.rollback()

# Add sample data
def add_sample_data(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM employees")
            count = cur.fetchone()[0]
            if count == 0:
                print("Adding sample data...")
                employees = [
                    ("EMP001", "IT"),
                    ("EMP002", "IT"),
                    ("EMP003", "Accounting"),
                    ("EMP004", "Accounting"),
                    ("EMP005", "Marketing"),
                    ("EMP006", "Marketing")
                ]
                for emp_id, dept in employees:
                    cur.execute(
                        "INSERT INTO employees (employee_id, department) VALUES (%s, %s)",
                        (emp_id, dept)
                    )
                
                today = datetime.now().date()
                emotions = ["happy", "neutral", "sad", "angry", "surprise", "fear", "disgust"]
                for emp_id, _ in employees:
                    num_sessions = random.randint(3, 5)
                    for _ in range(num_sessions):
                        days_ago = random.randint(0, 29)
                        session_date = today - timedelta(days=days_ago)
                        duration = random.randint(300, 1800)
                        dominant_emotion = random.choice(emotions)
                        cur.execute(
                            """
                            INSERT INTO sessions 
                            (employee_id, duration_seconds, dominant_emotion, session_date) 
                            VALUES (%s, %s, %s, %s)
                            RETURNING id
                            """,
                            (emp_id, duration, dominant_emotion, session_date)
                        )
                        session_id = cur.fetchone()[0]
                        dominant_percentage = random.uniform(40, 70)
                        remaining_percentage = 100 - dominant_percentage
                        other_emotions = [e for e in emotions if e != dominant_emotion]
                        selected_emotions = random.sample(other_emotions, random.randint(2, 4))
                        other_percentages = []
                        remaining = remaining_percentage
                        for _ in range(len(selected_emotions) - 1):
                            p = random.uniform(5, remaining - 5 * (len(selected_emotions) - len(other_percentages) - 1))
                            other_percentages.append(p)
                            remaining -= p
                        other_percentages.append(remaining)
                        cur.execute(
                            """
                            INSERT INTO emotion_details 
                            (session_id, emotion, count, percentage) 
                            VALUES (%s, %s, %s, %s)
                            """,
                            (session_id, dominant_emotion, int(dominant_percentage), dominant_percentage)
                        )
                        for emotion, percentage in zip(selected_emotions, other_percentages):
                            cur.execute(
                                """
                                INSERT INTO emotion_details 
                                (session_id, emotion, count, percentage) 
                                VALUES (%s, %s, %s, %s)
                                """,
                                (session_id, emotion, int(percentage), percentage)
                            )
                conn.commit()
                print("Sample data added successfully")
    except psycopg.Error as e:
        print(f"Error adding sample data: {e}")
        conn.rollback()

# Consolidate data into a single table
def consolidate_data(conn):
    try:
        with conn.cursor() as cur:
            # Clear existing data in consolidated_emotion_data to avoid duplicates
            cur.execute("TRUNCATE TABLE consolidated_emotion_data")
            
            # Insert data from employees, sessions, and emotion_details into consolidated_emotion_data
            cur.execute("""
                INSERT INTO consolidated_emotion_data (
                    employee_id,
                    department, 
                    session_id, 
                    emotion, 
                    count, 
                    percentage, 
                    session_date, 
                    time_stamp
                )
                SELECT 
                    e.employee_id,
                    e.department,
                    s.id AS session_id,
                    ed.emotion,
                    ed.count,
                    ed.percentage,
                    s.session_date,
                    EXTRACT(HOUR FROM s.created_at)::TEXT || ':' || 
                    EXTRACT(MINUTE FROM s.created_at)::TEXT || ':' || 
                    EXTRACT(SECOND FROM s.created_at)::TEXT AS time_stamp
                FROM 
                    employees e
                JOIN 
                    sessions s ON e.employee_id = s.employee_id
                JOIN 
                    emotion_details ed ON s.id = ed.session_id
            """)
            conn.commit()
            print("Data successfully consolidated into consolidated_emotion_data table")
    except psycopg.Error as e:
        print(f"Error consolidating data: {e}")
        conn.rollback()

# Main function
def main():
    conn = connect_db()
    init_database(conn)
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
    
    # Consolidate data before closing the connection
    consolidate_data(conn)
    
    if conn:
        conn.close()

if __name__ == "__main__":
    main()
