# Employee Wellbeing through Emotion Detection (v1 Tkinter version)

![WhatsApp Image 2025-07-09 at 20 18 00_7f54b435](https://github.com/user-attachments/assets/1883b461-33f0-4f58-8cce-34084cfbf144)


## Overview
The **Real-Time Emotion Monitoring and Analytics Dashboard** is a tool designed to improve employee well-being by analyzing facial expressions in real-time using deep learning. It uses an NVIDIA RTX 4060 GPU with CUDA 12.6 for fast emotion detection and model training. Unlike surveys, it gives instant insights into workforce morale. It processes 2-minute webcam feeds, detects emotions accurately, and stores data in PostgreSQL with separate `session_date` and `time_stamp` fields. The Tkinter GUI provides an easy-to-use interface for employees and admins, while a FastAPI endpoint is ready for future expansions. Privacy is maintained with implied consent via GUI use (explicit consent is planned for later) and data deletion after processing.

This `v1` release sets up a solid base with CUDA optimization and a working Tkinter frontend.

## Features

### CUDA-Accelerated Model Training
- Trains a modified ResNet18 and an enhanced CNN on the FER dataset for 25 epochs.
- Uses PyTorch with mixed precision (`torch.amp`) on an NVIDIA RTX 4060 for speed.
- Saves loss/accuracy plots in `saved_emotion_model/Assets/` with progress bars via `tqdm`.

### Real-Time Emotion Detection
- Captures a 2-minute webcam feed with OpenCV, processing up to 100 frames on the GPU with CUDA.
- Detects seven emotions (angry, disgust, fear, happy, neutral, sad, surprise) with confidence scores and color-coded boxes.
- Deletes temp files after processing for privacy.
- Adds timestamps to frames for tracking.

### Tkinter Frontend
- **Login Screen**:
  - Employees enter an ID and pick a department (IT, Accounting, Marketing, All).
  - Admins use `admin123` to access the admin dashboard.
  - Validates inputs and adds new employees to PostgreSQL.
- **Main Dashboard**:
  - Shows live webcam feed with emotions and confidence scores.
  - Displays a pie chart (via Matplotlib) of emotion distribution, updated every 30 frames.
  - Tracks session time and shows the dominant emotion.
  - Has buttons to start/stop the camera, upload videos, and test random FER images (shown as a comic strip).
- **Admin Dashboard**:
  - Two tabs:
    - **Employee Statistics**: Filter by department and date (Last 7 Days, Last 30 Days, All Time):
      - Pie chart of emotions.
      - Bar chart of department activity (sessions and average duration).
      - Table of employee details (ID, department, sessions, duration, dominant emotion).
    - **Department Overview**: Bar chart of top emotions across departments.
  - Allows downloading charts as PNGs and exporting data as CSV (`Emotions_Data.csv`).
- **Data Export**: Updates `Emotions_Data.csv` with session details (ID, department, emotion, count, percentage, date, time) on logout if requested.

### Database Storage
- Uses PostgreSQL tables (`employees`, `sessions`, `emotion_details`, `consolidated_emotion_data`) with split `session_date` (e.g., `2025-07-09`) and `time_stamp` (e.g., `20:18:45`).
- Stores employee ID, department, emotion counts, and percentages per session.
- Includes a `consolidated_emotion_data` table for easier queries.
- Relies on implied consent via GUI (explicit consent UI planned for later).

### API for Future Integration
- Offers a FastAPI endpoint (`/emotions`) to get emotion data as JSON, ready for web integration but not yet used by Tkinter.

## Directory Structure

```plaintext
Employee-Wellbeing-Emotion-Detection/
├── data/                  # FER dataset (not in repo)
│   ├── train/            # Training images by emotion
│   └── test/             # Testing images by emotion
├── saved_emotion_model/   # Model save directory
│   ├── Assets/           # Training plots (loss/accuracy)
│   └── Variables/        # Model weights (.pth)
├── train_models.py        # Training script with CUDA
├── emotion_detection_app.py # Tk Mockito frontend
├── real_time_detection.py # Backend for detection and DB storage
├── database_setup.sql     # PostgreSQL schema
├── api_server.py          # FastAPI server
├── Emotions_Data.csv      # Exported data
├── README.md              # Docs
├── requirements.txt       # Dependencies
└── .gitignore             # Ignored files
```

### Tech Stack

- **Programming Language**: Python 3.9+
- **Deep Learning**:
  - **PyTorch**: With CUDA for GPU tasks on RTX 4060.
  - **Torchvision**: Pretrained models and utilities.
- **Computer Vision**: OpenCV for video and face detection with CUDA support.
- **Frontend**:
  - **Tkinter**: GUI for login, video, and analytics.
  - **Matplotlib**: Charts for emotions and departments.
- **Database**: PostgreSQL with `psycopg` for connectivity.
- **API**: FastAPI with Uvicorn for future web use.
- **Utilities**:
  - Pillow: Image handling.
  - Numpy: Numerical operations.
  - Tqdm: Progress bars.
  - CSV: Data export.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
Ensure CUDA 12.6 and cuDNN are set up.

2. **Setup PostgreSQL**:
   
  Create database:
  ```bash
  CREATE DATABASE emotion_detection;
```
-Run database_setup.sql to create tables.
-Update DB_PARAMS in scripts with your credentials.

3.**Prepare Dataset**:
Get FER2013 from Kaggle.
Place in data/train/ and data/test/.

4.**Run Training**:
  ```bash
python train_models.py
```

5.**Run Frontend**:
```bash
python emotion_detection_app.py
```








