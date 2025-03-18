# Employee Wellbeing through Emotion Detection (v1)

## Overview
The **Real-Time Emotion Monitoring and Analytics Dashboard** is a cutting-edge tool designed to boost employee well-being by analyzing facial expressions in real-time using deep learning. This project leverages CUDA-accelerated computation on an NVIDIA RTX 4060 GPU with CUDA 12.6, enabling rapid training and inference of emotion detection models. It overcomes the shortcomings of traditional surveys by providing dynamic, immediate insights into workforce morale. The system processes video frames from a 2-minute feed, detects emotions with high accuracy, and stores results in PostgreSQL with split `date_stamp` and `time_stamp` fields—all while ensuring ethical use through consent and privacy via data deletion. A FastAPI endpoint delivers this data for frontend dashboards, empowering organizations to foster a healthier workplace.

This `v1` release establishes a CUDA-optimized foundation for emotion monitoring, with plans for future analytics enhancements.

## Features
- **CUDA-Accelerated Model Training:**
  - Trains ResNet50 (pretrained, adapted for grayscale) and a custom CNN on the FER dataset for 25 epochs each.
  - Utilizes CUDA via PyTorch with mixed precision training (`torch.amp`) to maximize GPU performance on the NVIDIA RTX 4060, speeding up computation.
  - Produces visualizations (loss/accuracy plots) saved for analysis, with progress bars (`tqdm`) for real-time training feedback.

- **Model Testing with GUI:**
  - Validates models on test images, using CUDA for fast inference to classify emotions.
  - Displays results in a Tkinter GUI popup, showing images with predicted emotions (e.g., happy, sad).

- **Real-Time Emotion Detection:**
  - Captures a 2-minute webcam video, processing frames on the GPU with CUDA for low-latency emotion detection using ResNet50.
  - Limits detection to 2 minutes for efficiency, deleting temporary files post-processing to ensure privacy.
  - Stamps frames with date/time for tracking, leveraging CUDA’s parallel processing for quick frame analysis.

- **Database Storage:**
  - Stores results in PostgreSQL with split `date_stamp` (e.g., `2025-03-18`) and `time_stamp` (e.g., `14:30:45`).
  - Records user ID, department, and emotion (0-6) per frame, with consent enforced via GUI.

- **API for Frontend Integration:**
  - Offers a FastAPI endpoint (`/emotions`) to fetch emotion data as JSON, lightweight and ready for dashboard integration.

## Directory Structure


Employee-Wellbeing-Emotion-Detection/
├── data/                  # FER dataset (not included in repo)
│   ├── train/            # Training images by emotion
│   └── test/             # Testing images by emotion
├── saved_emotion_model/   # Model save directory (empty in repo)
│   ├── Assets/           # Training visualizations
│   └── Variables/        # Trained model weights (.pth)
├── train_models.py        # CUDA-accelerated training script
├── test_model.py          # Testing script with GUI
├── real_time_detection.py # Real-time detection with database storage
├── database_setup.sql     # PostgreSQL schema
├── api_server.py          # FastAPI server
├── README.md              # Project docs
├── requirements.txt       # Dependencies
└── .gitignore             # Excluded files




## Tech Stack
- **Programming Language:** Python 3.9+
- **Deep Learning:**
  - **PyTorch:** Core framework with CUDA support for GPU-accelerated training and inference on the NVIDIA RTX 4060.
  - **Torchvision:** Supplies pretrained models and data utilities, optimized for CUDA.
- **Computer Vision:** OpenCV (`opencv-python`) for video capture and frame processing, enhanced by CUDA where applicable.
- **GUI:** Tkinter for user interfaces (input and results display).
- **Database:** PostgreSQL for structured data storage with split timestamps.
- **API:** FastAPI with Uvicorn for a high-performance API server.
- **Visualization and Utilities:**
  - **Matplotlib:** Plots training metrics.
  - **Tqdm:** Progress bars for training.
  - **Pillow (PIL):** Image handling.
  - **Psycopg2:** PostgreSQL connectivity.
- **Hardware:** NVIDIA RTX 4060 GPU with CUDA 12.6 and cuDNN, fully leveraged for parallel computation in training and real-time tasks.

## Requirements
Install via `requirements.txt`:



## Setup
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Setup PostgreSQL:**
   - `CREATE DATABASE emotion_db;`
   - Run `database_setup.sql` in your PostgreSQL client.
3. **Update Credentials:** Replace `your_password` in `real_time_detection.py` and `api_server.py`.
4. **Prepare Dataset:** Place FER dataset in `data/train/` and `data/test/`.
5. **Run Training:** `python train_models.py` (uses CUDA for speed).
6. **Test Model:** `python test_model.py` (CUDA-accelerated inference).
7. **Real-Time Detection:** `python real_time_detection.py` (CUDA for frame processing).
8. **Start API:** `uvicorn api_server:app --reload` (access at `http://127.0.0.1:8000/emotions`).

## Dataset
- **FER Dataset:** From [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013), not included due to size.

## Hardware
- Optimized for an NVIDIA RTX 4060 with CUDA 12.6 and cuDNN on Windows, leveraging CUDA for maximum GPU utilization.

## Version
`v1` as of March 18, 2025.

![WhatsApp Image 2025-03-18 at 22 01 45_e695ac03](https://github.com/user-attachments/assets/08b97566-433b-4e48-b7ca-8990511f1ce5)

![WhatsApp Image 2025-03-18 at 22 21 02_c9792316](https://github.com/user-attachments/assets/5db8533d-ff21-4b51-901e-961ed7f7ddfd)

![WhatsApp Image 2025-03-18 at 23 22 54_f4fe2dcc](https://github.com/user-attachments/assets/e7d38e2c-7e91-4e95-b3de-be7bb3ce7796)


