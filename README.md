# 📸 InstaCap AI
Automated Instagram Caption Generator

InstaCap AI is a full-stack Deep Learning application that automatically generates creative Instagram captions for any uploaded image. It uses a CNN-LSTM architecture (ResNet50 + LSTM) trained on over 20,000 celebrity Instagram posts.

The project features a high-performance FastAPI backend and a modern, animated React frontend optimized for mobile and desktop.

--------------------------------------------------------------------------------
🚀 Tech Stack

1. Deep Learning (The Brain)
   - PyTorch: Core framework for model training and inference.
   - ResNet50: Pre-trained Convolutional Neural Network for image feature extraction.
   - LSTM: Long Short-Term Memory network for text generation.
   - CUDA: GPU acceleration for training.

2. Backend (The Engine)
   - FastAPI: Asynchronous web server for serving predictions.
   - Uvicorn: ASGI server implementation.
   - Python-Multipart: Handling image uploads.

3. Frontend (The Face)
   - React.js: Component-based UI library.
   - Tailwind CSS: Utility-first styling framework.
   - Framer Motion: Production-ready animation library.
   - Lucide React: Icon library.

--------------------------------------------------------------------------------
📂 Project Structure

instagram_captioning_project/
│
├── backend/                 # 🐍 FastAPI Server
│   ├── app.py               # API Endpoints
│   ├── model_loader.py      # Model & Vocabulary Logic
│   └── requirements.txt     # Backend Dependencies
│
├── frontend/                # ⚛️ React UI
│   ├── src/
│   │   ├── components/      # UI Components
│   │   └── App.js           # Main Application Logic
│   └── package.json         # Frontend Dependencies
│
├── src/                     # 🧠 Training Scripts
│   ├── data_pipeline.py     # Data Download & Cleaning
│   ├── train.py             # Model Training Loop (50 Epochs)
│   └── inference.py         # CLI Testing Script
│
├── data/                    # 📊 Datasets
│   ├── raw/                 # Raw Kaggle Data
│   └── processed/           # Cleaned Images & CSVs
│
└── README.txt               # Project Documentation

--------------------------------------------------------------------------------
🛠️ Installation & Setup

[Prerequisites]
1. Python 3.8+
2. Node.js & npm (for the frontend)
3. Kaggle API Token (kaggle.json placed in your user config folder)
4. NVIDIA GPU (Recommended for training)

[Step 1: Data Preparation]
Download the dataset and prepare the images.
Command:
  cd src
  python data_pipeline.py

(This downloads the dataset from Kaggle, removes noisy text like emojis/hashtags, and resizes images to 256x256.)

[Step 2: Train the Model]
Train the CNN-LSTM model (approx. 2-4 hours on GPU).
Command:
  cd src
  python train.py

(This will create 'model_epoch_50.pth' in the root folder.)

[Step 3: Backend Setup]
Start the API server to listen for requests.
Command:
  cd backend
  pip install -r requirements.txt
  python app.py

(Server runs at: http://localhost:8000)

[Step 4: Frontend Setup]
Launch the user interface.
Command:
  cd frontend
  npm install
  npm start

(App runs at: http://localhost:3000)

--------------------------------------------------------------------------------
📱 How to Use

1. Open your browser to http://localhost:3000.
2. Wait for the startup animation to finish.
3. Drag & Drop an image or tap the upload area to select a photo.
4. Click "✨ Generate Caption".
5. Watch the AI generate a caption and copy it to your clipboard!

--------------------------------------------------------------------------------
⚠️ Known Limitations

- Dataset Bias: The model is trained on celebrity Instagram data. It tends to generate "influencer-style" captions (e.g., "Love this vibe", "Best friends") rather than descriptive text.
- Language: Supports English only.

--------------------------------------------------------------------------------
🤝 Credits

- Dataset: Instagram Images with Captions by Prithvi Jaunjale (Kaggle).
- Architecture: Based on the classic "Show and Tell" Neural Image Caption Generator.