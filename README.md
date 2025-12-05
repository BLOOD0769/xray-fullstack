# Chest X-Ray Analyzer (Full-Stack)

A full-stack web application for **classifying chest X-ray images** (e.g., Normal vs Pneumonia) and visualizing model attention using **Grad-CAM heatmaps**.

The user uploads an X-ray image, the backend model runs inference, and the frontend displays:

- Predicted class  
- Confidence scores for each class  
- Grad-CAM heatmap overlaid on the original X-ray  

---

## ✨ Features

- 🧠 Deep-learning model for chest X-ray classification  
- 📈 Confidence scores for each class (e.g., Normal, Pneumonia)  
- 🔥 Grad-CAM heatmap to show which regions influenced the model’s decision  
- 🌐 Full-stack setup (frontend + backend)  
- 🐳 Optional Docker / Docker Compose setup for easy deployment  

---

## 🧱 Tech Stack (example)

> Update this section to match your actual stack if it’s different.

**Backend**

- Python 3.x  
- Deep learning framework (e.g., TensorFlow / PyTorch)  
- Framework: Flask / FastAPI (or similar REST API)  
- Requirements managed via `requirements.txt`

**Frontend**

- React (or other JS framework)  
- Axios / Fetch for API calls  
- Modern UI with responsive layout

**DevOps**

- Docker & Docker Compose

---

## 📂 Project Structure

```
xray-fullstack/
├── backend/           # Backend API (model loading, prediction, Grad-CAM)
├── frontend/          # Frontend web app (UI for uploading and viewing results)
├── docker/            # (Optional) additional Docker configs/scripts
├── data/              # Local data, models, or temp files (usually git-ignored)
├── docker-compose.yml # Docker Compose definition
├── requirements.txt   # Python dependencies for backend
└── README.md
```
⚙️ Prerequisites

Before running locally, install:

Git

Python 3.9+

Node.js + npm (for the frontend)

Docker & Docker Compose (only if you want the Docker option)

🚀 Quick Start (Recommended: Docker Compose)

If you have Docker and Docker Compose installed, this is the easiest way:
```
# From the project root (xray-fullstack/)
docker-compose up --build
```

This will:

Build the backend image

Build the frontend image

Start all services defined in docker-compose.yml

After the services start, open:
```
http://localhost:3000
```

(or whatever port is configured in your docker-compose.yml) in your browser to use the app.

To stop:
```
docker-compose down
```
🧪 Running Without Docker

If you prefer to run frontend and backend manually:

1️⃣ Backend

From the project root:
```
cd backend

# (Optional) create and activate a virtual environment
# python -m venv venv
# venv\Scripts\activate      # on Windows
# source venv/bin/activate   # on macOS/Linux

# Install Python dependencies
pip install -r ../requirements.txt

# Set any required environment variables (example)
# set MODEL_PATH=../data/model.pth          # Windows PowerShell/cmd
# export MODEL_PATH=../data/model.pth       # macOS/Linux

# Run the backend (update this command to match your app)
# Example for Flask:
# python app.py
# Example for FastAPI:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Check the backend folder for the actual entry file (e.g., app.py, main.py) and update the command accordingly.

The backend will typically run on something like:
```
http://localhost:8000
```
2️⃣ Frontend

Open a new terminal window:
```
cd frontend

# Install JS dependencies
npm install

# Start development server
npm start
```

By default, React apps run at:
```
http://localhost:3000
```

Make sure the frontend is configured to call the backend API URL
(e.g., http://localhost:8000) in its environment/config file.

📘 How to Use the App

Open the frontend in your browser (e.g., http://localhost:3000).

Click on the upload area and choose a chest X-ray image (PNG/JPG/JPEG).

Click “Analyze X-Ray”.

The app will display:

Predicted Class (e.g., Normal / Pneumonia)

Confidence scores (rounded / percentages)

Grad-CAM heatmap showing important regions

⚠️ Disclaimer

This project is intended for educational and research purposes only and must not be used as a substitute for professional medical diagnosis.
Always consult a qualified medical professional for clinical decisions.
