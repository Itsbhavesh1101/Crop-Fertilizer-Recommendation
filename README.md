# 🌾 Crop & Fertilizer Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-00a393)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)

A high-end, intelligent AI-powered backend system that recommends the most suitable crops to grow and the best fertilizer to use, based on environmental parameters like temperature, humidity, rainfall, and soil characteristics.

This project uses **Random Forest Classifiers** to achieve over **95% accuracy** on both tasks and serves the models via a highly performant **FastAPI** backend.

---

## 🚀 Features
- **Crop Recommendation**: Predicts the optimal crop to cultivate based on Nitrogen, Phosphorus, Potassium levels, Temperature, Humidity, pH, and Rainfall.
- **Fertilizer Recommendation**: Suggests the best fertilizer type (Urea, DAP, 14-35-14, 28-28, 17-17-17, 20-20, 10-26-26) considering Temperature, Humidity, Moisture, Soil Type, Crop Type, and N-P-K values.
- **FastAPI Backend**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.
- **Scalable Architecture**: Code is split neatly into training scripts, API routing, and separate data/model directories.

---

## 📂 Project Structure

```
├── data/
│   ├── Crop_recommendation.csv       # Dataset for crop prediction
│   └── Fertilizer Prediction.csv     # Dataset for fertilizer prediction
├── models/                           # Directory containing trained pickled models & scalers (generated after training)
├── src/
│   ├── train.py                      # Script to train and save the Random Forest models
│   └── app.py                        # FastAPI application serving the inference endpoints
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/Itsbhavesh1101/Crop-Fertilizer-Recommendation.git
   cd Crop-Fertilizer-Recommendation
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Training the Models

Before running the API, you need to train the models to generate the `.sav` and `.pkl` artifacts.
Simply run the training script:

```bash
python src/train.py
```
*Expected Output:*
```text
Training Crop Recommendation Model...
Crop Model - Train Accuracy: 1.0000, Test Accuracy: 0.9932
Crop models and scaler saved successfully.

Training Fertilizer Recommendation Model...
Fertilizer Model - Train Accuracy: 1.0000, Test Accuracy: 0.9500
Fertilizer models, scaler, and artifacts saved successfully.
```

---

## 🌐 Running the API

Once models are trained, start the FastAPI server using Uvicorn:

```bash
uvicorn src.app:app --reload
```

The server will start at `http://127.0.0.1:8000`.

### API Documentation
FastAPI automatically generates interactive API documentation. Navigate to:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## 📡 API Endpoints

### 1. Predict Crop
**Endpoint:** `POST /predict_crop`

**Request Body (JSON):**
```json
{
  "N": 90.0,
  "P": 42.0,
  "K": 43.0,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

**Response:**
```json
{
  "recommended_crop": "Rice"
}
```

### 2. Predict Fertilizer
**Endpoint:** `POST /predict_fertilizer`

**Request Body (JSON):**
```json
{
  "temperature": 26.0,
  "humidity": 52.0,
  "moisture": 38.0,
  "soil_type": "Sandy",
  "crop_type": "Maize",
  "N": 37.0,
  "K": 0.0,
  "P": 0.0
}
```

**Response:**
```json
{
  "recommended_fertilizer": "Urea"
}
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📜 License
This project is licensed under the MIT License.