# Week 8 - Serving the Model as an API (EuroSAT Land-Cover Classification)

**Applied AI Lab | FastAPI | PyTorch | ResNet18 | Sentinel-2**

## Project Overview

This project deploys the Week 6 EuroSAT Land-Cover Classification model as a REST API using FastAPI.

The trained ResNet18 model takes RGB satellite images as input and predicts one of 10 land-cover categories from the EuroSAT dataset.

The API makes it possible to access the trained deep learning model through HTTP requests instead of running predictions directly inside a Jupyter Notebook.

## Model Information

| Property | Description |
|---|---|
| Model | ResNet18 |
| Framework | PyTorch |
| Dataset | EuroSAT |
| Satellite Source | Sentinel-2 |
| Input | RGB Satellite Image |
| Image Resolution | 64 × 64 |
| Number of Classes | 10 |
| Checkpoint | `week6_model.pt` |
| Task | Multi-class Land-Cover Classification |

## Land-Cover Classes

The API supports the following 10 classes:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

## API Features

- FastAPI REST API.
- PyTorch ResNet18 model inference.
- Satellite image upload.
- 10-class land-cover prediction.
- Class probability output.
- Prediction latency measurement.
- Automatic Swagger documentation.
- Model loading during application startup.
- API testing with Pytest.
- Performance benchmarking.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| FastAPI | REST API framework |
| PyTorch | Deep learning inference |
| Torchvision | Image preprocessing |
| ResNet18 | Classification architecture |
| Pillow | Image processing |
| NumPy | Numerical operations |
| Uvicorn | API server |
| Pytest | Automated testing |
| HTTPX | API testing |

## Project Structure

```text
p2-api/
│
├── app.py
├── bench.py
├── test_api.py
├── requirements.txt
├── README.md
│
├── week6_model.pt
├── sample.jpeg
│
└── .gitignore
```

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | API root information |
| GET | `/health` | Check API and model status |
| POST | `/predict` | Upload satellite image and predict land-cover class |

*The endpoint paths should match the actual routes implemented in `app.py`.*

### 1. Health Check

```http
GET /health
```

Checks whether the API is running and whether the trained model has loaded successfully.

### 2. Image Prediction

```http
POST /predict
```

Accepts an RGB satellite image and returns the predicted land-cover category, class probabilities, and inference latency.

### Example Prediction Response

```json
{
  "label": "Forest",
  "probabilities": {
    "AnnualCrop": 0.001,
    "Forest": 0.982,
    "HerbaceousVegetation": 0.004,
    "Highway": 0.001,
    "Industrial": 0.001,
    "Pasture": 0.002,
    "PermanentCrop": 0.002,
    "Residential": 0.001,
    "River": 0.003,
    "SeaLake": 0.003
  },
  "latency_ms": 45.2
}
```

*Illustrative response only. Actual predictions and latency depend on the uploaded image and hardware.*

## Installation and Setup

### 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd p2-api
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Environment

Windows CMD:

```cmd
.venv\Scripts\activate.bat
```

### 4. Install PyTorch CPU Version

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 5. Install Dependencies

```bash
pip install fastapi "uvicorn[standard]" python-multipart pillow numpy pytest httpx
```

Or install everything from the requirements file:

```bash
pip install -r requirements.txt
```

## Run the API

Start the FastAPI server:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Expected startup message:

```text
Model loaded successfully!
Classes: [
  'AnnualCrop',
  'Forest',
  'HerbaceousVegetation',
  'Highway',
  'Industrial',
  'Pasture',
  'PermanentCrop',
  'Residential',
  'River',
  'SeaLake'
]
Application startup complete.
```

## Access API Documentation

Once the server is running:

- API Root: http://127.0.0.1:8000/
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

Swagger UI allows users to upload satellite images and test predictions interactively.

## API Testing

Run automated tests:

```bash
python -m pytest -v
```

The test suite can validate:

- API availability.
- Model loading.
- Image prediction.
- Response structure.
- Class probabilities.
- Invalid image handling.
- Missing input handling.

## Performance Benchmarking

Run the benchmark script while the API server is running:

```bash
python bench.py
```

The benchmark measures prediction request performance and API response latency.

Recommended metrics to report:

| Metric | Description |
|---|---|
| p50 | Median response latency |
| p95 | 95th percentile latency |
| Maximum | Highest recorded latency |
| Throughput | Requests processed per second |

Actual benchmark values should be added after running `bench.py` on the target hardware.

## Model Deployment Workflow

1. Train the ResNet18 model using the EuroSAT dataset.
2. Save the trained model checkpoint.
3. Load the checkpoint when FastAPI starts.
4. Receive an RGB satellite image through the prediction endpoint.
5. Apply the same preprocessing used during model training.
6. Run PyTorch inference.
7. Calculate class probabilities.
8. Return the predicted land-cover category as JSON.

## Applications

- Agricultural land monitoring.
- Forest classification.
- Urban development monitoring.
- Water-body identification.
- Environmental analysis.
- Satellite image classification.

## Limitations

- Predictions depend on image quality and preprocessing.
- The model is trained on EuroSAT data and may not generalize to every satellite sensor.
- CPU inference performance depends on hardware.
- Predictions should be validated before being used in critical land-management decisions.

## Conclusion

This project demonstrates the deployment of a trained ResNet18 land-cover classification model through a FastAPI REST API.

It connects the Week 6 deep learning model with practical API development, allowing satellite images to be classified through HTTP requests.

The project demonstrates machine learning model serving, image preprocessing, inference, API testing, and performance benchmarking.

## Author

**Atteq ur Rehman**

Applied AI | Machine Learning | Computer Vision

## Dataset Reference

EuroSAT Dataset: https://github.com/phelber/EuroSAT

Kaggle Dataset: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
