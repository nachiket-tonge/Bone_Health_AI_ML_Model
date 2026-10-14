# Bone Health AI — ML Service

An AI-powered medical imaging and bone health assessment service that provides **fracture detection, bone age assessment, and bone mineral density (BMD) prediction** through machine learning models and REST APIs.

## Features

- **Fracture Detection:** Analyze bone X-ray images to predict potential fractures.
- **Bone Age Assessment:** Estimate bone age from relevant medical images.
- **Bone Mineral Density (BMD) Prediction:** Predict BMD-related results using patient information.
- **X-ray Validation:** Help identify whether an uploaded image is suitable for bone X-ray analysis.
- **Grad-CAM Visualization:** Generate visual explanations highlighting image regions that influence model predictions.
- **REST API:** Expose ML inference functionality through FastAPI.

## Technology Stack

- Python
- FastAPI
- PyTorch
- Torchvision
- timm
- NumPy
- Pandas
- scikit-learn
- OpenCV
- Pillow
- Joblib
- Uvicorn

## Project Structure

```text
Bone_Health_AI_ML_Model/
├── main.py
├── requirements.txt
├── routers/
│   ├── bmd.py
│   ├── bone_age.py
│   └── fracture.py
├── services/
│   ├── bmd_service.py
│   ├── bone_age_service.py
│   ├── fracture_service.py
│   └── xray_service.py
├── utils/
│   └── gradcam.py
├── models/                 # Download separately
├── dataset/                # Local datasets; not included
└── README.md
```

## Prerequisites

- Python 3.10 or a compatible Python version for the installed dependencies.
- pip
- Git
- Trained model files downloaded separately.
- Sufficient memory and storage for the selected ML models.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nachiket-tonge/Bone_Health_AI_ML_Model.git
cd Bone_Health_AI_ML_Model
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** PyTorch installation requirements can vary depending on your operating system and whether you use a CPU or a compatible GPU.

## Download Trained Models

The trained model files are hosted separately on Google Drive to keep large model artifacts out of the Git repository.

**[Download Bone Health AI Models from Google Drive](https://drive.google.com/drive/folders/1sqbvalWl_E7NYhHGjF9L8s-44u5AXBd8)**

### Setup

1. Open the Google Drive folder above.
2. Download the model files required by the prediction services.
3. Create the `models/` directory in the repository if it does not already exist.
4. Place the downloaded files in the appropriate locations expected by the service code.
5. Verify that the model filenames and paths match those referenced in the relevant files under `services/`.

> **Important:** The exact filenames and directory structure depend on the trained models and the paths configured in the source code. Ensure these match before starting inference.

## Run the ML API

Start the FastAPI application from the repository root:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The API should be available at:

```text
http://127.0.0.1:8000
```

FastAPI interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

The available routes depend on the routers registered in `main.py`.

## Integration

The ML service is designed to provide prediction functionality to the Bone Health AI application.

Typical integration flow:

1. A user submits an image or patient information through the application.
2. The application sends a request to the appropriate API endpoint.
3. The ML service validates the input and loads the required model.
4. The service performs inference.
5. The prediction result is returned to the calling application.

The frontend and backend should be configured to use the correct ML service URL for the local or deployed environment.

## Development and Testing

Check the available Python scripts before running them. The repository includes scripts for model inspection and testing, including:

- `check_model.py`
- `check_bone_age.py`
- `inspect_bmd.py`
- `inspect_bone_age.py`
- `test_bmd.py`
- `test_xray_detector.py`

Run a script from the repository root using the active virtual environment. Some scripts may require trained models or local test data.

## Important Notes

- Model files and datasets are downloaded or configured separately and are not included in the Git repository.
- Never commit private datasets, patient-identifiable images, credentials, or local environment files.
- Use compatible model files and preprocessing steps for each prediction service.
- Predictions are intended for research and decision-support development, not as a substitute for diagnosis by a qualified medical professional.

## License

Add the applicable license before redistributing this project or its trained model files.
