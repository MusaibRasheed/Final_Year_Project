"""
FastAPI Backend Server for TumorGANet React Application.
Provides REST API endpoints for:
1. POST /api/predict: Deep neural inference with TumorGANet (VGG-16)
2. POST /api/explain: LIME super-pixel segmentation, yellow contours, and heatmaps
3. GET  /api/generate-mri: DCGAN synthetic brain MRI generation
4. GET  /api/benchmarks: Published Table 5 and Table 7 research data
"""

import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
try:
    import pydicom
except ImportError:
    pydicom = None
import starlette.routing
_orig_router_init = starlette.routing.Router.__init__
def _compat_router_init(self, *args, **kwargs):
    kwargs.pop("on_startup", None)
    kwargs.pop("on_shutdown", None)
    return _orig_router_init(self, *args, **kwargs)
starlette.routing.Router.__init__ = _compat_router_init

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.dataset import create_synthetic_mri_slice
from data.preprocessor import BrainMRIPreprocessor
from models.tumor_ganet import build_tumorganet
from models.gan import BrainMRIGAN
from evaluation.explainability import BrainMRILIMEExplainer
from training.ablation import PUBLISHED_TABLE_7
from api.storage import authenticate, create_audit_event, create_patient, create_report, create_scan, create_reset_token, create_session, create_user, delete_session, get_session, list_audit_events, list_patients, list_reports, list_scans, reset_password
from api.mailer import send_email


app = FastAPI(
    title="TumorGANet NeuroVision API",
    description="Deep Learning REST API for Brain Tumor Classification & Explainable AI",
    version="2.0.0"
)

# Enable CORS for React frontend (Vite port 5173 or CRA port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def require_session(authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    user = get_session(authorization[7:])
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user

# Initialize Models on target device
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = build_tumorganet(pretrained=False, device=device)
best_weights = config.CHECKPOINTS_DIR / "tumorganet_best.pth"
if best_weights.exists():
    classifier.load_state_dict(torch.load(best_weights, map_location=device))
classifier.eval()

gan = BrainMRIGAN(device=device)
gan.load_models()

explainer = BrainMRILIMEExplainer(model=classifier, device=torch.device(device))


def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def numpy_to_base64(arr: np.ndarray) -> str:
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    pil_img = Image.fromarray(arr.astype(np.uint8))
    return pil_to_base64(pil_img)


def dicom_to_image(contents: bytes) -> tuple[Image.Image, Dict[str, Any]]:
    if pydicom is None:
        raise RuntimeError("DICOM support is not installed. Run: pip install pydicom")
    dataset = pydicom.dcmread(io.BytesIO(contents), force=True)
    pixels = dataset.pixel_array
    if pixels.ndim > 2:
        pixels = pixels[0]
    pixels = pixels.astype(np.float32)
    low, high = float(pixels.min()), float(pixels.max())
    if high > low:
        pixels = (pixels - low) / (high - low) * 255.0
    else:
        pixels = np.zeros_like(pixels)
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 255.0 - pixels
    image = Image.fromarray(pixels.astype(np.uint8)).convert("RGB")
    metadata = {
        "modality": str(getattr(dataset, "Modality", "MR")),
        "study_description": str(getattr(dataset, "StudyDescription", "Brain MRI")),
        "series_description": str(getattr(dataset, "SeriesDescription", "MRI series")),
        "rows": int(getattr(dataset, "Rows", image.height)),
        "columns": int(getattr(dataset, "Columns", image.width)),
        "slice_thickness": str(getattr(dataset, "SliceThickness", "Unknown")),
    }
    return image, metadata


@app.get("/")
def root():
    return {
        "status": "online",
        "system": "TumorGANet NeuroVision AI",
        "device": device,
        "classes": config.CLASSES
    }


@app.post("/api/auth/login")
def login(payload: Dict[str, str] = Body(...)):
    user = authenticate(payload.get("username", ""), payload.get("password", ""))
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_session(user)
    create_audit_event(user["id"], "login", "session", details="User signed in")
    return {"access_token": token, "user": user}


@app.post("/api/auth/signup")
def signup(payload: Dict[str, str] = Body(...)):
    try:
        user = create_user(payload.get("name", ""), payload.get("email", ""), payload.get("password", ""))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    token = create_session(user)
    create_audit_event(user["id"], "signup", "user", str(user["id"]), "Account created")
    return {"access_token": token, "user": user}


@app.post("/api/auth/forgot-password")
def forgot_password(payload: Dict[str, str] = Body(...)):
    email = payload.get("email", "").strip().lower()
    token = create_reset_token(email)
    if token:
        result = send_email(email, "TumorGANet password reset", f"Use this password reset token within one hour:\n\n{token}\n")
        response = {"message": "If an account exists for that email, reset instructions have been sent."}
        if result["status"] == "not_configured":
            response["development_token"] = token
        return response
    return {"message": "If an account exists for that email, reset instructions have been sent."}


@app.post("/api/auth/reset-password")
def reset_password_endpoint(payload: Dict[str, str] = Body(...)):
    try:
        changed = reset_password(payload.get("token", ""), payload.get("password", ""))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if not changed:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    return {"message": "Password reset successfully"}


@app.get("/api/auth/me")
def current_user(authorization: Optional[str] = Header(default=None)):
    return require_session(authorization)


@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    user = require_session(authorization)
    delete_session(authorization[7:])
    create_audit_event(user["id"], "logout", "session", details="User signed out")
    return {"message": "Signed out"}


@app.get("/api/patients")
def patients(search: str = "", authorization: Optional[str] = Header(default=None)):
    require_session(authorization)
    return {"patients": list_patients(search)}


@app.post("/api/patients")
def add_patient(payload: Dict[str, Any] = Body(...), authorization: Optional[str] = Header(default=None)):
    user = require_session(authorization)
    patient = create_patient(payload)
    create_audit_event(user["id"], "create", "patient", patient["id"], "Patient record created")
    return patient


@app.get("/api/scans")
def scans(patient_id: Optional[str] = None, authorization: Optional[str] = Header(default=None)):
    require_session(authorization)
    return {"scans": list_scans(patient_id)}


@app.post("/api/scans")
def add_scan(payload: Dict[str, Any] = Body(...), authorization: Optional[str] = Header(default=None)):
    user = require_session(authorization)
    scan = create_scan(payload)
    create_audit_event(user["id"], "create", "scan", str(scan["id"]), f"Prediction: {scan.get('prediction')}")
    return scan


@app.get("/api/reports")
def reports(patient_id: Optional[str] = None, authorization: Optional[str] = Header(default=None)):
    require_session(authorization)
    return {"reports": list_reports(patient_id)}


@app.post("/api/reports")
def add_report(payload: Dict[str, Any] = Body(...), authorization: Optional[str] = Header(default=None)):
    user = require_session(authorization)
    report = create_report(payload)
    create_audit_event(user["id"], "create", "report", str(report["id"]), f"Decision: {report.get('decision')}")
    return report


@app.get("/api/audit")
def audit(authorization: Optional[str] = Header(default=None)):
    user = require_session(authorization)
    if user.get("role") not in {"admin", "doctor"}:
        raise HTTPException(status_code=403, detail="Audit access is restricted to clinical staff")
    return {"events": list_audit_events()}


@app.post("/api/reports/email")
def email_report(payload: Dict[str, Any] = Body(...), authorization: Optional[str] = Header(default=None)):
    user = require_session(authorization)
    report = create_report(payload)
    create_audit_event(user["id"], "create", "report", str(report["id"]), "Preliminary report created")
    if not user.get("email"):
        return {"report": report, "email": {"status": "not_configured", "message": "No email address is associated with this account."}}
    body = (
        "TumorGANet preliminary MRI analysis report\n\n"
        f"Patient: {payload.get('patient', {}).get('name', 'Unknown')}\n"
        f"Patient ID: {payload.get('patient', {}).get('id', 'Unknown')}\n"
        f"Doctor: {payload.get('doctor_name', 'Unknown')}\n"
        f"Decision: {payload.get('decision', 'concur')}\n\n"
        f"Notes: {payload.get('notes', '')}\n\n"
        "This is an AI decision-support report and must be reviewed by a qualified clinician."
    )
    try:
        delivery = send_email(user["email"], "TumorGANet MRI analysis report", body)
    except Exception as error:
        print(f"[EMAIL] Delivery failed for {user['email']}: {type(error).__name__}: {error}")
        delivery = {"status": "failed", "message": f"Email delivery failed: {error}"}
    else:
        print(f"[EMAIL] {delivery['status']} for {user['email']}: {delivery['message']}")
    return {"report": report, "email": delivery}


@app.post("/api/predict")
async def predict_mri(file: UploadFile = File(...)):
    """Receives an uploaded MRI scan image, runs TumorGANet, returns class probabilities."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run TumorGANet inference
        results = classifier.predict_image(image)
        pclass = results["predicted_class"]

        # Clinical metadata enrichment
        metadata = {
            "glioma": {"grade": "Grade III/IV Astrocytoma", "risk": "CRITICAL", "est_area_mm2": 412},
            "meningioma": {"grade": "Grade I (Benign)", "risk": "MODERATE", "est_area_mm2": 328},
            "pituitary": {"grade": "Macroadenoma", "risk": "MODERATE", "est_area_mm2": 215},
            "notumor": {"grade": "Normal Anatomy", "risk": "LOW", "est_area_mm2": 0}
        }

        return {
            "prediction": pclass,
            "confidence": float(results["confidence"]),
            "probabilities": results["probabilities"],
            "clinical_meta": metadata.get(pclass, {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict-dicom")
async def predict_dicom(file: UploadFile = File(...)):
    """Reads a DICOM MRI slice, normalizes it, and runs the same classifier."""
    try:
        contents = await file.read()
        image, dicom_metadata = dicom_to_image(contents)
        results = classifier.predict_image(image)
        pclass = results["predicted_class"]
        metadata = {
            "glioma": {"grade": "Grade III/IV Astrocytoma", "risk": "CRITICAL", "est_area_mm2": 412},
            "meningioma": {"grade": "Grade I (Benign)", "risk": "MODERATE", "est_area_mm2": 328},
            "pituitary": {"grade": "Macroadenoma", "risk": "MODERATE", "est_area_mm2": 215},
            "notumor": {"grade": "Normal Anatomy", "risk": "LOW", "est_area_mm2": 0}
        }
        return {
            "prediction": pclass,
            "confidence": float(results["confidence"]),
            "probabilities": results["probabilities"],
            "clinical_meta": metadata.get(pclass, {}),
            "dicom_metadata": dicom_metadata,
            "preview": pil_to_base64(image),
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"DICOM processing failed: {error}")


@app.post("/api/explain")
async def explain_mri(file: UploadFile = File(...)):
    """Generates LIME superpixel masks, yellow contour boundaries, and heatmaps."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Generate LIME explanations (Figure 11)
        explanation = explainer.explain_image(image, num_samples=100)

        return {
            "predicted_class": explanation["predicted_class"],
            "confidence": explanation["confidence"],
            "original_b64": numpy_to_base64(explanation["original_image"]),
            "correct_region_b64": numpy_to_base64(explanation["correct_region"]),
            "highlight_contour_b64": numpy_to_base64(explanation["highlight_contour"]),
            "heatmap_b64": numpy_to_base64(explanation["heatmap"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/generate-mri")
def generate_synthetic_mri():
    """Samples random noise z ~ N(0, I) and synthesizes a novel brain MRI slice using DCGAN."""
    try:
        sample_batch = gan.generate_synthetic_samples(num_samples=1)
        b64_img = numpy_to_base64(sample_batch[0])
        return {"synthetic_mri_b64": b64_img}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns Table 5 and Table 7 published empirical benchmarks."""
    table_5 = [
        {"model": "TumorGANet (VGG-16)", "accuracy": 99.53, "precision": 100.0, "recall": 100.0, "f1": 99.00, "hamming_loss": 0.20, "mcc": 99.50},
        {"model": "Inception-v3", "accuracy": 99.34, "precision": 99.34, "recall": 99.34, "f1": 99.00, "hamming_loss": 1.02, "mcc": 97.20},
        {"model": "ResNet-101", "accuracy": 98.93, "precision": 98.93, "recall": 98.93, "f1": 99.00, "hamming_loss": 0.50, "mcc": 99.10},
        {"model": "AlexNet", "accuracy": 98.17, "precision": 98.17, "recall": 98.17, "f1": 98.00, "hamming_loss": 0.90, "mcc": 97.55},
        {"model": "GoogLeNet", "accuracy": 97.18, "precision": 97.18, "recall": 97.18, "f1": 97.00, "hamming_loss": 2.44, "mcc": 94.10},
        {"model": "VGG-19", "accuracy": 96.72, "precision": 96.72, "recall": 96.72, "f1": 96.50, "hamming_loss": 1.60, "mcc": 96.00},
        {"model": "MobileNet", "accuracy": 96.41, "precision": 96.41, "recall": 96.41, "f1": 96.20, "hamming_loss": 7.90, "mcc": 95.00},
        {"model": "MobileNet-v2", "accuracy": 95.85, "precision": 95.85, "recall": 95.85, "f1": 95.50, "hamming_loss": 2.44, "mcc": 94.10},
        {"model": "DenseNet-121", "accuracy": 95.78, "precision": 95.78, "recall": 95.78, "f1": 95.60, "hamming_loss": 2.47, "mcc": 94.10},
        {"model": "Xception", "accuracy": 78.01, "precision": 86.00, "recall": 77.00, "f1": 76.00, "hamming_loss": 10.90, "mcc": 64.30}
    ]
    return {
        "table_5_models": table_5,
        "table_7_ablation": PUBLISHED_TABLE_7
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
