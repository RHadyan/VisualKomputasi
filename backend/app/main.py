from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
from contextlib import asynccontextmanager

from .model import load_model, predict
from .gradcam import generate_gradcam
from .database import init_db, save_prediction, get_all_predictions, delete_prediction, get_prediction_count

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    await init_db()
    load_model()
    print("[APP] Server started successfully")
    yield
    # Shutdown
    print("[APP] Server shutting down")


app = FastAPI(
    title="Deteksi Struk Palsu API",
    description="API untuk mendeteksi keaslian struk/receipt menggunakan EfficientNetB0 + OCR",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images as static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Deteksi Struk Palsu API is running"}


@app.post("/api/predict")
async def predict_receipt(file: UploadFile = File(...)):
    """
    Upload a receipt image and get prediction result.
    
    Returns:
        - label: "REAL" or "FAKE"
        - confidence: prediction confidence (0-1)
        - visual_score: CNN model score
        - text_score: OCR validation score (if applicable)
        - hybrid_score: combined score (if applicable)
        - heatmap: base64-encoded LIME heatmap image
        - mode: "real" or "dummy"
    """
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read file content
    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum 10MB")

    # Save uploaded image
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    image_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # Run prediction
    try:
        result = predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Generate LIME explanation
    try:
        lime_result = generate_gradcam(image_bytes, result["label"])
        heatmap_base64 = lime_result["heatmap"]
        zona_stats = lime_result["zona_stats"]
        explanation = lime_result["explanation"]
    except Exception as e:
        print(f"[LIME] Failed: {e}")
        heatmap_base64 = None
        zona_stats = result.get("zona_stats")
        explanation = result.get("explanation")

    # Save to database
    prediction_id = await save_prediction(
        filename=file.filename,
        label=result["label"],
        confidence=result["confidence"],
        visual_score=result.get("visual_score"),
        text_score=result.get("text_score"),
        hybrid_score=result.get("hybrid_score"),
        heatmap_path=None,
        image_path=f"/uploads/{unique_filename}",
    )

    return {
        "id": prediction_id,
        "filename": file.filename,
        "label": result["label"],
        "confidence": result["confidence"],
        "visual_score": result.get("visual_score"),
        "text_score": result.get("text_score"),
        "hybrid_score": result.get("hybrid_score"),
        "heatmap": heatmap_base64,
        "image_url": f"/uploads/{unique_filename}",
        "mode": result.get("mode", "unknown"),
        "zona_stats": zona_stats,
        "explanation": explanation,
    }


@app.get("/api/history")
async def get_history(limit: int = 50, offset: int = 0):
    """Get prediction history."""
    predictions = await get_all_predictions(limit=limit, offset=offset)
    total = await get_prediction_count()
    return {
        "predictions": predictions,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.delete("/api/history/{prediction_id}")
async def delete_history(prediction_id: int):
    """Delete a prediction from history."""
    await delete_prediction(prediction_id)
    return {"status": "ok", "message": f"Prediction {prediction_id} deleted"}


@app.get("/api/model-info")
async def get_model_info():
    """Get information about the model."""
    from .model import _use_dummy

    return {
        "model_name": "EfficientNetB0 + EasyOCR Hybrid",
        "architecture": "EfficientNetB0 (Transfer Learning) + OCR Text Validation",
        "input_size": "224x224x3 (RGB)",
        "output": "Binary Classification (REAL / FAKE)",
        "preprocessing": "EfficientNet preprocess_input",
        "hybrid_logic": {
            "description": "CNN confidence > 0.8 → REAL, < 0.2 → FAKE, otherwise use OCR hybrid score",
            "formula": "final_score = 0.7 * visual_score + 0.3 * text_score",
        },
        "ocr_validation": {
            "description": "EasyOCR extracts text, validates receipt content",
            "checks": [
                "Presence of 'TOTAL' keyword (-0.3 if missing)",
                "At least 3 numbers found (-0.3 if fewer)",
            ],
        },
        "training": {
            "optimizer": "Adam (lr=1e-4)",
            "loss": "Binary Crossentropy",
            "epochs": "10 (with EarlyStopping, patience=3)",
            "dataset_split": "70% train / 15% val / 15% test",
        },
        "is_dummy_mode": _use_dummy,
        "note": "Model is in dummy mode. Place receipt_model.h5 in backend/models/ to enable real predictions."
        if _use_dummy
        else "Model is loaded and running real predictions.",
    }
