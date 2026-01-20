from fastapi import FastAPI, File, UploadFile, HTTPException
from app.image_utils import preprocess_image, extract_features
from app.genai_client import run_genai_reasoning
from app.confidence import compute_confidence
from app.schemas import AnalysisResponse

app = FastAPI(title="AI Image Analysis Service")

@app.post("/analyze-image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")

    image_bytes = await file.read()

    image = preprocess_image(image_bytes)
    features = extract_features(image)

    reasoning = run_genai_reasoning(features)
    confidence = compute_confidence(features, reasoning)

    return AnalysisResponse(
        summary=reasoning["summary"],
        detected_attributes=reasoning["attributes"],
        confidence_score=confidence
    )
