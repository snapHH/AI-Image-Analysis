import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def create_dummy_image():
    """
    Creates a simple in-memory JPEG image for testing.
    """
    import numpy as np
    import cv2

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[:] = (255, 255, 255)  # white image

    _, buffer = cv2.imencode(".jpg", image)
    return io.BytesIO(buffer.tobytes())


def test_analyze_image_success():
    image_file = create_dummy_image()

    response = client.post(
        "/analyze-image",
        files={"file": ("test.jpg", image_file, "image/jpeg")}
    )

    assert response.status_code == 200

    data = response.json()
    assert "summary" in data
    assert "detected_attributes" in data
    assert "confidence_score" in data

    assert isinstance(data["confidence_score"], float)
    assert 0.0 <= data["confidence_score"] <= 1.0


def test_analyze_image_invalid_file_type():
    response = client.post(
        "/analyze-image",
        files={"file": ("test.txt", b"not-an-image", "text/plain")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image format"


def test_analyze_image_missing_file():
    response = client.post("/analyze-image")

    assert response.status_code == 422  # validation error


def test_confidence_score_deterministic_range():
    image_file = create_dummy_image()

    response = client.post(
        "/analyze-image",
        files={"file": ("test.jpg", image_file, "image/jpeg")}
    )

    score = response.json()["confidence_score"]

    assert score >= 0.0
    assert score <= 1.0
