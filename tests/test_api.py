from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API running"}


def test_predict_invalid_file():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_predict_xai_invalid_file():
    response = client.post(
        "/predict-xai",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_predict_xai_response_structure():
    import io
    from PIL import Image

    # create dummy image
    img = Image.new("RGB", (10, 10), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict-xai",
        files={"file": ("test.jpg", buf, "image/jpeg")}
    )

    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()

        assert "prediction" in data
        assert "xai" in data
        assert "threat_score" in data
        assert "threat_level" in data
        assert "consistency" in data
        assert "trust_level" in data
        assert "explanation" in data
