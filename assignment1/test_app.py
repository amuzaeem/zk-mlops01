import pytest
from app import app  # Importing the Flask app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_index(client):
    """Test the index page"""
    response = client.get('/')
    assert response.status_code == 200


def test_predict(client):
    """Test the prediction endpoint"""
    response = client.post('/predict', json={
        'Age': 30,
        'Total_Bilirubin': 1.2,
        'Direct_Bilirubin': 0.2,
        'Alkaline_Phosphatase': 100,
        'Alanine_Aminotransferase': 20,
        'Aspartate_Aminotransferase': 20,
        'Total_Proteins': 7.0,
        'Albumin': 4.0,
        'Albumin_and_Globulin_Ratio': 1.0
    })
    assert response.status_code == 200
    assert 'prediction' in response.json
