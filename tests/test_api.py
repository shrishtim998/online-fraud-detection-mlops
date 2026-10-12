
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_TRANSACTION = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 5000,
    "oldbalanceOrg": 10000,
    "newbalanceOrig": 5000,
    "oldbalanceDest": 2000,
    "newbalanceDest": 7000,
}


def test_valid_transaction():
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 200

    data = response.json()
    assert data["is_fraud"] in [0, 1]
    assert 0 <= data["fraud_probability"] <= 1
    assert 0 <= data["threshold"] <= 1
    assert data["label"] in ["Fraud", "Legitimate"]


def test_missing_required_field():
    transaction = VALID_TRANSACTION.copy()
    transaction.pop("amount")

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


def test_wrong_data_type():
    transaction = VALID_TRANSACTION.copy()
    transaction["amount"] = "five thousand"

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


def test_negative_amount():
    transaction = VALID_TRANSACTION.copy()
    transaction["amount"] = -500

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


def test_unsupported_transaction_type():
    transaction = VALID_TRANSACTION.copy()
    transaction["type"] = "UNKNOWN"

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True
