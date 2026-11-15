import json

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_employees(tmp_path, monkeypatch):
    data_file = tmp_path / "employees.json"
    data_file.write_text(
        json.dumps(
            [
                {"name": "Ravi", "salary": 50000},
                {"name": "Priya", "salary": 60000},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.main.DATA_FILE", data_file)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_employees():
    response = client.get("/employees")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_employee_success():
    response = client.get("/employees/Ravi")
    assert response.status_code == 200
    assert response.json()["name"] == "Ravi"


def test_get_employee_not_found():
    response = client.get("/employees/Unknown")
    assert response.status_code == 404


def test_invalid_salary():
    response = client.post("/employees", json={"name": "Ravi", "salary": -5})
    assert response.status_code == 422


def test_empty_name():
    response = client.post("/employees", json={"name": "", "salary": 50000})
    assert response.status_code == 422


def test_create_employee():
    response = client.post("/employees", json={"name": "Kumar", "salary": 55000})
    assert response.status_code == 201
    assert response.json()["name"] == "Kumar"
