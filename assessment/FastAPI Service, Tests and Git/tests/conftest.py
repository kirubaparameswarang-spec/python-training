import json

import pytest
from fastapi.testclient import TestClient

import src.main as main
from src.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    data_file = tmp_path / "employees.json"
    data_file.write_text(
        json.dumps([{"name": "Ravi", "salary": 45000}]), encoding="utf-8"
    )
    monkeypatch.setattr(main, "DATA_FILE", data_file)
    return TestClient(app)
