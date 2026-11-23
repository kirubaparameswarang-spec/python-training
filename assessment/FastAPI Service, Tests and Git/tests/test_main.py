def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_employees(client):
    response = client.get("/employees")
    assert response.status_code == 200
    assert response.json() == [{"name": "Ravi", "salary": 45000}]


def test_get_employee_success(client):
    response = client.get("/employees/Ravi")
    assert response.status_code == 200
    assert response.json()["name"] == "Ravi"


def test_get_employee_not_found(client):
    response = client.get("/employees/Unknown")
    assert response.status_code == 404


def test_post_employee_success(client):
    response = client.post("/employees", json={"name": "Anu", "salary": 30000})
    assert response.status_code == 201
    assert response.json() == {"name": "Anu", "salary": 30000}


def test_invalid_salary(client):
    response = client.post("/employees", json={"name": "Anu", "salary": -5})
    assert response.status_code == 422


def test_empty_name(client):
    response = client.post("/employees", json={"name": "", "salary": 30000})
    assert response.status_code == 422
