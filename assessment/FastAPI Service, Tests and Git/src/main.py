import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Employee Service", version="1.0.0")
DATA_FILE = Path(__file__).resolve().parent.parent / "config" / "employees.json"


class EmployeeCreate(BaseModel):
    name: str = Field(min_length=1)
    salary: float = Field(gt=0)


def load_employees() -> list[dict]:
    with DATA_FILE.open(encoding="utf-8") as file:
        return json.load(file)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/employees")
def get_employees():
    return load_employees()


@app.get("/employees/{name}")
def get_employee(name: str):
    for employee in load_employees():
        if employee["name"].casefold() == name.casefold():
            return employee
    raise HTTPException(status_code=404, detail="Employee not found")


@app.post("/employees", status_code=201)
def create_employee(employee: EmployeeCreate):
    employees = load_employees()
    if any(item["name"].casefold() == employee.name.casefold() for item in employees):
        raise HTTPException(status_code=409, detail="Employee already exists")
    new_employee = employee.model_dump()
    employees.append(new_employee)
    DATA_FILE.write_text(json.dumps(employees, indent=2), encoding="utf-8")
    return new_employee
