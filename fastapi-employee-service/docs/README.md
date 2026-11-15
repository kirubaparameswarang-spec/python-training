# FastAPI Employee Service

A simple employee management REST API.

## Run

uvicorn src.main:app --reload

## Test

pytest --cov=src --cov-report=term-missing

## API

- GET /health
- GET /employees
- GET /employees/{name}
- POST /employees
