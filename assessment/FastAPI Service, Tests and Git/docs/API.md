# Employee Service API

Run locally with `python -m uvicorn src.main:app --reload`.

- `GET /health`: service health.
- `GET /employees`: list employees from `config/employees.json`.
- `GET /employees/{name}`: find an employee (404 when missing).
- `POST /employees`: create an employee. JSON requires a non-empty `name` and `salary` greater than zero. Returns 422 for invalid input.

Interactive API documentation: `/docs`.
