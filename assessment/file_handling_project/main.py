import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import openpyxl
import fitz

logging.basicConfig(level=logging.ERROR, format="%(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


def read_employees(path: Path) -> List[dict]:
    employees: List[dict] = []

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            row["salary"] = int(row["salary"])
            employees.append(row)

    return employees


def write_json(path: Path, data: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def write_high_earners(path: Path, employees: List[dict]) -> int:
    high_earners = [emp for emp in employees if emp["salary"] > 50000]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "name", "department", "salary"])

        writer.writeheader()
        writer.writerows(high_earners)

    return len(high_earners)


def create_salary_excel(path: Path, employees: List[dict]) -> int:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Employee Salaries"

    sheet.append(["ID", "Name", "Department", "Salary"])

    for emp in employees:
        sheet.append([emp["id"], emp["name"], emp["department"], emp["salary"]])

    workbook.save(path)

    return read_salary_excel(path)


def read_salary_excel(path: Path) -> int:
    workbook = openpyxl.load_workbook(path, data_only=True)
    sheet = workbook.active

    rows = list(sheet.iter_rows(min_row=2, values_only=True))

    workbook.close()
    return len(rows)


def extract_pdf_text(path: Path) -> str:
    document = fitz.open(path)

    try:
        if len(document) == 0:
            return ""

        page = document.load_page(0)
        return page.get_text()[:100]

    finally:
        document.close()


def read_file(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError:
        logger.error("File not found: %s", path.name)

    except json.JSONDecodeError as error:
        logger.error("Invalid JSON in %s: %s", path.name, error)

    return None


def find_employee(name: str, employees: List[dict]) -> Optional[dict]:
    for employee in employees:
        if employee["name"].lower() == name.lower():
            return employee

    return None


def format_employee_id(value: Union[int, str]) -> str:
    return str(value)


def main() -> None:
    employees_path = BASE_DIR / "employees.csv"
    json_path = BASE_DIR / "employees.json"
    high_earners_path = BASE_DIR / "high_earners.csv"
    excel_path = BASE_DIR / "salary.xlsx"
    pdf_path = BASE_DIR / "sample.pdf"

    employees = read_employees(employees_path)

    write_json(json_path, employees)

    high_count = write_high_earners(high_earners_path, employees)

    print(f"Rows read: {len(employees)} | " f"High earners written: {high_count}")

    excel_rows = create_salary_excel(excel_path, employees)

    print(f"salary.xlsx created and read back: " f"{excel_rows} rows")

    if pdf_path.exists():
        pdf_text = extract_pdf_text(pdf_path)
        print("PDF first 100 characters:", pdf_text)

    read_file(BASE_DIR / "missing.txt")


if __name__ == "__main__":
    main()
