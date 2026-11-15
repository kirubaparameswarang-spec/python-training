
import os

# Create project folder
folder = "yoho_employee_service"
os.makedirs(folder, exist_ok=True)

# Create employees.csv
csv_data = """name,department,salary
Arun,Sales,30000
Priya,HR,45000
Rahul,IT,60000
Divya,IT,75000
Karthik,Finance,90000
"""

with open(os.path.join(folder, "employees.csv"), "w") as file:
    file.write(csv_data)

# Create main.py
main_code = '''import csv

print("Yoho Employee Data Service")
'''

with open(os.path.join(folder, "main.py"), "w") as file:
    file.write(main_code)

print("Project created successfully!")
print("Folder:", folder)
print("Files: employees.csv, main.py")
