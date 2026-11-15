import csv
def calculate_bonus(salary, rate=0.1, *extras, **options):
    bonus = salary * rate + sum(extras)

    if "cap" in options:
        bonus = min(bonus, options["cap"])

    return bonus

class Employee:

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def get_details(self):
        return f"{self.name} - {self.salary}"

    @property
    def annual_salary(self):
        return self.salary * 12

def log_call(func):
    def wrapper(*args, **kwargs):
        print("Calling get_details")
        return func(*args, **kwargs)

    return wrapper

class Manager(Employee):

    def __init__(self, name, salary, team_size):
        super().__init__(name, salary)
        self.team_size = team_size

    @log_call
    def get_details(self):
        return (
            f"{self.name} - {self.salary} "
            f"- Team Size: {self.team_size}"
        )

employees = []


with open("employees.csv", "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        name = row["name"]
        department = row["department"]
        salary = int(row["salary"])

        employees.append(Employee(name, salary))

for employee in employees:

    if employee.salary < 50000:
        category = "Low"
    elif employee.salary < 70000:
        category = "Medium"
    else:
        category = "High"

    print(f"{employee.name} - {category} Salary")

employee = Employee("Arun", 30000)

manager = Manager("Divya", 75000, 5)
print(manager.get_details())
