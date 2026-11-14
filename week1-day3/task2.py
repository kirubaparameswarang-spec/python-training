import pandas as pd
data = {
    "id": [1, 2, 3, 4, 5, 6, 7, 8],
    "name": ["Arun", "Priya", "Rahul", "Divya",
             "Karthik", "Meena", "Suresh", "Anitha"],
    "department": ["Sales", "HR", "IT", "IT",
                   "Finance", "Sales", "IT", "HR"],
    "salary": [30000, 45000, 60000, 75000,
               90000, 52000, 68000, 48000],
    "city": ["Chennai", "Chennai", "Bangalore", "Chennai",
             "Mumbai", "Bangalore", "Chennai", "Mumbai"]
}

df1 = pd.DataFrame(data)
df1.to_csv("employees.csv", index=False)

df = pd.read_csv("employees.csv")
chennai = df.loc[df["city"] == "Chennai", "name"]

high_salary = df[df["salary"] > 50000]

it = df[df["department"] == "IT"]

print("\nChennai:", ", ".join(chennai))
print("Salary above 50000:", len(high_salary), "employees")
print("IT department:", ", ".join(it["name"]))