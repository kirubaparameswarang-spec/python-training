
import pandas as pd
import matplotlib.pyplot as plt

# Read employee data from CSV
df = pd.read_csv("employees.csv")

# --------------------------------
# 1. BAR CHART - Average Salary
# per Department
# --------------------------------

avg_salary = df.groupby("department")["salary"].mean()

plt.figure(figsize=(8, 5))

bars = plt.bar(avg_salary.index, avg_salary.values, color="skyblue")

plt.title("Average Salary per Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")

# Display values on bars
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()


# --------------------------------
# 2. HISTOGRAM - Salary Distribution
# --------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["salary"], bins=5, color="orange", edgecolor="black")

plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("histogram.png")
plt.show()


# --------------------------------
# 3. SCATTER PLOT - ID vs Salary
# Coloured by Department
# --------------------------------

plt.figure(figsize=(8, 5))

for department in df["department"].unique():
    dept_data = df[df["department"] == department]

    plt.scatter(
        dept_data["id"],
        dept_data["salary"],
        label=department
    )

plt.title("Employee ID vs Salary")
plt.xlabel("Employee ID")
plt.ylabel("Salary")
plt.legend(title="Department")

plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()


# --------------------------------
# 4. LINE CHART - Monthly Sales
# --------------------------------

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [12, 15, 14, 18, 22, 25]

plt.figure(figsize=(8, 5))

plt.plot(
    months,
    sales,
    marker="o",
    color="green",
    linestyle="-"
)

plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("line_chart.png")
plt.show()


print("All 4 charts saved successfully!")