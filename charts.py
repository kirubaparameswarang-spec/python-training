import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr"]
sales = [100, 150, 120, 180]
expenses = [80, 100, 90, 130]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(months, sales, marker="o")
axes[0, 0].set_title("Monthly Sales")

axes[0, 1].bar(months, expenses, color="orange")
axes[0, 1].set_title("Monthly Expenses")

axes[1, 0].pie(sales, labels=months, autopct="%1.1f%%")
axes[1, 0].set_title("Sales Distribution")

axes[1, 1].scatter(sales, expenses, color="green")
axes[1, 1].set_title("Sales vs Expenses")

plt.tight_layout()
plt.savefig("charts.png")
plt.show()