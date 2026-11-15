class Account:
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise Exception(
                f"Insufficient funds: balance {self.balance}, requested {amount}"
            )

        self.balance -= amount
        print(f"{self.name} withdrew {amount} -> balance {self.balance}")

class SavingsAccount(Account):

    def __init__(self, name, balance):
        super().__init__(name, balance)
        self.min_balance = 500

    # Method overriding
    def withdraw(self, amount):
        if self.balance - amount < self.min_balance:
            raise Exception(
                f"Insufficient funds: balance {self.balance}, requested {amount}"
            )

        self.balance -= amount
        print(f"{self.name} withdrew {amount} -> balance {self.balance}")

priya = SavingsAccount("Priya", 1500)

try:
    priya.withdraw(1200)
except Exception as e:
    print("Error:", e)

try:
    priya.withdraw(800)
except Exception as e:
    print("Error:", e)

print(isinstance(priya, Account))