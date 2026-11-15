class InsufficientFundsError(Exception):
    pass

class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"{self.owner} deposited {amount} -> balance {self.balance}")

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(
                f"Insufficient funds: balance {self.balance}, requested {amount}"
            )

    def show(self):
        print(self.owner, self.balance)


a = Account("Arun", 1000)

try:
    a.deposit(500)
    a.withdraw(2000)

except InsufficientFundsError as e:
    print("Error:", e)

finally:
    print("Transaction finished")