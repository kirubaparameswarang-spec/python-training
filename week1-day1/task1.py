integer_value = 100
float_value = 45.75
string_value = "Python Programming"
list_value = [10, 20, 30]
dict_value = {"name": "Arun", "age": 21}
set_value = {10, 20, 30}
tuple_value = (100, 200, 300)


def calculate_sales(sales, target):

    total = 0

    for value in sales:

        if value <= 0:
            print(f"Skipped invalid value {value}")
            continue

        total += value
        print(f"Added {value} total {total}")

        if total > target:
            break

    if total > target:
        print(f"Target reached. Stopped at {total}")

    elif total == target:
        print("Target reached")

    else:
        print("Target not reached")

    return total



sales = [12000, 0, 8500, 15000, -200, 9800]

calculate_sales(sales, 30000)
