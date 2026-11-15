prices = [500, 1200, 800, 2500, 300]
def apply_discount(price, discount=10):
    return price - (price * discount/100)
def total(*args):
    return sum(args)
def describe(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}:")
above_1000 = [p for p in prices if p > 1000]
with_gst = list(map(lambda p: round(p * 1.18, 2),prices))
below_1000= list(filter(lambda p: p < 1000,prices))

print(f"apply_discount(1000)-> {apply_discount(1000)}")
print(f"apply_discount(1000, 25)-> {apply_discount(1000, 25)}")
print(f"total(5, 10, 15)-> {total(5, 10, 15)}")
print(f"above 1000: {above_1000}")
print(f"with gst: {with_gst}")
print(f"below 1000: {below_1000}")