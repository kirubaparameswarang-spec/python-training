
import pandas as pd
customers = pd.DataFrame({
    'customer_id': [101, 102, 103, 105],
    'name': ['Arun', 'Priya', 'Rahul', 'Divya'],
    'city': ['Chennai', 'Mumbai', 'Chennai', 'Bangalore']
})

customers.to_csv('customers.csv', index=False)
orders = pd.DataFrame({
    'order_id': [1, 2, 3, 4],
    'customer_id': [101, 102, 103, 104],
    'amount': [500, 700, 300, 900]
})
inner = pd.merge(customers, orders, on='customer_id', how='inner')

print("Inner join rows:", len(inner))
left = pd.merge(
    customers, orders,
    on='customer_id',
    how='left',
    indicator=True
)

no_orders = left[left['_merge'] == 'left_only']

print("No orders:", ', '.join(no_orders['name']))
outer = pd.merge(
    customers, orders,
    on='customer_id',
    how='outer',
    indicator=True
)

unmatched = outer[outer['_merge'] == 'right_only']

for cid in unmatched['customer_id']:
    print("Order without customer: customer_id", cid)