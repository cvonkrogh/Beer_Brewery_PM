import pandas as pd

# Load the raw sales data
df = pd.read_csv('data/raw/sales_data.csv', sep=';')
print(f"Original data shape: {df.shape}")

# Filter the data to only include the specified beer types in column 'Grondstof' (column O)
filtered_beers = ["Hoop Bleke Nelis", "Hoop US Lager", "Hoop Kaper Tropical - Summer Session IPA"]
filtered_df = df[df['Grondstof'].isin(filtered_beers)]

# Remove rows with negative litres
filtered_df = filtered_df[filtered_df['Liter'] >= 0]
print(f"Filtered data shape (after removing negatives): {filtered_df.shape}")

# Calculate total litres per factuurnummer using a loop
order_totals = filtered_df.groupby('Factuurnummer')['Liter'].sum().to_dict()
filtered_df['order_total_litres'] = 0.0
for factuurnummer, total in order_totals.items():
    filtered_df.loc[filtered_df['Factuurnummer'] == factuurnummer, 'order_total_litres'] = total
print(f"Added order_total_litres column using loop. Rows unchanged: {filtered_df.shape[0]}")

# Split into large and regular orders based on order_total_litres
large_orders = filtered_df[filtered_df['order_total_litres'] > 300].copy()
regular_orders = filtered_df[filtered_df['order_total_litres'] <= 300].copy()

# Remove the temporary order_total_litres column to keep only original columns
large_orders.drop(columns=['order_total_litres'], inplace=True)
regular_orders.drop(columns=['order_total_litres'], inplace=True)

# Save the split data to separate CSV files
large_orders.to_csv('data/processed/large_orders.csv', index=False, sep=';')
regular_orders.to_csv('data/processed/regular_orders.csv', index=False, sep=';')

print(f"Large orders saved: {large_orders.shape[0]} rows")
print(f"Regular orders saved: {regular_orders.shape[0]} rows")
print("Data splitting complete.")