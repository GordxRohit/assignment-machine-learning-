import pandas as pd

# Read CSV file
df = pd.read_csv("sales.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Calculate Sales
df["Sales"] = df["Quantity"] * df["Price"]

print("\nData with Sales:")
print(df)

# Total sales
print("\nTotal Sales:")
print(df["Sales"].sum())

# Product with highest sales
print("\nProduct with Highest Sales:")
print(df.loc[df["Sales"].idxmax()])

# Category-wise total sales
print("\nCategory-wise Total Sales:")
print(df.groupby("Category")["Sales"].sum())

# Sort by sales descending
print("\nSales in Descending Order:")
print(df.sort_values("Sales", ascending=False))