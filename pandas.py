import pandas as pd

# Create student data
data = {
    "Name": ["Aman", "Riya", "Rohit", "Neha",
             "Arjun", "Priya", "Karan", "Sneha"],

    "Python": [80, 92, 75, 68, 88, 79, 95, 72],

    "DBMS": [85, 90, 78, 70, 92, 81, 94, 76],

    "Mathematics": [88, 95, 72, 65, 90, 85, 96, 74]
}

df = pd.DataFrame(data)

print("Student Data:")
print(df)

# Calculate total
df["Total"] = (
    df["Python"] +
    df["DBMS"] +
    df["Mathematics"]
)

# Calculate percentage
df["Percentage"] = df["Total"] / 3

print("\nData with Total and Percentage:")
print(df)

# Highest percentage
print("\nHighest Percentage:")
print(df["Percentage"].max())

# Student with highest percentage
print("\nStudent with Highest Percentage:")
print(df.loc[df["Percentage"].idxmax()])

# Students having percentage > 75
print("\nStudents with Percentage > 75:")
print(df[df["Percentage"] > 75])

# Sort by percentage descending
print("\nSorted by Percentage:")
print(df.sort_values("Percentage", ascending=False))