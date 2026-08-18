# ============================================================
# Assignment 2 - Question 1 & 2 only
# ============================================================

import numpy as np
import pandas as pd

# ============================================================
# QUESTION 1 : NumPy
# ============================================================

# Create array from 1 to 30
arr = np.arange(1, 31)
print("Original Array:\n", arr)

# Reshape into 5 × 6 matrix
matrix = arr.reshape(5, 6)

# (a) Display the matrix
print("\n(a) 5 × 6 Matrix:")
print(matrix)

# (b) Extract the third row
print("\n(b) Third Row:")
print(matrix[2])

# (c) Extract the second column
print("\n(c) Second Column:")
print(matrix[:, 1])

# (d) Find all elements divisible by 3
print("\n(d) Elements divisible by 3:")
print(matrix[matrix % 3 == 0])

# (e) Replace all elements greater than 20 with 0
matrix[matrix > 20] = 0
print("\n(e) Matrix after replacing >20 with 0:")
print(matrix)


# ============================================================
# QUESTION 2 : Pandas DataFrame
# ============================================================

# Create DataFrame of 8 students
data = {
    "Name": ["Amit", "Priya", "Rahul", "Sneha", "Vikram", "Ananya", "Rohan", "Neha"],
    "Python": [85, 92, 78, 88, 95, 70, 82, 90],
    "DBMS": [78, 85, 90, 82, 88, 75, 80, 87],
    "Mathematics": [90, 88, 85, 92, 80, 78, 88, 95]
}

df = pd.DataFrame(data)
print("\nOriginal DataFrame:")
print(df)

# (b) Total marks
df["Total"] = df["Python"] + df["DBMS"] + df["Mathematics"]
print("\n(b) Total Marks:")
print(df)

# (c) Percentage
df["Percentage"] = (df["Total"] / 300) * 100
print("\n(c) Percentage:")
print(df)

# (d) Student with highest percentage
print("\n(d) Student with Highest Percentage:")
print(df.loc[df["Percentage"].idxmax()])

# (e) Students with percentage > 75
print("\n(e) Students with Percentage > 75:")
print(df[df["Percentage"] > 75])

# Sort by Percentage (Descending)
print("\nSorted by Percentage (Descending):")
print(df.sort_values(by="Percentage", ascending=False))