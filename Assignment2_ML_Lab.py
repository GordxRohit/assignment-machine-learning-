# ============================================================
# Birla Institute of Technology, Mesra, Ranchi
# Department of Computer Science and Engineering
# MTECH I / Assignment 2
# Course: AI26510 Machine Learning Lab
# Date: 17-08-2026
# Working on NumPy and Pandas
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns

print("=" * 70)
print("ASSIGNMENT 2 - Machine Learning Lab (NumPy + Pandas)")
print("=" * 70)

# ============================================================
# QUESTION 1 : NumPy Array Operations
# ============================================================
print("\n" + "=" * 70)
print("QUESTION 1 : NumPy Array (1 to 30) → 5×6 Matrix")
print("=" * 70)

# Create NumPy array containing numbers from 1 to 30
arr = np.arange(1, 31)
print("\nOriginal 1-D Array (1 to 30):\n", arr)

# Reshape into 5 × 6 matrix
matrix = arr.reshape(5, 6)

# (a) Display the matrix
print("\n(a) 5 × 6 Matrix:")
print(matrix)

# (b) Extract the third row (index 2)
print("\n(b) Third Row:")
print(matrix[2])

# (c) Extract the second column (index 1)
print("\n(c) Second Column:")
print(matrix[:, 1])

# (d) Find all elements divisible by 3
print("\n(d) Elements divisible by 3:")
print(matrix[matrix % 3 == 0])

# (e) Replace all elements greater than 20 with 0
matrix_modified = matrix.copy()          # keep original safe
matrix_modified[matrix_modified > 20] = 0
print("\n(e) Matrix after replacing elements > 20 with 0:")
print(matrix_modified)

# ============================================================
# QUESTION 2 : Pandas DataFrame - Student Marks
# ============================================================
print("\n" + "=" * 70)
print("QUESTION 2 : Student Marks DataFrame (8 Students)")
print("=" * 70)

# Create DataFrame with 8 students and 3 subjects
data = {
    "Name": ["Amit", "Priya", "Rahul", "Sneha", "Vikram", "Ananya", "Rohan", "Neha"],
    "Python": [85, 92, 78, 88, 95, 70, 82, 90],
    "DBMS": [78, 85, 90, 82, 88, 75, 80, 87],
    "Mathematics": [90, 88, 85, 92, 80, 78, 88, 95]
}

df = pd.DataFrame(data)
print("\nOriginal DataFrame:")
print(df)

# (b) Calculate the total marks of each student
df["Total"] = df["Python"] + df["DBMS"] + df["Mathematics"]
print("\n(b) DataFrame with Total Marks:")
print(df)

# (c) Calculate the percentage (out of 300)
df["Percentage"] = (df["Total"] / 300) * 100
print("\n(c) DataFrame with Percentage:")
print(df)

# (d) Find the student with the highest percentage
highest = df.loc[df["Percentage"].idxmax()]
print("\n(d) Student with Highest Percentage:")
print(highest)

# (e) Display students having a percentage greater than 75
print("\n(e) Students with Percentage > 75:")
print(df[df["Percentage"] > 75])

# (d - repeated in question) Sort the DataFrame according to percentage in descending order
df_sorted = df.sort_values(by="Percentage", ascending=False)
print("\n(d) DataFrame sorted by Percentage (Descending):")
print(df_sorted)

# ============================================================
# PRACTICE QUESTION 1 : sales.csv Operations
# ============================================================
print("\n" + "=" * 70)
print("PRACTICE QUESTION 1 : sales.csv Analysis")
print("=" * 70)

# (a) Read the CSV file
sales = pd.read_csv("sales.csv")
print("\n(a) CSV file read successfully.")

# (b) Display the first five records
print("\n(b) First five records:")
print(sales.head())

# (c) Check for missing values
print("\n(c) Missing values in each column:")
print(sales.isnull().sum())

# (d) Create a new column Sales using Quantity × Price
sales["Sales"] = sales["Quantity"] * sales["Price"]
print("\n(d) DataFrame after adding 'Sales' column:")
print(sales)

# (e) Find the total sales
total_sales = sales["Sales"].sum()
print("\n(e) Total Sales:", total_sales)

# (f) Find the product with the highest sales
highest_sales_product = sales.loc[sales["Sales"].idxmax()]
print("\n(f) Product with Highest Sales:")
print(highest_sales_product[["Products", "Sales"]])

# (g) Find total sales for each category
category_sales = sales.groupby("Category")["Sales"].sum()
print("\n(g) Total Sales by Category:")
print(category_sales)

# (h) Sort the records according to sales in descending order
sales_sorted = sales.sort_values(by="Sales", ascending=False)
print("\n(h) Records sorted by Sales (Descending):")
print(sales_sorted)

# ============================================================
# PRACTICE QUESTION 2 : Seaborn Tips Dataset
# ============================================================
print("\n" + "=" * 70)
print("PRACTICE QUESTION 2 : Seaborn Tips Dataset")
print("=" * 70)

# Load the Tips dataset
tips = sns.load_dataset("tips")
print("\nTips dataset loaded successfully.")

# (a) Display first 50 observations
print("\n(a) First 50 observations:")
print(tips.head(50))

# (b) Check the type of the data structure
print("\n(b) Type of data structure:", type(tips))

# (c) Check the types of the features
print("\n(c) Data types of features:")
print(tips.dtypes)

# (d) Describe the dataset (five number summary + count, mean, std)
print("\n(d) Statistical Description of the dataset:")
print(tips.describe())

# (e) Display the detailed observations during Sunday
print("\n(e) Observations on Sunday:")
print(tips[tips["day"] == "Sun"])

# (f) Create another data frame sun_tips for Sunday only
sun_tips = tips[tips["day"] == "Sun"].copy()
print("\n(f) sun_tips DataFrame created (Sunday observations only).")
print("Shape of sun_tips:", sun_tips.shape)

# (g) Create a new feature ‘bill_per_person’
sun_tips["bill_per_person"] = sun_tips["total_bill"] / sun_tips["size"]
print("\n(g) Updated sun_tips with 'bill_per_person':")
print(sun_tips.head(10))

# (h) Calculate the tips percentage by every customer on Sunday
sun_tips["tip_percentage"] = (sun_tips["tip"] / sun_tips["total_bill"]) * 100
print("\n(h) Tip percentage for every customer on Sunday:")
print(sun_tips[["total_bill", "tip", "tip_percentage"]].head(10))

# (i) Compute the average percentage tips from the data set
# (Using the full tips dataset for overall average)
tips["tip_percentage"] = (tips["tip"] / tips["total_bill"]) * 100
avg_tip_percentage = tips["tip_percentage"].mean()
print("\n(i) Average Tip Percentage (entire dataset):", round(avg_tip_percentage, 2), "%")

# Also show average for Sunday only
avg_sun_tip = sun_tips["tip_percentage"].mean()
print("Average Tip Percentage on Sunday only:", round(avg_sun_tip, 2), "%")

print("\n" + "=" * 70)
print("ALL QUESTIONS COMPLETED SUCCESSFULLY!")
print("=" * 70)
