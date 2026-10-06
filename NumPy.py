import numpy as np

# Create array from 1 to 30
a = np.arange(1, 31)

# Reshape into 5 x 6
a = a.reshape(5, 6)

print("Array:")
print(a)

# Third row
print("\nThird Row:")
print(a[2])

# Second column
print("\nSecond Column:")
print(a[:, 1])

# Elements divisible by 3
print("\nElements divisible by 3:")
print(a[a % 3 == 0])

# Replace values greater than 20 with 0
a[a > 20] = 0

print("\nAfter replacing values > 20 with 0:")
print(a)