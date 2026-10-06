from math import log2

# Dataset
data = {
    "Outlook": [
        "Sunny", "Sunny", "Overcast", "Rain",
        "Rain", "Rain", "Overcast", "Sunny",
        "Sunny", "Rain", "Sunny", "Overcast",
        "Overcast", "Rain"
    ],

    "Temperature": [
        "Hot", "Hot", "Hot", "Mild",
        "Cool", "Cool", "Cool", "Mild",
        "Cool", "Mild", "Mild", "Mild",
        "Hot", "Mild"
    ],

    "Humidity": [
        "High", "High", "High", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "High"
    ],

    "Windy": [
        False, True, False, False,
        False, True, True, False,
        False, False, True, True,
        False, True
    ],

    "Play": [
        "No", "No", "Yes", "Yes",
        "Yes", "No", "Yes", "No",
        "Yes", "Yes", "Yes", "Yes",
        "Yes", "No"
    ]
}


# Entropy function
def entropy(values):

    total = len(values)

    counts = {}

    for value in values:
        counts[value] = counts.get(value, 0) + 1

    result = 0

    for count in counts.values():

        probability = count / total

        result -= probability * log2(probability)

    return result


# Information Gain function
def information_gain(data, attribute, target):

    total_entropy = entropy(data[target])

    attribute_values = set(data[attribute])

    weighted_entropy = 0

    for value in attribute_values:

        subset = []

        for i in range(len(data[target])):

            if data[attribute][i] == value:
                subset.append(data[target][i])

        weight = len(subset) / len(data[target])

        weighted_entropy += (
            weight * entropy(subset)
        )

    gain = total_entropy - weighted_entropy

    return gain


# Calculate information gain
attributes = [
    "Outlook",
    "Temperature",
    "Humidity",
    "Windy"
]

for attribute in attributes:

    gain = information_gain(
        data,
        attribute,
        "Play"
    )

    print(attribute, ":", gain)


# Find best attribute
best_attribute = max(
    attributes,
    key=lambda x: information_gain(
        data,
        x,
        "Play"
    )
)

print("\nBest Splitting Attribute:")
print(best_attribute)