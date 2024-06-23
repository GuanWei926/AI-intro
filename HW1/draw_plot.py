from tabulate import tabulate
import matplotlib.pyplot as plt
'''
data = [
    ["small data", "train data accuracy", "test data accuracy"],
    ["t=1", "81.0%", "48.0%"],
    ["t=2", "81.0%", "48.0%"],
    ["t=3", "88.0%", "53.0%"],
    ["t=4", "86.0%", "47.5%"],
    ["t=5", "88.5%", "54.0%"],
    ["t=6", "89.0%", "51.0%"],
    ["t=7", "90.0%", "54.5%"],
    ["t=8", "91.0%", "55.0%"],
    ["t=9", "90.0%", "57.5%"],
    ["t=10", "91.5%", "59.5%"]
]

# Create the table
table = tabulate(data, headers="firstrow", tablefmt="grid")

# Print the table
print(table)



# Train and test accuracy data
train_acc = [0.81, 0.81, 0.88, 0.86, 0.885, 0.89, 0.90, 0.91, 0.90, 0.915]
test_acc = [0.48, 0.48, 0.53, 0.475, 0.54, 0.51, 0.545, 0.55, 0.575, 0.595]
iterations = range(1, 11)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot train and test accuracy
ax.plot(iterations, train_acc, marker='o', color='orange', label='train')
ax.plot(iterations, test_acc, marker='o', color='green', label='test')

# Set axis labels and title
ax.set_xlabel('Iteration')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy for each Iteration')

# Set axis limits
ax.set_xlim(0, 11)  # Adjust x-axis limits if needed
ax.set_ylim(0.4, 1.1)  # Adjust y-axis limits if needed

# Add a grid
ax.grid(True)

# Add a legend
ax.legend()

# Display the plot
plt.show()
'''

#----------------------------------------------------------------
data = [
    [" ", "train data accuracy", "test data accuracy"],
    ["average", "84.49%", "81.73%"],
    ["median", "90.72%", "87.50%"],
    ["75th percentile", "90.44%", "84.29%"],
    ["25th percentile", "88.78%", "84.62%"],
]

# Create the table
table = tabulate(data, headers="firstrow", tablefmt="grid")

# Print the table
print(table)



# Train and test accuracy data
train_acc = [0.7147, 0.7147, 0.7576, 0.7659, 0.7604, 0.7604, 0.7687, 0.7839, 0.7909, 0.7853]
test_acc = [0.7179, 0.7179, 0.7596, 0.7692, 0.7660, 0.7468, 0.7724, 0.7821, 0.7853, 0.7788]
iterations = range(1, 11)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot train and test accuracy
ax.plot(iterations, train_acc, marker='o', color='orange', label='train')
ax.plot(iterations, test_acc, marker='o', color='green', label='test')

# Set axis labels and title
ax.set_xlabel('Iteration')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy for each Iteration')

# Set axis limits
ax.set_xlim(0, 11)  # Adjust x-axis limits if needed
ax.set_ylim(0.4, 1.1)  # Adjust y-axis limits if needed

# Add a grid
ax.grid(True)

# Add a legend
ax.legend()

# Display the plot
plt.show()