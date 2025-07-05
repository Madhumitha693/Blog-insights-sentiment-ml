import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["Random Forest", "SVM", "Naïve Bayes"]

# Accuracy values
accuracy = [0.80, 0.70, 0.61]

# Training time (Hypothetical values: Lower is better)
training_time = [5, 3, 1]  # Assume RF takes 5 sec, SVM 3 sec, NB 1 sec

# Creating figure and axes
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Accuracy (Bar chart)
ax1.bar(models, accuracy, color=['blue', 'green', 'red'], alpha=0.7, label="Accuracy")
ax1.set_ylabel("Accuracy", color="black", fontsize=12)
ax1.set_ylim(0, 1)  # Accuracy scale 0 to 1

# Create a second y-axis for training time
ax2 = ax1.twinx()
ax2.plot(models, training_time, color='black', marker='o', linestyle='dashed', label="Training Time")
ax2.set_ylabel("Training Time (Lower is Better)", color="black", fontsize=12)

# Title and legend
plt.title("Model Comparison: Accuracy vs Training Time", fontsize=14)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Show plot
plt.show()
