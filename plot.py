import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file with error handling
metrics_file = "training_metrics.csv"
try:

    data = pd.read_csv(metrics_file)
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()

# Ensure all numeric columns are converted properly
numeric_columns = ["Episode", "Score", "Invalid Moves", "Highest Tile", "Total Reward"]
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert, setting invalid entries to NaN

# Drop rows with any NaN values in numeric columns
data = data.dropna(subset=numeric_columns)

# Plot the episode vs score
plt.figure(figsize=(10, 6))
plt.plot(data["Episode"], data["Score"], label="Score")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Score vs Episode")
plt.legend()
plt.grid()
plt.show()

# Plot the episode vs highest tile
plt.figure(figsize=(10, 6))
plt.plot(data["Episode"], data["Highest Tile"], label="Highest Tile", color="orange")
plt.xlabel("Episode")
plt.ylabel("Highest Tile")
plt.title("Highest Tile vs Episode")
plt.legend()
plt.grid()
plt.show()

# Plot the episode vs invalid moves
plt.figure(figsize=(10, 6))
plt.plot(data["Episode"], data["Invalid Moves"], label="Invalid Moves", color="red")
plt.xlabel("Episode")
plt.ylabel("Invalid Moves")
plt.title("Invalid Moves vs Episode")
plt.legend()
plt.grid()
plt.show()

# Plot the episode vs total reward
plt.figure(figsize=(10, 6))
plt.plot(data["Episode"], data["Total Reward"], label="Total Reward", color="green")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward vs Episode")
plt.legend()
plt.grid()
plt.show()

# Plot the episode vs epsilon
plt.figure(figsize=(10, 6))
plt.plot(data["Episode"], data["Epsilon"], label="Epsilon", color="purple")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay Over Episodes")
plt.legend()
plt.grid()
plt.show()
