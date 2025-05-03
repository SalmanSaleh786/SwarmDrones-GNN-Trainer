import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Load CSV data ===
csv_path = "test_predictions.csv"
df = pd.read_csv(csv_path)

# Extract actual and predicted positions
actual_values = df[['actual_x', 'actual_y']].values
predictions = df[['predicted_x', 'predicted_y']].values

# === Estimate current positions as: current = actual - (predicted - actual) ===
# This assumes actual = current + delta, so current = actual - delta
current_positions = actual_values - (predictions - actual_values)

# === Direction classification ===
def get_direction(from_pos, to_pos, threshold=1e-3):
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]

    # Threshold small movements
    dx = 0 if abs(dx) < threshold else dx
    dy = 0 if abs(dy) < threshold else dy

    if dx == 0 and dy > 0:
        return 0  # NORTH
    elif dx == 0 and dy < 0:
        return 1  # SOUTH
    elif dx > 0 and dy == 0:
        return 2  # EAST
    elif dx < 0 and dy == 0:
        return 3  # WEST
    elif dx > 0 and dy > 0:
        return 4  # NORTHEAST
    elif dx < 0 and dy > 0:
        return 5  # NORTHWEST
    elif dx > 0 and dy < 0:
        return 6  # SOUTHEAST
    elif dx < 0 and dy < 0:
        return 7  # SOUTHWEST
    else:
        return 8  # STAY / UNDEFINED

# Apply direction classification
y_true = [get_direction(c, a) for c, a in zip(current_positions, actual_values)]
y_pred = [get_direction(c, p) for c, p in zip(current_positions, predictions)]

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred, labels=range(9))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["N", "S", "E", "W", "NE", "NW", "SE", "SW", "STAY"]
)
disp.plot(cmap=plt.cm.Blues)
plt.title("Drone Direction Confusion Matrix")
plt.show()

# === Scatter Plot with Connecting Lines ===
plt.figure(figsize=(8, 6))

for i in range(len(actual_values)):
    x_vals = [actual_values[i][0], predictions[i][0]]
    y_vals = [actual_values[i][1], predictions[i][1]]
    plt.plot(x_vals, y_vals, color='gray', linewidth=0.8, alpha=0.5)

plt.scatter(actual_values[:, 0], actual_values[:, 1], color='blue', label="Actual", zorder=3)
plt.scatter(predictions[:, 0], predictions[:, 1], color='red', alpha=0.6, label="Predicted", zorder=3)

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Actual vs. Predicted Positions (Connected)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
