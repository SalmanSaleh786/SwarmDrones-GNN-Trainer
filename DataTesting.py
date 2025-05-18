import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from DataTrainingColab import DroneGNN

# Load test dataset
test_data_list = torch.load("graphs_dataset_Testing.pt")#torch.load("graphs_dataset_for loss 0.99.pt")
#test_data_list=test_data_list[:30000]
cleaned_test_data_list = []

for i, data in enumerate(test_data_list):
    edge_index = data.edge_index

    # Ensure it's a tensor
    if not isinstance(edge_index, torch.Tensor):
        continue  # Skip bad sample

    # Ensure it's 2D and has 2 rows
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        continue  # Skip malformed edge_index

    cleaned_test_data_list.append(data)

print(f"Original dataset size: {len(test_data_list)}")
print(f"Cleaned dataset size: {len(cleaned_test_data_list)}")

test_data_list = cleaned_test_data_list  # Overwrite with cleaned data
print(len(test_data_list))
# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = test_data_list[0].x.shape[1]
hidden_dim = 154
output_dim = 2  # Predict next (x, y)
pos_dim = 2     #Number of position features at the beginning of x
model = DroneGNN(input_dim, hidden_dim, output_dim, pos_dim=pos_dim).to(device)
model.load_state_dict(torch.load("gat_model.pth", map_location=device))
model.eval()


predictions = []
actual_values = []

with torch.no_grad():  # Disable gradient tracking for testing
    for test_graph in test_data_list:
        test_graph = test_graph.to(device)  # Move to GPU if available
        pred = model(test_graph)  # Get predicted (x, y) positions
        predictions.append(pred.cpu().numpy())  # Store predictions
        actual_values.append(test_graph.y.cpu().numpy())  # Store actual values


# Convert lists to numpy arrays
import numpy as np
predictions = np.vstack(predictions)  # Stack predictions
actual_values = np.vstack(actual_values)  # Stack actual values

# # Scatter plot of actual vs predicted positions
#
# plt.figure(figsize=(8, 6))
#
# # Plot individual lines between actual and predicted points
# for i in range(len(actual_values)):
#     x_values = [actual_values[i][0], predictions[i][0]]
#     y_values = [actual_values[i][1], predictions[i][1]]
#     plt.plot(x_values, y_values, color='gray', linewidth=0.8, alpha=0.5)  # Connecting line
#
# # Scatter points
# plt.scatter(actual_values[:, 0], actual_values[:, 1], color='blue', label="Actual Positions", zorder=3)
# plt.scatter(predictions[:, 0], predictions[:, 1], color='red', alpha=0.6, label="Predicted Positions", zorder=3)
#
# plt.legend()
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.title("Actual vs. Predicted Drone Positions (with connecting lines)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(actual_values, predictions)
print(f"Test MSE: {mse:.4f}")


import pandas as pd

df = pd.DataFrame({
    "actual_x": actual_values[:, 0],
    "actual_y": actual_values[:, 1],
    "predicted_x": predictions[:, 0],
    "predicted_y": predictions[:, 1]
})

df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv")


