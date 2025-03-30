import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from DataTraining import DroneGNN

# Load test dataset
test_data_list = torch.load("graphs_dataset_testing.pt")  # Load your test data

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = test_data_list[0].x.shape[1]
hidden_dim = 128
output_dim = 2  # Predict next (x, y) coordinates
model = DroneGNN(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load("gat_model_0.88.pth", map_location=device))
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

# Scatter plot of actual vs predicted positions
plt.figure(figsize=(8, 6))
plt.scatter(predictions[:, 0], predictions[:, 1], color='red', alpha=0.6, label="Predicted Positions")
plt.scatter(actual_values[:, 0], actual_values[:, 1], color='blue', label="Actual Positions")
plt.legend()
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Actual vs. Predicted Drone Positions")
plt.show()


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


