import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import torch.optim as optim

# Load dataset
data_list = torch.load("graphs_dataset.pt")
print(f"Loaded {len(data_list)} graphs")
print(data_list[0])

# Define updated DroneGNN model
class DroneGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DroneGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)  # No activation for regression
        return x

# Model setup
input_dim = data_list[0].x.shape[1]
hidden_dim = 128
output_dim = 2  # Predict next (x, y) coordinates
model = DroneGNN(input_dim, hidden_dim, output_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# DataLoader
batch_size = 32
train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

# Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()  # Regression loss

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.train()
for epoch in range(2000):
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y.float())  # Ensure y is float for MSE
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "gnn_model.pth")
print("Training complete and model saved!")

# Prediction function
def predict_next_position(test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data)
    return out  # Returns predicted (x, y) position

# Example testing
test_graph = data_list[-1].to(device)
predicted_position = predict_next_position(test_graph)
print("Predicted positions for each node:", predicted_position)
