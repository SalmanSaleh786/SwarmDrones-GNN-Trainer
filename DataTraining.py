import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import torch.optim as optim

# Load dataset
data_list = torch.load("graphs_dataset.pt")
print(f"Loaded {len(data_list)} graphs")
print(data_list[0])

class DroneGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8):
        super(DroneGAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.conv4 = GATConv(hidden_dim, output_dim, heads=1, concat=False)  # Only one output layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)  # Last layer, no activation needed
        return x


# Model setup
input_dim = data_list[0].x.shape[1]
hidden_dim = 128
output_dim = 2  # Predict next (x, y) coordinates
model = DroneGAT(input_dim, hidden_dim, output_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# DataLoader
batch_size = 64
train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

# Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_fn = torch.nn.MSELoss()  # Regression loss

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.train()
import matplotlib.pyplot as plt

losses = []

for epoch in range(70):
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    losses.append(total_loss)
    #if (epoch + 1) % 10 == 0:  # Print loss every 10 epochs
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "gat_model.pth")
print("Training complete and model saved!")

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()



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
