import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

data_list = torch.load("graphs_dataset.pt")  # Load the stored graphs
print(f"Loaded {len(data_list)} graphs")  # If multiple graphs are stored
print(data_list[0])  # Print first graph to verify

class DroneGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DroneGNN, self).__init__()
        # Define GNN layers (using GCN, can replace with GAT, GIN, etc.)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

input_dim = data_list[0].x.shape[1]  # Number of node features
hidden_dim = 128
output_dim = 5  # 5 possible movement actions: North, South, East, West, Stop

model = DroneGNN(input_dim, hidden_dim, output_dim)


from torch_geometric.loader import DataLoader
batch_size = 48  # Adjust as needed
train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(400):  # Adjust epochs
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)  # Compare prediction with ground truth labels
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# Assuming you have a trained model
torch.save(model.state_dict(), "gnn_model.pth")

def predict_action(test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data)
        pred = out.argmax(dim=1)  # Get the predicted action
    return pred

test_graph = data_list[-1]  # Example: Use last stored graph for testing
predicted_action = predict_action(test_graph)
print("Predicted actions for each node:", predicted_action)
