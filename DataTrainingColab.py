from torch_geometric.nn import BatchNorm
import torch.optim as optim
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing
from torch.nn import Linear

class PositionOnlyEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, pos_dim=2):
        super(PositionOnlyEdgeConv, self).__init__(aggr='mean')  # mean aggregation

        # Linear layer will take full node features + aggregated position info from neighbors
        self.linear = Linear(in_channels + pos_dim, out_channels)
        self.pos_dim = pos_dim

    def forward(self, x, edge_index):
        # x has shape [num_nodes, in_channels]
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Only take first pos_dim features (e.g., position: x, y)
        return x_j[:, :self.pos_dim]

    def update(self, aggr_out, x):
        # aggr_out is [num_nodes, pos_dim], x is [num_nodes, in_channels]
        combined = torch.cat([x, aggr_out], dim=1)  # [num_nodes, in_channels + pos_dim]
        return self.linear(combined)


import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm

class DroneGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pos_dim=2):
        super(DroneGNN, self).__init__()

        self.conv1 = PositionOnlyEdgeConv(input_dim, hidden_dim, pos_dim)
        self.bn1 = BatchNorm(hidden_dim)

        self.conv2 = PositionOnlyEdgeConv(hidden_dim, hidden_dim, pos_dim)
        self.bn2 = BatchNorm(hidden_dim)

        self.conv3 = PositionOnlyEdgeConv(hidden_dim, hidden_dim, pos_dim)
        self.bn3 = BatchNorm(hidden_dim)

        self.conv4 = PositionOnlyEdgeConv(hidden_dim, hidden_dim, pos_dim)
        self.bn4 = BatchNorm(hidden_dim)

        self.conv5 = PositionOnlyEdgeConv(hidden_dim, hidden_dim, pos_dim)
        self.bn5 = BatchNorm(hidden_dim)

        self.conv6 = PositionOnlyEdgeConv(hidden_dim, output_dim, pos_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = F.relu(self.bn5(self.conv5(x, edge_index)))
        x = self.conv6(x, edge_index)

        return x
 # Prediction function
def predict_next_position(test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data.to(device))
    return out.cpu().numpy()
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

def predict_next_position(graph):
    model.eval()
    with torch.no_grad():
        output = model(graph)
    return output.cpu().numpy()

if __name__ == "__main__":
    # Load dataset
    data_list = torch.load("graphs_dataset.pt")
    print(f"Loaded {len(data_list)} graphs")

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dimensions
    input_dim = data_list[0].x.shape[1]   # Full feature vector length
    print('Input DIM:'+str(input_dim))
    hidden_dim = 128
    output_dim = 2                        # Predict next (x, y)
    pos_dim = 2                           # Number of position features at the beginning of x

    model = DroneGNN(input_dim, hidden_dim, output_dim, pos_dim=pos_dim).to(device)

    # Move all data to device
    data_list = [data.to(device) for data in data_list]

    # DataLoader
    batch_size = 64
    train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    losses = []
    model.train()
    for epoch in range(100):
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y.view(-1, 2).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "gat_model.pth")
    print("Training complete and model saved!")

    # Plot loss
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

    # Example prediction
    test_graph = data_list[-1].to(device)
    predicted_position = predict_next_position(test_graph)
    print("Predicted positions for each node:", predicted_position)
