import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class PositionOnlyEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(PositionOnlyEdgeConv, self).__init__(aggr='add')  # EdgeConv typically uses 'max'
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(6, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        pos_battery = x[:, [0, 1, 14]]  # Extract (x, y, battery)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index=edge_index, x=pos_battery)

    def message(self, x_i, x_j):
        diff = x_j - x_i
        combined = torch.cat([diff, x_i], dim=1)  # shape: [num_edges, 6]
        return self.mlp(combined)

    def update(self, aggr_out):
        return aggr_out


class DroneGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):#, dropout_rate=0.001):
        super(DroneGNN, self).__init__()
        self.conv1 = PositionOnlyEdgeConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = PositionOnlyEdgeConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = PositionOnlyEdgeConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.conv4 = PositionOnlyEdgeConv(hidden_dim, hidden_dim)
        self.bn4 = BatchNorm(hidden_dim)
        self.conv5 = PositionOnlyEdgeConv(hidden_dim, hidden_dim)
        self.bn5 = BatchNorm(hidden_dim)
        self.conv6 = PositionOnlyEdgeConv(hidden_dim, output_dim)
        #self.dropout = torch.nn.Dropout(dropout_rate)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        #x = self.dropout(x)
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
if __name__ == "__main__":
    # Load dataset
    data_list = torch.load("graphs_dataset.pt")
    print(f"Loaded {len(data_list)} graphs")

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    input_dim = data_list[0].x.shape[1]
    hidden_dim = 128
    output_dim = 2  # Predict next (x, y) coordinates
    model = DroneGNN(input_dim, hidden_dim, output_dim).to(device)

    # DataLoader
    batch_size = 64
    # Before loader
    data_list = [data.to(device) for data in data_list]

    train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    losses = []
    model.train()

    # Training loop
    for epoch in range(100):
        total_loss = 0
        for data in train_loader:  # now all data are already on cuda
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

    # Example testing
    test_graph = data_list[-1].to(device)
    predicted_position = predict_next_position(test_graph)
    print("Predicted positions for each node:", predicted_position)