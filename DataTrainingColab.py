import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing
from torch_geometric.data import DataLoader
from torch.nn import Linear, BatchNorm1d as BatchNorm
from torch.utils.data import random_split

# === GNN Layer ===
class PositionOnlyEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, pos_dim=2):
        super(PositionOnlyEdgeConv, self).__init__(aggr='mean')
        self.linear = Linear(in_channels + pos_dim, out_channels)
        self.pos_dim = pos_dim

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j[:, :self.pos_dim]  # Use only position info

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=1)
        return self.linear(combined)

# === GNN Model ===
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
        return self.conv6(x, edge_index)

# === Prediction ===
def predict_next_position(test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data.to(device))
    return out.cpu().numpy()

# === Main Execution ===
if __name__ == "__main__":
    # Load dataset
    data_list = torch.load("graphs_dataset_for loss 0.99.pt")#"graphs_dataset_Testing_4june.pt")#"graphs_dataset_for loss 0.99.pt")
    print(f"Loaded {len(data_list)} graphs")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model dimensions
    input_dim = data_list[0].x.shape[1]
    hidden_dim = 154
    output_dim = 2
    pos_dim = 2

    # Split data
    train_size = int(0.8 * len(data_list))
    test_size = len(data_list) - train_size
    train_data, test_data = random_split(data_list, [train_size, test_size])
    print(f"Train: {train_size}, Test: {test_size}")

    # DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model setup
    model = DroneGNN(input_dim, hidden_dim, output_dim, pos_dim=pos_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # === Training Loop ===
    num_epochs = 100
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y.view(-1, 2).float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                loss = loss_fn(out, data.y.view(-1, 2).float())
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

    print(f"Epoch {epoch+1:03d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "gat_model_forplot_0.0153_backup.pth")
    print("Training complete and model saved!")

    # === Plotting Loss ===
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training & Testing Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Example prediction
    test_graph = test_data[0].to(device)
    predicted_position = predict_next_position(test_graph)
    print("Predicted positions for each node:", predicted_position)
