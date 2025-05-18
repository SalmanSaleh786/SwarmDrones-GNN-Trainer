import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, BatchNorm

class DroneGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.01):
        super(DroneGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.bn4 = BatchNorm(hidden_dim)
        self.conv5 = GCNConv(hidden_dim, hidden_dim)
        self.bn5 = BatchNorm(hidden_dim)
        self.conv6 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
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
    data_list = torch.load("graphs_dataset_for loss 0.99.pt")
    print(f"Loaded {len(data_list)} graphs")

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = data_list[0].x.shape[1]
    hidden_dim = 128
    output_dim = 2  # Predict next (x, y) coordinates
    model = DroneGNN(input_dim, hidden_dim, output_dim).to(device)

    # DataLoader
    batch_size = 32
    train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    losses = []
    model.train()

    for epoch in range(400):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y.view(-1, 2).float())  # ✅ Fix shape
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(total_loss)  # ✅ Append loss correctly
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

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
