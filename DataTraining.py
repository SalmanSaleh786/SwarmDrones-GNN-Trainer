import torch
from torch_geometric.loader import DataLoader

from DronePathGCN import DronePathGCN

train_loader = DataLoader([data], batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DronePathGCN(in_channels=node_features.shape[1], hidden_channels=64, out_channels=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out, data.y.to(device))
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


#DATA EVALUATION
model.eval()
with torch.no_grad():
    pred = model(data.x.to(device), data.edge_index.to(device)).argmax(dim=1)
    accuracy = (pred == data.y.to(device)).sum().item() / data.y.size(0)
    print(f'Accuracy: {accuracy:.4f}')
