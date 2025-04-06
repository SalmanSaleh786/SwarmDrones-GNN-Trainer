import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class DroneGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DroneGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # Added extra layer
        self.conv4 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x
# === Inference server loop ===
def model_server(conn):
    print('Starting model server')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 16
    hidden_dim = 128
    output_dim = 2

    model = DroneGNN(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load("gat_model_0.88.pth", map_location=device))
    model.eval()

    print("[Model] Server ready!")
    while True:
        data = conn.recv()

        if isinstance(data, str):
            if data == "close":
                print("[Model] Server closing.")
                break
            else:
                print("[Model] Got unexpected string input:", data)
                continue

        try:
            graph_data = data.to(device)  # move to CPU or GPU
            output = model(graph_data)
            conn.send(output)
        except Exception as e:
            print("[Model] Error during prediction:", e)
            conn.send("error")

from multiprocessing import Process, Pipe
# Create a bidirectional pipe
parent_conn, child_conn = Pipe()

# Start the model server in a new process
p = Process(target=model_server, args=(child_conn,))
p.start()

# === Now communicate with it ===
# Prepare some dummy input graph
graph_input = {
    "x": [[...]],  # List of node features
    "edge_index": [[0, 1], [1, 0]]
}

# Send input to model server
parent_conn.send(graph_input)

# Receive output
output = parent_conn.recv()
print("Received prediction:", output)

# To stop the server
parent_conn.send("close")
p.join()
