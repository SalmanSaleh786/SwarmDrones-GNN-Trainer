import torch
from torch_geometric.data import Data
import os

# Define a mapping for actions to numerical labels
action_mapping = {'North': 0, 'South': 1, 'East': 2, 'West': 3}

# Function to process a single file and extract data
def process_data(data_line):

    # Example line: (0, (8, 2), ['.', '.', '%', '.'], [(4, 12), (4, 13), (16, 11)], ...)
    elements = data_line  # Convert string to tuple
    agentIndex, currDronePos, objectsAround, otherAgentPositions, wallCorners, battery, fire, food, score, action, nextPos = elements

    # Convert position to numerical features
    x_pos, y_pos = currDronePos
    obj_enc = [1 if obj == '%' else 0 for obj in objectsAround]  # Example encoding walls (modify as needed)

    # Encode other agents' positions (distance-based encoding)
    agent_distances = [abs(x_pos - ax) + abs(y_pos - ay) for ax, ay in otherAgentPositions]

    # Encode labels (action)
    action_label = action_mapping[action]

    # Combine node features
    node_features = torch.tensor([x_pos, y_pos, *obj_enc, *agent_distances, battery, fire, *food, score], dtype=torch.float)

    return node_features, action_label

def load_txt(file_path):
    with open(file_path, 'r') as f:
        allParsed = []
        for line in f:  # Read line by line
            parsedList=[]
            line = line.strip()
            firstIdx=line.find('(')
            if line.endswith(")"):  # Check if line starts and ends with ()
                line = line[firstIdx+1:-1]  # Remove first '(' and last ')'
                data = ast.literal_eval(f"({line})")
                # for item in data:
                #     parsedList.append(item)
                allParsed.append(data)
        return allParsed

import glob
# Load all files
data_folder = "/home/salmansaleh/PycharmProjects/GraphNeuralNetwork/logs/"
# Recursively get all .pkl files in subdirectories
files = glob.glob(os.path.join(data_folder, "**", "1_*.txt"), recursive=True)
files = files[:100]
print('Total Valid Files-', len(files))




# Load all data samples
all_missions = [load_txt(file) for file in files]

print(f"Loaded {len(all_missions)} samples from multiple folders.")
print("Example sample:", all_missions[0])  # Check first sample
nodes = []
edges = []
edge_attrs = []
labels = []
from collections import defaultdict

# Define a mapping for actions to numerical labels
action_mapping = {'North': 0, 'South': 1, 'East': 2, 'West': 3}

# Function to process a single file and extract data
def process_data(data_line):

    # Example line: (0, (8, 2), ['.', '.', '%', '.'], [(4, 12), (4, 13), (16, 11)], ...)
    elements = data_line  # Convert string to tuple
    agentIndex, currDronePos, objectsAround, otherAgentPositions, wallCorners, battery, fire, food, score, action, nextPos = elements

    # Convert position to numerical features
    x_pos, y_pos = currDronePos
    obj_enc = [1 if (obj == '%' or obj=='G' or obj=='P' or obj=='F') else 0 for obj in objectsAround]  # Example encoding walls (modify as needed)

    # Encode other agents' positions (distance-based encoding)
    agent_distances = [abs(x_pos - ax) + abs(y_pos - ay) for ax, ay in otherAgentPositions]

    # Encode labels (action)
    action_label = action_mapping[action]

    # Combine node features
    node_features = torch.tensor([x_pos, y_pos, *obj_enc, *agent_distances, battery, fire, *food, score], dtype=torch.float)

    return node_features, action_label

from torch_geometric.data import Data
import glob
import re
import ast

# Load all files
data_folder = "/home/salmansaleh/PycharmProjects/GraphNeuralNetwork/logs/"
# Recursively get all .pkl files in subdirectories
files = glob.glob(os.path.join(data_folder, "**", "1_*.txt"), recursive=True)
files = files[:10]
print('Total Valid Files-', len(files))


def load_txt(file_path):
    with open(file_path, 'r') as f:
        allParsed = []
        for line in f:  # Read line by line
            parsedList=[]
            line = line.strip()
            firstIdx=line.find('(')
            if line.endswith(")"):  # Check if line starts and ends with ()
                line = line[firstIdx+1:-1]  # Remove first '(' and last ')'
                data = ast.literal_eval(f"({line})")
                allParsed.append(data)
        return allParsed

# Load all data samples
all_missions = [load_txt(file) for file in files]

print(f"Loaded {len(all_missions)} samples from multiple folders.")
print("Example sample:", all_missions[0])  # Check first sample

from torch_geometric.data import Data

all_graphs = []  # Store multiple missions as separate graphs

for mission in all_missions:
    nodes = []
    edge_index = []
    obstacle_edges = []
    labels = []

    node_positions = {nd[1]: j for j, nd in enumerate(mission)}  # Fast lookup

    for j, missionIndex in enumerate(mission):
        node_feat, action_label = process_data(missionIndex)
        node_index = len(nodes)

        # Add node features and labels
        nodes.append(node_feat)
        labels.append(action_label)

        # Connect to previous node (time sequence)
        if j > 0:
            edge_index.append([node_index - 1, node_index])
            edge_index.append([node_index, node_index - 1])  # Bidirectional

        # Extract position
        x, y = missionIndex[1]
        directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]

        for i, obj in enumerate(missionIndex[2]):
            nx, ny = x + directions[i][0], y + directions[i][1]

            neighbor_idx = node_positions.get((nx, ny), None)

            if neighbor_idx is not None:
                if obj in {'.', '', ' ', 'F'}:  # Normal traversable edges
                    edge_index.append([node_index, neighbor_idx])
                else:  # Obstacle edges
                    obstacle_edges.append([node_index, neighbor_idx])

    # Convert lists to PyTorch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_features = torch.stack(nodes)
    labels = torch.tensor(labels, dtype=torch.long)

    # Create PyG Data object
    graph_data = Data(x=node_features, edge_index=edge_index, y=labels)

    # Store the graph
    all_graphs.append(graph_data)

# Print first graph structure
print(all_graphs[0])


import torch
from torch_geometric.data import Data

# Save graph to file
torch.save(all_graphs, "graph_dataset.pt")

# Load graph from file
loaded_data = torch.load("graph_dataset.pt")
print(loaded_data)
