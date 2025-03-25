import torch
from torch_geometric.data import Data
import os
import glob
import ast

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
                    if len(data)!=11:
                        raise Exception("Length of data is not 11!")

                    allParsed.append(data)
            return allParsed
# Function to process a single file and extract data
def process_data(data_line):

    # Example line: (0, (8, 2), ['.', '.', '%', '.'], [(4, 12), (4, 13), (16, 11)], ...)
    elements = data_line  # Convert string to tuple
    agentIndex, currDronePos, objectsAround, otherAgentPositions, wallCorners, battery, fire, food, score, action, nextPos = elements

    # Convert position to numerical features
    x_pos, y_pos = currDronePos
    obj_enc = [0 if obj in {'%', 'G', 'P'} else 1 if obj == 'F' else 0.5 for obj in objectsAround]
    walls_enc=[0 if obj==True else 1 for obj in wallCorners]
    max_agents = 3

    # Encode agent distances with fixed length
    agent_distances = [abs(x_pos - ax) + abs(y_pos - ay) for ax, ay in otherAgentPositions]

    # Truncate or pad to ensure fixed length
    if len(agent_distances) > max_agents:
        agent_distances = agent_distances[:max_agents]  # Truncate if too many
    else:
        agent_distances += [0] * (max_agents - len(agent_distances))  # Pad if too few


    # Encode labels (action)
    action_label = action_mapping[action]
    # agentIndex,
    # currDronePos,
    # objectsAroundCurrPos,
    # otherAgentPositions,
    # wallCorners,
    # data.agentStates[agentIndex].getBattery(),
    # self.isFireHere(currDronePos, data.layout),
    # self.isFoodNearby(currDronePos, data.layout),
    # data.score,
    # action
    # Combine node features
    node_features = torch.tensor([x_pos, y_pos, *obj_enc, *agent_distances, *walls_enc, battery, fire, *food, score], dtype=torch.float)

    return node_features, action_label

# Define a mapping for actions to numerical labels
action_mapping = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Stop':4}

# Load all files
data_folder = "/home/salmansaleh/PycharmProjects/GraphNeuralNetwork/logs/"
# Recursively get all .pkl files in subdirectories
files = glob.glob(os.path.join(data_folder, "**", "1_*.txt"), recursive=True)

# import re
# def extract_game_number(file_path):
#     """Extracts the first integer in the filename as the game number"""
#     match = re.search(r"/(\d+)_", file_path)  # Looks for the first number before an underscore
#     return int(match.group(1)) if match else float('inf')  # Default large number if no match
#
# # Sort by game number (as an integer)
# sorted_files = sorted(files, key=extract_game_number)
#
# # Print sorted file paths with only the game number for clarity
# for f in sorted_files:
#     game_no = extract_game_number(f)
#     print(f"Game {game_no}: {f}")

#files = files[:100]
print('Total Valid Files-', len(files))

# Load all data samples
all_missions = [load_txt(file) for file in files]

print(f"Loaded {len(all_missions)} samples from multiple folders.")
#print("Example sample:", all_missions[0])  # Check first sample

all_graphs = []  # Store multiple missions as separate graphs
missionNo=0
for mission in all_missions:
    missionNo=missionNo+1
    print('Processing Mission: ', missionNo)
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


# Save graph to file
torch.save(all_graphs, "graphs_dataset.pt")
print('Missions Saved!')
# Load graph from file
loaded_data = torch.load("graphs_dataset.pt")
print(loaded_data)
