import torch
from torch_geometric.data import Data
import os
import glob
import ast
import re
from collections import defaultdict


# Load text file
def load_txt(file_path):
    with open(file_path, 'r') as f:
        allParsed = []
        for line in f:
            line = line.strip()
            firstIdx = line.find('(')
            if line.endswith(")"):
                line = line[firstIdx + 1:-1]
                data = ast.literal_eval(f"({line})")
                if len(data) != 11:
                    raise Exception("Length of data is not 11!")
                allParsed.append(data)
        return allParsed


def process_data(data_line):
    elements = data_line
    agentIndex, currDronePos, objectsAround, otherAgentPositions, wallCorners, battery, fire, foodCorners, score, action, nextPos = elements
    x_pos, y_pos = currDronePos
    obj_enc = [0 if obj in {'%', 'G', 'P'} else 1 if obj == 'F' else 0.5 for obj in objectsAround]
    walls_enc = [0 if obj else 1 for obj in wallCorners]
    food_enc=[1 if obj else 0 for obj in foodCorners]
    normalized_battery = float(battery) / 100.0
    normalized_score=float(score)/10000
    normalized_x_pos=float(x_pos)/25
    normalized_y_pos=float(y_pos)/25
    #max_agents = 3
    #agent_distances = [abs(x_pos - ax) + abs(y_pos - ay) for ax, ay in otherAgentPositions]
    #agent_distances += [0] * (max_agents - len(agent_distances))

    node_features = torch.tensor([normalized_x_pos, normalized_y_pos, *obj_enc, *walls_enc, *food_enc, normalized_battery, normalized_score],
                                 dtype=torch.float)

    next_x, next_y = nextPos  # Extract next position
    normalized_next_x=float(next_x)/25
    normalized_next_y=float(next_y)/25
    next_pos_tensor = torch.tensor([normalized_next_x, normalized_next_y], dtype=torch.float)  # Convert to tensor

    return node_features, next_pos_tensor


#action_mapping = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Stop': 4}
#home/salmansaleh/PycharmProjects/GraphNeuralNetwork
data_folder = "/home/salmansaleh/PycharmProjects/GraphNeuralNetwork/logs/"
#files = glob.glob(os.path.join(data_folder, "**", "1_*.txt"), recursive=True)
files = glob.glob(os.path.join(data_folder, "**", "*.txt"), recursive=True)

# Store missions by game
missionsByGameDict = defaultdict(dict)

for file in files:
    match = re.search(r"(\d+)_(Drone\d+)", file.split("/")[-1])

    if match:
        game_no = int(match.group(1))
        drone_id = int(match.group(2)[-1])
        mission_data = load_txt(file)
        if game_no in missionsByGameDict and len(missionsByGameDict[game_no]) >= 4:
            print('duplicate gameNo:', game_no)

            # Delete all files associated with this gameNo
            for existing_file in files:
                if existing_file.startswith(f"{game_no}_"):
                    os.remove(existing_file)
                    print("Deleted:", existing_file)

            # Remove gameNo from dictionary
            del missionsByGameDict[game_no]
            print("Removed entry for gameNo:", game_no)

        else:
            missionsByGameDict[game_no][drone_id] = mission_data
    else:
        print('error filename:', file)

all_graphs = []  # Store separate graphs per timestep
gameIdx=0
for key in missionsByGameDict.keys():
    gameIdx=gameIdx+1
    print(gameIdx)
    missionsOfThisGame = missionsByGameDict[key]  # Get all drone missions for this game
    print('Processing Game:', key)

    max_steps = min(len(m) for m in missionsOfThisGame.values())  # Find max available steps
    previous_positions = {}  # Store node indices from the previous step

    for step in range(max_steps):  # Iterate over each timestep
        nodes = []
        edge_index = []
        labels = []
        drone_positions = {}

        # **Create node position lookup per timestep**
        node_positions = {}

        for drone_id, mission in missionsOfThisGame.items():
            if step >= len(mission):
                continue
            pos = mission[step][1]  # Get (x, y) position
            node_positions[pos] = len(nodes)  # Store node index at this position

        # Process each drone at this timestep
        for drone_id, mission in missionsOfThisGame.items():
            if step >= len(mission):
                continue

            node_feat, next_pos = process_data(mission[step])
            node_index = len(nodes)  # Assign node index

            nodes.append(node_feat)
            labels.append(next_pos)
            drone_positions[drone_id] = node_index  # Store node index for edge connections

            # **Add edge to previous state (time connection)**
            if drone_id in previous_positions:
                edge_index.append([previous_positions[drone_id], node_index])  # Past → Current
                edge_index.append([node_index, previous_positions[drone_id]])  # Current → Past

            #x, y = mission[step][1]
            #directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]  # Right, Bottom, Top, Left

            #if (len(mission)>step+1):
            #    nextIdx=mission[step+1][1]
            #    edge_index.append(node_index, nextIdx)

            #
            #for i, obj in enumerate(mission[step][2]):  # Check surrounding cells
            #     nx, ny = x + directions[i][0], y + directions[i][1]
            #
            #     neighbor_idx = node_positions.get((nx, ny))
            #
            #     if neighbor_idx is not None:
            #         if obj in {'.', '', ' ', 'F'}:  # Normal traversable edges
            #             edge_index.append([node_index, neighbor_idx])

        # **Store current positions for next timestep**
        previous_positions = drone_positions.copy()

        max_distance = 2  # Maximum allowed Manhattan distance

        drone_ids = list(drone_positions.keys())

        for i, drone_a in enumerate(drone_ids):
            for j, drone_b in enumerate(drone_ids):
                if i != j:  # Avoid self-loops
                    pos_a = mission[step][1]  # (x, y) of drone_a
                    pos_b = missionsOfThisGame[drone_b][step][1]  # (x, y) of drone_b

                    distance = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])  # Manhattan distance

                    if distance <= max_distance:  # Only connect if within range
                        edge_index.append([drone_positions[drone_a], drone_positions[drone_b]])
                        edge_index.append([drone_positions[drone_b], drone_positions[drone_a]])  # Bidirectional

        # Convert to tensors
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_features_tensor = torch.stack(nodes)
        labels_tensor = torch.tensor([list(label) for label in labels], dtype=torch.float)

#        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Create PyG Data object for this timestep
        graph_data = Data(x=node_features_tensor, edge_index=edge_index_tensor, y=labels_tensor)
        all_graphs.append(graph_data)

print(f"Total graphs created: {len(all_graphs)}")
torch.save(all_graphs, "graphs_dataset_Testing_4june.pt")
print("Missions Saved!")