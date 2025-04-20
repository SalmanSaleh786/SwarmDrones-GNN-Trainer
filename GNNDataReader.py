import torch
from torch_geometric.data import Data
from collections import defaultdict

def process_data(data_line):
    elements = data_line
    agentIndex, currDronePos, objectsAround, otherAgentPositions, wallCorners, battery, fire, foodCorners, score, _, _ = elements

    x_pos, y_pos = currDronePos
    obj_enc = [0 if obj in {'%', 'G', 'P'} else 1 if obj == 'F' else 0.5 for obj in objectsAround]
    walls_enc = [0 if obj else 1 for obj in wallCorners]
    food_enc = [1 if obj else 0 for obj in foodCorners]
    normalized_battery = float(battery) / 100.0
    normalized_score = float(score) / 10000
    normalized_x_pos = float(x_pos) / 25
    normalized_y_pos = float(y_pos) / 25

    node_features = torch.tensor([normalized_x_pos, normalized_y_pos, *obj_enc, *walls_enc, *food_enc, normalized_battery, normalized_score],
                                 dtype=torch.float)

    #next_x, next_y = nextPos
    #normalized_next_x = float(next_x) / 25
    #normalized_next_y = float(next_y) / 25
    #next_pos_tensor = torch.tensor([normalized_next_x, normalized_next_y], dtype=torch.float)

    return node_features#, next_pos_tensor

def convert_To_Graph(data, previous_positions):
    # agentIndex = data[0]
    edge_index = []
    node_features = []
    agent_to_node = {}

    missions = {
        0: data[1],
        1: data[2],
        2: data[3],
        3: data[4]
    }
    drone_positions = {}
    node_counter = 0
    for i in range(4):
        if len(missions[i]) > 0:
            node_feat = process_data(missions[i][0])
            node_features.append(node_feat)
            agent_to_node[i] = node_counter
            drone_positions[i] = missions[i][0][1]  # save (x, y)
        node_counter += 1

    # Time edges
    for drone_id, node_idx in agent_to_node.items():
        if drone_id in previous_positions:
            if len(previous_positions[drone_id])>0:
                prev_idx = previous_positions[drone_id][0]
                edge_index.append([prev_idx, node_idx])
                edge_index.append([node_idx, prev_idx])

    # Spatial edges
    drone_ids = list(agent_to_node.keys())
    for i in range(len(drone_ids)):
        for j in range(i + 1, len(drone_ids)):
            drone_a, drone_b = drone_ids[i], drone_ids[j]
            pos_a, pos_b = drone_positions[drone_a], drone_positions[drone_b]
            dist = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
            if dist <= 2:
                a_idx, b_idx = agent_to_node[drone_a], agent_to_node[drone_b]
                edge_index.append([a_idx, b_idx])
                edge_index.append([b_idx, a_idx])

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty(
        (2, 0), dtype=torch.long)
    x_tensor = torch.stack(node_features)

    return Data(x=x_tensor, edge_index=edge_index_tensor), agent_to_node
