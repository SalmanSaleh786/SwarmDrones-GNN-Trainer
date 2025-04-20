import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from multiprocessing.connection import Listener
import torch
import GNNDataReader


class Directions:
    NORTH = 'North'
    SOUTH = 'South'
    EAST = 'East'
    WEST = 'West'
    STOP = 'Stop'


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


import pickle
import struct


def getDirection(agent, next):
    if agent[1] < next[1]:
        return Directions.NORTH
    if agent[1] > next[1]:
        return Directions.SOUTH
    if agent[0] < next[0]:
        return Directions.EAST
    if agent[0] > next[0]:
        return Directions.WEST
    return Directions.STOP


def model_server():
    previous_positions_dict = \
        {
            0: [],
            1: [],
            2: [],
            3: []
        }

    print('Starting model server')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 16
    hidden_dim = 128
    output_dim = 2

    model = DroneGNN(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load("gat_model_0.88.pth", map_location=device))
    model.eval()

    address = ('localhost', 6000)  # You can use a port
    listener = Listener(address, authkey=b'password')
    print("[Model] Waiting for connection...")
    conn = listener.accept()
    print("[Model] Connected to", listener.last_accepted)
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
            agentIndex    = data[0]
            historyDrone1 = data[1]
            historyDrone2 = data[2]
            historyDrone3 = data[3]
            historyDrone4 = data[4]
            output_direction = Directions.STOP
            if   agentIndex == 0 and len(historyDrone1)>0:
                previous_positions_dict[agentIndex].append(historyDrone1[0][0])
            elif agentIndex == 1 and len(historyDrone2)>0:
                previous_positions_dict[agentIndex].append(historyDrone2[0][0])
            elif agentIndex == 2 and len(historyDrone3)>0:
                previous_positions_dict[agentIndex].append(historyDrone3[0][0])
            elif agentIndex == 3 and len(historyDrone4)>0:
                previous_positions_dict[agentIndex].append(historyDrone4[0][0])

            if historyDrone1 == [] or historyDrone2 == [] or historyDrone3 == [] or historyDrone4 == []:
                output_direction = Directions.STOP
            else:
                dataGraph, agent_to_node = GNNDataReader.convert_To_Graph(data, previous_positions_dict)
                dataGraph = dataGraph.to(device)
                output = model(dataGraph)
                if agentIndex not in agent_to_node:
                    print(f"[Model] Agent index {agentIndex} not found in the graph nodes!")
                    output_direction = Directions.STOP
                else:
                    node_index = agent_to_node[agentIndex]
                    predicted_next_pos = output[node_index]
                    predicted_next_pos = predicted_next_pos * 25
                    predicted_next_pos = tuple(predicted_next_pos.round().int().tolist())
                    current_pos =  data[1 + agentIndex][0][1]  # Adjust this based on actual position access
                    output_direction = getDirection(current_pos, predicted_next_pos)

            bytes_to_send = pickle.dumps(output_direction, protocol=2)
            conn.send_bytes(bytes_to_send)
        except Exception as e:
            print("[Model] Error during prediction:", e)
            conn.send("error")

import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)
model_server()
