import time
import copy
import torch
from torch import nn
from cloud_program import evaluate_cloud
from edge_model import net_glob_edge_auxiliary, net_glob_edge
from setup import prRed, prGreen, num_clients, device, lr, num_edges

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
count1 = 0

# ====================================================================================================
#                                  edge Side Program
# ====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

# optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

# Edge-side functions associated with Training and Testing
class Edge(object):
    def __init__(self, net_edge_model, net_edge_auxiliary_model, client_ep, lr, device):
        self.device = device
        self.lr = lr
        self.client_ep = client_ep
        # self.selected_clients = []
        self.client_ep_time = []
        self.edge_ep_time = []
        self.cloud_ep_time = [0 for i in range(client_ep - 1)]
        self.client_ep_loss = []
        self.edge_ep_loss = []
        # edge-side function associated with Training
        self.w_glob_edge = net_edge_model.state_dict()
        self.w_glob_edge_auxiliary = net_edge_auxiliary_model.state_dict()
        self.w_locals_edge = []
        self.w_locals_edge_auxiliary = []

        # client idx collector
        self.idx_collect = []
        self.l_epoch_check = False
        self.fed_check = False
        # Initialization of net_model_edge and net_edge (edge-side model)
        self.net_model_edge = [net_edge_model for i in range(int(num_clients / num_edges))]
        self.net_model_edge_auxiliary = [net_edge_auxiliary_model for j in range(int(num_clients / num_edges))]
        # self.net_edge = copy.deepcopy(self.net_model_edge[0]).to(device)
        # self.net_edge_auxiliary = copy.deepcopy(self.net_model_edge_auxiliary[0]).to(device)
        self.count1 = 0

    def train_edge(self, global_ep, fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
        if idx != 0 or idx != 1:
            idx -= 2
        net_edge = copy.deepcopy(self.net_model_edge[idx]).to(device)
        net_edge_auxiliary = copy.deepcopy(self.net_model_edge_auxiliary[idx]).to(device)
        net_edge.train()
        net_edge_auxiliary.train()
        optimizer_edge = torch.optim.Adam(net_edge.parameters(), lr=lr)
        optimizer_edge_auxiliary = torch.optim.Adam(net_edge_auxiliary.parameters(), lr=lr)

        # train and update
        optimizer_edge.zero_grad()
        optimizer_edge_auxiliary.zero_grad()

        fx_client = fx_client.to(device)
        y = y.to(device)

        # ---------forward prop-------------
        fx_edge = net_edge(fx_client)
        edge_fx = fx_edge.clone().detach().requires_grad_(True)

        # run net_auxiliary
        edge_fx_auxiliary = net_edge_auxiliary(edge_fx)

        # calculate loss
        local_loss = criterion(edge_fx_auxiliary, y)

        # --------backward prop--------------
        local_loss.backward()
        optimizer_edge.step()
        optimizer_edge_auxiliary.step()

        # Update the edge-side model for the current batch
        self.net_model_edge[idx] = copy.deepcopy(net_edge)
        self.net_model_edge_auxiliary[idx] = copy.deepcopy(net_edge_auxiliary)

        # count1: to track the completion of the local batch associated with one client
        self.count1 += 1

        if self.count1 == len_batch:
            self.count1 = 0

            # copy the last trained model in the batch
            w_edge = net_edge.state_dict()
            w_edge_auxiliary = net_edge_auxiliary.state_dict()

            # If one local epoch is completed, after this a new client will come
            if l_epoch_count == l_epoch - 1:

                self.l_epoch_check = True  # to evaluate_server function - to check local epoch has completed or not
                # We store the state of the net_glob_edge()
                self.w_locals_edge.append(copy.deepcopy(w_edge))
                self.w_locals_edge_auxiliary.append(copy.deepcopy(w_edge_auxiliary))

                # collect the id of each new user
                if idx not in self.idx_collect:
                    self.idx_collect.append(idx)
                    # print(idx_collect)

            # This is for federation process--------------------
            if len(self.idx_collect) == num_clients / num_edges:
                self.fed_check = True  # to evaluate_edge function  - to check fed check has hitted
                # Federation process at edge-Side------------------------- output print and update is done in evaluate_edge()
                # for nicer display

                w_glob_edge = FedAvg(self.w_locals_edge)
                w_glob_edge_auxiliary = FedAvg(self.w_locals_edge_auxiliary)

                # edge-side global model update and distribute that model to all clients ------------------------------
                net_glob_edge.load_state_dict(w_glob_edge)
                net_glob_edge_auxiliary.load_state_dict(w_glob_edge_auxiliary)
                self.net_model_edge = [net_glob_edge for i in range(int(num_clients / num_edges))]
                self.net_model_edge_auxiliary = [net_glob_edge_auxiliary for i in range(int(num_clients / num_edges))]

                self.w_locals_edge = []
                self.idx_collect = []

        return edge_fx, local_loss.item()

    def evaluate_edge(self, fx_client, y, idx, len_batch, batch_idx, ell, idx_count):
        if idx != 0 or idx != 1:
            idx -= 2
        net = copy.deepcopy(self.net_model_edge[idx]).to(device)
        net.eval()

        with torch.no_grad():
            fx_client = fx_client.to(device)
            y = y.to(device)
            # ---------forward prop-------------
            fx_edge = net(fx_client)

            # Sending activations to cloud
            evaluate_cloud(fx_edge, y, idx, len_batch, batch_idx, ell, idx_count)

        return