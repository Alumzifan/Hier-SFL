import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from cloud_program import train_cloud
from edge_model import net_glob_edge, net_glob_edge2, net_glob_edge_auxiliary, net_glob_edge2_auxiliary
from edge_program import criterion, Edge
from setup import prRed, device, client_ep, lr


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        # for i in range(len(self.idxs)):
        #     self.idxs[i] -= 1

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        features, label = self.dataset[self.idxs[item]]
        return features, label


# init two edges
Edge1 = Edge(net_glob_edge, net_glob_edge_auxiliary, client_ep, lr, device)
Edge2 = Edge(net_glob_edge2, net_glob_edge2_auxiliary, client_ep, lr, device)


# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, client_ep, idx, lr, device, edge_flag, dataset_train=None, dataset_test=None,
                 idxs=None,
                 idxs_test=None):
        self.idx = idx
        self.device = device
        self.edge_flag = edge_flag
        self.lr = lr
        self.client_ep = client_ep
        # self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=256, shuffle=True, pin_memory=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=256, shuffle=True, pin_memory=True)
        self.client_ep_time = []
        self.edge_ep_time = []
        self.cloud_ep_time = [0 for i in range(client_ep - 1)]
        self.client_ep_loss = []
        self.edge_ep_loss = []

    def train(self, global_ep, net, net_auxiliary):
        net.train()
        net_auxiliary.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)
        optimizer_client_auxiliary = torch.optim.Adam(net_auxiliary.parameters(), lr=self.lr)

        for iter in range(self.client_ep):
            len_batch = len(self.ldr_train)
            len_count = 0
            prRed('Client{} Train => Local Epoch: {}'.format(self.idx, iter))
            client_total_time = 0
            edge_total_time = 0
            cloud_total_time = 0
            client_total_loss = 0
            edge_total_loss = 0
            for batch_idx, (features, labels) in enumerate(self.ldr_train):
                batch_start_time = time.time()
                features = torch.reshape(features, [-1, 1, 141])
                features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer_client.zero_grad()
                optimizer_client_auxiliary.zero_grad()
                # ---------forward prop-------------
                fx = net(features)
                client_fx = fx.clone().detach().requires_grad_(True)

                # run net_auxiliary
                client_fx_auxiliary = net_auxiliary(client_fx)
                # calculate local loss
                local_loss = criterion(client_fx_auxiliary, labels)
                client_total_loss += local_loss.item()

                # --------backward prop -------------
                local_loss.backward()
                optimizer_client.step()
                optimizer_client_auxiliary.step()
                client_total_time += time.time() - batch_start_time
                if batch_idx == len_batch - 1:
                    self.client_ep_time.append(client_total_time)
                    self.client_ep_loss.append(client_total_loss / len_batch)
                    if (iter + 1) % 10 != 0 and iter != self.client_ep - 1:
                        self.edge_ep_time.append(0)
                        self.edge_ep_loss.append(0)

                if (iter + 1) % 10 == 0 and iter != self.client_ep - 1:
                    # Sending activations to edge
                    edge_start_time = time.time()
                    if self.edge_flag == 1:
                        edge_fx, edge_loss = Edge1.train_edge(global_ep, client_fx, labels,
                                                              iter, self.client_ep, self.idx,
                                                              len_batch)
                    else:
                        edge_fx, edge_loss = Edge2.train_edge(global_ep, client_fx, labels,
                                                              iter, self.client_ep, self.idx,
                                                              len_batch)
                    edge_total_time += time.time() - edge_start_time
                    edge_total_loss += edge_loss
                    if batch_idx == len_batch - 1:
                        self.edge_ep_time.append(edge_total_time)
                        self.edge_ep_loss.append(edge_total_loss / len_batch)
                if (iter + 1) % 10 == 0 and iter == self.client_ep - 1:
                    # Sending activations to edge and cloud
                    edge_start_time = time.time()
                    if self.edge_flag == 1:
                        edge_fx, edge_loss = Edge1.train_edge(global_ep, client_fx, labels, iter, self.client_ep,
                                                              self.idx, len_batch)
                    else:
                        edge_fx, edge_loss = Edge2.train_edge(global_ep, client_fx, labels, iter, self.client_ep,
                                                              self.idx, len_batch)
                    edge_total_time += time.time() - edge_start_time
                    edge_total_loss += edge_loss
                    if batch_idx == len_batch - 1:
                        self.edge_ep_time.append(edge_total_time)
                        self.edge_ep_loss.append(edge_total_loss / len_batch)
                    cloud_start_time = time.time()
                    train_cloud(global_ep, edge_fx, labels, iter, self.client_ep, self.idx, len_batch)
                    cloud_total_time += time.time() - cloud_start_time
                    if batch_idx == len_batch - 1:
                        self.cloud_ep_time.append(edge_total_time)
                    len_count += 1
                    if len_count != len_batch:
                        continue
                    else:
                        return net.state_dict(), net_auxiliary.state_dict(), \
                               self.client_ep_time, self.edge_ep_time, self.cloud_ep_time, self.client_ep_loss, self.edge_ep_loss

    def evaluate(self, net, ell, idx_count):
        net.eval()

        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (features, labels) in enumerate(self.ldr_test):
                features = torch.reshape(features, [-1, 1, 141])
                features, labels = features.to(self.device), labels.to(self.device)
                # ---------forward prop-------------
                fx = net(features)

                # Sending activations to edge
                if self.edge_flag == 1:
                    Edge1.evaluate_edge(fx, labels, self.idx, len_batch, batch_idx, ell, idx_count)
                else:
                    Edge2.evaluate_edge(fx, labels, self.idx, len_batch, batch_idx, ell, idx_count)

            # prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))

        return


# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
def dataset_iid(dataset, num_clients):
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# non_iid dataset
def dataset_non_iid(X_feature, y_label, num_clients):
    y_label = list(y_label)
    num_benigns = int(y_label.count(0) / num_clients)
    dict_users = {}
    dict_users[0] = set(i for i in range(len(X_feature)) if y_label[i] == 1 or y_label[i] == 2)
    dict_users[1] = set(i for i in range(len(X_feature)) if y_label[i] == 3 or y_label[i] == 4)
    dict_users[2] = set(i for i in range(len(X_feature)) if y_label[i] == 5 or y_label[i] == 6)
    dict_users[3] = set(i for i in range(len(X_feature)) if y_label[i] == 7)
    count = 0
    for i in range(num_clients):
        for j in range(num_benigns):
            while y_label[count] != 0:
                count += 1
            if count != len(X_feature) - 1:
                dict_users[i].add(count)
            count += 1
        if len(X_feature) - 1 in dict_users[i]:
            dict_users[i].discard(len(X_feature) - 1)
    # while count <= len(y_label) - 1:
    #     while y_label[count] != 0:
    #         count += 1
    #     dict_users[3].add(count)
    #     count += 1
    return dict_users
