import copy
import time
import numpy as np
from pandas import DataFrame
from client_model import net_glob_client, net_glob_client_auxiliary
from client_program import Client
from data_preprocess import dataset_train, dataset_test, dict_users, dict_users_test, dict_users_non_iid, dict_users_test_non_iid
from cloud_program import FedAvg, acc_train_collect, acc_test_collect, loss_train_collect, loss_test_collect
from edge_model import net_glob_edge, net_glob_edge_auxiliary, net_glob_edge2, net_glob_edge2_auxiliary
from setup import epochs, frac, num_clients, lr, device, program, client_ep, batch_size

# ------------ Training And Testing  -----------------
net_glob_client.train()
net_glob_client_auxiliary.train()
net_glob_edge.train()
net_glob_edge_auxiliary.train()
net_glob_edge2.train()
net_glob_edge2_auxiliary.train()
# copy weights
w_glob_client = net_glob_client.state_dict()
w_glob_client_auxiliary = net_glob_client_auxiliary.state_dict()
w_glob_edge = net_glob_edge.state_dict()
w_glob_edge_auxiliary = net_glob_edge_auxiliary.state_dict()
w_glob_edge2 = net_glob_edge2.state_dict()
w_glob_edge2_auxiliary = net_glob_edge2_auxiliary.state_dict()
# count training time
start_time = time.time()
clients_total_ep_time = []
edges_total_ep_time = []
cloud_total_ep_time = []
clients_total_ep_loss = []
edges_total_ep_loss = []
# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds
for iter in range(epochs):
    edge_flag = 0
    m = max(int(frac * num_clients), 1)
    idxs_users = np.random.choice(range(num_clients), m, replace=False)
    w_locals_client = []
    w_locals_client_auxiliary = []
    w_locals_edge = []
    w_locals_edge_auxiliary = []
    w_locals_edge2 = []
    w_locals_edge2_auxiliary = []
    clients_running_time = []
    edges_running_time = []
    cloud_running_time = []
    clients_ep_loss = []
    edges_ep_loss = []

    for idx in idxs_users:
        if idx == 0 or idx == 1:
            edge_flag = 1
        else:
            edge_flag = 2
        local = Client(net_glob_client, client_ep, idx, lr, device, batch_size, edge_flag, dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx])
        # Training ------------------
        w_client, w_client_auxiliary, w_edge, w_edge_auxiliary, client_running_time, edge_running_time, \
        cloud_running_time, client_ep_loss, edge_ep_loss = local.train(iter,
                                                    net=copy.deepcopy(net_glob_client).to(device),
                                                    net_auxiliary=copy.deepcopy(net_glob_client_auxiliary).to(device))
        w_locals_client.append(copy.deepcopy(w_client))
        w_locals_client_auxiliary.append(copy.deepcopy(w_client_auxiliary))
        if edge_flag == 1:
            w_locals_edge.append(copy.deepcopy(w_edge))
            w_locals_edge_auxiliary.append(copy.deepcopy(w_edge_auxiliary))
        else:
            w_locals_edge2.append(copy.deepcopy(w_edge))
            w_locals_edge2_auxiliary.append(copy.deepcopy(w_edge_auxiliary))
        clients_running_time.append(client_running_time)
        edges_running_time.append(edge_running_time)
        clients_ep_loss.append(client_ep_loss)
        edges_ep_loss.append(edge_ep_loss)

        # Testing -------------------
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)
        # else:
        #     local = Client(net_glob_client, client_ep, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
        #                    idxs=dict_users[idx], idxs_test=dict_users_test[idx])
        #     # Training ------------------
        #     w_client, w_client_auxiliary, w_edge, w_edge_auxiliary, client_running_time, edge_running_time, \
        #     cloud_running_time, client_ep_loss, edge_ep_loss = local.train(iter,
        #                                                 net=copy.deepcopy(net_glob_client).to(device),
        #                                                 net_auxiliary=copy.deepcopy(net_glob_client_auxiliary).to(device))
        #     w_locals_client.append(copy.deepcopy(w_client))
        #     w_locals_client_auxiliary.append(copy.deepcopy(w_client_auxiliary))
        #     w_locals_edge.append(copy.deepcopy(w_edge))
        #     w_locals_edge_auxiliary.append(copy.deepcopy(w_edge_auxiliary))
        #     clients_running_time.append(client_running_time)
        #     edges_running_time.append(edge_running_time)
        #     clients_ep_loss.append(client_ep_loss)
        #     edges_ep_loss.append(edge_ep_loss)
        #
        #     # Testing -------------------
        #     local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)

    # After serving all clients for its local epochs------------
    # Fed  Server: Federation process at Client-Side-----------
    print("-----------------------------------------------------------")
    print("------ FedServer: Federation process at Client-Side and Edge-Side ------- ")
    print("-----------------------------------------------------------")
    w_glob_client = FedAvg(w_locals_client)
    w_glob_client_auxiliary = FedAvg(w_locals_client_auxiliary)
    w_glob_edge = FedAvg(w_locals_edge)
    w_glob_edge_auxiliary = FedAvg(w_locals_edge_auxiliary)
    w_glob_edge2 = FedAvg(w_locals_edge2)
    w_glob_edge2_auxiliary = FedAvg(w_locals_edge2_auxiliary)

    # Update client-side and edge-side global model
    net_glob_client.load_state_dict(w_glob_client)
    net_glob_client_auxiliary.load_state_dict(w_glob_client_auxiliary)
    net_glob_edge.load_state_dict(w_glob_edge)
    net_glob_edge_auxiliary.load_state_dict(w_glob_edge_auxiliary)
    net_glob_edge2.load_state_dict(w_glob_edge2)
    net_glob_edge2_auxiliary.load_state_dict(w_glob_edge2_auxiliary)

    # calculate time and loss
    for i in range(client_ep):
        max_client_time = max(clients_running_time[0][i], clients_running_time[1][i], clients_running_time[2][i], clients_running_time[3][i])
        max_edge_time = max(edges_running_time[0][i], edges_running_time[1][i], edges_running_time[2][i], edges_running_time[3][i])
        max_client_loss = max(clients_ep_loss[0][i], clients_ep_loss[1][i], clients_ep_loss[2][i], clients_ep_loss[3][i])
        max_edge_loss = max(edges_ep_loss[0][i], edges_ep_loss[1][i], edges_ep_loss[2][i],edges_ep_loss[3][i])
        clients_total_ep_time.append(max_client_time)
        edges_total_ep_time.append(max_edge_time)
        cloud_total_ep_time.append(cloud_running_time[i])
        clients_total_ep_loss.append(max_client_loss)
        edges_total_ep_loss.append(max_edge_loss)

elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')

print("Training and Evaluation completed!")

# acc and loss of train and test
acc_train = []
acc_test = []
loss_train = []
loss_test = []
for i in range(len(acc_train_collect)):
    for j in range(client_ep - 1):
        acc_train.append(0)
        acc_test.append(0)
        loss_train.append(0)
        loss_test.append(0)
    acc_train.append(acc_train_collect[i])
    acc_test.append(acc_test_collect[i])
    loss_train.append(loss_train_collect[i])
    loss_test.append(loss_test_collect[i])

# ===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, epochs * client_ep + 1)]
# df = DataFrame({'round': round_process, 'acc_train': acc_train_collect, 'acc_test': acc_test_collect})
df = DataFrame({'round': round_process, 'client_train_time': clients_total_ep_time, 'client_local_loss': clients_total_ep_loss,
                'edge_train_time':edges_total_ep_time, 'edge_local_loss':edges_total_ep_loss,
                'cloud_train_time':cloud_total_ep_time, 'acc_train':acc_train, 'acc_test':acc_test, 'loss_train':loss_train,
                'loss_test':loss_test})
file_name = program + ".xlsx"
df.to_excel(file_name, sheet_name="v1_test", index=False)
