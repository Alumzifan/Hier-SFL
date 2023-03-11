import copy
import torch
from torch import nn
from setup import prRed, prGreen, num_clients, device, lr, criterion
from cloud_model import net_glob_cloud

# For Cloud Side Loss and Accuracy
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

count1 = 0
count2 = 0


# ====================================================================================================
#                                  Cloud Side Program
# ====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_cloud = net_glob_cloud.state_dict()
w_locals_cloud = []

# client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
# Initialization of net_model_cloud and net_cloud (cloud-side model)
net_model_cloud = [net_glob_cloud for i in range(num_clients)]
net_cloud = copy.deepcopy(net_model_cloud[0]).to(device)


# optimizer_cloud = torch.optim.Adam(net_cloud.parameters(), lr = lr)

# Cloud-side function associated with Training
def train_cloud(global_ep, fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_cloud, criterion, optimizer_cloud, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_cloud, w_glob_cloud, net_cloud
    global loss_train_collect_user, acc_train_collect_user, lr

    net_cloud = copy.deepcopy(net_model_cloud[idx]).to(device)
    net_cloud.train()
    optimizer_cloud = torch.optim.Adam(net_cloud.parameters(), lr=lr)

    # train and update
    optimizer_cloud.zero_grad()

    fx_client = fx_client.to(device)
    y = y.to(device)

    # ---------forward prop-------------
    fx_cloud = net_cloud(fx_client)

    # calculate loss
    loss = criterion(fx_cloud, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_cloud, y)

    # --------backward prop--------------
    loss.backward()
    optimizer_cloud.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # Update the server-side model for the current batch
    net_model_cloud[idx] = copy.deepcopy(net_cloud)

    # count1: to track the completion of the local batch associated with one client
    count1 += 1

    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)  # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Global Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, global_ep, acc_avg_train,
                                                                                      loss_avg_train))

        # copy the last trained model in the batch
        w_cloud = net_cloud.state_dict()

        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch - 1:

            l_epoch_check = True  # to evaluate_server function - to check local epoch has completed or not
            # We store the state of the net_glob_server()
            w_locals_cloud.append(copy.deepcopy(w_cloud))

            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)

            # print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train

            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)

            # collect the id of each new user
            if idx not in idx_collect:
                idx_collect.append(idx)
                # print(idx_collect)

        # This is for federation process--------------------
        if len(idx_collect) == num_clients:
            fed_check = True  # to evaluate_cloud function  - to check fed check has hitted
            # Federation process at Cloud-Side------------------------- output print and update is done in evaluate_cloud()
            # for nicer display

            w_glob_cloud = FedAvg(w_locals_cloud)

            # cloud-side global model update and distribute that model to all clients ------------------------------
            net_glob_cloud.load_state_dict(w_glob_cloud)
            net_model_cloud = [net_glob_cloud for i in range(num_clients)]

            w_locals_cloud = []
            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

    return


# Cloud-side functions associated with Testing
def evaluate_cloud(fx_client, y, idx, len_batch, ell):
    global net_model_cloud, criterion, batch_acc_test, batch_loss_test, check_fed, net_cloud, net_glob_cloud
    global loss_test_collect, acc_test_collect, count2, num_clients, acc_avg_train_all, loss_avg_train_all, w_glob_cloud, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train

    net = copy.deepcopy(net_model_cloud[idx]).to(device)
    net.eval()

    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        # ---------forward prop-------------
        fx_cloud = net(fx_client)

        # calculate loss
        loss = criterion(fx_cloud, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_cloud, y)

        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())

        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)

            batch_acc_test = []
            batch_loss_test = []
            count2 = 0

            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test,
                                                                                             loss_avg_test))

            # if a local epoch is completed
            if l_epoch_check:
                l_epoch_check = False

                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test

                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

            # if federation is happened----------
            if fed_check:
                fed_check = False
                print("------------------------------------------------")
                print("------ Federation process at Cloud-Side ------- ")
                print("------------------------------------------------")

                acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)

                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user = []

                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train,
                                                                                          loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user,
                                                                                         loss_avg_all_user))
                print("==========================================================")

    return