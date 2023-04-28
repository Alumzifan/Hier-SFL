import random
import numpy as np
import torch

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))

# ===================================================================
program = "result"
print(f"---------{program}----------")  # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


# ===================================================================
# No. of users
num_clients = 4
num_edges = 2
epochs = 50
client_ep = 100
frac = 1  # participation of clients; if 1 then 100% clients participate in SFLV1
lr = 0.001