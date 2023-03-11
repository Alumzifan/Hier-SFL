import os
from glob import glob
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# 攻击种类的标签值
ATTACK_TYPES = {
    'BENIGN': 0,
    'DrDoS_MSSQL': 1,
    'DrDoS_NTP': 2,
    'DrDoS_NetBIOS': 3,
    'DrDoS_SNMP': 4,
    'DrDoS_UDP': 5,
    'Syn': 6,
    'TFTP': 7,
}


# 根据file读取数据
def readData(file):
    print("Loading raw data...")
    raw_data = pd.read_csv(file, header=None, low_memory=False, nrows=150000)
    return raw_data


# 整合所有数据
def mergeData():
    MSSQL = readData("01-12/DrDoS_MSSQL.csv")

    # 剔除第一行属性特征名称
    # MSSQL = MSSQL.drop([0])
    NetBIOS = readData("01-12/DrDoS_NetBIOS.csv")
    # NetBIOS = NetBIOS.drop([0])
    NTP = readData("01-12/DrDoS_NTP.csv")
    # NTP = NTP.drop([0])
    SNMP = readData("01-12/DrDoS_SNMP.csv")
    # SNMP = SNMP.drop([0])
    UDP = readData("01-12/DrDoS_UDP.csv")
    # UDP = UDP.drop([0])
    Syn = readData("01-12/Syn.csv")
    # Syn = Syn.drop([0])
    TFTP = readData("01-12/TFTP.csv")
    # TFTP = TFTP.drop([0])
    frame = [MSSQL, NetBIOS, NTP, SNMP, UDP, Syn, TFTP]

    # 合并数据
    result = pd.concat(frame)
    list = clearDirtyData(result)
    result = result.drop(list)
    return result


# 清除数据集中的脏数据，第一行特征名称和含有Nan、Infiniti等数据的行数
def clearDirtyData(df):
    dropList = df[(df[21] == "") | (df[21] == "Infinity") | (df[22] == "Infinity")].index.tolist()
    return dropList


# raw_data = mergeData()
# file = '01-12/total.csv'
# raw_data.to_csv(file, index=False, header=False)

# 将时间戳数据调为浮点型
# raw_data = pd.read_csv('01-12/total.csv', header=None, low_memory=False)
# print(raw_data)
# print(len(raw_data))
# for i in range(len(raw_data)):
#     num_list = raw_data[2][i].split(':')
#     print(i)
#     num = int(num_list[0]) * 60 + float(num_list[1])
#     raw_data[2][i] = num
# print(raw_data)
# raw_data.to_csv('01-12/total.csv', index=False, header=False)



# 得到标签列索引
# last_column_index = raw_data.shape[1] - 1
# print(raw_data[last_column_index].value_counts())


# 将大的数据集根据标签特征分为8类，存储到lists集合中
def separateData(raw_data):
    # dataframe数据转换为多维数组
    lists = raw_data.values.tolist()
    temp_lists = []

    # 生成8个空的list集合，用来暂存生成的8种特征集
    for i in range(0, 8):
        temp_lists.append([])

    # 得到raw_data的数据标签集合
    label_set = lookData(raw_data)

    # 将无序的数据标签集合转换为有序的list
    label_list = list(label_set)

    for i in range(0, len(lists)):
        # 得到所属标签的索引号
        data_index = label_list.index(lists[i][len(lists[0]) - 1])
        temp_lists[data_index].append(lists[i])
    saveData(temp_lists, '01-12/expendData')
    return temp_lists


# 将lists分批保存到file文件路径下
def saveData(lists, file):
    label_set = lookData(raw_data)
    label_list = list(label_set)
    for i in range(0, len(lists)):
        save = pd.DataFrame(lists[i])
        file1 = file + label_list[i] + '.csv'
        save.to_csv(file1, index=False, header=False)


def lookData(raw_data):
    # 打印数据集的标签数据数量
    last_column_index = raw_data.shape[1] - 1
    print(raw_data[last_column_index].value_counts())

    # 取出数据集标签部分
    labels = raw_data.iloc[:, raw_data.shape[1] - 1:]

    # 多维数组转为以为数组
    labels = labels.values.ravel()
    label_set = set(labels)
    return label_set


# lists存储着8类数据集，将数据集数量少的扩充到至少不少于50000条，然后存储起来。
def expendData(lists):
    totall_list = []
    for i in range(0, len(lists)):
        while len(lists[i]) < 50000:
            lists[i].extend(lists[i])
        totall_list.extend(lists[i])
    saveData(lists, '01-12/expendData')
    save = pd.DataFrame(totall_list)
    file = '01-12/expendData/total_extend.csv'
    save.to_csv(file, index=False, header=False)


# file = '01-12/total.csv'
# raw_data = pd.read_csv(file, header=None, low_memory=False)
# lists = separateData(raw_data)
# expendData(lists)


# ==============================================================
# Custom dataset prepration in Pytorch format
# 自定义数据集
class CICData(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        m_data = self.data_frame.iloc[idx]
        m_data = m_data.astype('float')
        X = torch.tensor(list(m_data[:-1]))
        y = torch.tensor(int(m_data[-1]))

        return X, y