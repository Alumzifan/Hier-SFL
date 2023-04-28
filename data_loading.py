import os
from glob import glob
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# 攻击种类的标签值
ATTACK_TYPES = {
    'Zero_Day': 0,
    'Adware': 1,
    'Backdoor': 2,
    'Trojan_Banker': 3,
    'Trojan_Dropper': 4,
    'FileInfector': 5,
    'No_Category': 6,
    'PUA': 7,
    'Ransomware': 8,
    'Riskware': 9,
    'Scareware': 10,
    'Trojan_SMS': 11,
    'Trojan_Spy': 12,
    'Trojan': 13,
}

categories = ['Adware', 'Backdoor', 'Trojan_Banker', 'Trojan_Dropper', 'FileInfector', 'No_Category', 'PUA',
              'Ransomware', 'Riskware', 'Scareware', 'Trojan_SMS', 'Trojan_Spy', 'Trojan', 'Zero_Day']

# 根据file读取数据
def readData(file):
    print("Loading raw data...")
    raw_data = pd.read_csv(file, header=None, low_memory=False)
    return raw_data


# 整合所有数据
def mergeData():
    Adware = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/Adware_after_reboot_Cat.csv")
    last_col = Adware.shape[1] - 1
    last_3_col = Adware.shape[1] - 3
    # 删除最后一列和倒数第三列
    Adware = Adware.drop([last_3_col, last_col], axis=1)
    # 删除第一行属性名称
    Adware = Adware.drop([0], axis=0)

    Backdoor = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/Backdoor_after_reboot_Cat.csv")
    Backdoor = Backdoor.drop([last_3_col, last_col], axis=1)
    Backdoor = Backdoor.drop([0], axis=0)

    Banker = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/Trojan_Banker_after_reboot_Cat.csv")
    Banker = Banker.drop([last_3_col, last_col], axis=1)
    Banker = Banker.drop([0], axis=0)

    Dropper = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/Trojan_Dropper_after_reboot_Cat.csv")
    Dropper = Dropper.drop([last_3_col, last_col], axis=1)
    Dropper = Dropper.drop([0], axis=0)

    FileInfector = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/FileInfector_after_reboot_Cat.csv")
    FileInfector = FileInfector.drop([last_3_col, last_col], axis=1)
    FileInfector = FileInfector.drop([0], axis=0)

    NoCategory = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/No_Category_after_reboot_Cat.csv")
    NoCategory = NoCategory.drop([last_3_col, last_col], axis=1)
    NoCategory = NoCategory.drop([0], axis=0)

    PUA = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/PUA_after_reboot_Cat.csv")
    PUA = PUA.drop([last_3_col, last_col], axis=1)
    PUA = PUA.drop([0], axis=0)

    Ransomware = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/Ransomware_after_reboot_Cat.csv")
    Ransomware = Ransomware.drop([last_3_col, last_col], axis=1)
    Ransomware = Ransomware.drop([0], axis=0)

    Riskware = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/Riskware_after_reboot_Cat.csv")
    Riskware = Riskware.drop([last_3_col, last_col], axis=1)
    Riskware = Riskware.drop([0], axis=0)

    Scareware = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/Scareware_after_reboot_Cat.csv")
    Scareware = Scareware.drop([last_3_col, last_col], axis=1)
    Scareware = Scareware.drop([0], axis=0)

    SMS = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/Trojan_SMS_after_reboot_Cat.csv")
    SMS = SMS.drop([last_3_col, last_col], axis=1)
    SMS = SMS.drop([0], axis=0)

    Spy = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/Trojan_Spy_after_reboot_Cat.csv")
    Spy = Spy.drop([last_3_col, last_col], axis=1)
    Spy = Spy.drop([0], axis=0)

    Trojan = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/Trojan_after_reboot_Cat.csv")
    Trojan = Trojan.drop([last_3_col, last_col], axis=1)
    Trojan = Trojan.drop([0], axis=0)

    Zeroday = readData("AndMal2020-Dynamic-BeforeAndAfterReboot/Zero_Day_after_reboot_Cat.csv")
    Zeroday = Zeroday.drop([last_3_col, last_col], axis=1)
    Zeroday = Zeroday.drop([0], axis=0)

    frame = [Adware, Backdoor, Banker, Dropper, FileInfector, NoCategory, PUA, Ransomware, Riskware, Scareware,
             SMS, Spy, Trojan, Zeroday]

    # 合并数据
    result = pd.concat(frame)
    result = clearDirtyData(result)
    return result


# 清除数据集中的值全为0的列
def clearDirtyData(df):
    df = df.loc[:, ~(df == 0).all(axis=0)]
    # 重整列序列
    df.index = range(len(df))
    return df


# raw_data = mergeData()
# file = 'AndMal2020-Dynamic-BeforeAndAfterReboot/total.csv'
# print(raw_data)
# raw_data.to_csv(file, index=False, header=False)


# 将大的数据集根据标签特征分为14类，存储到lists集合中
def separateData(raw_data):
    # dataframe数据转换为多维数组
    lists = raw_data.values.tolist()
    temp_lists = []

    # 生成14个空的list集合，用来暂存生成的14种特征集
    for i in range(0, 14):
        temp_lists.append([])

    # 得到raw_data的数据标签集合
    label_set = lookData(raw_data)

    # 将无序的数据标签集合转换为有序的list
    label_list = list(label_set)

    for i in range(0, len(lists)):
        # 得到所属标签的索引号
        data_index = label_list.index(lists[i][len(lists[0]) - 1])
        temp_lists[data_index].append(lists[i])
    saveData(temp_lists, 'AndMal2020-Dynamic-BeforeAndAfterReboot/expendData')
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


# lists存储着15类数据集，将数据集数量少的扩充到至少不少于1000条，然后存储起来。
def expendData(lists):
    total_list = []
    for i in range(0, len(lists)):
        while len(lists[i]) < 1000:
            lists[i].extend(lists[i])
        # lists[i] = lists[i][0:1000]
        total_list.extend(lists[i])
    saveData(lists, 'AndMal2020-Dynamic-BeforeAndAfterReboot/expendData')
    save = pd.DataFrame(total_list)
    file = 'AndMal2020-Dynamic-BeforeAndAfterReboot/expendData/total_extend.csv'
    save.to_csv(file, index=False, header=False)


# file = 'AndMal2020-Dynamic-BeforeAndAfterReboot/total.csv'
# raw_data = readData(file)
# raw_data.columns = [i for i in range(raw_data.shape[1])]
# print(raw_data)
# lists = separateData(raw_data)
# expendData(lists)


# ==============================================================
# Custom dataset prepration in Pytorch format
# 自定义有标签数据集
class CICData(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        m_data = self.data_frame.iloc[idx]
        m_data = m_data.astype('float')
        X = torch.tensor(list(m_data[:-1]))
        y = torch.tensor(int(m_data.iloc[-1]))

        return X, y

