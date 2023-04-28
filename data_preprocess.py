import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from data_loading import CICData, categories
from setup import num_clients

# 加载数据
raw_data_filename = "AndMal2020-Dynamic-BeforeAndAfterReboot/expendData/total_extend_all.csv"
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_filename, header=None, low_memory=False)

# 随机抽取比例，当数据集比较大的时候，可以采用这个，可选项
raw_data = raw_data.sample(frac=0.1)
print(raw_data)

# 查看标签数据情况
last_column_index = raw_data.shape[1] - 1
print("print data labels:")
print(raw_data[last_column_index].value_counts())

# 将非数值型的数据转换为数值型数据
print("Transforming data...")
cat = pd.Categorical(raw_data[last_column_index], categories=categories)
raw_data[last_column_index], attacks = pd.factorize(cat)

# 对原始数据进行切片，分离出特征和标签
features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]

# 特征数据标准化，这一步是可选项
# 获取每一列特征名
names = raw_data.columns
# 创建标准化对象
scaler = preprocessing.StandardScaler()
# scaler = preprocessing.MinMaxScaler()
# 处理数据
features = scaler.fit_transform(features)
features = pd.DataFrame(features)
labels = labels.reset_index(drop=True)
labels = pd.DataFrame(labels)
raw_data = pd.concat([features, labels], axis=1)

# raw_data = scaler.fit_transform(raw_data)
# raw_data = pd.DataFrame(raw_data, columns=names)

# raw_data = preprocessing.scale(raw_data)
# raw_data = pd.DataFrame(raw_data)

# 将多维的标签转为一维的数组
labels = labels.values.ravel()

# 将数据分为训练集和测试集,并打印维数
df = pd.DataFrame(raw_data)
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2, stratify=labels)
X_train.to_csv('AndMal2020-Dynamic-BeforeAndAfterReboot/expendData/train_data.csv', index=False, header=False)
X_test.to_csv('AndMal2020-Dynamic-BeforeAndAfterReboot/expendData/test_data.csv', index=False, header=False)

print("X_train,y_train:", X_train.shape, y_train.shape)
print("X_test,y_test:", X_test.shape, y_test.shape)

dataset_train = CICData('AndMal2020-Dynamic-BeforeAndAfterReboot/expendData/train_data.csv', transform=None)
dataset_test = CICData('AndMal2020-Dynamic-BeforeAndAfterReboot/expendData/test_data.csv', transform=None)


# ----------------------------------------------------------------
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


dict_users = dataset_iid(dataset_train, num_clients)
dict_users_test = dataset_iid(dataset_test, num_clients)
dict_users_non_iid = dataset_non_iid(X_train, y_train, num_clients)
dict_users_test_non_iid = dataset_non_iid(X_test, y_test, num_clients)
