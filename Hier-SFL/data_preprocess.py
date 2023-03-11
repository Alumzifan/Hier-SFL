import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from client_program import dataset_iid, dataset_non_iid
from data_loading import CICData
from setup import num_clients

# 加载数据
raw_data_filename = "01-12/expendData/total_extend.csv"
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
raw_data[last_column_index], attacks = pd.factorize(raw_data[last_column_index], sort=True)

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
X_train.to_csv('01-12/expendData/train_data.csv', index=False, header=False)
X_test.to_csv('01-12/expendData/test_data.csv', index=False, header=False)

print("X_train,y_train:", X_train.shape, y_train.shape)
print("X_test,y_test:", X_test.shape, y_test.shape)

dataset_train = CICData('01-12/expendData/train_data.csv', transform=None)
dataset_test = CICData('01-12/expendData/test_data.csv', transform=None)

# ----------------------------------------------------------------
dict_users = dataset_iid(dataset_train, num_clients)
dict_users_test = dataset_iid(dataset_test, num_clients)
dict_users_non_iid = dataset_non_iid(X_train, y_train, num_clients)
dict_users_test_non_iid = dataset_non_iid(X_test, y_test, num_clients)
