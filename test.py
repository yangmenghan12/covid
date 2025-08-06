import matplotlib.pyplot as plt
import torch
import numpy as np
import csv
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch import optim
import time
from sklearn.feature_selection import SelectKBest, chi2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_feature_importance(feature_data, label_data, k =4,column = None):
    """
    此处省略 feature_data, label_data 的生成代码。
    如果是 CSV 文件，可通过 read_csv() 函数获得特征和标签。
    这个函数的目的是， 找到所有的特征种， 比较有用的k个特征， 并打印这些列的名字。
    """
    model = SelectKBest(chi2, k=k)      #定义一个选择k个最佳特征的函数
    feature_data = np.array(feature_data, dtype=np.float64)
    # label_data = np.array(label_data, dtype=np.float64)
    X_new = model.fit_transform(feature_data, label_data)   #用这个函数选择k个最佳特征
    #feature_data是特征数据，label_data是标签数据，该函数可以选择出k个特征
    print('x_new', X_new)
    scores = model.scores_                # scores即每一列与结果的相关性
    # 按重要性排序，选出最重要的 k 个
    indices = np.argsort(scores)[::-1]        #[::-1]表示反转一个列表或者矩阵。
    # argsort这个函数， 可以矩阵排序后的下标。 比如 indices[0]表示的是，scores中最小值的下标。

    if column:                            # 如果需要打印选中的列
        k_best_features = [column[i+1] for i in indices[0:k].tolist()]         # 选中这些列 打印
        print('k best features are: ',k_best_features)
    return X_new, indices[0:k]                  # 返回选中列的特征和他们的下标。

class CovidDataset(Dataset):
    def __init__(self, file_path, mode="train", all_feature=False, feature_dim=6):
        with open(file_path, "r") as f:
            ori_data = list(csv.reader(f))
            column = ori_data[0]
            csv_data = np.array(ori_data[1:])[:, 1:].astype(float)
        feature = np.array(ori_data[1:])[:, 1:-1]
        label_data = np.array(ori_data[1:])[:, -1]
        if all_feature:
            col = np.array([i for i in range(0, 93)])
        else:
            _, col = get_feature_importance(feature, label_data, feature_dim, column)
        col = col.tolist()
        if mode == "train":        #逢5取1.
            indices = [i for i in range(len(csv_data)) if i % 5 != 0]
            data = torch.tensor(csv_data[indices, :-1])
            self.y = torch.tensor(csv_data[indices, -1])
        elif mode == "val":
            indices = [i for i in range(len(csv_data)) if i % 5 == 0]
            data = torch.tensor(csv_data[indices, :-1])
            self.y = torch.tensor(csv_data[indices, -1])
        else:
            indices = [i for i in range(len(csv_data))]
            data = torch.tensor(csv_data[indices])
        data = data[:, col]
        self.data = (data- data.mean(dim=0, keepdim=True))/data.std(dim=0, keepdim=True)
        self.mode = mode
    def __getitem__(self, idx):
        if self.mode != "test":
            return self.data[idx].float(),  self.y[idx].float()
        else:
            return self.data[idx].float()

    def __len__(self):
        return len(self.data)




class MyModel(nn.Module):
    def __init__(self, inDim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(inDim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):      #模型前向过程
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        if len(x.size()) > 1:
            return x.squeeze(1)

        return x


def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path):
    model = model.to(device)

    plt_train_loss = [] #记录所有轮次的loss
    plt_val_loss = []

    min_val_loss = 9999999999999

    for epoch in range(epochs):   #冲锋的号角
        train_loss = 0.0
        val_loss = 0.0
        start_time = time.time()

        model.train()     #模型调为训练模式
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)
            pred = model(x)
            train_bat_loss = loss(pred, target,model)
            train_bat_loss.backward()
            optimizer.step() #更新模型的作用
            optimizer.zero_grad()
            train_loss += train_bat_loss.cpu().item()
        plt_train_loss.append(train_loss / train_loader.__len__())


        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_bat_loss = loss(pred, target,model)
                val_loss += val_bat_loss.cpu().item()
        plt_val_loss.append(val_loss/ val_loader.__len__())
        if val_loss < min_val_loss:
            torch.save(model, save_path)
            min_val_loss = val_loss

        print("[%03d/%03d] %2.2f sec(s) Trainloss: %.6f |Valloss: %.6f"% \
              (epoch, epochs, time.time()-start_time, plt_train_loss[-1], plt_val_loss[-1]))


    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss")
    plt.legend(["train", "val"])
    plt.show()



def evaluate(sava_path, test_loader,device,rel_path ):   #得出测试结果文件
    model = torch.load(sava_path).to(device)
    rel = []
    with torch.no_grad():
        for x in test_loader:
            pred = model(x.to(device))
            rel.append(pred.cpu().item())
    print(rel)
    with open(rel_path, "w",newline='') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(["id","tested_positive"])
        for i, value in enumerate(rel):
            csvWriter.writerow([str(i), str(value)])
    print("文件已经保存到"+rel_path)


all_feature = False
if all_feature:
    feature_dim = 93
else:
    feature_dim = 6

train_file = "covid.train.csv"
test_file = "covid.test.csv"

train_dataset = CovidDataset(train_file, "train",all_feature=all_feature, feature_dim=feature_dim)
val_dataset = CovidDataset(train_file, "val",all_feature=all_feature, feature_dim=feature_dim)
test_dataset = CovidDataset(test_file, "test",all_feature=all_feature, feature_dim=feature_dim)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# for batch_x, batch_y in train_loader:
#     print(batch_x, batch_y)

#
# predy = model(batch_x)

#
# file = pd.read_csv(train_file)
# print(file.head())


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

config = {
    "lr": 0.001,
    "epochs": 20,
    "momentum": 0.9,
    "save_path": "model_save/best_model.pth",
    "rel_path": "pred.csv"
}

def mseLoss_with_reg(pred, target, model):
    loss = nn.MSELoss(reduction='mean')
    ''' Calculate loss '''
    regularization_loss = 0                    # 正则项
    for param in model.parameters():
        # TODO: you may implement L1/L2 regularization here
        # 使用L2正则项
        # regularization_loss += torch.sum(abs(param))
        regularization_loss += torch.sum(param ** 2)                  # 计算所有参数平方
    return loss(pred, target) + 0.00075 * regularization_loss             # 返回损失。


model = MyModel(inDim=feature_dim).to(device)
loss = mseLoss_with_reg
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

train_val(model, train_loader, val_loader, device, config["epochs"], optimizer, loss, config["save_path"])

evaluate(config["save_path"], test_loader, device, config["rel_path"])