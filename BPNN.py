import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# 导入包
import matplotlib.pyplot as plt
import numpy as np


# 构造数据
# 位置 （2维：x,y一一对应）


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32, skiprows=1)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('all.csv')
train_loader = DataLoader(dataset=dataset,
                          batch_size=64,
                          shuffle=True,
                          num_workers=0)
test_dataset = DiabetesDataset('west.csv')

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(7, 9)
        self.linear2 = torch.nn.Linear(9, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x


model = Model()
criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

mseloss = []
n = 0
for epoch in range(200):
    for i, data in enumerate(train_loader, 0):
        n += 1
        inputs, labels = data  # 1. Prepare data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())  # 2. Forward
        mseloss = np.append(mseloss,loss.item())
        optimizer.zero_grad()
        loss.backward()  # 3. Backward
        optimizer.step()  # 4. Update

x = np.arange(0,n)
y = mseloss

c = 'red'
c = 'r'
c = '#FF0000'
# 大小（0维）: 线宽
lw = 1

fig, ax = plt.subplots()
# 在生成的坐标系下画折线图
ax.plot(x, y, c, linewidth=lw)
# 显示图形
plt.show()

