### 动手学深度学习之一：线性回归、Softmax与分类模型、多层感知器

本文是《动手学深度学习PyTorch版》的知识点整理与学习笔记。课程地址为[https://github.com/ShusenTang/Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch)。

#### 1 机器学习模型基本要素

一个机器学习模型的基本要素主要包含**数据集**、**模型**、**损失函数**和**优化函数**。

数据集通常指我们收集到的一系列真实数据，包含样本的真实标签与用来预测的特征集合。在这个给定的数据集上，我们需要构建一个合适的模型，将特征集合空间映射到标签值空间中。对于分类问题，模型的输出通常是{0, 1}二元变量；对于回归问题，模型输出则为实数。

在确定好模型之后，我们的目标便是估计模型中的参数。但是，现实生活中的数据总是存在着或多或少的噪声，且理论模型需要一定的条件假设，通常没有一个模型能够完美地拟合现实情况。所以，我们需要做一定的妥协，找出一组最好的参数，使得在给定数据集上，模型预测的结果与真实结果的偏差相对最小。我们用损失函数来表示这种偏差。

对于回归模型，我们常使用**平方损失**。它在评估索引为 $i$ 的样本误差的表达式为：
$$
l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2
$$

$$
L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2
$$

> **注意**：平方损失与一般的均方误差(MSE)不同，差了系数1/2。

对于分类模型，我们常使用交叉熵损失。其计算公式如下：
$$
H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)}
$$

$$
\ell(\boldsymbol{\Theta}) = \frac{1}{n} \sum_{i=1}^n H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right )
$$

当模型与损失函数的形式较为简单时，我们可以对上述的最小化误差问题求闭式解，来获取模型的最优参数。但是，即便是最简单的线性回归问题 $Y=A^TX$，其闭式解为 $\hat A=(X^TX)^{-1}X^TY$，包含了求逆这类复杂操作，模型的计算复杂度达到了 $O(n^3)$。当训练集样本量很大时，模型运算效率低下。而对于一些复杂模型，我们甚至难以求出其显式解。因而，我们常使用优化模型来迭代求最优解。在深度学习中，小批量随机梯度下降模型被广泛使用。其迭代公式为：
$$
(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b)
$$

$\eta$ 表示在每次优化中的学习率(也即步长)大小。
$\mathcal{B}$ 表示小批量计算中的批量大小。

#### 2 线性回归模型

在PyTorch中，线性回归模型针对每一条样本表示为：
$$
y = x W + b
$$

##### 2.1 从零开始实现线性回归模型

从零开始实现线性回归模型主要包含以下步骤：
- 生成数据集
- 按批量大小读取数据集
- 初始化模型参数并计算梯度
```python
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
```
- 定义模型：线性回归
- 定义损失函数：平方损失
```python
def squared_loss(y_hat, y):
	return (y_hat - y.view(y_hat.size())) ** 2 / 2
```
> y_hat的形状是[n, 1]，而y的形状是[n]，两者相减得到的结果的形状是[n, n]，相当于用y_hat的每一个元素分别减去y的所有元素，所以无法得到正确的损失值。
> y_hat.view(-1)的形状是[n]。y.view(y_hat.shape)和y.view(-1, 1)的形状都是[n, 1]。
> 所以上式可以改为：(y_hat.view(-1) - y) ** 2 / 2 或 (y_hat - y.view(y_hat.shape)) ** 2 / 2 或 
> (y_hat - y.view(-1, 1)) ** 2 / 2。

- 定义优化函数：随机梯度下降迭代公式
- 训练模型

  在每一次迭代中，1. 求预测值；2.正向传播求损失；3.梯度清零；4.反向传播求梯度；5.sgd一步迭代优化。
```python
# super parameters init
lr = 0.03
num_epochs = 5

net = linreg
loss = squared_loss

# training
for epoch in range(num_epochs):  # training repeats num_epochs times
    # in each epoch, all the samples in dataset will be used once
    
    # X is the feature and y is the label of a batch sample
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum() 
        # reset parameter gradient
        w.grad.data.zero_()
        b.grad.data.zero_()
        # calculate the gradient of batch sample loss 
        l.backward()  
        # using small batch random gradient descent to iter model parameters
        sgd([w, b], lr, batch_size)  
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
```

##### 2.2 pytorch版简洁实现

 pytorch版简洁实现线性回归模型主要包含如下步骤：

- 生成数据集：与从零开始完全相同
- 读取数据集：使用DataLoader函数

```python
import torch.utils.data as Data

batch_size = 10

# combine featues and labels of dataset
dataset = Data.TensorDataset(features, labels)

# put dataset into DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # whether shuffle the data or not
    num_workers=2,              # read data in multithreading
)
```
- 定义模型：包含\__init__() ，forward()函数
```python
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()      # call father function to init 
        self.linear = nn.Linear(n_feature, 1)  # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`

    def forward(self, x):
        y = self.linear(x)
        return y
    
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
```

- 初始化模型参数：使用init模块
- 定义损失函数：调用MSELoss()函数
- 定义优化函数：调用SGD函数
- 训练模型：1. 求预测值；2. 正向传播求损失；3. 梯度清零；4. 反向传播求梯度；5.sgd一步迭代优化。

```python
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # reset gradient, equal to net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
```

#### 2 Softmax与分类模型

#### 3 多层感知机
