# 线性神经网络

## 线性回归-预测

线性回归-自变量与因变量的线性关系

线性模型-对特征进行线性变换

* 损失函数（最小二乘法）

二阶范数

* 线性回归解析解

`w∗=(X⊤X)−1X⊤y.`

* 随机梯度下降

矩阵求导等应用

* 小批量梯度下降

设置batch，减少计算梯度对内存的占用

* 矢量化加速-numpy，torch等开源库

* 正态分布与平方损失

通过假设噪声分布为高斯分布，得出极大似然估计下的解与最小二乘法的解相同

* 全连接层：每个输入都与每个输出相连
* 输入层的输入数称为特征维度dim

### 0实现

`y.reshape(-1,1)`此处的-1代表的是x，reshape通过另一个参数与y原本的性质计算-1处的值

此时代表将y转换为列向量

`torch.normal(means, std, out=None)`

eg：`X = torch.normal(0, 1, (3, 2))`	out即具体的输出张量，三行两列

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

定义data分组的迭代器：

其中`random.shuffle(x)`会将列表x中的顺序打乱，之后将每个batch的index读取进一个tensor，用yield来修饰以便在for循环中迭代

```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

由于w和b需要梯度下降法，因此保留过程中的梯度`requires_grad=True`,w.shape=(2,1)

优化算法写法：

```python
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size # /batch_size just for mean
            param.grad.zero_()
```

`no_grad()`更新时不要进行梯度计算，保留梯度的变量会将梯度保存在`x.grad`中

pytorch不会自动把梯度归0，因此在每次重启梯度时要用`x.grad.zero_()`

训练过程：

```python
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

注意不需要梯度的地方使用`no_grad()`，以及在每次训练时的`backward()`

`sum().backward()`只是一个简化计算的方法，与实际意义无关

### Pytorch实现

数据的生成相同，pytorch对数据迭代器有更好的支持

```python
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

Dataset与Dataloader更加便捷

关于模型，torch内部的nn包提供标准方便的神经网络写法

```python
# nn是神经网络的缩写
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

`nn.Linear`代表输入层，唯一需要的参数就是输入dim与输出层，而`nn.Sequential()`是一个便于管理神经网络层的list类工具，因此`net[0]`就是linear层。

由于线性层内置了weight和bias的知识，因此可以直接修改。`normal_(),fill_()`可以通过调用直接改变该层参数。

同样，torch也内置了许多常见的损失函数和优化方法，如此处使用的MSE（平方范数）与SGD（梯度下降）

optim包提供这样的功能，`net.parameters()`一次性选中神经网络的全部需要训练的参数，而lr只需要自定义学习率即可

```python
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

写法稍微有点奇怪，不过无伤大雅

考虑到鲁棒性，torch中也需要每次训练后手动清零内部的梯度

对于每一个小批量，我们会进行以下步骤:

- 通过调用`net(X)`生成预测并计算损失`l`（前向传播）。
- 通过进行反向传播来计算梯度。
- 通过调用优化器来更新模型参数。

## Softmax回归-分类