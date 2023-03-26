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

回归与分类的区别：单连续数值输出->多个输出，输出i为预测第i类的置信度

softmax操作子将置信度转换为概率，衡量概率的区别通过交叉熵进行，在化简后为-log(y^_y)，梯度就是真实概率与预测概率的区别
$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{其中}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}
$$
同样，在分类问题中也可以使用小批量样本的矢量化

O=XW+b, Y=softmax(O)

#### Softmax函数

softmax函数给出了一个向量$ \hat{\mathbf{y}} $，

我们可以将其视为“对给定任意输入$\mathbf{x}$的每个类的条件概率”。

例如，$ \hat{y}_1$=$P(y=\text{猫} \mid \mathbf{x})$。

假设整个数据集$\{\mathbf{X}, \mathbf{Y}\}$具有$n$个样本，

其中索引$ i $的样本由特征向量$\mathbf{x}^{(i)}$和独热标签向量$\mathbf{y}^{(i)}$组成。

我们可以将估计值与实际值进行比较：
$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$
根据最大似然估计，我们最大化$P(\mathbf{Y} \mid \mathbf{X})$，相当于最小化负对数似然：



$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})

= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$



其中，对于任何标签$\mathbf{y}$和模型预测$\hat{\mathbf{y}}$，损失函数为：

$$
l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j.
$$
这里使用交叉熵作为损失函数，符合极大似然估计（和线性回归也有类似之处）

关于softmax函数的导数，涉及到稍微有一点点复杂的数学运算，不做收录
$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$
得到梯度即为softmax模型分配的概率与实际情况之间的差异。

关于信息论的知识在此处值收录这段比喻：

- 如果把熵$H(P)$想象为“知道真实概率的人所经历的惊异程度”，那么什么是交叉熵？

交叉熵**从**$P$**到**$Q$，记为$H(P, Q)$。我们可以把交叉熵想象为“主观概率为$Q$的观察者在看到根据概率$P$生成的数据时的预期惊异”。当$P=Q$时，交叉熵达到最低。在这种情况下，从$P$到$Q$的交叉熵是$H(P, P)= H(P)$。（就像写MDAD时发现准确率接近50%时我的惊异最大w他妈的根本没有起到分类的作用啊（恼））

简而言之，我们可以从两方面来考虑交叉熵分类目标：

（i）最大化观测数据的似然；（ii）最小化传达标签所需的惊异。

#### 数据集

终于来到了大家都喜欢的`torchvision,transforms`这些包的使用了（喜

来过一遍流程罢

```python
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

最简单的预处理：图片文件改变格式为tensor

`transform`参数：在读取时默认的操作

此时分为训练集和测试集

```python
>>> mnist_train[0][0].shape
torch.Size([1, 28, 28])
```

为单通道28x28的图片,`[0][0]`的原因是第一阶为区分images和labels，第二阶为images内部的排序

```python
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
```

通过DataLoader可以读取图片内容和标签内容

```python
batch_size = 256

def get_dataloader_workers():  
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

通过DataLoader也可以设置小批量的读取方法，此处也设置了打乱顺序为true

读取数据集的内容大概到此为止，输出了API为DataLoader，以供神经网络的训练

#### Softmax 0-1

首先读入数据，这里的load_data是上方函数的总和

```python
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

由于无法直接处理图片特征，将图片展平为向量（丢失一些信息）

```python
num_inputs = 784 # 28x28 image
num_outputs = 10 # 10 types

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

这里的W为784列，10行，而输入为784列

***关于tensor的求和函数* **

```python
>>> X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
>>> X.sum(0, keepdim=True), X.sum(1, keepdim=True)

(tensor([[5., 7., 9.]]),
 tensor([[ 6.],
         [15.]]))
```

`sum()`中的dim具体指的是在操作的时候，将哪个维度的shape值变为1

0-将行数变为1，1-将列数变为1

回想一下，实现softmax由三个步骤组成：

1. 对每个项求幂（使用`exp`）；
2. 对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
3. 将每一行除以其规范化常数，确保结果的和为1。

因此批量的softmax操作函数如下：

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
```

每一行代表一个样本在各个种类的置信度

实现softmax回归如下：

```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

X为256x784的矩阵

继续定义损失函数：

***列表的高级索引***

```python
>>> y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
# 对0号样本，拿出第0个元素，对1号样本，拿出第2个元素
tensor([0.1000, 0.5000])
```

因此可以改进交叉熵损失函数：

```python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
# 由于y的0-1性，在这里省去，使公式更简洁
cross_entropy(y_hat, y)
```

继续实现工具函数：除了损失函数，也需要正确率函数

```python
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    # 判断：y_hat为二元矩阵，且列数大于1
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        # 找到每一行元数最大的项
    # 类型转换cmp
    cmp = y_hat.type(y.dtype) == y
    # 这里的转化是因为bool不可以sum，要转换为y对应的int型
    return float(cmp.type(y.dtype).sum())
  
accuracy(y_hat, y) / len(y)
```

扩展工具函数，使其可以计算某个模型的精度

```python
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)
	
  	# 当实例对象做p[key] 运算时，会调用类中的方法__getitem__
    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    # data_iter将为之前封装好的Dataloader迭代器
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式，不会计算梯度，不反向传播
    metric = Accumulator(2)  # 正确预测数、预测总数
    # Accumulator 累加器，使用方法见下（似乎只是对list改了一点API）
    with torch.no_grad():
        for X, y in data_iter:
          	# y.numel()为样本的总数
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

---

在完成了工具函数之后，开始搭建训练框架

```python
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数，正向传播
        y_hat = net(X)
        # 计算损失函数
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            # 清零优化器内已有的梯度
            updater.zero_grad()
            # 依据损失函数进行一次反向传播
            l.mean().backward()
            # 依据优化器进行一次更新梯度
            # updater封装了优化器，在优化器初始化时已经与模型绑定
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
```

*优化器的梯度清零是一个trick，由于在torch中会默认把梯度叠加，因此如果不清零的话会起到扩大效果的作用

同时，如果不清零的话会浪费空间（多保存计算图）

之后在对训练函数进一步抽象，给出训练所需的函数（迭代）

```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

在训练后，也给出进行实际预测的接口：

```python
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    # 真实标号
    trues = d2l.get_fashion_mnist_labels(y)
    # 预测标号
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

#### softmax-模板

同样，softmax也可以利用开源的库进行更快速的实现

1. 数据调用同上，没什么好改的

2. 构造模型：

```python
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状(保留第0维，其他均展开成向量，0维为batch维)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        
# apply会对每一层运行内部函数
net.apply(init_weights);

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

利用已有的API，可以更快的搭建模型