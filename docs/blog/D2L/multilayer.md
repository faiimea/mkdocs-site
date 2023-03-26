# 多层感知机

## What is MLT

由于线性层无法解决非线性的问题，因此通过在网络中加入一个或多个隐藏层来克服线性模型的限制， 使其能处理更普遍的函数关系类型。

最简单的方法是将许多全连接层堆叠在一起。 每一层都输出到上面的层，直到生成最后的输出。 这种架构称为**多层感知机**

同样，如果隐藏层中只包含线性变换，则无论堆叠多少layer都无济于事，因为都会等效为一个线性层。

为了发挥多层架构的潜力， 我们还需要一个额外的关键要素： 在仿射变换之后对每个隐藏单元应用非线性的*激活函数*（activation function）σ。 激活函数的输出（例如，σ(⋅)）被称为*活性值*（activations）。 一般来说，有了激活函数，就不可能再将我们的多层感知机退化成线性模型：

 事实上，通过使用更深（而不是更广）的网络，我们可以更容易地逼近许多函数。 

### 常见激活函数

*激活函数*（activation function）通过计算加权和并加上偏置来确定神经元是否应该被激活， 它们将输入信号转换为输出的可微运算。 大多数激活函数都是非线性的。 

***修正线性单元*（Rectified linear unit，*ReLU*）**

ReLU函数被定义为该元素与0的最大值：**$$\operatorname{ReLU}(x) = \max(x, 0).$$**

ReLU函数通过将相应的活性值设为0，仅保留正元素并丢弃所有负元素。在此之上延伸了pReLU等函数

使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。 这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题。

***挤压函数*（squashing function，*sigmoid*）**

它将范围（-inf, inf）中的任意输入压缩到区间（0, 1）中的某个值：**$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$**

当人们逐渐关注到到基于梯度的学习时， sigmoid函数是一个自然的选择，因为它是一个平滑的、可微的阈值单元近似。 其实是一个阶跃函数的平滑版本。

BTW，输入接近于0时，sigmoid接近于线性函数

注意，当输入为0时，sigmoid函数的导数达到最大值0.25； 而输入在任一方向上越远离0点时，导数越接近0。

以及还有tanh函数（类似于sigmoid）等

## MLT：0-1

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 用张量表示参数
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)
  
loss = nn.CrossEntropyLoss(reduction='none')

# 优化器的使用，params可用model.parameters()代替 lr is learn_rate
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

d2l.predict_ch3(net, test_iter)
```

## MLT-模板

```python
# 添加了2个全连接层（之前我们只添加了1个全连接层）。 第一层是隐藏层，它包含256个隐藏单元，并使用了ReLU激活函数。 第二层是输出层。
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 模型选择

### 概念

我们的目标是发现某些模式， 这些模式捕捉到了我们训练集潜在总体的规律。 如果成功做到了这点，即使是对以前从未遇到过的个体， 模型也可以成功地评估风险。 如何发现可以泛化的模式是机器学习的根本问题。

将模型在训练数据上拟合的比在潜在分布中更接近的现象称为***过拟合***（overfitting）， 用于对抗过拟合的技术称为***正则化***（regularization）。（在实验中调整模型架构或超参数时会发现： 如果有足够多的神经元、层数和训练迭代周期， 模型最终可以在训练集上达到完美的精度，此时测试集的准确性却下降了。）

***训练误差***（training error）是指， 模型在训练数据集上计算得到的误差。 ***泛化误差***（generalization error）是指， 模型应用在同样从原始样本的分布中抽取的无限多数据样本时，模型误差的期望。

在我们目前已探讨、并将在之后继续探讨的监督学习情景中， 我们假设训练数据和测试数据都是从相同的分布中独立提取的。 这通常被称为***独立同分布假设***（i.i.d. assumption）， 这意味着对数据进行采样的过程没有进行“记忆”。然而，实际情况下并不会总是iid，因此会出现泛化误差。

在考虑模型复杂性时，**可调整参数的数量，参数可能的取值范围，训练样本的数量**都会影响到最后模型的拟合程度

### 模型选择

*事实上在使用时的test_data很多时候都是验证数据集

#### 验证集

将我们的数据分成三份， 除了训练和测试数据集之外，还增加一个*验证数据集*（validation dataset）， 也叫*验证集*（validation set）。通过训练集来进行模型的训练，验证集调整超参数，测试集进行最终测试

#### K折交叉验证

当训练数据稀缺时，我们甚至可能无法提供足够的数据来构成一个合适的验证集。 这个问题的一个流行的解决方案是采用K*折交叉验证*。 这里，原始训练数据被分成K个不重叠的子集。 然后执行K次模型训练和验证，每次在K−1个子集上进行训练， 并在剩余的一个子集（在该轮中没有用于训练的子集）上进行验证。 最后，通过对K次实验的结果取平均来估计训练和验证误差。

当然也会增加训练的时间，不过减小过拟合的可能

### 拟合

**模型容量需要匹配数据复杂度**

首先，我们要注意这样的情况：训练误差和验证误差都很严重， 但它们之间仅有一点差距。 如果模型不能降低训练误差，这可能意味着模型过于简单（即表达能力不足）， 无法捕获试图学习的模式。

 此外，由于我们的训练和验证误差之间的*泛化误差*很小， 我们有理由相信可以用一个更复杂的模型降低训练误差。 这种现象被称为*欠拟合*（underfitting）。

另一方面，当我们的训练误差明显低于验证误差时要小心， 这表明严重的*过拟合*（overfitting）。 注意，*过拟合*并不总是一件坏事。 特别是在深度学习领域，众所周知， 最好的预测模型在训练数据上的表现往往比在保留（验证）数据上好得多。 最终，我们通常更关心验证误差，而不是训练误差和验证误差之间的差距。

统计学习的**VC维**，代表模型可以记住的最大数据量，即可以完美分类的最大数据集

如2维输入的感知机，可以完美解决三个点的分类，而对四个点则不行（xor），但在深度学习中难以计算VC维

同时，实际数据集的大小和复杂度也会影响到拟合的效果（样本个数，每个样本的元素数，时空结构，多样性）

（如果没有足够的数据，简单的模型可能更有用。 对于许多任务，深度学习只有在有数千个训练样本时才优于线性模型。 从一定程度上来说，深度学习目前的生机要归功于 廉价存储、互联设备以及数字化经济带来的海量数据集。）

### 权重衰减

> 一种常见的正则化技术

收集更多的训练数据想必可以一定程度上缓解过拟合，但相比于正则化总是更昂贵的

在多项式回归中，可以通过调整拟合多项式的阶数来限制模型容量（即限制特征数量），然而在高维数据输入时，由于大量交叉项的存在，使阶数的操作对网络的性能影响很大，可能在过简单和过复杂之间循环

在训练参数化机器学习模型时， ***权重衰减***（weight decay）是最广泛使用的正则化的技术之一， 它通常也被称为L2*正则化*。 这项技术**通过函数与零的距离来衡量函数的复杂度**， 因为在所有函数f中，函数f=0（所有输入都得到值0） 在某种意义上是最简单的。

//考虑到出于实践的目的学习，忽略在权值衰减中对L2范数正则化效果的数学证明，集中在实用性上

一种简单的方法是通过线性函数$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$中的权重向量的某个范数来度量其复杂性，例如$\| \mathbf{w} \|^2$。

要保证权重向量比较小，最常用方法是将其范数作为惩罚项加到最小化损失的问题中。将原来的训练目标**最小化训练标签上的预测损失**，调整为**最小化预测损失和惩罚项之和**。与罚函数的思路相似

因此，新的损失函数如下：

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

而通过梯度下降的数学推导，得到新的迭代式如下：
$$
\begin{aligned}

\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).

\end{aligned}
$$
即对于原来的w，在每次迭代时进行一次权重的衰减缩放。（通常，偏置项不会被正则化）

```python
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            # 在这里更新损失函数之后再反向传播
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())
```

至于效果。。反正是好了一些吧，不收录了

### 暂退法(Dropout)

泛化性和灵活性之间的这种基本权衡被描述为*偏差-方差权衡*（bias-variance tradeoff）。 

线性模型有很高的偏差：它们只能表示一小类函数。 然而，这些模型的方差很低：它们在不同的随机数据样本上可以得出相似的结果。

深度神经网络位于偏差-方差谱的另一端。 与线性模型不同，神经网络并不局限于单独查看每个特征，而是学习特征之间的交互。 

对于泛化性，我们期待模型在未知的数据中也能有不错的表现：经典泛化理论认为，为了缩小训练和测试性能之间的差距，应该以简单的模型为目标，参数的范数也代表了一种有用的简单性度量（权重衰减本质上还是对简单性的讨论）。

简单性的另一个角度是平滑性，即函数不应该对其输入的微小变化敏感。 例如，当我们对图像进行分类时，我们预计向像素添加一些随机噪声应该是基本无影响的。Tikhonov正则化描述了这一点。

因此，在训练过程中加入噪声也有助于解决过拟合问题：在训练过程中，在计算后续层之前向网络的每一层注入噪声。 因为当训练一个有多层的深层网络时，注入噪声只会在输入-输出映射上增强平滑性。这个想法被称为***暂退法***（dropout），因为我们从表面上看是在训练过程中丢弃（drop out）一些神经元。
$$
\begin{aligned}

h' =

\begin{cases}

​    0 & \text{ 概率为 } p \\

​    \frac{h}{1-p} & \text{ 其他情况}

\end{cases}

\end{aligned}
$$
根据此模型的设计，其期望值保持不变，即E[h′]=h。

在一层线性模型时，暂退法效果不好，而在具有隐藏层的MLP中可以使用dropout：**以一定概率删除隐藏层中的神经元，并撤销其联系**

```python
import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    # 与X大小相同的bool向量，转换为浮点数
    mask = (torch.rand(X.shape) > dropout).float()
    # 随机舍弃X中的一些元素
    return mask * X / (1.0 - dropout)
```

```python
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5
# net的class写法，不用torch的接口
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

使用更简单的API时：

```python
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

后面给了一些前向传播和反向传播的数学推导，证明了BP算法可以减小计算量

“因此，在训练神经网络时，在初始化模型参数后， 我们交替使用前向传播和反向传播，利用反向传播给出的梯度来更新模型参数。 注意，反向传播重复利用前向传播中存储的中间值，以避免重复计算。 带来的影响之一是我们需要保留中间值，直到反向传播完成。 这也是训练比单纯的预测需要更多的内存（显存）的原因之一。 此外，这些中间值的大小与网络层的数量和批量的大小大致成正比。 因此，使用更大的批量来训练更深层次的网络更容易导致*内存不足*（out of memory）错误。”

### 数值稳定性&模型初始化

初始化方案的选择在神经网络学习中起着举足轻重的作用， 它对保持数值稳定性至关重要。

**梯度消失与梯度爆炸**

在深层网络中传播误差时，由于要进行大量的矩阵乘法，可能会出现梯度爆炸（过高）与梯度消失（过低）的情况。不稳定梯度带来的风险不止在于数值表示； 不稳定梯度也威胁到我们优化算法的稳定性。 我们可能面临一些问题。 要么是*梯度爆炸*（gradient exploding）问题： 参数更新过大，破坏了模型的稳定收敛； 要么是*梯度消失*（gradient vanishing）问题： 参数更新过小，在每次更新时几乎不会移动，导致模型无法学习。

sigmoid函数是梯度消失的一个常见原因：当sigmoid函数的输入很大或是很小时，它的梯度都会消失。 此外，当反向传播通过许多层时，除非我们在刚刚好的地方， 这些地方sigmoid函数的输入接近于零，否则整个乘积的梯度可能会消失。因此更稳定的ReLU更受欢迎。

**权重初始化**

需要在合理值区间内随机初始参数，由于训练开始的时候更容易出现数值不稳定的情况（远离最优解时函数情况复杂）

使用标准正态分布进行初始化难以保证复杂网络的初始化需求。

考虑到大量层间运算后，采用*Xavier初始化*，以保证经过每层的运算后，前向传播（下一层数据）与后向传播（梯度）的输出都是期望为0，方差为1的分布。数学上无法满足，因此用Xavier初始化进行金丝，得到一些比较好的初始化分布。

同时，在判断激活函数优劣性时也可以用Xavier判断，可以发现在0处最好满足y=x，通过对常见函数的Taylor展开发现ReLU，tanhx等函数均满足，而sigmoid不满足，因此会产生梯度问题

*slice函数，在k折验证中用于对列表进行裁切