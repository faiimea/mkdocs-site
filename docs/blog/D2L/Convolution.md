# 卷积神经网络

## 卷积

我们之前讨论的多层感知机十分适合处理表格数据，其中行对应样本，列对应特征。 对于表格数据，我们寻找的模式可能涉及特征之间的交互，但是我们不能预先假设任何与特征交互相关的先验结构。 此时，多层感知机可能是最好的选择，然而对于高维感知数据，这种缺少结构的网络可能会变得不实用。

然而，如今人类和机器都能很好地区分猫和狗：这是因为图像中本就拥有丰富的结构，而这些结构可以被人类和机器学习模型使用。 ***卷积神经网络*（convolutional neural networks，CNN）**是机器学习利用自然图像中一些已知结构的创造性方法。

## 从全连接层到卷积

例如，在之前猫狗分类的例子中：假设我们有一个足够充分的照片数据集，数据集中是拥有标注的照片，每张照片具有百万级像素，这意味着网络的每次输入都有一百万个维度。 即使将隐藏层维度降低到1000，这个全连接层也将有10^6×10^3=10^9个参数。 想要训练这个模型将不可实现，因为需要有大量的GPU、分布式优化训练的经验和超乎常人的耐心。

因此，需要想办法在缩减神经网络的连接数的同时，尽可能提取图像识别的特征。考虑到在真实生活中，人类试图从图片中分离物体时的一些**常识**，作为压缩信息的假设。卷积神经网络正是将***空间不变性*（spatial invariance）**的这一概念系统化，从而基于这个模型使用较少的参数来学习有用的表示。

1. ***平移不变性*（translation invariance）**：不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“平移不变性”。
2. ***局部性*（locality）**：神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。

经过对全连接层公式（4-dim权重，2-dim输入，2-dim输出）的一系列数学推导与化简，可以得到一种新的运算方式，应用在神经网络中，称为卷积层
$$
[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.
$$
当然，称之为卷积是因为这种对于图像的运算与信号处理中的卷积运算有类似之处。

在数学中，两个函数（比如$f, g: \mathbb{R}^d \to \mathbb{R}$）之间的“卷积”被定义为
$$
(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.
$$
也就是说，卷积是当把一个函数“翻转”并移位$\mathbf{x}$时，测量$f$和$g$之间的重叠。事实上，在信号处理的过程中，我会这样去理解卷积运算：f是一个变化的输入信号，g代表这个信号在系统中随时间的衰减过程，而f*g描述了一个信号在经过系统后某一时刻的值，是该时刻之前所有的信号在此刻的衰减后的值之和。

当为离散对象时，积分就变成求和。例如，对于由索引为$\mathbb{Z}$的、平方可和的、无限维向量集合中抽取的向量，我们得到以下定义：
$$
(f * g)(i) = \sum_a f(a) g(i-a).
$$
对于二维张量，则为$f$的索引$(a, b)$和$g$的索引$(i-a, j-b)$上的对应加和：
$$
(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b)
$$
这看起来类似于图像卷积，但有一个主要区别：这里不是使用$(i+a, j+b)$，而是使用差值。然而，这种区别是表面的，因为我们总是可以匹配卷积层和数学卷积之间的符号。我们在卷积层中的原始定义更正确地描述了**互相关**（cross-correlation）

BTW，由于现代大多数图片都是RGB的三通道数据，因此实际的输入是三维张量，对应的权重与偏置，也会对应的改变维度

## 图像卷积

卷积层实际上表达的运算是**互相关运算**，但和卷积差别不大。由于基本已经理解了，在这里省略关于卷积运算的解释。

注意，输出大小略小于输入大小。这是因为卷积核的宽度和高度大于1，而卷积核只与图像中每个大小完全适合的位置进行互相关运算。所以，输出大小等于输入大小$n_h \times n_w$减去卷积核大小$k_h \times k_w$，即：$$(n_h-k_h+1) \times (n_w-k_w+1).$$

卷积运算的实现基于最基本的语法：

```python
import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
          	# 这里的*是H乘法，即每个元素对应的积，而不是矩阵乘法(mm)
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
```

### 卷积层

卷积层对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出。 所以，卷积层中的两个被训练的参数是**卷积核权重**和**标量偏置**。 就像我们之前随机初始化全连接层一样，在训练基于卷积层的模型时，我们也随机初始化卷积核权重。

```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

### 边缘检测

利用卷积运算可以实现边缘检测（检测的是卷积核对应位置的元素的突变），如垂直的边缘可以用[[1.0,-1.0]]的卷积核检测。而如果使用机器学习来得出检测所需的卷积核，流程如下：

```python
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
# 调用了pytorch对应的Conv2d层
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核（手写梯度下降）
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
```

训练后卷积核的权重为`tensor([[ 1.0486, -0.9313]])`

### 特征映射与感受野

（一些概念）

输出的卷积层有时被称为***特征映射*（feature map）**，因为它可以被视为一个输入映射到下一层的空间维度的转换器。 在卷积神经网络中，对于某一层的任意元素x，其***感受野*（receptive field）**是指在前向传播期间可能影响x计算的所有元素（来自所有先前层）。

### 填充与步幅

卷积的输出形状取决于输入形状和卷积核的形状。

有时，在应用了连续的卷积之后，我们最终得到的输出远小于输入大小。如此一来，原始图像的边界丢失了许多有用信息。而*填充*是解决此问题最有效的方法；

有时，我们可能希望大幅降低图像的宽度和高度。例如，如果我们发现原始的输入分辨率十分冗余。*步幅*则可以在这类情况下提供帮助。

#### 填充

如上所述，在应用多层卷积时，我们常常丢失边缘像素。 由于我们通常使用小卷积核，因此对于任何单个卷积，我们可能只会丢失几个像素。 但随着我们应用许多连续卷积层，累积丢失的像素数就多了。 解决这个问题的简单方法即为*填充*（padding）：在输入图像的边界填充元素（通常填充元素是0）。

通常，如果我们添加ph行填充（大约一半在顶部，一半在底部）和pw列填充（左侧大约一半，右侧一半），则输出形状将为
$$
(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1)
$$
基本上都会设置ph=kh-1，以此直观上看起来并没有造成信息的损失。

（很好理解，同样的原因，卷积核一般设置为奇数）

```python
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

`Conv2d`提供了不错的API，这里的含义是使用高度为5，宽度为3的卷积核，高度和宽度两边的填充分别为2和1。

#### 步幅

在计算互相关时，卷积窗口从输入张量的左上角开始，向下、向右滑动。 在前面的例子中，我们默认每次滑动一个元素。 但是，有时候为了高效计算或是缩减采样次数，卷积窗口可以跳过中间位置，每次滑动多个元素。我们将每次滑动元素的数量称为*步幅*（stride）。
$$
\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor
$$
（可以理解为xxx+1，也是上面公式把1换为sh的结果）

```python
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape

torch.Size([2, 2])
```

## 多输入多输出通道

当我们添加通道时，我们的输入和隐藏的表示都变成了三维张量。例如，每个RGB输入图像具有3×h×w的形状。我们将这个大小为3的轴称为*通道*（channel）维度。

### 多输入通道

（没有太多新奇的东西）

当输入包含多个通道时，需要构造一个与输入数据具有相同输入通道数的卷积核，以便与输入数据进行互相关运算。最后将每个通道得出的核函数求和，得到最后的输出。

```python
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    # zip是好用的生成元组的方法，将X和K各自分为元素再合成tuple
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

### 多输出通道

在最流行的神经网络架构中，随着神经网络层数的加深，我们常会增加输出通道的维数，通过减少空间分辨率以获得更大的通道深度。直观地说，我们可以将每个通道看作对不同特征的响应。而现实可能更为复杂一些，因为每个通道不是独立学习的，而是为了共同使用而优化的。因此，多输出通道并不仅是学习多个单通道的检测器。

用ci和co分别表示输入和输出通道的数目，并让kh和kw为卷积核的高度和宽度。为了获得多个通道的输出，我们可以为每个输出通道创建一个形状为ci×kh×kw的卷积核张量，这样卷积核的形状是co×ci×kh×kw。在互相关运算中，每个输出通道先获取所有输入通道，再以对应该输出通道的卷积核计算出结果。

```python
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起（参数设为0，新增第0位堆入）
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```

即当ci=kh=kw=2时

```python
K = torch.stack((K, K + 1, K + 2), 0)
K.shape

>>> torch.Size([3, 2, 2, 2])
```

### 1x1卷积层

看起来似乎没有多大意义。 毕竟，卷积的本质是有效提取相邻像素间的相关特征，而1×1卷积显然没有此作用。然而，1x1卷积是具有实际意义的

因为使用了最小窗口，1×1卷积失去了卷积层的特有能力——在高度和宽度维度上，识别相邻元素间相互作用的能力。 其实1×1卷积的唯一计算发生在通道上。1×1卷积层需要的权重维度为co×ci，再额外加上一个偏置。

**1×1卷积层通常用于调整网络层的通道数量和控制模型复杂性。**

具体实现实际上和全连接层是一样的，因此省略。

## 池化层

起源于图像卷积对位置的高度敏感性与需求的空间不变性之间的冲突。

本节将介绍*池化*（pooling）层，它具有双重目的：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。

池运算是确定的，训练不会改变其操作内容。由于基本已经理解了，忽略here。常用的包括最大池化层，平均池化层等。当然，与卷积层一样，汇聚层也可以改变输出形状。和以前一样，我们可以通过填充和步幅以获得所需的输出形状。默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同（目的就是为了提取信息，不会重复运算一个像素点）。

```python
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

采用nn内置的`MaxPool2d`模块捏

由于卷积层在通道上对输入汇总，而池化层只是对每层的运算，因此输出维和输入维是一样的。

## LeNet

作为本章的结尾，构建一个完整的卷积神经网络，来对MNIST进行分类：

总体来看，LeNet（LeNet-5）由两个部分组成：

- 卷积编码器：由两个卷积层组成;
- 全连接层密集块：由三个全连接层组成。

![../_images/lenet.svg](https://zh.d2l.ai/_images/lenet.svg)

事实上采用深度学习的框架实现LeNet并不困难：

```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

而在训练时，只需要注意将数据，变量和模型存入合适的设备中（GPU）

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        # 找到数据存放的设备
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
    
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
          	# 常规操作，进行训练
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```python
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```