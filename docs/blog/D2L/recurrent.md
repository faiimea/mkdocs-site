# 循环神经网络

到目前为止，我们遇到过两种类型的数据：表格数据和图像数据。 对于图像数据，我们设计了专门的卷积神经网络架构来为这类特殊的数据结构建模。 换句话说，如果我们拥有一张图像，我们需要有效地利用其像素位置， 假若我们对图像中的像素位置进行重排，就会对图像中内容的推断造成极大的困难。

最重要的是，到目前为止我们默认数据都来自于某种分布， 并且所有样本都是独立同分布的 （independently and identically distributed，i.i.d.）。 然而，大多数的数据并非如此。 例如，文章中的单词是按顺序写的，如果顺序被随机地重排，就很难理解文章原始的意思。 同样，视频中的图像帧、对话中的音频信号以及网站上的浏览行为都是有顺序的。 因此，针对此类数据而设计特定模型，可能效果会更好。

另一个问题来自这样一个事实： 我们不仅仅可以接收一个序列作为输入，而是还可能期望继续猜测这个序列的后续。 例如，一个任务可以是继续预测2,4,6,8,10,…。 这在时间序列分析中是相当常见的，可以用来预测股市的波动、 患者的体温曲线或者赛车所需的加速度。 同理，我们需要能够处理这些数据的特定模型。

**简言之，如果说卷积神经网络可以有效地处理空间信息， 那么本章的*循环神经网络*（recurrent neural network，RNN）则可以更好地处理序列信息。 循环神经网络通过引入状态变量存储过去的信息和当前的输入，从而可以确定当前的输出。**

许多使用循环网络的例子都是基于文本数据的，因此我们将在本章中重点介绍语言模型。 在对序列数据进行更详细的回顾之后，我们将介绍文本预处理的实用技术。 然后，我们将讨论语言模型的基本概念，并将此讨论作为循环神经网络设计的灵感。 最后，我们描述了循环神经网络的梯度计算方法，以探讨训练此类网络时可能遇到的问题。

## 序列模型

### 统计工具

处理序列数据需要统计工具和新的深度神经网络架构。例如在股票价格中：

其中，用$x_t$表示价格，即在**时间步**（time step）$t \in \mathbb{Z}^+$时，观察到的价格$x_t$（ t是离散的）。假设一个交易员想在$t$日的股市中表现良好，于是通过以下途径预测$x_t$：
$$
x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)
$$
即$x_i$之间是不独立的。

### 自回归模型

通常来说，对于序列的预测可以转换为曾经学习过的回归模型。然而存在一个问题，即输入数据的数量随着t的改变而改变，因此需要近似方法。简单来说，归结为以下两种策略

第一种策略，假设在现实情况下相当长的序列$x_{t-1}, \ldots, x_1$可能是不必要的，因此我们只需要**满足某个长度为$\tau$的时间跨度**，即使用观测序列$x_{t-1}, \ldots, x_{t-\tau}$。当下获得的最直接的好处就是参数的数量总是不变的，至少在$t > \tau$时如此，这就使我们能够训练一个上面提及的深度网络。这种模型被称为**自回归模型**（autoregressive models），因为它们是对自己执行回归。

第二种策略，是保留一些对过去观测的总结$h_t$，并且同时更新预测$\hat{x}_t$和总结$h_t$。这就产生了基于$\hat{x}_t = P(x_t \mid h_{t})$估计$x_t$，以及公式$h_t = g(h_{t-1}, x_{t-1})$更新的模型。由于$h_t$从未被观测到，这类模型也被称为**隐变量自回归模型**（latent autoregressive models）。（在信息论中，总结$h_t$被称为状态$S_t$）

一个常见的假设是虽然特定值$x_t$可能会改变，但是序列本身的动力学不会改变。这样的假设是合理的，因为新的动力学一定受新的数据影响，而我们不可能用目前所掌握的数据来预测新的动力学。

统计学家称不变的动力学为**静止的**（stationary）。因此，整个序列的估计值都将通过以下的方式获得：

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

### 马尔可夫模型

在信息论中已经熟悉了的马尔可夫模型与马尔可夫条件，在这里不做赘述

### 序列模型训练

生成一些数据：使用正弦函数和一些可加性噪声来生成序列数据， 时间步为1,2,…,1000。

```python
%matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

接下来，我们将这个序列转换为模型的*特征－标签*（feature-label）对。 基于嵌入维度τ，我们将数据映射为数据对yt=xt 和xt=[xt−τ,…,xt−1]。 这比我们提供的数据样本少了τ个， 因为我们没有足够的历史记录来描述前τ个数据样本。 一个简单的解决办法是：如果拥有足够长的序列就丢弃这几项； 另一个方法是用零填充序列。 在这里，我们仅使用前600个“特征－标签”对进行训练。

```python
tau = 4
# 代表矩阵的高宽
features = torch.zeros((T - tau, tau))
for i in range(tau):
  	# 把features[:, i]这个列向量=x的向量数据
    features[:, i] = x[i: T - tau + i]
# x从tau后的数据，reshape为高不知，宽为1的矩阵
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
```

在这里，我们使用一个相当简单的架构训练模型： 一个拥有两个全连接层的多层感知机，ReLU激活函数和平方损失。

```python
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

由于训练损失很小，因此我们期望模型能有很好的工作效果。 让我们看看这在实践中意味着什么。 首先是检查模型预测下一个时间步的能力， 也就是***单步预测*（one-step-ahead prediction）**。

```python
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
```

由于始终可以获得正确数据的反馈，单步预测的效果不错，然而考虑到大多数真实情况（如股价）具有时效性，在一个时间之后的数据是无法获取的，因此需要改变预测方法

通常，对于直到xt的观测序列，其在时间步t+k处的预测输出$x_{t}$+k 称为**k*步预测*（k-step-ahead-prediction）**。 由于我们的观察已经到了$x_{604}$，它的k步预测是$x_{604}$+k。 换句话说，我们必须使用我们自己的预测（而不是原始数据）来进行多步预测。

经过几个预测步骤之后，预测的结果很快就会衰减到一个常数。事实是由于错误的累积：假设在步骤$1$之后，我们积累了一些错误$\epsilon_1 = \bar\epsilon$。于是，步骤$2$的输入被扰动了$\epsilon_1$，结果积累的误差是依照次序的$\epsilon_2 = \bar\epsilon + c \epsilon_1$，其中$c$为某个常数，后面的预测误差依此类推。因此误差可能会相当快地偏离真实的观测结果。

![../_images/output_sequence_ce248f_96_0.svg](https://zh.d2l.ai/_images/output_sequence_ce248f_96_0.svg)

 虽然“4步预测”看起来仍然不错，但超过这个跨度的任何预测几乎都是无用的。

## 文本预处理

相比于向量序列，文本数据需要经过预处理才能解析。这些步骤通常包括：

1. 将文本作为字符串加载到内存中。
2. 将字符串拆分为词元（如单词和字符）。
3. 建立一个词表，将拆分的词元映射到数字索引。
4. 将文本转换为数字索引序列，方便模型操作。

以下是使用python进行文本处理的标准步骤（其实还是比较方便的）

### 读取数据集

```python
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # python re为正则表达式模块 re.sub为替换，在这里忽略标点符号与大写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])
```

### 词元化

下面的`tokenize`函数将文本行列表（`lines`）作为输入， 列表中的每个元素是一个文本序列（如一条文本行）。 每个文本序列又被拆分成一个词元列表，*词元*（token）是文本的基本单位。 最后，返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（string）。

```python
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

### 词表

词元的类型是字符串，而**模型需要的输入是数字**，因此这种类型不方便模型使用。 现在，让我们构建一个字典，通常也叫做*词表*（vocabulary）， 用来将字符串类型的词元映射到从0开始的数字索引中。 我们先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计， 得到的统计结果称之为*语料*（corpus）。 然后根据每个唯一词元的出现频率，为其分配一个数字索引。 很少出现的词元通常被移除，这可以降低复杂性。 另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”。 我们可以选择增加一个列表，用于保存那些被保留的词元， 例如：填充词元（“<pad>”）； 序列开始词元（“<bos>”）； 序列结束词元（“<eos>”）。

现在，我们可以将每一条文本行转换成一个数字索引列表。

整合以上功能后将词元字符串映射为数字索引

## 语言模型与数据集

我们了解了如何将文本数据映射为词元，以及将这些词元可以视为一系列离散的观测，例如单词或字符。

假设长度为$T$的文本序列中的词元依次为$x_1, x_2, \ldots, x_T$。于是，$x_t$（$1 \leq t \leq T$）可以被认为是文本序列在时间步$t$处的观测或标签。

在给定这样的文本序列时，**语言模型**（language model）的目标是估计序列的联合概率$$P(x_1, x_2, \ldots, x_T).$$

例如，只需要一次抽取一个词元$x_t \sim P(x*_t \mid x_*{t-1}, \ldots, x_1)$，

### 学习语言模型

通过全概率公式将一个序列的联合概率转换为多个条件概率的积

为了训练语言模型，我们需要计算单词的概率， 以及给定前面几个单词后出现某个单词的条件概率。 这些概率本质上就是语言模型的参数。由最基本的条件概率公式，可以得出：
$$
\hat{P}(\text{learning} \mid \text{deep}) = \frac{n(\text{deep, learning})}{n(\text{deep})}
$$
其中$n(x)$和$n(x, x')$分别是单个单词和连续单词对的出现次数。

不幸的是，由于连续单词对“deep learning”的出现频率要低得多，所以估计这类单词正确的概率要困难得多。特别是对于一些不常见的单词组合，要想找到足够的出现次数来获得准确的估计可能都不容易。而对于三个或者更多的单词组合，情况会变得更糟。许多合理的三个单词组合可能是存在的，但是在数据集中却找不到。

除非我们提供某种解决方案，来将这些单词组合指定为非零计数，否则将无法在语言模型中使用它们。

如果数据集很小，或者单词非常罕见，那么这类单词出现一次的机会可能都找不到。

拉普拉斯平滑（即在所有计数中添加一个小常数）可以在一定程度上帮助解决，但是这样的模型很容易变得无效

### 马尔可夫模型与语法

