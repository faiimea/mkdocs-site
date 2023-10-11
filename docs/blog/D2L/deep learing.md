# 深度学习计算

## 层和块

回顾MLP

```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

> 层（1）接受一组输入， （2）生成相应的输出， （3）由一组可调整参数描述。

对于多层感知机而言，整个模型及其组成层都是这种架构。 整个模型接受原始输入（特征），生成输出（预测）， 并包含一些参数（所有组成层的参数集合）。 同样，每个单独的层接收输入（由前一层提供）， 生成输出（到下一层的输入），并且具有一组可调参数， 这些参数根据从下一层反向传播的信号进行更新。

这样层组以各种重复模式排列的架构普遍存在。为了实现这些复杂的网络，我们引入了神经网络*块*的概念。 *块*（block）可以描述单个层、由多个层组成的组件或整个模型本身。

在这个例子中，我们通过实例化`nn.Sequential`来构建我们的模型， 层的执行顺序是作为参数传递的。

简而言之，`nn.Sequential`定义了一种特殊的`Module`， 即在PyTorch中表示一个块的类， 它维护了一个由`Module`组成的有序列表。  注意，两个全连接层都是`Linear`类的实例， `Linear`类本身就是`Module`的子类。 

### 自定义块

每个块需要实现的基本功能：

1. 将输入数据作为其前向传播函数的参数。
2. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。**通常这是自动发生的**。
4. 存储和访问前向传播计算所需的参数。
5. 根据需要初始化模型参数。

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```

Thanks to OOP，这里不需要人工实现复杂的初始化与反向传播，只需要调用父类的nn.Module

块的一个主要优点是它的多功能性。 我们可以子类化块以创建层（如全连接层的类）、 整个模型（如上面的`MLP`类）或具有中等复杂度的各种组件。 

### 顺序块

`Sequential`的设计是为了把其他模块串起来。每个顺序块需要额外实现的功能为：

1. 一种将块逐个追加到列表中的函数；
2. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```

*关于python的args，`*args`的类型是元组，可以遍历

*enumerate函数：将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标

```python
>>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 下标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

### 在前向传播时执行自定义代码

当需要更强的灵活性时，我们需要定义自己的块。 例如，我们可能希望在前向传播函数中执行Python的控制流。 此外，我们可能希望执行任意的数学运算， 而不是简单地依赖预定义的神经网络层。

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

### 块的嵌套

规范定义的`nn.Module`子类可以任意的嵌套，如：

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

## 参数管理

我们从已有模型中访问参数。 当通过`Sequential`类定义模型时， 我们可以通过索引来访问模型的任意层。 这就像模型是一个列表一样，每层的参数都在其属性中。

```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
# net是个列表，元素为内部的层
print(net[2].state_dict())

OrderedDict([('weight', tensor([[ 0.3016, -0.1901, -0.1991, -0.1220,  0.1121, -0.1424, -0.3060,  0.3400]])), ('bias', tensor([-0.0291]))])
```

*命名原因：实际上层是一个状态机，在这里查看的时此刻的状态

输出的结果告诉我们一些重要的事情： 首先，这个全连接层包含两个参数，分别是该层的权重和偏置。 两者都存储为单精度浮点数（float32）。

### 目标参数

注意，每个参数都表示为参数类的一个实例。 要对参数执行任何操作，首先我们需要访问底层的数值。

下面的代码从第二个全连接层（即第三个神经网络层）提取偏置， 提取后返回的是一个参数类实例，并进一步访问该参数的值。

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

<class 'torch.nn.parameter.Parameter'>
Parameter containing:
tensor([-0.0291], requires_grad=True)
tensor([-0.0291])
```

参数是复合的对象，包含值、梯度和额外信息。 这就是我们需要显式参数值的原因(.data)。 除了值之外，我们还可以访问每个参数的梯度。 在上面这个网络中，由于我们还没有调用反向传播，所以参数的梯度处于初始状态。

```python
>>> net[2].weight.grad == None
True
```

同时，正如之前所使用的一样，也可以一次性访问所有参数

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

# name - parameters
('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
```

同样，知道了这样的存储方式后，也可以使用字典的调用方式

```python
net.state_dict()['2.bias'].data
```

在分层嵌套的层中，也可以像通过嵌套列表索引一样访问它们。

```python
rgnet[0][1][0].bias.data
```

### 参数初始化

默认情况下，PyTorch会根据一个范围均匀地初始化权重和偏置矩阵， 这个范围是根据输入和输出维度计算出的。 PyTorch的`nn.init`模块提供了多种预置初始化方法。

```python
def init_normal(m):
    if type(m) == nn.Linear:
        # normal_ 代表会对参数进行修改（调用内置的初始化器）
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

当然，也可以对不同的层调用不同的初始化函数：

```python
net[0].apply(init_xavier)
net[2].apply(init_42)
```

### 自定义初始化

有时，深度学习框架没有提供我们需要的初始化方法。此时可以自定义初始化方法：

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
# :2表示前两行
```

当然也可以直接自己去设置参数

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

### 参数绑定

有时我们希望在多个层间共享参数： 我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。

事实上，后续的卷积层中虽然没有显式的调用shared，但是隐含了参数共享的思想，即每一层的神经元运算一致（卷积核），以此体现图像识别中的局部性原理

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
# seq中的两个share层的参数不变
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

## 自定义层

层也是module的一个子类（），即定义一个像linear一样的层出来：

```python
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

此后可以合并到更复杂的模型中。

而带有参数的层需要考虑到，过去层的参数都是`Parameter`的实例，同样需要实现这一点。

```python
class MyLinear(nn.Module):
  	# init需要参数
    def __init__(self, in_units, units):
        super().__init__()
        # 调用nn.Para
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
    
linear = MyLinear(5, 3)
linear.weight
```

## 读写文件

到目前为止，我们讨论了如何处理数据， 以及如何构建、训练和测试深度学习模型。 然而，有时我们希望保存训练的模型， 以备将来在各种环境中使用（比如在部署中进行预测）。 

此外，当运行一个耗时较长的训练过程时， 最佳的做法是定期保存中间结果， 以确保在服务器电源被不小心断掉时，我们不会损失几天的计算结果。 

因此，现在是时候学习如何加载和存储权重向量和整个模型了。

### 张量IO

对于单个张量，我们可以直接调用`load`和`save`函数分别读写它们。 这两个函数都要求我们提供一个名称，`save`要求将要保存的变量作为输入。

```python
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')

>>>x2
tensor([0, 1, 2, 3])
```

也可以存储一个**张量列表**和**字符串映射到张量的字典**，然后把它们读回内存。

### 参数IO

如果需要保存整个模型，如果把每一层的向量都分开保存就不太好，使用内置函数，保存全部的权重。

```python
torch.save(net.state_dict(), 'mlp.params')
```

为了恢复模型，我们实例化了原始多层感知机模型的一个备份（具体的模型没有保存，因此要自己实例化）。 这里我们不需要随机初始化模型参数，而是直接读取文件中存储的参数。

```python
clone = MLP()
# load_state_dict函数读取模型
clone.load_state_dict(torch.load('mlp.params'))
# eval 进入测试模式
clone.eval()
```

## GPU

在PyTorch中，每个数组都有一个设备（device）， 我们通常将其称为环境（context）。 默认情况下，所有变量和相关的计算都分配给CPU。 有时环境可能是GPU。 当我们跨多个服务器部署作业时，事情会变得更加棘手。 通过智能地将数组分配给环境， 我们可以最大限度地减少在设备之间传输数据的时间。 例如，当在带有GPU的服务器上训练神经网络时， 我们通常希望模型的参数在GPU上。

我们可以指定用于存储和计算的设备，如CPU和GPU。 默认情况下，张量是在内存中创建的，然后使用CPU计算它。

在PyTorch中，CPU和GPU可以用`torch.device('cpu')` 和`torch.device('cuda')`表示。 应该注意的是，`cpu`设备意味着所有物理CPU和内存， 这意味着PyTorch的计算将尝试使用所有CPU核心。 然而，`gpu`设备只代表一个卡和相应的显存。 如果有多个GPU，我们使用`torch.device(f'cuda:{i}')` 来表示第i块GPU（i从0开始）。 另外，`cuda:0`和`cuda`是等价的。

在创建张量时，可以通过更改参数使其存储在GPU上：

```python
X = torch.ones(2, 3, device=torch.device('cuda'))
X
```

然而，由于在计算时的数组需要存储在同一个位置，因此假如有一个cuda0的X和一个cuda1的Y，二者不可以直接运算，需要复制到同一个GPU上才能运算（称为复制）

```python
Z = X.cuda(1)
print(X)
print(Z)
Y+Z #替代X+Z
```

类似的，神经网络也可以指定设备

```python
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=torch.device('cuda'))
net(X)

>>> net[0].weight.data.device
device(type='cuda', index=0)
```

### GPU vs 并行

人们使用GPU来进行机器学习，因为单个GPU相对运行速度快。 **但是在设备（CPU、GPU和其他机器）之间传输数据比计算慢得多**。 这也使得并行化变得更加困难，因为我们必须等待数据被发送（或者接收）， 然后才能继续进行更多的操作。 这就是为什么拷贝操作要格外小心。 根据经验，多个小操作比一个大操作糟糕得多。

 此外，一次执行几个操作比代码中散布的许多单个操作要好得多。 如果一个设备必须等待另一个设备才能执行其他操作， 那么这样的操作可能会阻塞。 

最后，当我们打印张量或将张量转换为NumPy格式时， 如果数据不在内存中，框架会首先将其复制到内存中， 这会导致额外的传输开销。 更糟糕的是，它现在受制于全局解释器锁，使得一切都得等待Python完成。

