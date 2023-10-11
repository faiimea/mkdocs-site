# 计算机视觉

近年来，深度学习一直是提高计算机视觉系统性能的变革力量。 无论是医疗诊断、自动驾驶，还是智能滤波器、摄像头监控，许多计算机视觉领域的应用都与我们当前和未来的生活密切相关。 可以说，最先进的计算机视觉应用与深度学习几乎是不可分割的。 有鉴于此，本章将重点介绍计算机视觉领域，并探讨最近在学术界和行业中具有影响力的方法和应用。

我们研究了计算机视觉中常用的各种卷积神经网络，并将它们应用到简单的图像分类任务中。 本章开头，我们将介绍两种可以改进模型泛化的方法，即*图像增广*和*微调*，并将它们应用于图像分类。 由于深度神经网络可以有效地表示多个层次的图像，因此这种分层表示已成功用于各种计算机视觉任务，例如*目标检测*（object detection）、*语义分割*（semantic segmentation）和*样式迁移*（style transfer）。 秉承计算机视觉中利用分层表示的关键思想，我们将从物体检测的主要组件和技术开始，继而展示如何使用*完全卷积网络*对图像进行语义分割，然后我们将解释如何使用样式迁移技术来生成像本书封面一样的图像。 最后在结束本章时，我们将本章和前几章的知识应用于两个流行的计算机视觉基准数据集。

## 图像增广

大型数据集是成功应用深度神经网络的先决条件。 图像增广在对训练图像进行一系列的随机变化之后，生成相似但不同的训练样本，从而**扩大了训练集的规模**。 此外，应用图像增广的原因是，随机改变训练样本可以减少模型对某些属性的依赖，从而**提高模型的泛化能力**。

例如，我们可以以不同的方式裁剪图像，使感兴趣的对象出现在不同的位置，减少模型对于对象出现位置的依赖。 我们还可以调整亮度、颜色等因素来降低模型对颜色的敏感度。 

*事实上，很多时候都是从实际情况出发来对训练集进行图像增广，因此会涉及到工程经验

在`torchvision.transforms`库中，有很多对图像的处理方法

* 翻转和裁剪

```python
# 各有50%的几率使图像向左或向右翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())

# 各有50%的几率向上或向下翻转
apply(img, torchvision.transforms.RandomVerticalFlip())

# 随机裁剪一个面积为原始面积10%到100%的区域，该区域的宽高比从0.5～2之间随机取值。 然后，区域的宽度和高度都被缩放到200像素。
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

* 改变颜色，亮度，对比度，饱和度，色调

```python
# 随机更改图像的亮度
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
# brightness:亮度
# contrast:对比度
# saturation:饱和度
# hue:色调
```

在具体应用时，一般采用多种图像增广方法结合(`torchvision.transforms.Compose`)，并且在读入图片数据集时也常常需要将图片转换为张量，因此：

```python
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)

train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

# 这里的augs将调用上面的train_augs
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

事实上，采用图像增广的神经网络在精度与正则化上的表现均更好，可以通过多次迭代保证训练的图像数据增加，因为每次迭代时使用的数据集都经过了随机变换。

## 微调 - 迁移学习

由于在实际应用时，很难获得如imagenet等丰富且有标号的数据集，因此可能很难获得预期的效果。不过，可以采用迁移学习的方法，）将从*源数据集*学到的知识迁移到*目标数据集*，当目标数据集比源数据集小得多时，微调有助于提高模型的泛化能力。

迁移学习中的常见技巧:*微调*（fine-tuning）。微调包括以下四个步骤。

1. 在源数据集（例如ImageNet数据集）上预训练神经网络模型，即*源模型*。
2. 创建一个新的神经网络模型，即*目标模型*。这将复制源模型上的所有模型设计及其参数（输出层除外）。**我们假定这些模型参数包含从源数据集中学到的知识，这些知识也将适用于目标数据集。**我们还假设源模型的输出层与源数据集的标签密切相关；因此不在目标模型中使用该层。
3. 向目标模型添加输出层，其输出数是目标数据集中的类别数。然后随机初始化该层的模型参数。
4. 在目标数据集（如椅子数据集）上训练目标模型。**输出层将从头开始进行训练，而所有其他层的参数将根据源模型的参数进行微调。**

抽象来说，迁移学习将神经网络分为两个部分：特征提取与分类，分类模块与数据集相关，因此需要根据实际应用重新训练，而特征提取模块可以是通用的，在大数据集上表现良好的模块代表其可以有效的识别某些特征，因此也可以代入到新的网络中。

下面给出了一个利用resnet与迁移学习实现热狗识别的例子

```python
%matplotlib inline
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 下载图片
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')

# 图片导入内存中
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# 使用RGB通道的均值和标准差（from Imagenet），以标准化每个通道
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# 首先从图像中裁切随机大小和随机长宽比的区域，然后将该区域缩放为224×224输入图像
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

# 使用在ImageNet数据集上预训练的ResNet-18作为源模型，pretrained自动下载预训练的模型参数
pretrained_net = torchvision.models.resnet18(pretrained=True)

# 预训练的源模型实例包含许多特征层和一个输出层fc。 此划分的主要目的是促进对除输出层以外所有层的模型参数进行微调。pretrained_net.fc = Linear(in_features=512, out_features=1000, bias=True)
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)

# 只对输出层进行初始化
nn.init.xavier_uniform_(finetune_net.fc.weight);

# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
    
train_fine_tuning(finetune_net, 5e-5)
```

## 目标检测

在图像分类任务中，我们假设图像中只有一个主要物体对象，我们只关注如何识别其类别。 然而，很多时候图像里有多个我们感兴趣的目标，我们不仅想知道它们的类别，还想得到它们在图像中的具体位置。 在计算机视觉里，我们将这类任务称为*目标检测*（object detection）或*目标识别*（object recognition）。

### 边界框

在目标检测中，我们通常使用*边界框*（bounding box）来描述对象的空间位置。 边界框是矩形的，由矩形左上角的以及右下角的x和y坐标决定。 另一种常用的边界框表示方法是边界框中心的(x,y)轴坐标以及框的宽度和高度。

### 锚框

目标检测算法通常会在输入图像中采样大量的区域，然后判断这些区域中是否包含我们感兴趣的目标，并调整区域边界从而更准确地预测目标的*真实边界框*（ground-truth bounding box）。 不同的模型使用的区域采样方法可能不同。 这里我们介绍其中的一种方法：以每个像素为中心，生成多个缩放比和宽高比（aspect ratio）不同的边界框。 这些边界框被称为*锚框*（anchor box）

由于和目的相悖，在这里忽略具体实现锚框的代码，只描述思路

#### 预处理流程

对输入图像，以图像的每个像素为中心生成不同形状的锚框，用缩放比和宽高比作为标签。

在已知目标的真实边界框的条件下，通过*杰卡德系数*（Jaccard）量化锚框和真实边界框之间的相似性，杰卡德系数是他们交集的大小除以他们并集的大小。使用交并比（IOU）来衡量锚框和真实边界框之间、以及不同锚框之间的相似度。

#### 训练

在训练集中，我们将每个锚框视为一个训练样本。 为了训练目标检测模型，我们需要每个锚框的*类别*（class）和*偏移量*（offset）标签，其中前者是与锚框相关的对象的类别，后者是真实边界框相对于锚框的偏移量。 在预测时，我们为每个图像生成多个锚框，预测所有锚框的类别和偏移量，根据预测的偏移量调整它们的位置以获得预测的边界框，最后只输出符合特定条件的预测边界框。

目标检测训练集带有*真实边界框*的位置及其包围物体类别的标签。 要标记任何生成的锚框，我们可以参考分配到的最接近此锚框的真实边界框的位置和类别标签。

通过匈牙利算法，建立（锚框，真实边界框）的矩阵，每次找到其中IOU最大的量，作为一组确定的“锚框-真边框”对，并从矩阵中舍弃这一行与这一列，再迭代，直到所有真实边界框都被分配为止，此时没有被分配的锚框，将通过阈值判断是否要分配真实边界框。

被分配的锚框为*正类*锚框，通过公式计算和对应的真实框的偏移量，其他为*背景*（background），没有对应的物体。 背景类别的锚框通常被称为*负类*锚框，负类锚框的偏移量被标记为零。

#### 抑制预测边界框

在预测时，我们先为图像生成多个锚框，再为这些锚框一一预测类别和偏移量。 一个*预测好的边界框*则根据其中某个带有预测偏移量的锚框而生成。

当有许多锚框时，可能会输出许多相似的具有明显重叠的预测边界框，都围绕着同一目标。 为了简化输出，我们可以使用*非极大值抑制*（non-maximum suppression，NMS）合并属于同一目标的类似的预测边界框。具体的NMS算法细节不做描述。



---

emm，事实上我开始怀疑目前对于计算机视觉的学习的必要性了

本来进行d2l的目的只是为了掌握torch框架的使用，与在NSEC中SGX可能要用到的序列分类，以及未来语音接口中可能的图像分类任务，不过看起来关于目标检测，风格迁移等明显的cv任务并不在任务栈中，所以准备放个坑，等到之后有时间再填

（才不是因为数学推导太多导致我很烦）