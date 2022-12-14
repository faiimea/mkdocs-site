# Voice Liveness Detection with Sound Field Dynamics

## 术语表

* Automatic speaker verification(ASV):  自动语音识别
* Sound Field Dynamics (SFD): 声场动力学
* Amplitude(Amp): 信号幅度

## Introduction

### Insight
人类和扬声器之间的本质差异是发声孔径大小的变化。具体来说，人类需要动态地张开和闭合嘴来发出语音命令，而扬声器保持固定的孔径大小。直观地说，人类的时变开口比扬声器带来更为动态的声场。通过检查声场的动态水平，我们可以区分声音的活跃度，即回避命令是来自真实用户的嘴巴还是扬声器。

### Challenge
* 如何描述声场的动态水平
* 考虑到阵列中通常有多个麦克风，需要正确处理所有麦克风通道以便于测量
* 根据我们测量的特征，设计一种有效的方法来区分人类和扬声器

### Solution
声场动力学SFD基于不同麦克风之间能量比的时间波动

优点：

（i） 语音内容被取消，攻击者很难操纵语音来愚弄我们的系统。

（ii）这种相对划分消除了绝对声音强度的影响，因此SFD独立于音量。

(iii)此外，SFD基本上取决于声源的物理孔径大小变化，而与声源位置无关。

Based on the extracted SFDfeatures, we design a deep learning model with a self-attention mechanism to further fuse multiple channels and differentiate humans and loudspeakers


## 声场动力学

### 声场与方向性

衍射效应与声音叠加和干扰一起产生了声音的方向性。声音方向性取决于两个因素：信号频率f和孔径大小a
## System

## Implementation

## Evaluation

