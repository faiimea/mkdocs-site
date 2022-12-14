# DeHiREC: Detecting Hidden Voice Recordersvia ADC Electromagnetic Radiation

## 术语表
* EMR electromagnetic radiations 电磁辐射
* ADC analog-to-digital converter 模数转换器
* MSoC mixed signal system-on-chips 嵌入芯片上混合信号系统
* SNR signal-to-noise ratio 信噪比
* EMI electromagnetic interference 电磁干扰
* PGA programmablegain amplifier 可编程增益放大器
* AGC automatic gain control 自动增益控制
* LPF low-pass filter 低通滤波器
* RF 射频信号
* LNA low noiseamplifier 低噪声放大器
* SDR software defined radio 软件无线电
* TPR True Positive Rate 真阳率

## 背景
### 语音记录器
语音记录器是一种将模拟语音转换为数字信号的电子设备，实现原理略

语音记录器中常用的ADC有 successive-approximation ADC和 Sigma-Delta ADC（通过过采样和噪声整形将高频噪声集中在模拟信号中，然后通过数字滤波器将其去除，分辨率更高）

当新的转换开始时，输入信号由S-H电路采样，然后与参考电压进行多次比较。最后，SAR ADC存储结果并输出数字信号，直到整个转换完成

### EMR of MSoC
由于MSoC由数字、模拟和电源电路组成，因此MSoC内的时变电流将始终产生EMR，当电流波动时，MSoC仍不可避免地会以时钟频率产生EMR。

当我们采用长时间FFT窗口时，峰值将出现在一个范围内，大约1秒。现代时钟发生器使用扩频技术来重塑时钟的能量分布，当我们捕获时钟信号产生的射频泄漏时，我们会发现功率分布在不同采样之间发生变化。


## 准备工作

### 目标
从电磁辐射中检测离线隐藏语音记录器

### 问题
1: 语音记录器的EMR特点

语音记录器的EMR峰值以相等的间隔出现在设备的时钟频率周围，MSoC中的ADC是辐射的主要来源，嵌入MSoC中的ADC产生的共享EMR模式，频谱上的等间隔表示ADC时钟频率。

2: 存在类似频谱的其他设备干扰时，如何验证EMR由语音记录器产生

由于EMR的强度与电流幅度密切相关，我们的关键思想是主动改变流经ADC的电流，同时测量EMR变化以进行相关性。

通过EMR催化方法，用超声波或电磁干扰将信号注入麦克风，主动改变ADC辐射的EMR强度并衍生变异的独特特征以识别隐藏的语音记录器。

3: 如何有效测量功率较低的语音记录器的极弱EMR

提出了一种增强的信号处理算法，称为自适应折叠，该算法使用相位对准方法折叠频谱，以累积EMR峰值并增强检测。

### 检测语音记录器的EMR-实验
当语音记录器离线记录时，即不通过无线信道传输数据时，我们使用近场探测器来检测从语音记录器泄漏的EMR信号。

我们推断记录过程引入了额外的EMR信号。通过上述初步实验，我们对从语音记录器发出的EMR信号的特征进行了表征，这些特征可以进一步用于执行隐藏语音记录器检测。

话音记录器的EMR信号主要来源于ADC的EMR（即基带）调制到的系统时钟EMR（如载波），同一芯片内的系统时钟和ADC的EMR信号将耦合并散发到外部。

### 识别来自语音记录器的EMR
我们需要从功率谱中分离或验证属于语音记录器EMR的分量。所发出的EMR信号的强度与来自麦克风的波动电流高度相关。因此，信号识别的问题可以被转化为找到一种主动激励ADC模块的输入电流的方法，从而使EMR的强度相应地改变。

增加麦克风的输出/增加传输电流，一旦我们发现一个与语音记录器特征相似的可疑信号，我们将注入EMI信号进行二次确认。如果主动刺激后EMR信号的强度增加，我们确定它是一个语音记录器。

### 增强弱EMR信号的强度
我们使用折叠算法，该算法通常用于放大周期性天文信号。对于未知频谱，折叠算法的目标是搜索周期为T的信号，然后根据时间窗T划分频谱，并折叠所有小段。当n等于信号的周期时，信号的能量在折叠后将变得更强，而噪声由于其随机性将被抵消。
在我们的场景中，当窗口T等于记录器的ADC时钟频率时，折叠后的EMR的信噪比将大大提高。

自适应折叠算法，核心是在折叠之前对齐每个片段中峰的相位。在每个段中找到峰值，并根据偏差移动功率谱，以便峰值可以位于相同的位置。

## DeHiREC
### Step 1:初步筛选
通过捕获和处理周围环境中的RF信号，DeHiREC首先分析是否存在由录音机引起的可疑频谱特征。
### Step 2:刺激与验证
一旦出现可疑的EMR信号，我们开始进行EMR催化验证。
利用EMR催化与自适应折叠和SNR计算确定是否为隐藏的语音记录器

## conclusion
本文首次尝试检测环境中隐藏的录音机。我们发现ADC产生的共享EMR模式，可用于检测隐藏的语音记录器。为了唯一地识别录音机，我们设计了EMR催化，这是一种主动刺激方法，可以触发ADC的EMR对EMI的反应。因此，我们设计了DeHiREC，这是第一个能够检测隐藏录音机的概念验证系统。我们的评估表明，DeHiREC可以在21个电子设备的干扰下检测记录仪。此外，我们设想我们的方法对检测其他低功耗和无线隐藏设备具有深远意义，我们将进一步研究我们工作的潜在应用。

## Comment
Summary:
In this paper, the author proposes a roof-of-concept system called DeHiREC to detect offline hidden voice recorder. In order to cope with the characteristics of miniaturization and disguise of hidden recorders, the author first determined the unique mode of EMR signals sent by the recorder, and located the source of EMR signals (MSoC). Then, the author designed the EMR catalytic algorithm to recognize the EMR from the voice recorder through the active stimulus signal. Finally, the author used the adaptive folding algorithm to enhance the strength of the weak EMR signal and improve the signal-to-noise ratio. The author tested it under different external environment and equipment conditions to show its effectiveness.

Advantage:
This paper focuses on a common security threat in reality, and proposes the first proof of concept system to detect offline hidden voice recorder from EMR.
In this paper, the environmental interference, measurement factors and equipment differences that may exist in the real situation are discussed and tested extensively to show the practicability of the system.
In this paper, the real layout of the system equipment in the actual situation is given, which meets the requirements for the concealment of the detection system in the threat model.

Disadvantages:
When there is an authorized recording device similar to the hidden voice recorder, it will greatly interfere with the system's judgment, and the solutions not given in the article are relatively vague (such as how to implement whitelist filtering).

