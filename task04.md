## task04

### 深度学习

#### 发展趋势

最早可追溯到1958年，感知机（线性模型）的出现。2009年GPU的发展很关键，使用GPU矩阵运算节省了很多的时间。有了2012年Alexnet深度神经网络赢得了ILSVRC image competition冠军，深度学习变得越来越火。

#### 三个步骤

- Step1：神经网络（Neural network）

神经网络里面的节点，类似我们的神经元，可以有很多不同的连接方式，形成不同的结构，有不同的参数即权重和偏差，

##### 完全连接前馈神经网络

概念：前馈（feedforward）也称前向，从信号流向来理解就是输入信号进入网络后，信号流动是单向的，即信号从前一层流向后一层，一直到输出层，其中任意两层之间的连接并没有反馈（feedback），亦即信号没有从后一层又返回到前一层

- 为什么叫全链接呢？
  - 因为layer1与layer2之间两两都有连接，所以叫做Fully Connect；
- 为什么叫前馈呢？
  - 因为现在传递的方向是由后往前传，所以叫做Feedforward。

* 那什么叫做Deep呢？
  * Deep = Many hidden layer，有很多的隐藏层

##### 矩阵计算

计算方法：sigmoid（权重w * 输入+ 偏移量b）= 输出

##### 本质：通过隐藏层进行特征转换

把隐藏层通过特征提取来替代原来的特征工程，在最后一个隐藏层输出的就是一组新的特征（相当于黑箱操作）而对于输出层，其实是把前面的隐藏层的输出当做输入（经过特征提取得到的一组最好的特征）然后通过一个多分类器（可以是softmax函数）得到最后的输出y。



- Step2：模型评估（Goodness of function）

一般采用损失函数来反应模型的好差，所以对于神经网络来说，采用交叉熵（cross entropy）函数来对y和$\hat{y}$的损失进行计算，接下来就是调整参数，让交叉熵越小越好。

对于损失，不能只计算一笔数据的，而是要计算整体所有训练数据的损失，然后把所有的训练数据的损失都加起来，得到一个总体损失L。接下来在function set里面找到一组函数能最小化这个总体损失L，或者是找一组神经网络的参数$\theta$，来最小化总体损失L



- Step3：选择最优函数（Pick best function）

用梯度下降找到最优的函数和最好的一组参数。

具体流程：$\theta$是一组包含权重和偏差的参数集合，随机找一个初试值，接下来计算一下每个参数对应偏微分，得到的一个偏微分的集合$\nabla{L}$就是梯度,有了这些偏微分，我们就可以不断更新梯度得到新的参数，这样不断反复进行，就能得到一组最好的参数使得损失函数的值最小



##### 一个通用的理论： 

对于任何一个连续的函数，都可以用足够多的隐藏层来表示。那为什么我们还需要‘深度’学习呢，直接用一层网络表示不就可以了？



### 梯度下降

- 给到$\theta$ (weight and bias)
- 先选择一个初始的$\theta^0$，计算 $\theta^0$ 的损失函数（Loss Function）设一个参数的偏微分
- 计算完这个向量（vector）偏微分，然后就可以去更新的你$ \theta$
- 百万级别的参数（millions of parameters）
- 反向传播（Backpropagation）是一个比较有效率的算法，让你计算梯度（Gradient） 的向量（Vector）时，可以有效率的计算出来

#### 链式法则

![img](https://datawhalechina.github.io/leeml-notes/chapter14/res/chapter14-2.png)

- 连锁影响(可以看出x会影响y，y会影响z)
- BP主要用到了chain rule

### 反向传播

1. 损失函数(Loss function)是定义在单个训练样本上的，即一个样本的误差，如果想要分类，预测的类别和实际类别的区别，是一个样本的，用L表示。
2. 代价函数(Cost function)是定义在整个训练集上面的，也就是所有样本的误差的总和的平均，也就是损失函数的总和的平均，有没有这个平均其实不会影响最后的参数的求解结果。
3. 总体损失函数(Total loss function)是定义在整个训练集上面的，也就是所有样本的误差的总和。也就是平时我们反向传播需要最小化的值。

计算梯度分成两个部分

- 计算$\frac{\partial z}{\partial w}$（Forward pass的部分）
- 计算$\frac{\partial l}{\partial z}$ ( Backward pass的部分 )



### Forward Pass

首先计算$\frac{\partial z}{\partial w}$（Forward pass的部分）： 

根据求微分原理，forward pass的运算规律就是：

$\frac{\partial z}{\partial w_1} = x_1$ ,$\frac{\partial z}{\partial w_2} = x_2$

计算得到的$x_1$和$x_2$恰好就是输入的$x_1$和$x_2$直接使用数字，更直观地看到运算规律：



### Backward Pass

(Backward pass的部分)这就很困难复杂因为我们的l是最后一层： 那怎么计算 \frac{\partial l}{\partial z}∂z∂l （Backward pass的部分）这就很困难复杂因为我们的ll是最后一层：

计算所有激活函数的偏微分，激活函数有很多，这里使用Sigmoid函数为例

这里使用链式法则（Chain Rule）的case1，计算过程如下：

$\frac{\partial l}{\partial z} = \frac{\partial a}{\partial z}\frac{\partial l}{\partial a} \Rightarrow {\sigma}'(z)$

$\frac{\partial l}{\partial a} = \frac{\partial z'}{\partial a}\frac{\partial l}{\partial z'} +\frac{\partial z''}{\partial a}\frac{\partial l}{\partial z''}$

* case 1 : Output layer

假设$\frac{\partial l}{\partial z'}$和$\frac{\partial l}{\partial z''}$是最后一层的隐藏层 也就是就是y1与y2是输出值，那么直接计算就能得出结果

但是如果不是最后一层，计算$\frac{\partial l}{\partial z'}$和$\frac{\partial l}{\partial z''}$的话就需要继续往后一直通过链式法则算下去

* case 2 : Not Output Layer

我们要继续计算后面绿色的$\frac{\partial l}{\partial z_a}$和$\frac{\partial l}{\partial z_b}$,然后通过继续乘$w_5$和$w_6$得到$\frac{\partial l}{\partial z'}$，但是要是$\frac{\partial l}{\partial z_a}$和$\frac{\partial l}{\partial z_b}$都不知道，那么我们就继续往后面层计算，一直到碰到输出值，得到输出值之后再反向往输入那个方向走。

从最后一个$\frac{\partial l}{\partial z_5}$和$\frac{\partial l}{\partial z_6}$看，因为$\frac{\partial l}{\partial z_a}$和$\frac{\partial l}{\partial z_b}$比较容易通过output求出来，然后继续往前求$\frac{\partial l}{\partial z_3}$和$\frac{\partial l}{\partial z_4}$，再继续求$\frac{\partial l}{\partial z_1}$和$\frac{\partial l}{\partial z_2}$ 



### 总结

我们的目标是要求计算$\frac{\partial z}{\partial w}$（Forward pass的部分）和计算$\frac{\partial l}{\partial z}$ ( Backward pass的部分 )，然后把$\frac{\partial z}{\partial w}$和$\frac{\partial l}{\partial z}$相乘，我们就可以得到$\frac{\partial l}{\partial w}$,所有我们就可以得到神经网络中所有的参数，然后用梯度下降就可以不断更新，得到损失最小的函数
