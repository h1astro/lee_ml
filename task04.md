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



- Step2：模型评估（Goodness of function）
- Step3：选择最优函数（Pick best function）

