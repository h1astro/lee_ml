## task03

### 误差的来源

#### 偏差 

所有可能的训练数据集训练出的所有模型的输出的平均值与真实模型的输出值之间的差异。

#### 方差

不同的训练数据集训练出的模型输出值之间的差异。

#### 偏差大-欠拟合

模型不能适配训练样本，有一个很大的偏差。

应该重新设计模型。将更多的函数加进去，考虑更多次幂、更复杂的模型。 强行再收集更多的data去训练，没有什么帮助的，因为设计的函数集本身就不好，再找更多的训练集也不会更好。

#### 方差大-过拟合

模型很好的适配训练样本，但在测试集上表现很糟，有一个很大的方差。

简单粗暴的方法：更多的数据

但是很多时候不一定能做到收集更多的data。可以对数据增强：旋转、缩放、随机截取、亮度、水平翻转、数值翻转等操作。

#### 交叉验证

拿未用来给模型做训练的数据集，测试模型的性能，以便减少类似过拟合和选择偏差等问题。

### 梯度下降法

#### Tip1:调整学习率

1.手动慢慢调整

2.自适应调整: 随着次数的增加，通过一些因子来减少学习率

- 通常刚开始，初始点会距离最低点比较远，所以使用大一点的学习率
- update好几次参数之后呢，比较靠近最低点了，此时减少学习率
- 比如 $\eta^t =\frac{\eta^t}{\sqrt{t+1}}$，t 是次数。随着次数的增加，$\eta^t$ 减小

3.Adagrad 是什么？

每个参数的学习率都把它除上之前微分的均方根。

#### Tip2：随机梯度下降法

损失函数不需要处理训练集所有的数据，只需要计算某一个例子的损失函数Ln，就可以赶紧update 梯度。

#### Tip3：特征缩放

两个输入的分布的范围很不一样，建议把他们的范围缩放，使得不同输入的范围是一样的。