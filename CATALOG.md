# 点还在动 · 目录

目录记录现有源码与历史机制。新作按 [创作约定](docs/CREATIVE_PRINCIPLES.md) 编排连续的 D2L 动态艺术作品，以群体运动、几何形变与章节衔接为主，保留极简章节标识。当前试作见 [49 段完整连续作品](docs/CONTINUOUS_STUDY.md)。下表保留历史机制描述；完整作品时间线和本次交付状态见制作记录。

跳过：第 1–2 章、所有「从零实现」、简洁 API / Kaggle / 附录、没有独立几何机制的工程节（存盘、GPU 列表等）。

教材：[动手学深度学习](https://zh.d2l.ai/)

## 已过审

此处记录人工过审的成片；后文为源码与机制目录。自动渲染、视频规格和解码检查通过，不会自动升级为“已过审”。可用 `python -m scripts.review_all` 生成当前工作树的预览验证报告，或加 `--profile final` 验证成片。

| 集 | 机制 | 成片 |
| --- | --- | --- |
| 3.1 | 小批量 SGD 把错线拽正，留下噪声残差 | [Release 03.1](https://github.com/siddhartha-yz/neural-choreography/releases/tag/03.1) |
| 3.4 | 未训练前向 \(o=Wx+b\) → softmax，和为 1 | [Release 03.4](https://github.com/siddhartha-yz/neural-choreography/releases/tag/03.4) |

## 第 3 章其余

3.2 / 3.3 / 3.6 / 3.7 是实现。3.5 是 Fashion-MNIST 读入。都不做正片。

## 第 4 章 · 多层感知机

| 集 | D2L | 机制 |
| --- | --- | --- |
| 4.1 | [多层感知机](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html) | 隐层 + ReLU：\(z=W_1x+b_1\) 可负 → \(h=\mathrm{relu}(z)\) → \(o=W_2h+b_2\)。不训练 |
| 4.4 | [欠拟合和过拟合](https://zh.d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html) | 低容量欠拟合、高容量穿过训练点却偏离旅行者 |
| 4.5 | [权重衰减](https://zh.d2l.ai/chapter_multilayer-perceptrons/weight-decay.html) | 高容量曲线被 \(\lambda\|w\|^2\) 拉回光滑 |
| 4.6 | [暂退法](https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html) | 训练时隐单元随机置零，推断时全开 |
| 4.7 | [正向传播、反向传播和计算图](https://zh.d2l.ai/chapter_multilayer-perceptrons/backprop.html) | 同一条路上正向再反向，链式法则在几何上倒走 |
| 4.8 | [数值稳定性和模型初始化](https://zh.d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) | 激活饱和 / 梯度消失的几何（第二波） |
| 4.2 3 9 10 | 从零、简洁、分布偏移、房价 | 不做正片 |

## 第 5 章 · 深度学习计算

| 集 | D2L | 机制 |
| --- | --- | --- |
| 5.1 | [层和块](https://zh.d2l.ai/chapter_deep-learning-computation/model-construction.html) | 层套进块、块再套进 Sequential：旅行者 \(x\) 走进外盒，逐层变成下一盒的输入。不训练 |

5.2 / 5.3 / 5.4 / 5.5 / 5.6 是参数访问、延后初始化、自定义层 API、读写文件、GPU。都不做正片。

## 第 6 章 · 卷积

| 集 | D2L | 机制 |
| --- | --- | --- |
| 6.2 | [互相关 / 卷积层](https://zh.d2l.ai/chapter_convolutional-neural-networks/conv-layer.html) | 小核在图上滑动，一个输出格被点亮 |
| 6.3 | [填充和步幅](https://zh.d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html) | 同一核，输出格子变少或边缘补零 |
| 6.4 | [多输入输出通道](https://zh.d2l.ai/chapter_convolutional-neural-networks/channels.html) | 几个薄片合成一根输出 |
| 6.5 | [汇聚层](https://zh.d2l.ai/chapter_convolutional-neural-networks/pooling.html) | 窗口取最大或平均，分辨率下降 |
| 6.6 | [LeNet](https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html) | 旅行者走完卷积→汇聚→全连接（不训练） |
| 6.1 | 从全连接看卷积 | 可并入 6.2，不单集 |

## 第 7 章 · 现代卷积

| 集 | D2L | 机制 |
| --- | --- | --- |
| 7.2 | [使用块的网络（VGG）](https://zh.d2l.ai/chapter_convolutional-modern/vgg.html) | 同一卷积块反复叠：块末 \(2\times 2\) 汇聚，高宽减半、通道翻倍。不训练 |
| 7.3 | [网络中的网络（NiN）](https://zh.d2l.ai/chapter_convolutional-modern/nin.html) | \(1\times 1\) 卷积当逐像素 MLP，最后全局平均汇聚压成类别向量。不训练 |
| 7.4 | [含并行连结的网络（GoogLeNet）](https://zh.d2l.ai/chapter_convolutional-modern/googlenet.html) | Inception：同一输入并行走 \(1\times 1/3\times 3/5\times 5/\)汇聚，通道维拼接。不训练 |
| 7.5 | [批量规范化](https://zh.d2l.ai/chapter_convolutional-modern/batch-norm.html) | 小批量上减均值除标准差，再 \(\gamma\odot\hat x+\beta\) 拉回可学尺度 |
| 7.6 | [残差网络（ResNet）](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html) | 残差块：主路拟合 \(f(x)-x\)，捷径把 \(x\) 加回来，\(y=\mathrm{relu}(F(x)+x)\)。不训练 |
| 7.7 | [稠密连接网络（DenseNet）](https://zh.d2l.ai/chapter_convolutional-modern/densenet.html) | 稠密块：每层输出在通道维拼到所有前层，\(x\leftarrow[x,f(x)]\)，不是相加。不训练 |

7.1 AlexNet 是加深的 LeNet 名录，不做正片。

## 第 8 章 · 循环网络

| 集 | D2L | 机制 |
| --- | --- | --- |
| 8.4 | [循环神经网络](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn.html) | \(h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b)\)：隐状态沿时间带走历史。不训练 |
| 8.7 | [通过时间反向传播](https://zh.d2l.ai/chapter_recurrent-neural-networks/bptt.html) | 把循环沿时间展开，梯度沿同一条链倒走；过长则截断以免爆炸/消失 |

8.1 是序列统计工具。8.2 / 8.3 是预处理和语料。8.5 / 8.6 是从零与简洁实现。都不做正片。

## 第 9 章 · 现代循环

| 集 | D2L | 机制 |
| --- | --- | --- |
| 9.1 | [门控循环单元（GRU）](https://zh.d2l.ai/chapter_recurrent-modern/gru.html) | 更新门 \(\mathbf{Z}_t\) 在旧隐状态与候选之间做凸组合：\(\mathbf{H}_t=\mathbf{Z}_t\odot\mathbf{H}_{t-1}+(1-\mathbf{Z}_t)\odot\tilde{\mathbf{H}}_t\) |
| 9.2 | [长短期记忆网络（LSTM）](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html) | 记忆元 \(\mathbf{C}_t=\mathbf{F}_t\odot\mathbf{C}_{t-1}+\mathbf{I}_t\odot\tilde{\mathbf{C}}_t\)，输出门再放出 \(\mathbf{H}_t\) |
| 9.3 | [深度循环神经网络](https://zh.d2l.ai/chapter_recurrent-modern/deep-rnn.html) | 隐状态同时交给下一时间步和下一层：\(\mathbf{H}_t^{(l)}\) 吃 \(\mathbf{H}_t^{(l-1)}\) 与 \(\mathbf{H}_{t-1}^{(l)}\)。不训练 |
| 9.4 | [双向循环神经网络](https://zh.d2l.ai/chapter_recurrent-modern/bi-rnn.html) | 前向 \(\overrightarrow h_t\) 与后向 \(\overleftarrow h_t\) 在每个时间步拼接。不训练 |
| 9.6 | [编码器-解码器架构](https://zh.d2l.ai/chapter_recurrent-modern/encoder-decoder.html) | 变长输入压成固定形状状态，再由解码器逐词吐出变长输出。不训练 |
| 9.8 | [束搜索](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html) | 每步只留束宽 \(k\) 条候选，其余剪掉；贪心是 \(k=1\) |

9.5 是翻译数据集。9.7 是 seq2seq 训练流程。都不做正片。

## 第 10 章 · 注意力

| 集 | D2L | 机制 |
| --- | --- | --- |
| 10.1 | [注意力提示](https://zh.d2l.ai/chapter_attention-mechanisms/attention-cues.html) | 查询对键，权重点亮对应的值；热图行=查询、列=键 |
| 10.2 | [注意力汇聚：Nadaraya-Watson 核回归](https://zh.d2l.ai/chapter_attention-mechanisms/nadaraya-waston.html) | 查询越靠近哪个键，那个值的权重越大，加权合成 \(\hat y\) |
| 10.3 | [注意力评分函数](https://zh.d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html) | 评分 \(a(q,k)\) → softmax 成权重 → 值的加权和；加性 vs 缩放点积是两种 \(a\) |
| 10.4 | [Bahdanau 注意力](https://zh.d2l.ai/chapter_attention-mechanisms/bahdanau-attention.html) | 解码器当前隐状态当查询，编码器各步当键值，注意力对齐源序列 |
| 10.5 | [多头注意力](https://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html) | \(h\) 组投影后并行做注意力，头拼接再线性变回 |
| 10.6 | [自注意力和位置编码](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html) | 同一组词元同时当 \(q,k,v\)，每个位置直接连到全序列（自注意力热图） |

10.7 Transformer 是把前几节拼成编码器–解码器名录，不做正片。

## 第 11 章 · 优化

| 集 | D2L | 机制 |
| --- | --- | --- |
| 11.2 | [凸性](https://zh.d2l.ai/chapter_optimization/convexity.html) | 凸碗上任意两点连线都在曲面上方；非凸则有鞍和坑 |
| 11.3 | [梯度下降](https://zh.d2l.ai/chapter_optimization/gd.html) | 沿 \(-\nabla f\) 走下损失面，步长太大越过谷底 |
| 11.4 | [随机梯度下降](https://zh.d2l.ai/chapter_optimization/sgd.html) | 每步只用一个样本的噪声梯度，轨迹在谷里抖动 |
| 11.6 | [动量法](https://zh.d2l.ai/chapter_optimization/momentum.html) | 速度项带着点冲过窄谷，少在垂直方向振荡 |
| 11.7 | [AdaGrad算法](https://zh.d2l.ai/chapter_optimization/adagrad.html) | 各坐标累积平方梯度，走得多的轴步长被 \(1/\sqrt{s}\) 压小 |
| 11.8 | [RMSProp算法](https://zh.d2l.ai/chapter_optimization/rmsprop.html) | 用平方梯度的滑动平均代替累积，步长不再单调缩死 |
| 11.10 | [Adam算法](https://zh.d2l.ai/chapter_optimization/adam.html) | 动量平滑一阶矩 + RMSProp 缩放二阶矩，偏置校正后迈步 |
| 11.11 | [学习率调度器](https://zh.d2l.ai/chapter_optimization/lr-scheduler.html) | 同一条下降路上，学习率按计划变小（或热身再衰减） |

11.1 是优化挑战综述。11.5 小批量 SGD 已在 3.1，且含从零/简洁。11.9 Adadelta 太接近 11.8。都不做正片。

## 第 12 章 · 计算性能

12.1–12.7 是编译器、异步、并行、硬件、多 GPU、参数服务器。整章不做正片。

## 第 13 章 · 计算机视觉

| 集 | D2L | 机制 |
| --- | --- | --- |
| 13.3 | [目标检测和边界框](https://zh.d2l.ai/chapter_computer-vision/bounding-box.html) | 物体用矩形框住：两角 \((x_1,y_1,x_2,y_2)\) 与中心+宽高互相转 |
| 13.4 | [锚框](https://zh.d2l.ai/chapter_computer-vision/anchor.html) | 格子上铺不同形状的锚框，与真框算 \(\mathrm{IoU}=|A\cap B|/|A\cup B|\) |
| 13.5 | [多尺度目标检测](https://zh.d2l.ai/chapter_computer-vision/multiscale-object-detection.html) | 粗特征图放大感受野、细特征图盯小物体，多层格子同时铺锚框 |
| 13.10 | [转置卷积](https://zh.d2l.ai/chapter_computer-vision/transposed-conv.html) | 一个输入格乘核后铺开，输出比输入更大（单个输入的贡献铺到输出，重叠处相加；不是逆运算） |

13.1 / 13.2 是增广与微调。13.6 / 13.9 是数据集。13.7 / 13.8 / 13.11 / 13.12 是检测与分割名录。13.13 / 13.14 是 Kaggle。都不做正片。

## 第 14 章 · NLP 预训练

| 集 | D2L | 机制 |
| --- | --- | --- |
| 14.1 | [词嵌入（word2vec）](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html) | 跳元：中心词向量 \(\mathbf{v}_c\) 拉近窗口内上下文 \(\mathbf{u}_o\)；独热彼此正交 |
| 14.7 | [词的相似性和类比任务](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/similarity-analogy.html) | 词向量平行四边形：\(\mathrm{king}-\mathrm{man}+\mathrm{woman}\approx\mathrm{queen}\) |

14.2 近似训练、14.3 / 14.9 数据集、14.4 / 14.10 预训练实现、14.5 GloVe、14.6 子词、14.8 BERT 架构。都不做正片。

## 第 15 章 · NLP 应用

| 集 | D2L | 机制 |
| --- | --- | --- |
| 15.5 | [自然语言推断：使用注意力](https://zh.d2l.ai/chapter_natural-language-processing-applications/natural-language-inference-attention.html) | 前提每个词查询假设的键，注意力对齐后再做推断 |

15.1 / 15.4 是数据集。15.2 / 15.3 / 15.6 / 15.7 是情感与 BERT 微调名录。都不做正片。

## 第 16–18 章

第 16 章是附录：Jupyter / SageMaker / EC2 / GPU 选型 / 贡献 / `d2l` API。整章不做正片。在线 [zh.d2l.ai](https://zh.d2l.ai/) 2.0.0 目录到第 16 章为止，没有第 17、18 章编号（推荐系统、GAN 未进本章目录）。不发明集号。

质量约定以 `docs/CREATIVE_PRINCIPLES.md` 为准：章节标识克制、结构身份可追踪、多层运动与连续衔接。数学和工程检查不能自动替代艺术反馈。
