# 10.2 片外说明

这一页给看完默片的人。**不进视频。**

教材：[D2L 10.2 注意力汇聚：Nadaraya-Watson 核回归](https://zh.d2l.ai/chapter_attention-mechanisms/nadaraya-waston.html)

## 先说问题

手里已经有一堆带标签的点：每个键 \(k_i\) 旁边挂着一个值 \(v_i\)。现在来了一个新的查询 \(q\)，你要吐出一个数。

平均所有 \(v_i\) 太笨：远处的点和近处的点一样说话。Nadaraya–Watson 的办法是——查询越靠近哪个键，那个值的权重越大，再加权求和。权重来自高斯核再 softmax，所以非负，且和为 1。这就是注意力汇聚。模型没有参数要训，片里的曲线就是这个估计器。

默片只演这一件：光晕旅行者坐在 \(q\) 上，训练点上的针跟着远近亮暗；\(\hat{y}\) 落在核回归曲线上。没有旁白。蓝黄只是点的身份，不是两类。

## 画面里在干什么

1. **中文片头（4 秒）**：内容名称、章节号和一句极简讲解。
2. **训练点先停住**。键是 1-D 的 \(x\)，值是 \(y\)。真实关系是 \(2\sin x+x^{0.8}\)，再加噪声。九点。
3. **圈住查询**。旅行者坐在横轴上，左侧固定写 \(q\)。这是查询，不是某个训练点的坐标。
4. **针长在点上**：\(\alpha_i=\mathrm{softmax}(-\frac12\|q-k_i\|^2)\)。\(q=1.00\) 时最近的键 \(k=0.90\) 最亮，\(\alpha=0.247\)。右边几乎不亮。
5. **\(\hat{y}\) 落在曲线上**。\(q=1.00\) 时 \(\hat{y}=+2.672\)。底下写 \(\sum\alpha_i=1.000\)。公式 \(\hat{y}=\sum\mathrm{softmax}(-\frac12\|q-k_i\|^2)\,v_i\) 出现一次。
6. **查询挪到第二处** \(q=3.60\)。重量跟着跑到右边，最大针变成 \(k=3.40\) 上的 \(\alpha=0.229\)，\(\hat{y}=+2.249\)。还是同一条核回归曲线，没有训练。

## 刻意没演

平均汇聚当对照、注意力热图、可学习的带宽 \(w\)。下一集试点是 [10.3 注意力评分函数](https://zh.d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html)：还是加权求和，换成加性和缩放点积两种 \(a(q,k)\)。

## B 站可粘贴

**标题**

点还在动 · 10.2 Nadaraya-Watson

**简介**

查询靠近哪个键，那个值就更重。没有旁白。

《动手学深度学习》10.2：https://zh.d2l.ai/chapter_attention-mechanisms/nadaraya-waston.html

圈住的样本会在后面几集再出现。源码与成片：https://github.com/siddhartha-yz/neural-choreography/releases/tag/10.2
