# 14.7 片外说明

这一页给看完默片的人。**不进视频。**

教材：[D2L 14.7 词的相似性和类比任务](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/similarity-analogy.html)

## 先说问题

词被写成向量之后，加减法开始有语义。书里的类比是：给出 \(a:b::c:d\) 的前三个词，找 \(d\)，使得

\[
\mathrm{vec}(d)\approx\mathrm{vec}(c)+\mathrm{vec}(b)-\mathrm{vec}(a).
\]

「man 之于 woman，如同 king 之于 queen」就是 \(\mathrm{king}-\mathrm{man}+\mathrm{woman}\approx\mathrm{queen}\)。几何上这是平行四边形：从 king 走出 woman 相对 man 的那一条边，落点应该靠近 queen。

默片只演这一件：四个玩具二维词向量，一条平行四边形，合成点带着光晕停在 queen 旁边。没有旁白。向量是写死的，不是训出来的。不把词表按余弦排成一张表。

光晕圈的是**合成向量 \(\hat y\)**，不是 queen 自己。\(\hat y\) 和 queen 之间那截虚线就是 \(\approx\)：靠近，不相等。

## 画面里在干什么

1. **中文片头（4 秒）**：内容名称、章节号和一句极简讲解。
2. **描边坐标格**，轴上是 \(x_1,x_2\)。二维，所以平行四边形能画在纸上。
3. **四个词元**：man、woman、king、queen。蓝与白是身份，不是「男/女」两类。
4. **公式只出现一次**：`king − man + woman ≈ queen`。
5. **先画 man→woman、man→king**，再从 king 把同一条偏移走出去。黄点 \(\hat y\) 从 king 滑到 \((2.02,\,1.75)\)。
6. **光晕圈住 \(\hat y\)**。左侧固定口袋写 \(ŷ_1=2.02\)、\(ŷ_2=1.75\)，就是这个点的两个分量，不是一张向量清单。
7. **补上 woman→\(\hat y\)**，平行四边形合上。虚线连到 queen。停住时四点、平行四边形、旅行者都还在。

源数组（numpy：\(\hat y=\mathrm{king}-\mathrm{man}+\mathrm{woman}\)）：

\[
\begin{aligned}
\mathrm{man}&=(0.30,\,0.35),&
\mathrm{woman}&=(1.80,\,0.48),\\
\mathrm{king}&=(0.52,\,1.62),&
\mathrm{queen}&=(1.84,\,1.52),\\
\hat y&=(2.02,\,1.75).
\end{aligned}
\]

## 刻意没演

余弦 \(k\) 近邻表、GloVe / fastText 加载、BERT、首都-国家或比较级那些其它类比。下一集若走 15.5，才轮到注意力对齐句子。

## B 站可粘贴

**标题**

点还在动 · 14.7 词的类比

**简介**

king 减 man 加 woman，平行四边形落在 queen 旁边。没有旁白。

《动手学深度学习》14.7：https://zh.d2l.ai/chapter_natural-language-processing-pretraining/similarity-analogy.html

圈住的是合成向量。源码与成片：https://github.com/siddhartha-yz/neural-choreography/releases/tag/14.7
