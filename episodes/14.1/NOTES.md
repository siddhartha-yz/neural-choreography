# 14.1 片外说明

这一页给看完默片的人。**不进视频。**

教材：[D2L 14.1 词嵌入（word2vec）](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html)

## 先说问题

独热把每个词放在互相垂直的轴上，点积是 0，谈不上谁跟谁更近。跳元（skip-gram）换了一种几何：中心词向量 \(\mathbf{v}_c\) 用来生成窗口里的上下文，训练就是把对应的 \(\mathbf{u}_o\) 拉近，让 \(\mathbf{u}_o^\top\mathbf{v}_c\) 变大。

默片只演这一件：四个 2-D 词向量先停在轴上，光晕圈住中心 \(\mathbf{v}_c\)，黄线标出窗口对 \((\mathbf{v}_c,\mathbf{u}_o)\)，然后六步真实的 softmax SGD 把 \(\mathbf{u}_o\) 拽过来。\(\mathbf{u}_n\)、\(\mathbf{u}_k\) 不在窗口里，停在远处。没有旁白。

## 画面里在干什么

1. **中文片头（4 秒）**：内容名称、章节号和一句极简讲解。
2. **四根向量先停在轴上**：\(\mathbf{v}_c\) 在 \(e_1\)，\(\mathbf{u}_o\) 在 \(e_2\)，一开始 \(\mathbf{u}_o^\top\mathbf{v}_c=0\)。这是独热彼此正交的 2-D 缩影，不是 4 维独热本身。
3. **圈住旅行者** \(\mathbf{v}_c\)。左下固定口袋写 \(s_o=\mathbf{u}_o^\top\mathbf{v}_c\)、\(s_n=\mathbf{u}_n^\top\mathbf{v}_c\)，开始都是 \(+0.00\)。
4. **公式出现一次**：\(P(w_o\mid w_c)\propto\exp(\mathbf{u}_o^\top\mathbf{v}_c)\)。黄线连上窗口对，不连 \(\mathbf{u}_n\)。
5. **六步 SGD**：损失是 \(-\log\mathrm{softmax}(U\mathbf{v}_c)_o\)，学习率 \(0.22\)。\(\mathbf{u}_o\) 走向 \(\mathbf{v}_c\)，\(s_o\) 升到 \(+2.18\)；\(\mathbf{u}_n\) 更远，\(s_n\) 落到 \(-1.37\)。
6. **停住时四根向量都在**。最终 \(\mathbf{v}_c\approx(0.982,\,1.009)\)，\(\mathbf{u}_o\approx(0.709,\,1.467)\)，距离 \(0.534\)（开始是 \(1.697\)）。\(\mathbf{u}_n\)、\(\mathbf{u}_k\) 仍在南边和西边。

数只来自同一份 numpy。没有负采样近似，词表就是这四个。

## 刻意没演

CBOW、负采样、层次 softmax、整张词表、预训练好的 embedding。14.7 才是类比平行四边形。

## B 站可粘贴

**标题**

点还在动 · 14.1 词嵌入

**简介**

中心向量把窗口里的上下文拉近。没有旁白。

《动手学深度学习》14.1：https://zh.d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html

圈住的样本会在后面几集再出现。源码与成片：https://github.com/siddhartha-yz/neural-choreography/releases/tag/14.1
