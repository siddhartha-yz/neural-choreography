# Neural Choreography

以神经网络的状态传递、几何形变与群体运动编排连续动画。中文系列名 **点还在动**，内容参考 D2L。

内容参考《动手学深度学习》。目标是一整串可连续欣赏的神经网络动画作品，以群体运动、几何形变、空间感与章节衔接构成观看体验。中文主题、章节编号和极简提示保留在动画角落，默认静音。

## 创作约定

以 [创作约定](docs/CREATIVE_PRINCIPLES.md) 为准：先编排相邻章节的运动母题、视觉层次与形态交接，再制作连续动态样片。取消上一轮以提问、预测和因果讲解为核心的教学分镜要求。固定主角、固定片头和固定时长也不作为新作模板。

当前 [十四章节串联样片](docs/CONTINUOUS_STUDY.md) 将 07.4 多尺度分支、07.5 批量归一化、07.6 残差修正、07.7 稠密特征拼接、08.4 循环流动、08.7 反向回传、09.1 门控汇合、09.2 贯通记忆主干、09.3 逐层状态传递、09.4 双向拼接、09.6 编码生成、09.8 候选分叉、10.1 查询匹配与 10.2 加权汇聚放在同一个连续舞台。07.4–07.7 已改为连续三维特征面编舞，艺术效果待实际审看；已有 49 集技术迁移不等于已完成整部作品。

```bash
python -m scripts.render_continuous --profile final --output media/continuous-study/video.mp4
```

默认成片规格为 1920×1080、60 fps、H.264、静音。MP4 不进 Git 历史。`episodes/` 保留旧场景，`zanim_scenes/` 为当前原生实现，`experiments/` 为独立技法试验。

## 从这两集开始

| 集 | D2L | 机制 |
| --- | --- | --- |
| `episodes/03.1` | [3.1 线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html) | 直线用小批量 SGD 在点里找位置；高亮样本的残差缩短；留下噪声残差 |
| `episodes/03.4` | [3.4 softmax回归](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html) | 前向 \(\mathbf{o}=W\mathbf{x}+b\) → softmax；分数可负，概率非负且和为 1；不塞交叉熵、不假装已训练 |

系列从第 3 章开始。第 1–2 章和「从零实现」小节不做正片。现有 2→2→2 前向片在 `experiments/`，更接近 4.1 的一小段，只当技法试验。

仓库包含 49 集场景源码，涵盖第 3–15 章的选定机制：[完整目录](CATALOG.md)。源码存在、自动渲染通过与人工过审是不同状态；已过审成片见目录顶部的 Release 链接。

## Zanim 迁移与新视觉

已完成 **49 / 49 集** 的原生 Zanim 重写。统一入口 `scripts.render` 对全集使用 Zanim；`episodes/` 保留原 Manim 实现用于数学对照，直接执行旧场景不会生成新版。

- 新版采用深蓝背景、青绿／珊瑚强调色、统一中文字体和固定信息布局。
- 通道采用分层视图；转置卷积的贡献按数值高度叠加；优化器沿实际损失曲面运动。循环状态、门控贡献和注意力权重用数值柱形展示，多尺度检测用分层平面展示。3D 的空间含义直接写在画面里；拟合曲线、计算图、边界框与词向量关系使用原生二维图形。
- 每集保留中文标题和极简讲解。多数章节采用 4 秒片头；08.4 动效样片把标题直接融入连续场景。计算数据独立于渲染器，优化器轨迹逐点对照原版验证。
- 实测环境：Linux x86_64、Python 3.12、Zanim 0.7.0rc1 发布 wheel。仓库 HEAD 所写 rc2 尚未在本次读取的 Release 列表中提供，因此依赖固定到已验证的 rc1；其 Surface3D 使用 Y 轴表示高度。
- 若 PATH 上没有 Typst，统一入口使用 `typst==0.15.0` Python 绑定适配器编译中文矢量文字。仍需要 FFmpeg 和 Noto Sans CJK SC 字体。当前完整渲染验证在 Linux 上完成。

```bash
python -m scripts.render 13.10 --profile final
python -m scripts.render 11.6 --profile final
python -m scripts.render_zanim 06.4 --time 22 --output media/channel-frame.png
```

**动效二次迭代**：08.4 以黑底、蓝色网格和极简字幕展示整个状态空间的连续变换：循环权重旋转并压缩网格，当前输入推动网格，tanh 将网格弯曲收拢。黄、粉两点表示不同初始记忆，在相同的三次输入下逐渐靠近；结尾标明倍率后局部放大。每个网格点均按原 RNN 模型独立计算，源码见 `zanim_scenes/rnn_flow.py`。其余 48 集仍为第一轮动效，迁移完成不等于视觉质量已经验收。

原生源码位于 `zanim_scenes/`；迁移状态以 `scripts/engines.py` 为准。`episodes/` 中保留旧实现用于数学对照。所有 49 集均已注册到原生渲染入口。

## 渲染

```bash
sudo apt-get install -y $(cat apt-packages.txt)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

建议使用 Python 3.12；上面的系统包列表面向提供 Python 3.12 的 Debian/Ubuntu 环境。其他 Python 版本需安装匹配版本的 venv 包。

在仓库根目录运行，默认是快速预览（854×480、15 fps）：

```bash
.venv/bin/python -m scripts.render 03.1
.venv/bin/python -m scripts.render 03.1 --profile final
```

成片为 1920×1080、60 fps。预览和成片分别写入 `media/preview/` 与 `media/final/`，视频名为 `03.1-preview.mp4` / `03.1-final.mp4`；Manim 会在目录下再创建视频子目录。用 `--output` 更换媒体目录，`--dry-run` 查看命令。49 集的脚本不再覆盖分辨率和帧率，各集 README 中的直接 Manim 命令只生成旧版；新版统一使用 `scripts.render`。

## 检查与抽帧

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff check --select E9,F63,F7,F82 episodes experiments scripts tests
.venv/bin/python -m unittest discover -s tests -v
```

检查覆盖全部场景导入，以及 SGD 损失下降、softmax 归一化与平移不变性、反向传播与有限差分一致、边界框坐标往返、转置卷积重叠累加。未安装 Manim 时，数学检查会明确标记为跳过；CI 会安装完整依赖后执行。

渲染后传入实际视频路径，生成关键帧联系表和 JSON 报告（需要 `ffmpeg` / `ffprobe`）：

```bash
.venv/bin/python -m scripts.qa_video \
  media/preview/videos/scene/480p15/03.1-preview.mp4 \
  --profile preview --output media/review/03.1
```

默认抽取开场、0.5 秒、四分之一、中点、四分之三及最后一帧；可用 `--times 2 8 16` 指定关键时刻，最后一帧始终保留。用浏览器打开输出目录的 `index.html`，检查标签遮挡、符号可读性、数字自洽和结尾光晕主角是否仍在。联系表使用缩小的帧，精细检查仍需查看原视频。

工具检查时长有效性、对应模式的尺寸与帧率、H.264 编码及无音轨，并完整解码视频以发现损坏。元数据不合格返回退出码 1，解码或执行失败返回 2；自动检查通过仍需人工看帧、观看完整视频确认运动和节奏。每次抽帧的人工审核状态都为 `pending`。用 `--frame-width 1920` 导出清晰的关键帧（不超过原视频宽度），用 `--copy-video` 在审核页面中附上可播放的视频副本。

GitHub Actions 会运行检查并渲染 3.1、9.8 的完整预览，把视频和联系表上传为构建产物。它不自动发布 Release，也不自动标记“已过审”。

全量预览审核可在本地运行：

```bash
.venv/bin/python -m scripts.review_all --jobs 2
# 只检查指定集：
.venv/bin/python -m scripts.review_all --episodes 03.1 09.8 --jobs 2
# 全集成片验证（耗时较长）：
.venv/bin/python -m scripts.review_all --profile final --jobs 2
```

批量工具为每集隔离渲染缓存，个别失败不会中断其余集。默认在 `media/review-all/index.html` 展示进度与末帧，按集保存关键帧、JSON 报告和运行日志；任意一集失败或规格不合格时返回非零退出码。`--output` 可指定审核输出目录；每次运行的总览只包含本次选择的集数。

成片模式使用独立的 `media/batch-final/` 缓存，审核页面默认写到 `media/review-final/`，附带 1920 像素宽的关键帧与可播放的成片副本。点击关键帧可查看原尺寸图片；预览与成片的自动验证都不替代人工过审。

## 许可

MIT。内容结构对齐 D2L，动画是独立的无声可视化，不是教材再版。
