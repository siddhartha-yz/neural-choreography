# Neural Choreography · 点还在动

一部以神经网络的状态传递、几何形变与群体运动编排的连续动画作品。内容参考《动手学深度学习》，以黑底、克制配色、清晰连接与多层运动表达结构，保留中文主题、章节编号和一句极简提示。

**Zanim 完整版：49 段，11 分 21 秒，1920×1080，60 fps，H.264，静音。** 第 3–15 章按教材顺序连续播放，覆盖原目录全部选定机制。播放页提供章节定位、全屏、纯享与整片下载；MP4 内也写入 49 个章节。

- [下载整片与离线播放器](https://github.com/siddhartha-yz/neural-choreography/releases/tag/zanim-complete-v1)
- [完整目录](CATALOG.md)
- [时间线、机制与制作记录](docs/CONTINUOUS_STUDY.md)
- [创作约定](docs/CREATIVE_PRINCIPLES.md)
- [原创配乐试听与复现](docs/SOUNDTRACK.md)：先制作两分钟音画草稿，已发布的静音整片保留。
- [第 23 版三十段制作记录](docs/archive/CONTINUOUS_STUDY_V23.md)

本版保留已有连续片段，补齐注意力、优化器、计算机视觉与语言后段。Softmax 明确展示三个正响应共同除以总量、总长收为一；多层感知机采用 2→4→2 全连接图，信号汇入、ReLU 归零和再分发都发生在连接结构上。片尾收束到与开场呼应的粒子环。

完整范围、数学检查、导出通过与艺术认可分别记录。新增片段不因工程验证通过而自动标为人工过审。第 22 版的 04.4–04.6 已获用户认可；早期单集人工过审状态保留在目录中。

## 安装

本版验证环境为 Linux x86_64、Python 3.12、Zanim 0.7.0rc1。依赖固定在 `requirements-zanim.txt`；Windows/macOS 提供对应 wheel，但完整电影的本次导出在 Linux 上验证。

需要 FFmpeg / ffprobe，以及 Noto Sans CJK SC 中文字体。Typst 使用已固定的 Python 绑定适配器，不必另外安装命令行编译器。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-zanim.txt
```

Debian/Ubuntu 可安装 `ffmpeg fonts-noto-cjk python3.12-venv`。使用其他系统时安装对应工具与字体。

## 渲染完整作品

在仓库根目录运行：

```bash
.venv/bin/python -m scripts.render_continuous --profile final \
  --output media/complete/video.mp4 --timeline media/complete/timeline.json
.venv/bin/python -m scripts.deliver_continuous media/complete/video.mp4 \
  --timeline media/complete/timeline.json
```

打开 `media/complete/index.html` 观看。交付命令要求全片解码、视频规格和时间线检查通过后才生成播放器、版本记录和章节文件。

用 `--profile preview` 快速查看，或用 `--time 27.8 --output media/frame.png` 生成单帧。`--start` / `--end` 可只导出指定区间；区间需落在 60 fps 帧格上。每次视频导出会验证原生随机访问，并将优化后的渲染区间与原场景做逐像素对照。

## 数学与工程检查

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff check --select E9,F63,F7,F82 zanim_scenes scripts tests
.venv/bin/python -m unittest discover -s tests -v
```

本轮环境通过 64 项检查，覆盖 RNN 递推、门控与反向梯度、注意力归一化、原始 MLP 输出、卷积贡献、49 段唯一覆盖、绘图边界与渲染区间一致性。全片规格和完整解码由交付命令检查；这些检查不自动批准艺术效果。

## 源码

- `zanim_scenes/continuous.py`：整部作品入口与章节时间线。
- `zanim_scenes/classification_art.py`：共同归一化和全连接 MLP。
- `zanim_scenes/finale_art.py`：10.3–15.5 的十九段与收尾。
- `zanim_scenes/models/`：与渲染分离的原数值模型。
- `scripts/render_continuous.py`、`scripts/deliver_continuous.py`：渲染与交付。
- `episodes/`：保留早期 Manim 单集，供历史对照。

`python -m scripts.render 13.10 --profile final` 仍可生成早期 Zanim 独立单集；完整作品以 `render_continuous` 为准。MP4 不进 Git 历史，本地输出与 Release 保存成片。

第 1–2 章、实现与 API 小节、数据集、工程工具和纯架构名录不做正片，取舍见目录。数学示例与跨章构图独立；未训练模型不声称完成真实任务。

## 许可

MIT。内容结构参考 D2L，动画为独立作品。
