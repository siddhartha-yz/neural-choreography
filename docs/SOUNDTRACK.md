# 点还在动 · 配乐草稿

已发布的 `zanim-complete-v1` 是 49 段静音作品。在两分钟钢琴夜曲样片之后，用户要求将此方向扩展到整片，并明确要求调性与编排有差异。完整钢琴版覆盖相同的 681.05 秒画面，静音版和历史试听版均保留。工程检查不能代替音乐审听。

## 钢琴夜曲完整版

179 小节、3245 个音符事件，仍为独奏钢琴。开场保留样片前 24 小节的演奏时序，将原样片的闭合尾声改写为转调连接。后续增加三个主题，并在六个调性区域之间展开，七个音乐段落有不同的伴奏密度、音区、力度与句法。

| 时间 | 音乐段落 | 调性与编排 |
| --- | --- | --- |
| 0:00–2:00 | 夜曲主题 | 降 A 大调，连接到降 D；保留样片的歌唱式开场 |
| 2:00–4:00 | 空间舒展 | 降 D 大调；新旋律、宽幅波浪琶音、高低音区呼应 |
| 4:00–5:44 | 往复与记忆 | F 小调；较低的旋律音区、稀疏伴奏与新的小调主题 |
| 5:44–7:12 | 光线交织 | 降 E 大调；明亮主题与较轻的内声部回答 |
| 7:12–9:16 | 推进与高潮 | C 小调；四连音伴奏、较强力度与八度强调，再连接到 E 大调 |
| 9:16–10:21 | 明亮的远景 | E 大调；新的明亮色彩与宽幅琶音，随后回到降 A |
| 10:21–11:21 | 主题归来与收束 | 降 A 大调；主题重现、放慢的尾声、最后和弦随粒子环与片尾标题释放 |

乐句跨章节连接。速度随句法舒展；音乐只在较大的内容转折处对应画面，不给每个标题配置一次重新开场。全片音符与踏板一次连续渲染，未用两分钟音频循环拼接。

复现完整版：

```bash
python -m scripts.compose_film_nocturne \
  --soundfont /absolute/path/SalamanderGrandPiano-V3+20200602.sf2 \
  --timeline media/complete/timeline.json --output media/piano-complete
python -m scripts.package_score \
  --video media/complete/video.mp4 \
  --timeline media/complete/timeline.json --output media/piano-complete
```

打包命令输出 49 段播放器、带 AAC 音轨的 `neural-choreography-piano.mp4`、24-bit/48 kHz 立体声 WAV、MP3、章节文件、时间线和版本记录。`nocturne.mid` 与 `score.json` 保存演奏时序，播放器另提供七个音乐段落的定位。钢琴采样来源与许可随页面、音视频元数据和离线包一起保留。

本地：`../outputs/neural-choreography-piano-final/index.html`。远程：[钢琴夜曲版](https://github.com/siddhartha-yz/neural-choreography/releases/tag/piano-edition-v1)。

验证包含原样片前 90 秒的未母带音频逐采样一致性、全片音视频解码、40,863 帧、49 个章节、48 kHz 立体声音轨、全片响度与真峰值，以及九处原静音片的画面像素对照。版本记录分别保存文件检查结果和完整配乐待审听状态。

## 草稿 02 · 独奏钢琴夜曲

用户反馈第一版不够好听，指定肖邦夜曲式的方向。第二版重新写作，采用独奏钢琴；主题与和声为原创，没有转录肖邦现有曲目。

降 A 大调出发，中段转向 C 小调，采用 12/8 拍的低音与宽幅分解和弦。右手承担歌唱式旋律，局部加入倚音，主题重现时加入回音装饰。28 小节按 A–B–A′–尾声展开，手写的句尾延缓、中段推进与触键力度取代固定章节循环。左右手略有先后，踏板随和声更换；结尾让终止和弦自然释放。

旋律、音符事件、踏板与节奏伸缩保存在 `score.json`；`nocturne.mid` 保存实际演奏时序，便于继续编辑或换用钢琴音源。两分钟试听完整覆盖原片开头十段，音乐乐句不强制卡在每个章节标题上。

音源使用 Alexander Holm 的 [Salamander Grand Piano](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html)，本次采用 Roberto / FreePats 转换的 V3+20200602 SF2。采样作者和 CC BY 3.0 链接写入试听页、文件元数据与记录；原音库不修改，不提交到 Git。实际钢琴音色仍是采样渲染，此阶段没有真人钢琴演奏录音。

安装 FluidSynth 共享库（本次 Linux 环境使用 `libfluidsynth.so.3`）与 `requirements-score.txt`。下载并解压 [SF2 音库](https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3+20200602.tar.xz)，在仓库根目录运行：

```bash
python -m scripts.compose_nocturne \
  --soundfont /absolute/path/SalamanderGrandPiano-V3+20200602.sf2 \
  --output media/nocturne-study
python -m scripts.package_score \
  --video media/complete/video.mp4 \
  --timeline media/complete/timeline.json --output media/nocturne-study
```

本地新样片：`../outputs/soundtrack-nocturne-v2/index.html`。原电子草稿和静音完整版均保留。响度、完整解码与原片像素对照仍由打包命令检查，夜曲风格及音乐审听等待实际反馈。

## 草稿 01 · 历史电子版本

主题为清澈、缓慢展开的电子室内乐：柔和键音承担旋律，持续弦音连接和声，短拨音建立脉动，高音玻璃质感提供少量声部回应。没有人声、旁白或外部录音；所有音源由数值合成生成，没有搬用参考项目的音乐或曲谱。

80 BPM、4/4 拍，四小节正好 12 秒，贴合本片开头十段的时间线。调性以 D 小调与开放九度和弦为主，共同音保持连续，主旋律在相邻段落中改变音区与方向。转场不统一添加冲击声。结尾四秒渐隐仅用于试听截段，不代表完整电影的音乐结尾。

| 画面时间 | 内容 | 音乐编排 |
| --- | --- | --- |
| 0–12 | 线性回归 | 稀疏主题，脉动稍后进入 |
| 12–24 | Softmax | 同一主题向共同音收拢，收为一与成环处点缀高音 |
| 24–36 | 多层感知机 | 不同音区的声部依次回应 |
| 36–48 | 欠拟合与过拟合 | 主题展开，和声保持连接 |
| 48–60 | 权重衰减 | 减少拨音密度与旋律力度 |
| 60–72 | 暂退法 | 脉动中留出间隙 |
| 72–84 | 正反传播 | 拨音方向反转，高音声部向下回答 |
| 84–96 | 初始化与稳定性 | 留出呼吸，准备再次展开 |
| 96–108 | 层和块 | 主旋律返回，声部重新汇合 |
| 108–120 | 卷积 | 脉动更清晰，预示后续空间结构 |

这是音乐与画面形态的创作对应，不表示声音在计算或证明模型行为。

## 复现

在仓库根目录运行，提供完整电影的现有时间线。需要 Python、NumPy、SciPy、FFmpeg 与 ffprobe。

```bash
python -m pip install -r requirements-score.txt
python -m scripts.compose_score \
  --timeline media/complete/timeline.json --output media/score-study
python -m scripts.package_score \
  --video media/complete/video.mp4 \
  --timeline media/complete/timeline.json --output media/score-study
```

`compose_score` 写入五条分轨、未母带混音与 `score.json` 音符事件记录。`package_score` 以两遍响度测量生成 24-bit/48 kHz 立体声 WAV、192 kbps MP3 和带音轨的两分钟 MP4。目标响度为 −18 LUFS、真峰值上限 −1.5 dBTP；编码后的结果再测量。

视频流直接复制，播放器保留十段定位、纯享、下载和静音对比；原 11 分 21 秒的静音文件不被覆盖。打包时检查完整解码、7200 帧、1080p60、音频规格、十段章节以及三处原片像素一致性，结果写入 `report.json`。这些项目验证文件与同步，不判定音色、旋律或艺术效果已经通过审听。

本地样片：`../outputs/soundtrack-study-v1/index.html`。试听页需点击播放才能开启声音。
