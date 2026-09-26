# 13.10 转置卷积

机制：一个输入格乘核后铺到更大的输出上（单个输入的贡献铺到输出，重叠处相加；不是逆运算）。片里只写旅行者那一格：\(x\cdot K\) 盖在 \(Y[i:i+h,j:j+w]\)。核不训练。空着的输出格保持空。

片外说明（不进视频）：[NOTES.md](NOTES.md)

D2L：https://zh.d2l.ai/chapter_computer-vision/transposed-conv.html

## Render

```bash
python -m manim episodes/13.10/scene.py Episode1310 \
  -r 1920,1080 --fps 60 --format mp4
```

成片约 31 秒，H.264，无音轨。MP4 放 GitHub Release，不进 Git。
