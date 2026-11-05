# bag_vis

将 ROS bag 中的**事件相机**与**六维力传感器**数据合成为 1920×1080、30 fps 的对比视频。

## 输出内容

```
┌────────────────┬────────────────────────────┐
│                │   FX-FY 二维力向量          │
│  事件图像       │   当前向量 + 最近轨迹       │
│  顺时针旋转90°  │   FX / FY / |FXY| 数值      │
│  480×640 竖放   ├────────────────────────────┤
│                │   FZ 随时间折线             │
│                │   当前时刻竖线 + FZ 标注     │
└────────────────┴────────────────────────────┘
```

- 事件图：红色 ON、蓝色 OFF。事件率较低时使用“最后事件时间”衰减图；事件率很高时自动改用“事件计数密度”图，避免噪声把画面冲爆。
- 力向量：`(fx, fy)` 从原点画箭头，并显示最近 2.5 秒轨迹。
- FZ 曲线：完整曲线为暗灰色，已播放部分为绿色；黄色竖线为当前时刻，旁边标注当前 `fz` 数值。
- 每个 bag 单独导出一个 mp4。

## 环境

脚本依赖 ROS Noetic 的 `rosbag`，以及 `numpy`、`opencv-python`、`ffmpeg`。
当前机器上应使用系统 Python：

```bash
/usr/bin/python3 bag_vis.py --help
```

> 注意：`python3` 默认指向 miniforge，里面没有 `rosbag`/`numpy`，请使用 `/usr/bin/python3`。

## 使用

处理 `data/` 下所有 `.bag`：

```bash
/usr/bin/python3 bag_vis.py
```

指定输出目录：

```bash
/usr/bin/python3 bag_vis.py --data-dir data --out-dir output
```

只处理某个 bag：

```bash
/usr/bin/python3 bag_vis.py --bags data/test1.bag
```

先导出某一时刻的预览图（不编码视频）：

```bash
/usr/bin/python3 bag_vis.py --preview 30
```

只渲染一段用于检查：

```bash
/usr/bin/python3 bag_vis.py --start 28 --end 32 --out-dir output/clips
```

## 常用参数

| 参数 | 说明 |
|---|---|
| `--fps` | 帧率，默认 30 |
| `--crf` | x264 质量，默认 20，越小越清晰、文件越大 |
| `--event-mode` | `auto` / `surface` / `density`，默认自动选择 |
| `--event-tau` | 事件衰减时间（秒）；不填则自动估计 |
| `--start` / `--end` | 只渲染 bag 内指定时间段 |
| `--preview SECONDS` | 只输出该时刻的 PNG |

## 实现说明

- 事件消息通过 `read_messages(..., raw=True)` 读取并按二进制直接解析，每个事件 13 字节，避免几亿个 Python 对象导致内存/速度问题。
- 力话题自动识别：
  - `geometry_msgs/WrenchStamped` → `wrench.force.x/y/z`
  - `force_data_process/Force` → `fx/fy/fz`
- 时间轴使用 bag 的 header 时间戳；事件按时间戳逐帧累积，力数据在帧时刻线性插值。
- 视频帧通过管道直接送入 ffmpeg（H.264 / yuv420p / faststart），不生成中间图片序列。
