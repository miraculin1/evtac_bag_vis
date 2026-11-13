# bag_vis

将 ROS bag 中的**事件相机**与**六维力传感器**数据合成为 1920×1080、30 fps 的对比视频。

## 输出内容

```
┌────────────────┬────────────────────────────┐
│  事件图 +       │  TORQUE / TIME (Nm)        │
│  半透明 fx-fy   │  tx / ty / tz 三条曲线      │
│  HUD            ├────────────────────────────┤
│  480×640 竖放   │  FZ / TIME                 │
└────────────────┴────────────────────────────┘
```

- 事件图：红色 ON、蓝色 OFF。事件率较低时使用“最后事件时间”衰减图；事件率很高时自动改用“事件计数密度”图，避免噪声把画面冲爆。
- 事件图中央叠加半透明 `fx-fy` HUD：
  - 当前力向量箭头、最近 2.5 秒轨迹、十字坐标和 `fx / fy / |fxy|` 数值。
  - 方向与旋转后的事件图一致：`fx` 负方向向右，`fy` 负方向向下。旧 bag 可用 `--force-orientation old` 恢复原来的 `fx+` 向下、`fy-` 向右。
  - 触觉端力传感器测量的是反作用力，因此力与扭矩均取反后再显示。
- 右上角为扭矩时间曲线：`tx / ty / tz` 三条曲线，当前时刻竖线、当前点和图例数值。
- FZ 曲线：完整曲线为暗灰色，已播放部分为绿色；黄色竖线为当前时刻，旁边标注当前 `fz` 数值。由于力已取反，`fz` 曲线位于负半轴。
- 每个 bag 单独导出一个 mp4。

> `session06` 的事件流中还包含一个约 **15 Hz 的全局闪烁**。主视频使用 200 ms 的事件密度窗口，会把闪烁平均掉；为此额外提供了下面的专用可视化。

## session06 闪烁机制专用可视化

`flicker_vis.py` 单独分析 `session06` 事件流中的周期性闪烁：

- 以 0.1 ms 分辨率统计全局事件数，检测到约 850 次 burst。
- 周期 **66.70 ms（14.99 Hz）**，每次 burst 是一个持续约 1–2 ms 的全局闪光，随后有一个约 15 ms 的恢复尾巴。
- 以 128 个相位对事件进行平均，得到闪烁周期内的相位平均事件图像。
- 恢复时间判据：burst 峰值后，连续 2 个 1 ms 分箱回落到安静期 P95 阈值以下。
  - 安静期基线中位数：**19 events/ms**
  - 恢复阈值：**49 events/ms**（细化 burst 峰值后的安静期 P95）
  - 中位恢复时间：**15 ms**，P90：**20 ms**
- FFT 显示 15 Hz 基频以及 30 / 45 / 60 Hz ... 谐波。

输出：

```text
output/session06_slow_press_flicker_analysis.png   静态诊断图
output/session06_slow_press_flicker.mp4            闪烁分析视频
```

运行：

```bash
/usr/bin/python3 flicker_vis.py
```

常用参数：

```bash
# 只生成静态诊断图
/usr/bin/python3 flicker_vis.py --no-video

# 指定视频时间段
/usr/bin/python3 flicker_vis.py --start 20 --duration 20
```

视频内容：

- 左侧：相位平均事件图，慢速循环展示一个 66.7 ms 闪烁周期；下方 filmstrip 显示 8 个代表相位。
- 右上：全局事件率示波器（1 ms 分箱，对数纵轴），可以看到每 66.7 ms 一个尖峰、恢复阈值线，以及每个 burst 后的恢复区间阴影。
- 右中：burst 触发的平均波形（0.1 ms 分箱），标出恢复阈值和 15 ms 恢复点。
- 右下：左侧为事件率 FFT，右侧为恢复时间直方图（中位 15 ms、P90 20 ms）。


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

直接指定 folder 路径（位置参数或 `--folder` 均可）：

```bash
/usr/bin/python3 bag_vis.py /path/to/bags --out-dir output
/usr/bin/python3 bag_vis.py --folder /path/to/bags --out-dir output
```

指定输出目录：

```bash
/usr/bin/python3 bag_vis.py --data-dir data --out-dir output
```

只处理某个 bag：

```bash
/usr/bin/python3 bag_vis.py data/test1.bag
/usr/bin/python3 bag_vis.py --bags data/test1.bag
```

多个 bag 并行处理（每个 bag 一个线程，带独立进度条）：

```bash
# -j 0 为自动（按 bag 数量和 CPU 核数，最多 8 个）
/usr/bin/python3 bag_vis.py /path/to/bags -j 4 --out-dir output
```

递归搜索子目录：

```bash
/usr/bin/python3 bag_vis.py /path/to/bags --recursive
```

先导出某一时刻的预览图（不编码视频）：

```bash
/usr/bin/python3 bag_vis.py --preview 30
```

只渲染一段用于检查：

```bash
/usr/bin/python3 bag_vis.py --start 28 --end 32 --out-dir output/clips
```

## 事件-力时间软同步

`tac_bag/` 中的事件流相对力变化存在约 0.4 s 的延迟。`sync_calibrate.py` 用
“事件计数 vs 力变化率”的窗口化互相关估计每个 bag 的延迟，并写出 JSON：

```bash
/usr/bin/python3 sync_calibrate.py tac_bag -j 4 \
  -o calibration/event_force_sync.json
```

当前 `tac_bag/` 的校准结果：

- `cl_train03_fast_press`: **+376 ms**
- `cl_val02_mixed_slide`: **+493 ms**
- 全局中位数：**+413 ms**（9/11 个 bag 质量合格）
- `cl_train01_null`、`cl_train02_slow_press` 质量不足，自动回退到全局值

`bag_vis.py` 通过 `--sync-calib` 消费校准文件：

```bash
/usr/bin/python3 bag_vis.py tac_bag -j 4 \
  --sync-calib calibration/event_force_sync.json \
  --out-dir out_bright
```

符号约定：正值表示**事件滞后力**，`bag_vis.py` 会把力时间轴向前移动该值。
力曲线、fz/torque 曲线和 fx-fy HUD 会一起对齐；不会修改原始 bag。

也可以手动覆盖或禁用：

```bash
# 手动指定 400 ms
/usr/bin/python3 bag_vis.py tac_bag/cl_train03_fast_press.bag --force-shift 0.4

# 不做时间对齐
/usr/bin/python3 bag_vis.py tac_bag/cl_train03_fast_press.bag --no-sync
```

## 常用参数

| 参数 | 说明 |
|---|---|
| `paths` | 位置参数，可以是 bag 文件、folder 或 glob |
| `--folder` / `--data-dir` | 存放 `.bag` 的目录，默认 `data` |
| `--bags` | 显式指定 bag 路径，兼容旧用法 |
| `--recursive` | folder 模式下递归搜索子目录 |
| `-j` / `--jobs` | 并行处理的 bag 数量；`0` 为自动 |
| `--preset` | x264 preset，默认 `veryfast`；要更小文件可用 `medium` |
| `--encoder` | `libx264`（默认）或 `h264_nvenc`（NVIDIA GPU） |
| `--fps` | 帧率，默认 30 |
| `--crf` | 视频质量，默认 20，越小越清晰、文件越大 |
| `--event-mode` | `auto` / `surface` / `density`，默认自动选择 |
| `--event-tau` | 事件衰减时间（秒）；不填则自动估计 |
| `--force-orientation` | `new`（默认，`fx-` 向右、`fy-` 向下）或 `old`（`fx+` 向下、`fy-` 向右） |
| `--sync-calib JSON` | 使用 `sync_calibrate.py` 生成的校准文件 |
| `--force-shift SECONDS` | 手动力时间偏移，优先于校准文件 |
| `--no-sync` | 禁用事件/力时间对齐 |
| `--start` / `--end` | 只渲染 bag 内指定时间段 |
| `--preview SECONDS` | 只输出该时刻的 PNG |

## 实现说明

- 事件消息通过 `read_messages(..., raw=True)` 读取并按二进制直接解析，每个事件 13 字节，避免几亿个 Python 对象导致内存/速度问题。
- 事件按帧批量拼接，避免每来一条 1 kHz 消息就复制一次缓冲区。
- 事件表面使用 float32 相对时间戳，并复用中间缓冲区；density 模式只对事件覆盖的 ROI 做直方图统计。
- 布局中的网格、坐标标签和完整暗色曲线只绘制一次，之后每帧复制静态图层；标签框和游标只混合局部 ROI，不再整幅复制 1080p 画布。
- 力话题自动识别：
  - `geometry_msgs/WrenchStamped` → `wrench.force.x/y/z` + `wrench.torque.x/y/z`
  - `force_data_process/Force` → `fx/fy/fz` + `tx/ty/tz`
- 时间轴使用 bag 的 header 时间戳；事件按时间戳逐帧累积，力/扭矩数据在帧时刻线性插值。
- 视频帧通过管道直接送入 ffmpeg（H.264 / yuv420p / faststart），不生成中间图片序列；默认 `-preset veryfast`。
- 多个 bag 通过 `ThreadPoolExecutor` 并行处理，每个 bag 输出独立进度条。
- `sync_calibrate.py` 使用 10 ms 分箱、50 ms 平滑、5 s 窗口 / 2.5 s 步长的互相关，取窗口延迟中位数和 MAD；质量不足时使用全局中位数。
- `bag_vis.py` 在渲染时把力时间轴平移校准值，不修改原始 bag；力曲线和事件时间轴因此对齐。
