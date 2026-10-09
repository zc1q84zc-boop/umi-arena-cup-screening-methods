# 腕相机视觉对齐 v3 · 2026-10-08

控制台在 squirrel_5090 上的新建在线运行默认通过
`yubi_isaac_sim_env.run_visual_aligned` 启动当前修正版控制流程。
视觉配置 ID：`wrist_visual_aligned_v3`。

## 已应用

- 分别拟合左右腕的固定相机安装位姿与等距鱼眼焦距；相机仍跟随当前 GPU 刚体位姿。
- 夹爪改为黑色粗糙材质和红色指端饰面，保留电动夹爪的实际 CAD 指形。
- 隐藏训练手套画面中不存在的电机外壳、法兰适配件的可见网格，保留对应碰撞体。
- 白色程序化布面、柔和顶灯、固定三维实验室桌架与显示器背景。
- 固定杯盘颜色响应：杯子的少量发光填充近似透光蓝塑料，盘子的线性漫反射颜色为 `[0.24, 0.35, 0.26]`。
- Dome 强度 2200；每次 reset 后重新应用物体外观。没有逐帧追踪物体、图像贴背景、裁切到物体或在线改相机。

## 实际 RTX 渲染对照

训练参考为 episode 259632。使用帧 0、15、30 的固定腕姿态复现；15 为离线拟合的保留帧。
15／30 的关节来自先前参考回放，因此存在回放跟踪误差，不能把下列数字视为实物手眼标定精度。

| 三个姿态的平均中心误差，像素 | 修改前 | 修改后 |
|---|---:|---:|
| 左腕杯子 | 31.53 | 3.46 |
| 左腕盘子 | 39.45 | 9.04 |
| 右腕杯子 | 33.95 | 8.72 |
| 右腕盘子 | 77.31 | 8.96 |

| 初始帧固定桌面区域，灰度 0–255 | 训练画面 | 修改前 | 修改后 |
|---|---:|---:|---:|
| 左腕 | 223 | 123 | 229 |
| 右腕 | 234 | 125 | 225 |

初始帧杯子／盘子区域 RGB 中位数：

| 区域 | 训练 | 修改后 |
|---|---|---|
| 左腕杯子 | 172, 198, 207 | 157, 190, 200 |
| 左腕盘子 | 178, 201, 183 | 175, 194, 180 |
| 右腕杯子 | 140, 182, 202 | 149, 185, 195 |
| 右腕盘子 | 152, 173, 157 | 157, 181, 163 |

初始渲染中三个 RGB 通道同时超过 250 的像素比例为 0。
物体色块检测只用于离线对照，不进入策略输入或控制反馈。

## 保留的物理与控制

USD 审计核对了 1131 项物理属性、255 项物理关系、87 个碰撞体的几何和物理材质绑定，均一致；背景没有新增碰撞体。
四个外壳／适配件网格只改变 render visibility，不关闭 PhysX 碰撞。
三组前后对照的 reset 杯盘位置差为 0。

当前源手坐标变换、初始双手关节、杯盘位置、双爪映射、连续控制、限速限加速度、官方任务提示词和模型权重均沿用现有流程。
独立入口装饰当前 runner，避免复制旧控制循环。manifest／report 记录视觉配置及新增文件 SHA256。

## 文件与复现

包内实现：

- `yubi_isaac_sim_env/run_visual_aligned.py`
- `yubi_isaac_sim_env/visual_alignment.py`
- `yubi_isaac_sim_env/wrist_rig_visual.py`
- `yubi_isaac_sim_env/wrist_camera_visual_aligned_v3.json`
- `yubi_isaac_sim_env/assets/visual_alignment/cloth.png`

本地审计：`track1_console/sim_validation/wrist_visual_alignment_20261008/`。
其中 `wrist_comparison_initial.png`、`wrist_comparison_all.jpg` 为训练／修改前／修改后的对照，
`render_metrics.json`、`usd_visual_validation.json`、`integration_verification.json` 保存数值与验证结果。

服务器修正版目录：`/home/lrl/dual-franka-yubi-isaac-sim-console-tuned-v1`。
运行入口的参数沿用 `python -m yubi_isaac_sim_env.run`，将模块名改为 `yubi_isaac_sim_env.run_visual_aligned`。
`run_tuned_online.sh.pre_visual_v3_20261008` 保存升级前的启动脚本。
原来的 `yubi_isaac_sim_env.run` 可用于旧外观的 A/B 渲染。

启动入口的一步保持检查已输出头部和双腕三路 640×480、30 fps、各 4 帧视频，report 状态 completed。
本次没有进行新的模型成功率评估。

## 精度边界

相机参数来自固定场景和视频色块的离线近似，尚未做实物手眼标定。背景为简化三维实验室。
手套与电动夹爪、人手前臂与 Franka 的形状差异仍存在；杯子的透光外观使用固定填充近似。
视觉对齐是否改善 π0.5／LingBot／OpenWAM，需要另做同提示词、同 reset 的闭环对照。
