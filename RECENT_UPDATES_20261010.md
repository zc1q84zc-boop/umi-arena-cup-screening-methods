# 最新进展：仿真、控制与训练集筛选（2026-10-10）

后续更新：[π0.5 同轮10k/20k/30k比较与控制台提速](FAST_PARALLEL_PI05_20261010.md)。三组当前仍在运行，未确认整轮成功。

本次更新包含可复现源码、配置、公开 CAD 和汇总数据。当前 4090 工作区是
`/home/claude/Corl_Track_1/umi_workspace_zhangchi`。

## 1. 杯子闭环调整

2026-10-10 晚间修正：恢复双腕夹爪主体和法兰转接外壳显示，消除头部录像中的悬空外观。已部署并完成三路静态渲染与 USD 物理一致性核查；[画面与证据](WRIST_HOUSING_VISIBILITY_20261010.md)。新外观尚未重新评估模型成功率。

| 项目 | 当前实现 / 实验结果 | 源码或证据 |
| --- | --- | --- |
| 高精度弹性杯 | 3 GPa、1 mm 薄壳、240 Hz 物理、128 次求解；参数仍为仿真估计 | [PVC 数值配置](console/simulator_profiles/tuned_v1/yubi_isaac_sim_env/pvc_numerics.py) |
| 三路共享渲染 | 同一物理时刻只获取一组图像，录像与模型观测复用；RTX 调用 216→54，采集部分 2.404→0.827 s（2.91×） | [源码](console/simulator_profiles/tuned_v1/yubi_isaac_sim_env/shared_camera_render.py)、[汇总](results/shared_render_20261009.json) |
| 平滑响应 | 可分别设置手臂与夹爪增益；本轮采用 4/4，速度上限 0.8 rad/s、加速度 1.5 rad/s² | [控制器](console/simulator_profiles/tuned_v1/yubi_isaac_sim_env/continuous_targets.py)、[参数选择](results/smoothing_parameter_choice_20261010.json) |
| 平滑对照 | 增益 8 的冻结动作实验改善跟踪但增大杯子形变；8/4 的实时实验未完成取回，因此最终保留 4/4 | [对照汇总](results/smoothing_comparison_20261010.json)、[实验脚本](console/deployments/smoothing_response_20261010/) |
| 左手夹持余量 | 左手闭合命令额外减小 0.005 rad，接近释放区时渐退；不改右手或手臂笛卡尔目标 | [夹持映射](console/deployments/jaw_margin_20261010/jaw_margin.py)、[0/0.005/0.010/0.020 rad 对照](results/jaw_margin_comparison_20261010.json) |
| 阶段顺序 | 右手放盘→松开→向上退 12 cm→回到起点并稳定→左手取回；两段操作采用官方 prompt | [完整任务入口](console/deployments/sequential_home005_20261010_v2/)、[阶段控制](console/deployments/sequential_home005_20261010_v2/sequencing.py) |
| 判定与步数 | 放盘中心半径 45 mm；取回半径仍为 35 mm。在线模式可运行到成功、手动停止或报错 | [放盘配置](console/simulator_profiles/tuned_v1/yubi_isaac_sim_env/config.json)、[取回判定](console/simulator_profiles/tuned_v1/yubi_isaac_sim_env/two_stage_task.py) |

45 mm 是用户选择的本地仿真诊断阈值，未声称为比赛官方要求。录像 30 Hz、模型请求 10 Hz；240 Hz 是本次高精度杯的物理频率。同步推理和渲染开销仍会使墙钟时间长于仿真时间。

归位门同时要求位置误差 ≤5 mm、方向误差 ≤2°、最大关节误差 ≤0.02 rad、
最大关节速度 ≤0.05 rad/s、开度 ≥0.95，并连续稳定 0.5 s。归位和非活动手臂
保持属于控制器介入，独立记录，不能当作完全由模型自主完成。

### 最新完整任务实验

模型 `pi05-cup-intersection-30000`，初态 `online_aligned_v2`，seed 42，
左手余量 0.005 rad，三路同步录制。用户停止后正常写出完整录像。

| 指标 | 本轮结果 |
| --- | ---: |
| 控制周期 / 每路录像帧 | 2,226 / 6,677 |
| 录像时长 | 222.567 s |
| 右手放盘 | 通过 |
| 右手归位时间 / 位置误差 | 75.946 s / 0.327 mm |
| 首次左手模型请求 | 76.013 s（右手归位之后） |
| 最大左手额外闭合 | 0.005 rad |
| 杯子最大形变 | 4.706 mm |
| 最终取回 XY 误差 / 阈值 | 51.090 mm / 35 mm |
| 完整任务成功 | **未通过** |

这是一次实验记录，不能推导总体成功率。[汇总 JSON](results/sequential_home005_20261010.json)
包含阶段请求数、官方 prompt、归位门实测值与停止原因。

[三路完整录像：手机播放与下载](https://umi-recording-20261010-home005.steven-robotics-ai.chatgpt.site)。约 1:16 开始左手阶段。

## 2. 新的任务 2–5 仿真环境

[`arena_tasks_v1`](console/simulator_profiles/arena_tasks_v1/README.md) 复用现有双 Franka/YUBI、
腕相机和控制器，新增四个任务：

| 任务 | 官方子步骤 | 主要新增物理对象 |
| --- | ---: | --- |
| 笔筒取放 `pens` | 12 | 开口笔筒、六支笔 |
| YUBI 零件分拣 `sps` | 15 | 九格盒、15 件零件 |
| USB-C 插接 `cable` | 1 | 支架、PCB、开口插座、插头、32 段线链 |
| 手机装盒 `phone` | 8 | 手机、空腔盒、可取下套盖、内托 / 隔板 |

空腔保留独立壁面碰撞体；可操作物体为动态刚体。没有吸附、自动抓取或成功时
移动物体的辅助。Toyota 官方零件 CAD 使用固定 revision，公开 STL 与许可证、
来源路径和哈希一并保留。机器人组合 USD 仍需获授权的 Isaac Sim Panda 资产。

四个场景已完成 USD、GPU 物理运行、三路同步录像和接口检查；每个场景检查录像
181 帧/路。场景本身可运行，不代表策略已经完成四个任务。此 profile 使用
120 Hz 物理、64/8 次刚体求解，与高精度杯的 240 Hz/128 次配置分别记录。

输入接口保持两路当前腕图像、7D 双手相对位姿、2D 夹爪状态、当前 prompt；
主相机与物体真值只供录像和评测。运行方法与限制见 profile README；
页面源码在 [arena_console.py](console/deployments/arena_tasks_20261010/arena_console.py)，
[验证汇总](results/arena_tasks_validation_20261010.json) 保留各场景的实测状态。

## 3. 训练集筛选更新

[训练集筛选进展](TRAINING_DATA_SCREENING_20261010.md) 包含方法、输入范围、
保留 / 隔离统计、原语覆盖及当前 IK 状态。

| 数据范围 | 输入 episode / 帧 | 质量保留 episode / 帧 | 质量＋IK 交集状态 |
| --- | ---: | ---: | --- |
| 杯子 | 3,975 / 827,323 | 3,423 / 746,914 | 已生成：2,781 / 600,087 |
| 手机装盒 | 14,726 / 3,169,018 | 2,978 / 599,292 | 待同事原始 IK 方法 |
| 链条版 SPS | 14,280 / 1,569,835 | 3,273 / 383,446 | 待同事原始 IK 方法 |

链条版 SPS 与新场景中的 YUBI 零件版 SPS 分别登记。当前数据发布版没有笔筒、
YUBI 零件版 SPS、MCU USB-C 插接的准确对应任务。新任务未启动训练；
质量中间视图不能标成已完成两套筛选的最终训练集。

## 4. 当前部署与复现

双 4090 采用 GPU0 仿真、GPU1 推理。工作区迁至
`/home/claude/Corl_Track_1/umi_workspace_zhangchi`；共享模型实体仍在
`/home/claude/workspace/` 的原模型目录，没有复制到公开仓库。
相关启动、控制台 transport 与 service 模板已同步路径。

已获服务器访问权限的队员可 SSH 转发已部署的页面：

```bash
ssh -N -L 8774:127.0.0.1:8774 -L 8775:127.0.0.1:8775 squirrel_4090_2
```

同一台电脑打开 `http://127.0.0.1:8774/`（杯子）或
`http://127.0.0.1:8775/`（新任务场景）。SSH 别名和身份由团队管理员提供。
新 clone 保持模型就绪检查，需目标机私有权重和实际验证记录才能启用相应策略。
归位＋0.005 rad 实验采用独立运行入口，其 README 说明依赖和停止方式。

公开包检查结果见 [2026-10-10 源码验证](results/public_source_validation_20261010.json)。
此前的 [2026-10-09 快照](RECENT_UPDATES_20261009.md) 与早期 5090 文档保留为历史记录。
