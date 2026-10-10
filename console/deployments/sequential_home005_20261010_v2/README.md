# 右手完全归位后再运行左手：0.005 rad 完整任务实验

模型 `pi05-cup-intersection-30000`，初态 `online_aligned_v2`，seed 42。
模型负责右手放盘和左手取回；中间松开、上退 12 cm、回初始关节，以及非活动手臂保持，
属于显式控制器介入。`sequencing.py` 和录像审计分别记录其来源。

归位必须满足位置 ≤5 mm、方向 ≤2°、关节误差 ≤0.02 rad、关节速度 ≤0.05 rad/s、
开度 ≥0.95，并保持 0.5 s，才允许首次左手模型请求。
右手放盘阶段保留原模型双手输出；归位时左手保持放盘结束姿态；左手取回时右手保持初始关节。

`jaw_margin.py` 对左手闭合增加 0.005 rad 余量，在开度 0.60–0.65 间渐退，释放时归零。
物理 240 Hz / 128 次求解、共享三路渲染、手臂和夹爪响应增益 4/4。
运行到完整任务成功、用户停止或报错，不设置固定模型步数上限。

## 实验依赖

`full_task_sequential.py` 在进程内装饰现有 runner，不覆盖常规仿真源码。
入口要求已经部署的 Isaac Sim runtime、tuned profile 和其适配器、
GPU1 的 π0.5 18861 服务，并检查服务身份、公共运行锁和 GPU0 可用显存。
`run_full_task.sh` 还引用此前夹持实验的隔离 Python 环境和运行状态目录；
维护者需要在自己的授权部署中配置这些依赖及新的输出路径。
它不是 clone 后可在普通 CPU 机器上直接运行的训练命令。

2026-10-10 目标机诊断部署目录为
`/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/deployment_diagnostics/sequential_home005_20261010_v2`。
`run_full_task.sh` 的输出目录必须尚不存在。停止时写入本次启动的 `--stop-file`，
等待 runner 完成三路录像封装；不要停止其它用户的服务。

## 本轮结果

[公开汇总](../../../results/sequential_home005_20261010.json) 中右手放盘通过、归位门通过，
左手第一请求发生在归位之后。最终 XY 误差 51.090 mm，未通过 35 mm 取回阈值。
用户停止后共 2,226 控制周期、6,677 帧/路，每路 222.567 秒。

[三路录像](https://umi-recording-20261010-home005.steven-robotics-ai.chatgpt.site)
支持手机播放和下载。纯控制器逻辑可以直接导入 `sequencing.py` 和 `jaw_margin.py`；
`prepare_mobile.py` 是转为 H.264/yuv420p/faststart 的 FFmpeg 处理脚本，需要私有原录像输入。
