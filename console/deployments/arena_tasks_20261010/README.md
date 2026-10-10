# 任务 2–5 仿真扩展（4090，2026-10-10）

复用当前 `tuned_v1` 的双 Franka/YUBI、桌面碰撞、手根/TCP 变换、夹爪映射、相机和控制器，新增独立的 `arena_tasks_v1`。任务包括笔筒取放（12 步）、零件分拣（15 步）、USB-C 插接（1 步）和手机装盒（8 步）。原语顺序与 prompt 来自官方评测，输入为官方两路腕图像、双手相对位姿、夹爪状态和 prompt；主视角与物体真值只用于录像/评测。

七类 YUBI CAD 使用 Toyota 官方固定 revision 和校验哈希，左右手指、左右 flap 分别使用对应 STL。笔筒、九格盒、手机盒/可取下套盖保留空腔。CAD 固体使用实际重心和主惯性轴，质量仍待实测。线缆用 32 段受限球铰链近似。

本机入口：[任务页面](http://127.0.0.1:8775/)；[杯子控制台](http://127.0.0.1:8774/) 已增加导航链接。场景检查使用 GPU0，复用公共运行锁并拒绝占用中的 GPU，不调用 GPU1 的策略服务。新页面服务为 `umi-arena-tasks-4090.service`，本机转发为 `umi-arena-tasks-4090-forward.service`。旧页面变更前后快照保存在目标机本目录的 `simulation.before.html` / `simulation.after.html`。

67 项 CPU 测试与四个 USD 组合检查通过。每个场景录制 181 帧/路、30 Hz，检查时间戳同步、有限刚体状态和桌面穿透。笔筒、所有 15 件 SPS 零件、手机/内托/套盖另做 1200 步物理目标放置检查。诊断 identity policy 验证实际相机输入、官方字段和 16 行动作执行；不代表训练模型解决了任务。

最终结果、代码哈希与远端记录路径见 `validation.json`，本机录像和状态在 `../../sim_validation/arena_tasks_20261010/`。USB 已检查线链运行和开口插座结构，机器人驱动插接及接触力校准尚待验证。其余尺寸、摩擦、软材料与模型任务成功率也需要后续实物/策略验收。

复核本机记录：

```bash
PYTHONPATH=track1_console/simulator_profiles/arena_tasks_v1 \
  umi_cup_screening_showcase/.cache/public-tests-20261009/bin/python \
  track1_console/deployments/arena_tasks_20261010/verify_artifacts.py
```

环境和运行参数的详细说明见 [profile README](../../simulator_profiles/arena_tasks_v1/README.md)。场景改动已部署到 4090；2026-10-10 的本次更新将源码、公开 CAD 和汇总发布到 GitHub。目标机原始验证输出仍在私有工作区，公开汇总见仓库 results/arena_tasks_validation_20261010.json。
