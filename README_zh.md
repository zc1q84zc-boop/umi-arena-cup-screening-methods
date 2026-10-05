# UMI Arena 杯子任务数据筛选展示包

本包提供**筛选方法、控制台与仿真改动源码、汇总统计**。不含原始视频、Parquet、模型
checkpoint、筛选清单、人工标注原表或逐帧结果。在线控制代码含少量公开 practice
片段 ID 作为固定参考配置，但不含片段内容。代码与汇总材料存放在
公开 GitHub 仓库，受限数据不在仓库中。

筛选范围是成功的“Place the cup on the plate, then put it back to its
original position”任务，共 3,975 个 episode、827,323 帧（30 Hz）。先逐帧检查
时间戳、姿态与动作一致性、位置/方向跳变和运动导数，再合并先前用户标记的
14 个动作跳变 episode。按照保守、可逆的训练数据政策，机器触发任一规则或
有人为标记的 episode 暂时隔离，原始数据不删除、不改写。

| 结果 | Episode | 帧 |
| --- | ---: | ---: |
| 输入 | 3,975 | 827,323 |
| 保留 | 3,423 | 746,914 |
| 隔离待复核 | 552 | 80,409 |

机器标记 551 个，人为标记 14 个，其中重合 13 个；共记录 1,580 个候选帧事件。
规则及假设见 [METHODS.md](METHODS.md)，更详细的汇总见
[aggregate_summary.json](results/aggregate_summary.json)，程序在 `src/`。

**重要限制：**隔离代表训练质量筛选，不等于证明这些轨迹在三种真机上无法执行。
当前未完成三机实测标定后的连续 IK、关节限位、碰撞、夹爪映射与动力学核验；
未触发规则也不代表可执行。代码复现需要另外取得 AIRoA 数据与人工标注的
授权访问。请遵守[原数据的限制访问条款](https://huggingface.co/datasets/airoa-org/yubi-corl2026-umi-arena)。

如需逐条查看，见[SSH 按需查看器](viewer/README.md)：GitHub 仍只存代码，
原始数据留在 A100；在工作站点开某条 episode 时，才持久缓存那条轨迹和
请求播放的机位视频。`.cache/` 已排除在 Git 之外，不缓存完整数据集或 checkpoint。

[训练与推理手册](TRAINING_AND_INFERENCE.md)逐项列出 π0.5、LingBot、OpenWAM 的训练
配方、A100 原始 checkpoint、squirrel RTX 5090 推理副本及获授权队员的启动/登记步骤。
GitHub 克隆不包含权重，也不自动取得服务器权限。

[模型调试交接指南](MODEL_DEBUGGING.md)说明控制台怎么下载、连接哪台服务器、
各模型权重位于哪里、如何启动/停止和检查运行结果，以及之前的相机、初态、夹爪、
平滑控制和完整任务判定优化。[控制台](console/README.md)现可查看三路视频、双手
轨迹和夹爪曲线，切换 π0.5、LingBot、OpenWAM 的离线回放，并通过本机 8772
页面启动 squirrel 上的 Isaac Sim 闭环实验。[仿真源码覆盖包](simulator_overlay/README.md)
提供本地改动的可复现源码。仓库不含 practice 原始记录、生成的视频或权重；
有授权的使用者须自行接入这些数据与服务器。
