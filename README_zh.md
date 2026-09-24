# UMI Arena 杯子任务数据筛选展示包

本包仅供展示**筛选方法、源代码与汇总统计**。不含原始视频、Parquet、模型
checkpoint、逐 episode 编号/清单、人工标注原表或逐帧结果；尚未上传 GitHub。

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
