# OpenWAM：交集版配对合并数据完整遍历

**2026-10-09 结果更新：正式训练已完成。** 最终 5,069 次参数更新，实际读到全部 160,338 个不同训练窗口和全部 162,196 个阶段平衡索引；`full_pass_completed=true`，实际顺序与完整置乱计划一致。正式训练约 7.72 小时。最终 checkpoint 已迁移至 5090 并接入控制台，GPU 启用等待显存释放；详见 [部署记录](../../console/deployments/intersection_20261009/README.md)。下文启动时的快照保留为历史记录。

用户于 2026-10-09 授权：检查 A100 空卡，有空闲即训练 OpenWAM，至少完整遍历一次。

远端工作目录：`openwam-a100:/mnt/data/benyun/workspace/openwam_cup_fullpass_20261009`。
本目录保存该次运行的独立脚本和审计记录。原模型仓库、数据源及既有 checkpoint 不修改。

## 数据与输入

- 复用 `/mnt/data/benyun/workspace/openwam_cup_merged_official_20261008/prepared_v2`。
- 使用质量筛选与 IK 可解筛选的交集，并沿用此前确认的相邻右手、左手 episode 配对合并方案。
- 共 1,186 条合并录制，对应 2,372 个源 episode；按录制 UUID 划分训练 1,068 条、验证 118 条。
- 训练有 160,338 个不同的 10 Hz 控制窗口，验证有 17,366 个窗口。
- 两个阶段平衡后的训练索引有 162,196 项，其中 1,858 项是用于阶段平衡的重复窗口。
- 输入只用左右腕相机，按左、右固定顺序拼为 640×256 图像；状态是双手相对位姿 7 维与两路夹爪状态 2 维。
- 使用当前阶段的官方英文 prompt。未来动作及视频监督在阶段末尾裁剪并加 mask，避免把下一阶段动作标成当前 prompt 的答案。
- 动作为双臂 16 维、未来 10 Hz 的 SE(3) 合成增量，action horizon 32；9 帧视频监督按 4 个控制步间隔采样。
- 状态和动作四元数采用与随模型保存的推理 contract 一致的单位化及符号规范化。归一化统计只由训练录制计算。

## 完整遍历如何保证

`CoverageSampler` 对完整训练索引做 seed 42 的无放回置乱，只有最后一批进行补齐。有效 batch 固定为 32：

| 项目 | 数量 |
| --- | ---: |
| 不同训练窗口 | 160,338 |
| 阶段平衡后的索引 | 162,196 |
| 最后一批补齐 | 12 |
| 一轮实际样本呈现数 | 162,208 |
| 参数更新次数 | 5,069 |

双卡每卡 batch 和梯度累积由实测吞吐、显存余量确定：`2 × 每卡 batch × 累积次数 = 32`。
取消旧流程的 24 小时时间预算上限。`max_steps` 在 OpenWAM 上游表示微步，脚本据此换算，不把微步误记为参数更新。

正式训练每一步跨卡收集实际样本索引，检查其顺序与预定完整置乱一致，并统计不同窗口及平衡索引的读取次数。
只有训练结束且全部索引、全部不同窗口都被实际消费，`training/coverage.json` 才会写入 `full_pass_completed=true`。
完成时另存 `coverage_counts.npz`，可逐项复查。启动训练不表示完整遍历已经结束，也不等于闭环任务成功。

## 流程与参数

1. 仅检查并使用 GPU 1、2。GPU 0 不使用；GPU 3、4 上的 π0.5 训练保持运行。
2. 新数据上的真实前向、反向和参数更新短测，先测每卡 batch 8；显存允许时测 16，失败时回退 4、2。
3. CPU 审计数据来源、训练/验证划分、有限值、归一化、视频路径、真实图像解码、阶段末尾 mask、Accelerate 跨卡采样覆盖。
4. 用选定配置更新两次，实际保存 checkpoint 后读回，核对结构、有限值、动作权重确实变化及归一化文件存在。
5. 从 foundation 重新初始化正式训练，丢弃短测和 smoke 的更新。

固定配置：BF16、ZeRO 2、CPU optimizer offload、梯度检查点、常数学习率 `1e-5`、seed 42；冻结 video backbone，训练约 10.21 亿 action/proprio 参数。每卡 2 个数据读取 worker。

每 1,250 次参数更新保存模型权重，预定在 1,250 / 2,500 / 3,750 / 5,000 / 5,069 次保存。
上游 checkpoint 文件名以微步计数，实际换算见 `training_plan.json`。保存完整模型权重、配置及归一化统计；本轮未启用 optimizer state 快照，不能据此声称支持精确恢复优化器状态。

## 远端结果文件

- `pipeline_status.json` / `pipeline.log`：当前阶段、进程 PID 和错误。
- `dataset_audit.json`：数据来源及 CPU 采样覆盖检查。
- `benchmarks.json`：实际 batch、显存及吞吐。
- `smoke_readback.json`：checkpoint 保存读回及真实权重更新检查。
- `training_plan.json`：最终 batch、微步、参数更新次数、吞吐推算时间及 checkpoint 间隔。
- `training.log` / `training/steps.jsonl`：正式训练进展、loss、gradient norm 和显存。
- `training/coverage.json` / `training/coverage_counts.npz`：实际完整遍历证据。
- `training/checkpoints/`：本轮 checkpoint。

耗时按选定 batch 的实测吞吐计算，另需模型初始化、checkpoint I/O 和运行波动时间。不会因为估计耗时较长而减少所需样本数。

## 实测结果

2026-10-09 04:35（北京时间）两组双卡测试均完成了 8 次真实参数更新，计时去掉前 2 步：

| 每卡 batch | 双卡样本/秒 | 每卡峰值保留显存 | 一轮纯计算估计 |
| --- | ---: | ---: | ---: |
| 8 | 4.645 | 31.38 GiB | 9.70 小时 |
| 16 | 5.623 | 39.92 GiB | 8.01 小时 |

选用每卡 batch 16、双卡、梯度累积 1，有效 batch 32。CPU 数据审计已通过，包括实际 Accelerate 跨卡采样顺序与完整覆盖验证。上述耗时是短测推算，不是完成承诺。

两次 smoke 更新的 loss、gradient norm 及实际跨卡采样顺序检查通过。约 25 GB 的完整权重保存到 NFS 用了约 4 分钟，随后读回检查的 12 个动作参数矩阵均为有限值、形状正确，并且全部相对 foundation 有真实更新；归一化文件存在。

正式训练于 **2026-10-09 04:42:33（北京时间）** 启动，torchrun PID **2345418**，GPU **1、2**；后台流程 PID **1977067**。`training_launch.json` 保存完整启动命令。
训练从 foundation 重新初始化，smoke 更新未混入正式模型。预计纯计算约 8 小时，加上 5 次 checkpoint 的共享存储写盘及运行波动，可按约 **8–10 小时**预估。
本段启动时间表示后台正式进程已创建，实际参数更新和样本覆盖请以远端 `training/steps.jsonl` 与 `training/coverage.json` 为准。

**2026-10-09 04:45:17（北京时间）启动核验：**正式训练已完成前 3 次参数更新，实际消费 96 个不同训练窗口，跨卡顺序逐步验证通过。三步 loss 及 gradient norm 均为有限值，gradient norm 均大于 0。GPU 1、2 的 rank PID 分别为 2346314、2346315；π0.5 PID 2095710 仍在 GPU 3、4 正常运行。
本轮 checkpoint 目录为 `/mnt/data/benyun/workspace/openwam_cup_fullpass_20261009/training/checkpoints/2026-10-09_04-44-22/`。
本地 `training_snapshot.json` 是此时的证据快照，**完整一轮尚未结束**，当前 `full_pass_completed=false` 符合预期；实时进度以远端文件为准。

## 本轮修复及保留记录

最初双卡短测在加载模型前因旧单进程脚本同时创建目录而退出；已改为跨进程安全的创建方式，配置仅 rank 0 写入，随后用新目录 `benchmark_b8_v2` 重启，失败日志保留。
初次 CPU 审计的交集 manifest 路径带了多余的 `selection/`，改正路径后 `dataset_audit_v2.log` 记录检查通过；没有修改数据。
