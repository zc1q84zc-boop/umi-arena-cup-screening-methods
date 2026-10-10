# π0.5 检查点比较与控制台提速 · 2026-10-10

目标机为 `squirrel_4090_2` 的双 RTX 4090，杯子控制台端口 8774。
[三路录像和下载](https://umi-recording-20261010-home005.steven-robotics-ai.chatgpt.site)
分别标注进行中片段、完整录像与旧版试验。

## 已应用设置

| 参数 | 原设置 | natural_fast_v1 |
| --- | ---: | ---: |
| 手臂 / 夹爪响应增益 | 4 / 4 | 6 / 4 |
| 关节速度上限 | 0.8 rad/s | 1.2 rad/s |
| 关节加速度上限 | 1.5 rad/s² | 2.4 rad/s² |
| 物理频率 / 求解迭代 | 240 Hz / 128 | 240 Hz / 128 |
| 共享三路渲染预热 | 每组 5 帧 | 首组 5 帧，后续每组 1 帧 |

保留左手闭合余量 0.005 rad、释放时归零、右手归位门及恢复腕部外壳的
`wrist_visual_aligned_v3_housing_visible`。未放大位置指令或改变物理时钟。
0.5 rad 控制目标阶跃达到 1% 误差耗时 1.017 → 0.704 秒，约缩短 31%；
这是指令响应检查，未直接测量机器人动作完成时间。
静态 10 组共享图像耗时 1.004 → 0.404 秒，RGB 平均差约 2.7–3.2 / 255。
该短测不能代替动态相机 A/B 检验，也不能当作完整任务提速倍数。

## 同轮训练检查点

| 模型 ID | 4090 私有目录后缀 | 常规服务端口 |
| --- | --- | ---: |
| `pi05-cup-intersection-10000` | `pi05/10000` | 18864 |
| `pi05-cup-intersection-20000` | `pi05/20000` | 18865 |
| `pi05-cup-intersection-30000` | `pi05/30000` | 18861 |

共同目录：`/home/claude/workspace/umi_cup_intersection_models_4090_20261009`。
A100 训练源：`/mnt/data/benyun/workspace/pi05_cup_intersection_20261008/checkpoints/pi05_cup_intersection/pi05_cup_intersection_v2/{10000,20000,30000}`。
杯子质量＋IK交集包含 2,781 episode、600,087 原始帧。
新增 10k/20k 的 38 个权重及资产文件共 24,881,110,204 字节，迁移后逐文件 SHA-256 一致。

模型输入为双腕 RGB、9D 双手相对状态和官方阶段指令，输出为 16D 动作。
本训练版本第 0 行已对齐未来 100 ms；执行时只取第 0 行，保持三个 30 Hz 控制周期。
模型请求为 **10 Hz 仿真时间**，并非每秒墙钟十次。
新增两检查点的两个官方阶段服务检查都通过；JIT 后的单次实测约 85 / 117 ms。
这些单次请求耗时不属于吞吐基准。

## 当前结果与比较范围

固定初态 `online_aligned_v2`、seed 42、外观、杯子材质、速度和成功判定。
三组从初态运行完整任务，录制头部、左腕、右腕 640×480 / 30 fps。
30k 已放盘、完成右手归位并取回桌面，但距原中心约 35.6 mm，超过当前 35 mm 阈值；
10k / 20k 尚未通过放盘。三组仍在运行，**未确认整轮成功**。
45 mm 放盘与 35 mm 取回是本地诊断判定，不宣称为比赛官方完整规则。
单轮固定 seed 也不等于成功率统计。

最初误选旧版 cup-clean 10k/20k 的试验因版本纠正结束；其录像不纳入同轮训练比较。
当前完整任务不设任意步骤或时间上限，成功、人工停止、不可恢复落桌或报错时结束。
模型负责抓取与移动；中间释放、上退、归位及非活动手臂保持属于显式控制器介入。

## 双卡并行

独立场景和推理进程分配到两张卡。最近 180 秒的三场景采样中，GPU0 / GPU1
平均利用率约 81% / 72%，峰值均到 100%；显存峰值为 18,806 / 23,226 MiB。
GPU1 余量较小。物理步进是已测主要耗时，GPU并非持续满载。
双卡增加独立试验吞吐；单个 GPU 物理场景仍使用一张卡。
[NVIDIA 性能说明](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/reference_material/sim_performance_optimization_handbook.html#multi-gpu-support)
区分多相机渲染和单卡 GPU 物理。不同并发负载下的墙钟时间不能直接作为检查点速度排名。

## 源码与证据

- [入口、依赖和部署说明](console/deployments/fast_parallel_pi05_20261010/README.md)
- [控制目标阶跃](results/fast_governor_step_response.json)
- [静态共享渲染对比](results/fast_render_smoke_comparison.json)
- [同轮实验配置](results/fast_experiment_plan.json)
- [双卡采样汇总](results/fast_gpu_summary_three_trials.json)
- [进行中视频恢复校验](results/fast_preview_verification.json)

已通过 14 项相关 CPU 检查、两 GPU 渲染短测和新增两模型的真实推理请求。
控制台目录显示三检查点可用；完整原生页面启动链尚需在当前 GPU 试验结束后验证。
公开仓库保留源码和汇总；私有权重、归一化资产及原始示范另行配置。
