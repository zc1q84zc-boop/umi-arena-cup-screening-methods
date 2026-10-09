# π0.5 cup intersection v2 — 2026-10-08

**2026-10-09 结果更新：30,000 步训练已完成，最终 checkpoint 写盘完成，流程退出码为 0。** 10k、20k、30k checkpoint 均已保留；最终 30k 已迁移至 5090 并接入控制台，GPU 启用等待显存释放。固定验证 loss 在 10k 为 0.049089、30k 为 0.067746，需另做闭环评测；详见 [部署记录](../../console/deployments/intersection_20261009/README.md)。

用户指定：暂不配对，寻找空闲 A100，训练 30,000 optimizer steps，在完成 10,000 / 20,000 / 30,000 步时各保存 checkpoint。

远程目录：`/mnt/data/benyun/workspace/pi05_cup_intersection_20261008`。
现有第一版目录和共享 OpenPI 源码不修改。所有交集 episode 独立使用，包括同一次录制仅保留一个阶段的 episode，不要求两阶段齐全。UUID 只用于防止训练和验证泄漏，不用于配对或拼接。

正式训练于 2026-10-08 17:28（北京时间）在 A100 3、4 号卡启动，PID 2095710，后台流程 PID 2054178。OpenPI commit 为 `15a9616a00943ada6c20a0f158e3adb39df2ccac`，共享仓库工作树干净。

启动前检查已通过：96 条真实样本（batch 32）；两步全参数梯度更新（loss 1.0903、0.9406，梯度有限）；64 条独立验证窗口 loss 1.06295；完整测试 checkpoint 保存及重新加载；训练/验证两个 split 下两种阶段 prompt 共四次推理均产生有限 `(32,16)` 动作。测试 checkpoint 写盘耗时 511 秒，不计入正式 30,000 步。

| 项目 | 第一版 | 本版 |
| --- | --- | --- |
| 数据 | 质量筛选 3,423 episodes / 746,914 帧 | 质量筛选与 IK 参考筛选交集，2,781 episodes / 600,087 原始帧 |
| 指令 | 两阶段都强制使用整任务 prompt | 源数据每个 episode 对应的官方阶段 prompt |
| 动作时间 | 直接使用源 30 Hz 相对动作列，previous-pose timing | 将当前观测之后 3 个 30 Hz 局部 SE(3) 增量合成为一个 10 Hz 因果动作；旋转与平移联合合成 |
| episode 尾部 | 最后一帧动作重复填充 | 最后不足 3 帧的短动作后，位姿保持、夹爪保持；不重复最后运动、不跨 episode |
| 验证 | 没有独立验证集 | 按 UUID、seed 42 固定约 90/10 分割；仅训练集计算归一化；固定 256 条验证窗口计算 EMA flow-matching loss |
| 检查 | 已有第一版 smoke | 全量 allowlist、相机、时序与 SE(3) 审核；真实 batch；两步梯度及保存；重新加载并推理后才开始正式训练 |

固定的可比训练设置：pi05_base 重新开始全参数微调；两张 A100 / FSDP 2；global batch 32，无梯度累积；BF16 模型计算；AdamW 默认 beta=(0.9,0.95)、eps=1e-8、weight_decay=1e-10、clip=1；warmup 1,000、peak LR 5e-5、cosine decay 至 5e-6；EMA 0.99；seed 42；horizon 32、内部 action/state pad 32。训练 30,000 步，共 960,000 次窗口呈现，不代表 960,000 个不同窗口。

已完成的全量预处理：2,504 个训练 episode（1,395 UUID），277 个验证 episode（155 UUID）；训练窗口 181,072，验证窗口 19,664。训练阶段 prompt 分布为右手 1,245 episode、左手 1,259 episode；验证为右手 140、左手 137。30k × batch 32 相当于约 5.30 次训练窗口池呈现。shuffle=True、drop_last=True，每个完整 pass 丢弃的 16 条会随重排变化。

合成动作相对源绝对端点的全量最大数值误差：平移 3.78e-9 m、旋转 7.79e-7 度。输入状态四元数保留源数据的符号表示，与官方推理适配器一致。

两版都已经使用左右腕相机、9 维状态（手间相对位姿 7 + 两个夹爪 2）、16 维双臂输出，base 相机槽 mask=False。这些不是本版新增改进。绝对单手位姿只用于离线动作审核，不进入训练输入。

10 Hz 重采样是本版对训练目标的设计选择，不能称为官网要求必须如此。官网回放示例使用 30 Hz / previous timing；本版回放必须显式匹配 action_hz=10 / causal-next timing。官方输入输出仍遵循 https://umi-arena.airoa.io/submission-format。

验证 loss 是固定随机种子的离线诊断，不能代替闭环任务成功率。IK 参考筛选也不代表碰撞、连续性或目标机器人可执行性的全面认证。

检查点保留模型、EMA/优化器训练状态及归一化 assets。正式目录：`checkpoints/pi05_cup_intersection/pi05_cup_intersection_v2/{10000,20000,30000}`。两步测试 checkpoint 独立位于 `smoke_checkpoints`，不会混入正式结果。

`status.json` 给出当前流程阶段，`train.log` 给出训练进度，`validation.jsonl` 记录验证结果，`allocation.json` 记录实际使用的 GPU。仅选择 GPU 1–7 中无计算进程、显存占用 <500 MiB、利用率 ≤2% 的两张卡，每阶段重新检查，GPU 0 不使用。
