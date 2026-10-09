# OpenWAM / π0.5 交集版迁移至 5090

2026-10-09：两次训练已完成，最终推理 checkpoint 已迁移并接入本地仿真控制台。5090 GPU 被其他用户的服务占用，目前两个新模型选项显示在控制台中，但尚未启用；释放显存后需逐个完成 5090 加载和真实推理检查。

控制台：<http://127.0.0.1:8772/>。

## 训练结果

| 模型 | 最终 checkpoint | 数据使用 | 完成证据 |
| --- | --- | --- | --- |
| OpenWAM | `checkpoint_step_5069.safetensors` | 交集 episode 配对合并；训练 1,068 条录制，验证 118 条 | 5,069 次更新；160,338 / 160,338 不同训练窗口实际消费；`full_pass_completed=true` |
| π0.5 | `30000/params` | 交集 episode 独立使用；训练 2,504 episodes，验证 277 episodes | `completed_steps=30000`、`phase=complete`、训练退出码 0、完整最终 checkpoint |

OpenWAM 使用两张 A100、每卡 batch 16、无梯度累积，有效 batch 32；阶段平衡索引共 162,196 项，最后一批补齐 12 项，实际呈现 162,208 次。正式训练耗时 27,802 秒（约 7.72 小时）。实际读取顺序与无放回完整置乱计划一致，不能仅用 step × batch 代替覆盖证明。

π0.5 的 10k、20k、30k checkpoint 均保留在 A100。本次按用户要求迁移最终 30k 推理权重，未迁移 optimizer/train_state。固定 256 条验证窗口的 EMA flow-matching loss 在 10k 为 0.049089、30k 为 0.067746，后期上升；这项离线诊断不能判断闭环任务成功率。部署没有将 30k 暗中替换成 10k。

训练证据见 `training_completion.json`；完整源文件与 SHA-256 见 `transfer_manifest.json`。

## 部署位置与控制台选项

5090 主机：`squirrel_5090`，GPU 0。迁移根目录：

```text
/home/lrl/workspace/umi_cup_intersection_models_20261009
```

| 控制台名称 | 模型 ID | 权重相对路径 | systemd 用户服务 | 推理端口 |
| --- | --- | --- | --- | --- |
| OpenWAM 交集合并版 · 完整一轮 5069 | `openwam-cup-intersection-fullpass-5069` | `openwam/5069` | `umi-intersection-openwam-5069.service` | 18862 |
| π0.5 交集版 30k · 官方双阶段 / 10 Hz | `pi05-cup-intersection-30000` | `pi05/30000` | `umi-intersection-pi05-30000.service` | 18861 |

共迁移 275 个文件、37,277,943,669 字节（34.72 GiB），全部通过 SHA-256 校验。下载过程中遇到零字节 Python 文件，修正下载器后复用已验证的大文件并完成剩余文件；`transfer_result.json` 的秒数仅表示恢复阶段，不能当作整个迁移耗时。

两套推理服务按需加载，启动前检查 GPU 计算进程。新适配器安装到 deploy 与 tuned 仿真目录，控制台通过独立启动脚本调用；旧 checkpoint、旧推理协议和原有模型选项保留。模型元数据与图像接口按本轮训练保存的 contract 加载。

## 输入和动作时序

- 当前左右腕 RGB，不使用中央相机作为模型输入。
- 双手相对位姿 7 维和两路夹爪状态 2 维，不向模型提供绝对单手位姿或未来观测。
- 当前阶段使用对应官方提示词：
  - `Pick up the cup with your right hand and set it on the plate`
  - `Pick up the cup with your left hand and return the cup to its original position`
- 模型输出双臂 16 维动作。训练标签已经将当前观测之后的三个 30 Hz 局部 SE(3) 增量合成为一个未来 100 ms 动作，部署直接使用第 0 行，不再次跳过或合成三行。
- 每次观测仅积分一个未来 100 ms 动作，随后三个 30 Hz 仿真关节控制周期保持相同端点；夹爪也使用与该动作对齐的第 0 行。

提示词来源：<https://umi-arena.airoa.io/evaluation>，本轮训练前于 2026-10-08 检查。10 Hz 是本轮训练设计，不能称为官网强制频率。

## 已通过的检查与尚待完成的检查

已完成：

- 迁移全部 275 个文件 SHA-256 验证。
- 控制台 28 项测试、新时序与输入协议 3 项测试；5090 上使用真实 tuned 适配器帮助函数的相同 3 项检查通过。
- 5090 CPU 导入检查：OpenWAM PyTorch 2.8.0+cu128 / Transformers 5.17.0；π0.5 JAX 0.5.3 / Flax 0.10.2。
- A100 与 5090 使用的 11 个核心 OpenPI 源文件 SHA-256 一致，见 `openpi_source_check.json`。
- 两个最终 checkpoint 在 A100 上真实加载，分别使用两种官方提示词进行推理；共 4 次请求均返回有限、单位四元数的 `(1,16)` 动作，模型标识、episode/step、prompt 和未来动作时序全部匹配。见 `openwam_a100_preflight.json`、`pi05_a100_preflight.json`。

A100 单次实测：π0.5 首次 JIT 请求约 17.10 秒、随后约 109 ms；OpenWAM 首次约 3.80 秒、随后约 1.19 秒。每个阶段仅一次请求，这些数值不是吞吐基准，也不是 5090 性能。

尚待完成：5090 GPU 加载和推理检查。当前 GPU 被 `dzq` 的 OpenWAM 服务占用（PID 2326374，端口 18849，约 24.4 GiB）；两个新用户服务保持停止，控制台策略为 `ready=false`，未停止该用户的服务。

释放显存并重新检查无其他计算进程后，可逐个加载服务，运行 `scripts/probe_inference.py` 两阶段检查；通过后将相应控制台策略启用。两套模型不能同时常驻同一张 32 GB 卡。GPU 推理检查通过也不等于完整任务成功，尚未做新模型的闭环任务评测。

临时 A100 验证服务和权重传输 HTTP 服务在验证结束后清理，仅保留训练结果、迁移结果和审计记录；清理证据见 `cleanup.json`。
