# 杯子任务：训练方法、私有模型位置与授权控制台使用

## 当前版本 · 2026-10-10

[最新仿真、控制和部署](RECENT_UPDATES_20261010.md) 与 [训练集筛选](TRAINING_DATA_SCREENING_20261010.md) 已更新。新任务尚未启动训练；手机装盒与链条版 SPS 只有质量中间视图，最终 IK 交集待完成。下文保留此前训练和5090部署记录。

## 2026-10-09 更新

下文保留 2026-10-05 的第一版训练与部署记录。最新版本见 [仿真、模型和部署更新](RECENT_UPDATES_20261009.md)：新增质量与 IK 交集数据上的 π0.5 30k、OpenWAM 配对合并完整一轮 5069，两个新模型均采用当前双腕、9D 官方状态、阶段 prompt 和未来 10 Hz/16D 动作。新模型使用独立 ID 和时序适配，不能与下文旧 OpenWAM 的头部图像/20D 配方混用。实际完整遍历证据、验证 loss 与公开训练代码见更新文档。

本文记录已经完成的训练及现有部署，供**分别获得数据、A100 和 squirrel RTX 5090 访问权限**的队员复核。GitHub 只提供方法、控制台和适配代码；它不会分发权重、原始视频、Parquet、逐 episode 结果、SSH 配置或密钥。文中的路径是 2026-10-05 核对的既有位置，不是公共下载地址，也不是重启训练的指令。

## 1. 三个模型使用了什么数据

输入是 UMI Arena 中成功的“把杯子放到盘子上再放回”任务：3,975 个 episode、827,323 帧（30 Hz）。保守筛选隔离 552 个 episode／80,409 帧，正式训练只用 **3,423 个 episode／746,914 帧**。筛选规则和汇总见 [METHODS.md](METHODS.md) 与 [results/aggregate_summary.json](results/aggregate_summary.json)。A100 上的私有 `selection/manifest.json` 是训练时的权威 allowlist；共享原始 LeRobot packed Parquet/视频的软链接本身**不是**过滤后的数据。训练加载器必须按 episode ID 过滤，并在加载后验证这两个精确计数；不得以“路径叫 clean”代替验证。

这些模型的练习集回放与训练数据有重叠，只能用于检查观测、动作格式和轨迹，**不能**作为独立泛化成功率。训练中的未来帧监督也不能在在线推理时变成未来真实观测输入。

## 2. 已完成的训练配方

| 模型 | A100 训练方式 | 主要输入、目标和优化 | 已完成保存 |
| --- | --- | --- | --- |
| π0.5 | OpenPI π0.5 基座；GPU 2、3，FSDP 2 卡；global batch 32；action horizon 32；30,000 步 | 当前双腕图像、双臂状态和杯任务文字；LeRobot episode 过滤；余弦学习率，1,000 步 warm-up，峰值 `5e-5`，末值 `5e-6`；训练前做数据/梯度 smoke | 完整 10k、20k、30k checkpoint；只有这三个完整步数实际落盘 |
| LingBot VLA2 6B | GPU 1–4，4 进程 FSDP2；**全模型微调**（视觉编码器未冻结）；10,000 步 | 左右腕相机，左右手绝对状态、相对姿态动作及下一帧夹爪目标；PA 阶段文字而非完整两阶段标题；32-action chunk；global batch 32（每卡 micro batch 1、梯度累积 8）；AdamW，常数 `5e-5` | 5k、10k distributed checkpoint，另有各自完整 HF 权重导出 |
| OpenWAM Alpha | GPU 5 单卡；**仅 action/proprio 微调，视频骨干冻结**；10,000 步 | 当前头部图作推理条件，训练视频/动作跨度 33 帧、视频 stride 4；双臂 20 维原始状态/动作映射至预训练 80-slot 布局（左 0–9、右 34–43）；batch 1、累积 1、bf16、ZeRO-2、CPU optimizer offload、`1e-5` | 5k、10k **weights-only** safetensors；不是完整 optimizer-resume checkpoint |

上述 GPU 编号只描述历史训练，不授权占用当前同号 GPU。A100 GPU0 和其他人的作业不可碰；训练已经结束，**不要直接重跑历史 `train` 命令或覆盖既有 checkpoint**。

历史脚本与实际参数可在已授权的 A100 上核对：

```text
/mnt/data/benyun/workspace/pi05_cup_clean_20260923/scripts/train_cup_clean_pi05.py
/mnt/data/benyun/workspace/pi05_cup_clean_20260923/train.log
/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/scripts/{verify_clean_inputs.py,launch_lingbot.sh,train_lingbot_clean.py,launch_openwam.sh,train_openwam_cup.py,openwam_cup_dataset.py}
/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/lingbot/config.yaml
/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/{lingbot,openwam}/train.log
```

历史 π0.5 入口是 `train_cup_clean_pi05.py --mode train`，显式传入 work-dir、authorized dataset root、manifest-dir、OpenPI root、官方 evaluation root 和 `--batch-size 32`；脚本会拒绝非 3,423／746,914 或非两张可见 GPU。LingBot 的 `launch_lingbot.sh train` 会先验证 allowlist、归一化统计和 GPU 空闲，再启动 4 进程 trainer。OpenWAM 的 `launch_openwam.sh train` 会先验证 prepared manifest 和 GPU 空闲，再运行单卡训练；正式保存关闭 full optimizer snapshot，以免共享存储产生巨量 I/O。**这三个入口是记录和审计参考，不是新队员登录后应执行的步骤。** 若要重新训练，应先在隔离的新工作区复制配方，重新验证数据许可、基座模型版本、GPU 所有权及不会覆盖旧结果。

## 3. 权重实际在哪里

`openwam-a100` 的原始训练结果与 `squirrel_5090` 的在线推理副本是两套不同目录。下面是精确的私有路径；GitHub `console/checkpoints.json` 只登记名称和元数据。

| 模型 | A100 原始训练输出 | squirrel 在线推理副本 |
| --- | --- | --- |
| π0.5 10k/20k/30k | `/mnt/data/benyun/workspace/pi05_cup_clean_20260923/checkpoints/pi05_cup_clean_success/pi05_cup_clean_v1/{10000,20000,30000}` | `/home/lrl/workspace/umi_cup_models_5090_20260928/pi05/{10000,20000,30000}/inference_export` |
| LingBot 5k/10k | `/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/lingbot/checkpoints/checkpoints/global_step_{5000,10000}/hf_ckpt`（同级还有 distributed checkpoint） | `/home/lrl/workspace/umi_cup_models_5090_20260928/lingbot/{5000,10000}/hf_ckpt` |
| OpenWAM 5k/10k | `/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/openwam/checkpoints/2026-09-25_18-55-32/checkpoint_step_{5000,10000}.safetensors` | `/home/lrl/workspace/umi_cup_models_5090_20260928/openwam/{5000,10000}/checkpoint_step_{5000,10000}.safetensors` |

π0.5 的 `inference_export` 是用于推理的导出，不含完整训练优化器状态。LingBot 的 HF 目录需要三个 safetensors 分片、索引和配置文件一起存在。任何迁移先在源端/目标端逐文件比较相对路径、大小与 SHA-256；不能仅凭目录存在或总大小判定完成。**不要从 A100 或 squirrel 把权重复制到公开仓库或未经授权的队员电脑。**

2026-10-05 检查时，squirrel 的七个模型 user-service 都是 `inactive`，但 unit-file 状态为 `enabled`。这只是当时状态，重启后不能据此假定 GPU 空闲；先看 `nvidia-smi` 和 service/PID 身份，再启动控制台实验。本仓库不会自动改变服务器的开机策略。

## 4. 已授权队员如何像现有工作站一样打开控制台

前提是管理员已经为该队员配置各自的 Tailscale/局域网、跳板机、SSH 身份，以及私有 squirrel 部署；不要共享他人的私钥。`ssh squirrel_5090`、`ssh openwam-a100` 是示例别名，具体 HostName、ProxyJump、端口由管理员单独提供。网页无需输入 SSH 密码，也不保存凭据。先在**要运行网页的电脑**上验证：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 squirrel_5090 'hostname; nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader'
ssh -o BatchMode=yes -o ConnectTimeout=8 openwam-a100 'hostname'
```

只需要使用已经部署在 squirrel 的模型时，控制台不会重新训练，也不需要把权重下载到本机：

```bash
git clone https://github.com/zc1q84zc-boop/umi-arena-cup-screening-methods.git
cd umi-arena-cup-screening-methods/console
python3 sim_console.py
```

在**运行该命令的同一台电脑**打开 `http://127.0.0.1:8772/`。控制台默认经 SSH 访问 squirrel 的私有 `/home/lrl/dual-franka-yubi-isaac-sim-deploy`，由该机 GPU0 运行 Isaac Sim 和所选模型；本机浏览器只看结果并发送控制请求。不要把 8772、18783–18813 等控制/推理端口直接发布到公网。如果控制台在另一台团队控制机上运行，需由管理员提供单独的安全端口转发，浏览器里的 `127.0.0.1` 始终是**浏览器所在机器**。

在网页中先运行 2–3 步 `Hold`，确认场景、三路视频和报告，再选择已标为可执行的模型和对应 squirrel 后端，按“启动推理服务”，做 3–10 步因果短测，最后扩大到完整任务。运行完成后只停止本任务的推理服务；不要结束未知 PID。网站的“导入模型”含义是**选择已在私有服务器登记并验收的 checkpoint**，不是把任意本地权重拖进浏览器。离线轨迹比较另见 [`console/README.md`](console/README.md)，它需要单独授权的回放数据。

### 在私有 squirrel 上登记一个新 checkpoint（维护者）

1. 在新的私有目录存放完整权重及模型特有代码/环境，验证哈希、显存需求和 GPU0 无其他使用者；绝不覆盖现有 checkpoint。
2. 复制并审查本仓库 `console/deploy_servers/` 与 `console/squirrel_deployment/` 的相应模板，在 squirrel 上创建**独立、只监听 127.0.0.1** 的推理服务。把它作为未启用自动启动的专用服务，先核对启动脚本、checkpoint、模型 ID、端口和进程身份。
3. 在 squirrel 的私有 `.../adapters/` 放入对应在线适配器，逐项核对图像顺序/尺寸、状态单位与坐标系、语言提示、动作行时间语义、夹爪开度、未来观测隔离和安全门槛。模型加载成功不等于动作正确。
4. 在控制机的 `console/sim_console.py` 登记新的 `INFERENCE_BACKENDS`（主机、GPU、service、port、模型 ID、确切私有 checkpoint 路径）；在 `console/sim_policy_registry.json` 登记策略，先保持 `ready:false`。需要离线曲线时再把**元数据**加进 `console/checkpoints.json`，回放报告留在 Git 忽略目录。
5. 通过身份检查、`/health`、当前渲染图像与状态的 3–10 步因果短测，核对三路视频、有限动作、关节限位和服务退出。维护者确认后才能把该入口改为 `ready:true`，再进行同种子完整任务比较；未观测到稳定抓取/完整放盘放回，不得报告成功。

参考 [控制台部署边界](console/DEPLOYMENT.md) 和 [模型调试交接](MODEL_DEBUGGING.md)。当前 LingBot 5k/10k 因在线动作时序与训练提示词不一致而**暂时禁用**；仓库有本地修正，但截至本文日期尚未在 squirrel 完成配对部署和因果重测，不能绕过 `ready:false`。

## 5. 在线输入与结果边界

| 模型 | 在线模型观测 | 请求/执行与已知边界 |
| --- | --- | --- |
| π0.5 | 当前左右腕 RGB、双臂当前状态 | action chunk 中姿态/夹爪分别按标签时序对齐，仿真 30 Hz 连续目标；实际手眼外参仍是暂定值 |
| LingBot | 当前左右腕 RGB、双臂状态、相应 PA 阶段提示词 | 旧服务的姿态 row 0 为已观察增量；修正后取姿态 row 1–3 配夹爪 row 0–2；远端未复测前禁用 |
| OpenWAM | 当前头部 RGB、20 维双臂状态 | 视频骨干冻结；左右腕视频仅用于审计，不是隐含模型输入；当前帧条件推理，不能读取未来记录帧 |

页面的固定步数不是训练步数。当前网页每 10 Hz 请求一次模型，动作和三路录像按 30 Hz，物理仿真按 60 Hz；推理同步等待可能让墙钟运行比视频时长慢。所有现有多模型闭环只证明**能运行**，没有观察到完整杯子任务成功。甚至已知杯位受控抓取仍受指爪/相机托架碰撞几何影响，因此不能仅把失败归因于模型大小或训练步数。
