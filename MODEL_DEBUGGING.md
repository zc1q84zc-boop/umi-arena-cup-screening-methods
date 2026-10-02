# Track 1 双臂模型调试交接

本页给已获团队服务器与数据访问权限的同事使用。代码可以从本仓库下载；**模型权重、官方数据、运行视频、SSH 密钥和 NVIDIA 原版 Panda 资产不在 GitHub**。这里记录的是 2026-10-02 的代码与路径，不是对远端当前在线状态的保证。每次实验先做下面的连通性和 GPU 检查。

## 1. 哪台机器做什么

| 位置 | 用途 | 路径或入口 |
| --- | --- | --- |
| 你的电脑或团队控制机 | 运行本仓库的网页控制台；浏览器只连本机 | `console/sim_console.py`，`http://127.0.0.1:8772/` |
| `squirrel_5090`（用户 `lrl`，RTX 5090 物理 GPU 0） | 当前在线模型推理、双 Franka/YUBI Isaac Sim、录像 | `/home/lrl/dual-franka-yubi-isaac-sim-deploy/`；模型根目录 `/home/lrl/workspace/umi_cup_models_5090_20260928/` |
| `openwam-a100`（用户 `benyun`） | 原始训练 checkpoint 与数据工作区；当前网页的在线推理只选择 squirrel | `/mnt/data/benyun/workspace/` 下的训练目录，见下表 |

`squirrel_5090` 和 `openwam-a100` 是团队 SSH 别名，需先取得网络、跳板机及各自主机的授权。请让管理员提供本人的 SSH 配置和密钥，**不要复制别人的私钥**。公开仓库不发布内网地址或跳板机凭据。示例配置：

```sshconfig
Host squirrel_5090
    HostName <管理员提供的内网地址>
    User lrl
    ProxyJump <管理员提供的跳板机别名>
    IdentityFile ~/.ssh/<本人获授权的密钥>
```

先检查连通性和 GPU 使用者，避免覆盖其他人的任务：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 squirrel_5090 'hostname; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader'
ssh squirrel_5090 'test -d /home/lrl/dual-franka-yubi-isaac-sim-deploy/repo && test -d /home/lrl/workspace/umi_cup_models_5090_20260928 && echo deployment-present'
```

如果 SSH 超时，先检查团队网络/Tailscale、跳板机和 SSH 授权。不要把“网页打不开”误判为模型故障。2026-10-02 从原控制机进行的 SSH 连通性检查出现超时，因此本次发布只核对了本地源码和已保存的实验记录，未声称远端服务当前在线。

## 2. 模型在哪里

下面是推理用模型的**现有服务器路径**，不是 GitHub 下载地址。`squirrel_5090` 的模型根目录为 `/home/lrl/workspace/umi_cup_models_5090_20260928`。

| 控制台模型 | squirrel 相对路径 | 后端端口 | 2026-10-02 网页状态 |
| --- | --- | ---: | --- |
| π0.5 10k | `pi05/10000/inference_export` | 18784 | 可执行实验 |
| π0.5 20k | `pi05/20000/inference_export` | 18785 | 可执行实验 |
| π0.5 30k | `pi05/30000/inference_export` | 18783 | 可执行实验 |
| LingBot VLA2 5k | `lingbot/5000/hf_ckpt` | 18810 | **暂时禁用**：时序/提示词修正待远端部署和重测 |
| LingBot VLA2 10k | `lingbot/10000/hf_ckpt` | 18811 | **暂时禁用**：同上 |
| OpenWAM Alpha 5k | `openwam/5000/checkpoint_step_5000.safetensors` | 18812 | 可执行实验；未抓起杯子 |
| OpenWAM Alpha 10k | `openwam/10000/checkpoint_step_10000.safetensors` | 18813 | 可执行实验；未抓起杯子 |

原始训练 checkpoint 留在 A100：π0.5 是 `/mnt/data/benyun/workspace/pi05_cup_clean_20260923/checkpoints/pi05_cup_clean_success/pi05_cup_clean_v1/{10000,20000,30000}`；LingBot 和 OpenWAM 在 `/mnt/data/benyun/workspace/umi_cup_multimodel_20260925/`。squirrel 上的 π0.5 是单独的推理导出，不是完整训练状态。`console/checkpoints.json` 和 `console/sim_console.py` 是可审计的登记来源；网页不会自动下载或迁移权重。

## 3. 下载并使用在线控制台

普通电脑不需要安装 Isaac Sim；仿真在 squirrel 的 GPU 上运行。控制台本身只需 Python 3.10+、`ssh` 与 `scp`，默认只监听本机回环地址。

```bash
git clone https://github.com/zc1q84zc-boop/umi-arena-cup-screening-methods.git
cd umi-arena-cup-screening-methods/console
python3 sim_console.py
```

在**同一台电脑**打开 <http://127.0.0.1:8772/>。使用已经运行的团队控制机时，可由管理员提供控制机别名，再建立本机端口转发；不要把 8772 直接暴露在公网。

首次测试按这个顺序：

1. 页面显示 `squirrel_5090` 的连接与 GPU 状态后，选 `Hold`、固定 2–3 步验证场景和录像。
2. 选择一个标为可执行的模型及其 squirrel 后端，点击“启动推理服务”，等模型健康检查通过。
3. 先跑 3–10 步，检查首步输入图、`online_adapter.jsonl`、三视角 MP4、`joints.csv` 和 `report.json`；确认动作有限、位姿坐标、夹爪方向和帧数对齐。
4. 再做同初态、同任务判定和同种子的闭环比较。固定步数用于可比实验；“直到任务成功”没有 150 步硬上限，可手动停止并正常写出视频。
5. 离开前用控制台“退出推理服务”停止自己启动的模型；不要手动杀掉不属于本实验的 GPU 进程。

每次运行使用新目录。远端写在 `.../dual-franka-yubi-isaac-sim-deploy/runs/console_<run-id>/`，控制台把报告、审计、CSV、预览图和录像复制到本地 `console/sim_runs/<run-id>/`。`sim_runs/` 不会提交到 GitHub。策略请求设定为 **10 Hz**，动作与录像为 **30 Hz**，物理仿真为 **60 Hz**；同步等待推理时实际完成频率可能低于 10 Hz。

可选的真实记录/离线预测控制台在 <http://127.0.0.1:8768/>。它要求另行获准的 practice 数据和回放结果；准备方法见 [`console/README.md`](console/README.md)。离线回放是在已录制观测上预测，不能当作闭环成功率。

## 4. 之前的优化和当前边界

- **相机方向**：左右腕相机绕视轴旋转 180°，使夹爪从图像下方进入，消除之前上下颠倒。视角、手套/夹爪外形和实验室背景仍有视觉域差，图像旋转不能消除这些差异。
- **初态对齐**：以训练回放同一帧的左右手位姿校准仿真起点。已有首帧检查为左手 `1.71 mm / 0.34°`、右手 `1.80 mm / 0.20°`；这是所选参考条件的残差，不是已测量的真实手眼外参。另一个镜像手到工具候选变换在 [`console/calibration_mirrored_provisional.json`](console/calibration_mirrored_provisional.json) 明确标为 provisional，不能冒充实测外参。
- **夹爪和安全**：YUBI mimic jaw 的传动符号已修正为相反方向开合；30 Hz 动作仍受 4 cm 位移门槛、关节限位与运行前 GPU 所有权检查约束。提高摩擦、夹紧力或动作尺度的实验被标为辅助诊断，不算纯模型成绩。
- **连续控制和平滑**：仿真内维持 60 Hz 连续关节目标，增加三点轨迹预览和 Panda 驱动阻尼。对同一批 120 步模型输出，实际关节速度反向 `173 → 5`，关节加速度 P95 `2.377 → 1.195 rad/s²`；这是控制层重放比较，不是任务成功率提升。同步推理等待仍可能使画面短暂停顿。
- **完整任务判定**：`plate_return` 要求先把杯子放上盘，再放回原位并松手；只放上盘的旧 `is_success` 不能当作完整任务成功。停止按钮会写入 `user_stop` 和录像，不用靠杀进程终止。
- **多模型因果输入**：π0.5 用当前左右腕图和双臂状态；LingBot 用当前左右腕图，OpenWAM 用当前头部图。LingBot 旧服务把已观察姿态增量和下一步夹爪目标错配，也用了与微调不一致的提示词；本仓库的本地修正尚未在 squirrel 重测，所以两个 LingBot 入口禁用。OpenWAM 的输入布局和因果时序已核对，但仍未稳定抓取。详见 [`console/DEPLOYMENT.md`](console/DEPLOYMENT.md)。

在相同初态的 120 步 π0.5 比较中，10k 的工具原点曾接近初始杯中心到 `5.22 cm`，20k/30k 分别为 `17.16/17.29 cm`，但三组都没有完成放盘。这个距离不是指垫到杯壁的接触距离。此前 OpenWAM 5k 的 600 步闭环也未抬起杯子。下一步调试应先看真实/仿真相机差异、手到工具外参和指垫接触几何，再做同条件重复试验。

## 5. 复现仿真改动与增加模型

[`simulator_overlay/`](simulator_overlay/README.md) 保存相机、初态、连续控制、任务判定等修改后的源码和测试。它基于公开 [dual-franka-yubi-isaac-sim](https://github.com/StevenLiudw/dual-franka-yubi-isaac-sim) 的 `609e6da` 版本，提供检查过的覆盖脚本。它**没有**包含 NVIDIA 原版 Panda USD、Isaac Sim 安装包或模型权重。新机器要按上游 README 从获许可的 Isaac 安装中配置 Panda 资产并重建组合 USD。现有 squirrel 部署可能有额外本地诊断文件，覆盖前应由维护者对比，不要直接覆盖共享机器。

新增模型时，先在私有工作区部署独立推理进程，然后在 `console/sim_policy_registry.json` 中登记可信适配器脚本。适配器必须使用**当前或过去**的仿真观测，输出 world-frame YUBI 工具目标（米、wxyz 四元数）、`[0,1]` 夹爪开度及 `action_dt_s`；先验明图像顺序、裁剪/归一化、状态历史、语言提示、动作标签时序和坐标变换，再把 `ready` 设为 true。只做模型加载或 3 步 smoke test 不能声称抓取成功。

## 6. 常见问题

| 现象 | 先检查 |
| --- | --- |
| 页面显示 SSH 超时 | 团队网络、跳板机、`ssh squirrel_5090`；不要先重启模型。 |
| 模型不可选 | `sim_policy_registry.json` 的 `ready`、后端健康检查、模型路径和服务身份；LingBot 当前故意禁用。 |
| GPU 被占用 | `nvidia-smi` 的 PID/进程名；控制台只允许身份检查通过的本实验服务，不会强杀别人任务。 |
| 画面停一下但手臂轨迹平滑 | 区分同步推理等待与关节控制振铃；分别看请求耗时和关节 CSV。 |
| 靠近杯子但抓不住 | 检查指垫到杯壁的实际距离、双侧接触、夹爪开度、相机和手眼外参；工具原点接近不等于抓取。 |
| 只有 `is_success=true` | 查看 `full_task_success`、`task_stage` 和 `task_objective`，确认是否完成放回。 |

公开仓库只供代码审查和授权复现。官方数据、训练视频、权重及基于单条记录的审计输出继续保留在各自受控工作区。
