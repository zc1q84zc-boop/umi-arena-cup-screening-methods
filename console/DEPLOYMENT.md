# 在线控制台代码与部署边界

先读仓库根目录的[模型调试交接](../MODEL_DEBUGGING.md)。本目录包含网页、SSH 调度器、模型适配器、私有服务器上的推理服务脚本和参考启动脚本；不含可直接运行的模型或 Isaac Sim 安装。

## 目录用途

| 文件 | 运行位置 | 作用 |
| --- | --- | --- |
| `sim_console.py`, `simulation.{html,js,css}` | 控制机 | 回环网页、模型选择、健康检查、运行/停止与录像下载 |
| `sim_policy_registry.json`, `checkpoints.json` | 控制机 | 可执行策略与离线 checkpoint 元数据；`ready:false` 不可启动 |
| `pi05_isaac_online_adapter.py`, `lingbot_isaac_online_adapter.py`, `openwam_isaac_online_adapter.py`, `online_calibration.py` | squirrel 的 `.../adapters/` | 把实时仿真图像/状态转换为模型输入，再把动作转换为 world-frame YUBI 目标 |
| `pi05_online_server.py`, `deploy_servers/{lingbot,openwam}_sim_server.py` | squirrel 私有模型工作区 | 各模型 HTTP 推理服务，只监听本机 loopback |
| `squirrel_deployment/` | squirrel 私有模型工作区 | 已验证的模型启动脚本与 π0.5 user service 样例；路径是原部署路径，换机器须检查 |
| `variants/` | squirrel `.../variants/` | 非模型的抓取接触物理诊断；不能当作纯模型成绩 |

控制机默认把 `ssh squirrel_5090` 用作远端。若管理员给了不同别名，可以设置 `UMI_SIM_HOST`；部署根目录不是默认路径时设置 `UMI_SIM_REMOTE_ROOT`，同时检查 `sim_policy_registry.json` 内的绝对脚本路径和各服务的 checkpoint 路径。网页固定在 `127.0.0.1:8772`，POST 只接受同源 JSON；不要通过反向代理公开写操作。

```bash
UMI_SIM_HOST=squirrel_5090 \
UMI_SIM_REMOTE_ROOT=/home/lrl/dual-franka-yubi-isaac-sim-deploy \
python3 sim_console.py
```

`sim_console.py` 不是一键部署器。它假设远端已有 `repo/`、`adapters/`、`runs/`、`run_headless.sh`、模型推理环境、服务单元以及合法的 NVIDIA Panda 资产。共享机更新前请比对现有文件、停止本实验服务并由维护者验证；本仓库发布不会修改 squirrel。远端的模型服务使用 GPU 0，`console.lock` 防止本控制台的并发仿真；模型启动前检查 GPU 进程归属。

## 模型适配边界

- π0.5：左右腕 640×480 实时 RGB 与左右工具/夹爪状态；检查当前帧、动作 chunk 的 30 Hz→10 Hz 执行时序和 `source_hand`↔`world_tool` 双向变换。10k、20k、30k 各有独立推理导出。
- LingBot：左右腕图；旧服务把姿态行 `t` 的**已观察运动**和夹爪行 `t` 的**下一步目标**配对，还用了训练时没有的完整双阶段提示词。本仓库的服务代码改用姿态行 `1..3` 对夹爪行 `0..2`，使用训练样本的 pickup primitive prompt，并返回时序标记；适配器拒绝没有该标记的响应。**截至 2026-10-02 远端未部署重测，两个网页入口保持 `ready:false`。**
- OpenWAM：当前头部图和 20 维状态；训练/服务的 320×384 头图填充、33 帧历史、四帧间隔、next-frame 目标、20→80 slot 对应和归一化已经对照。没有证据表明当前失败来自一个特定时序 bug；可执行不等于抓取成功。

推理端只收当前/历史仿真观测，不得读取未来回放帧。模型和仿真共享 squirrel 的同一张 RTX 5090，故加载前先确认显存；服务按模型选择自动启动，运行后只停止身份匹配的本实验进程。A100 原始 checkpoint 路径用于溯源和受控回放，网页当前不把 A100 当作在线后端。

## 新同事的最小验收

1. `python3 -m unittest -v test_server.py test_sim_console.py test_online_calibration.py test_pi05_online_timing.py test_lingbot_pose_frame.py test_openwam_pose_frame.py` 检查不依赖权重的代码约束。
2. 在浏览器跑 2 步 `Hold`，看 `report.json`、三路录像、`joints.csv` 和首帧预览；这是仿真/记录基线。
3. 启动一个已标记可执行的模型服务，先跑 3 步，核对首帧图像哈希变化、动作有限值、姿态坐标、夹爪方向、健康检查的模型 ID 和输入时序。
4. 逐步扩大到完整任务；比较模型必须使用相同参考初态、种子、场景与任务判定。只有 `full_task_success` 才代表“放盘再放回”的完整仿真任务完成。

如果服务没有达到健康状态，不要把 `ready` 改成 true 绕过检查。模型权重与数据仍由各服务器的授权管理者提供。
