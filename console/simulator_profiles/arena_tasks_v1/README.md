# UMI Arena 后续四个任务环境

在当前 `tuned_v1` 上建立任务 2–5，共享双 Franka/YUBI 资产、桌面碰撞体、腕相机 CAD 安装与 v3 外观、相机同步渲染、固定夹爪开度映射、关节参考调节和轨迹 IK。杯子环境及其运行服务保持原样。公共依赖通过相邻目录的相对符号链接复用，`env.py` 只增加可选物体注册及 reset 钩子；默认杯子接口兼容。

| 任务 ID | 场景 | 官方子步骤 | 新增物体 |
|---|---|---:|---|
| `pens` | 笔筒取放 | 12 | 开口笔筒，两支铅笔、两支圆珠笔、两支记号笔 |
| `sps` | YUBI 零件分拣 | 15 | 九格分拣盒，15 件零件，使用第 1–8 格 |
| `cable` | USB-C 插接 | 1 | YUBI 支架、PCB/开口插座、插头和 32 段柔性线链 |
| `phone` | 手机装盒 | 8 | 手机、空腔盒体、可取下的套盖、内托/隔板 |

任务顺序与物体参照：[官方评测](https://umi-arena.airoa.io/evaluation)及其中四段视频，核对日期 2026-10-10。七类零件采用 Toyota YUBI v1.1.2 官方 STL，固定 revision `ce5a191c9189fbc057a808876a0daedef1519310`，哈希、路径及 CERN-OHL-W-2.0 来源见 `arena_tasks/assets/source_manifest.json`。具体零件是否与视频中的硬件版本相同仍待确认；齿轮和其余物品为参数化近似。

## 已实现的物理与接口

- 开口笔筒、分拣盒、手机盒/盖和 USB 插口采用分开的壁面碰撞体，保留空腔；没有吸附、自动抓取或插接成功时的位姿修正。
- 所有可操作物体是动态刚体；线缆为带有限摆角的球铰链胶囊段。CCD 和 120 Hz GPU PhysX 用于细小碰撞，策略/动作 10 Hz，录像 30 Hz。
- CAD 密闭网格按均匀密度计算实际重心和主惯性轴；总质量仍待称量。物体采用 64/8 次位置/速度求解，sleep 阈值为 0.0001 m²/s²，stabilization 阈值保持 PhysX 默认 0.00001 m²/s²，不启用额外场景 stabilization。
- episode reset 后物体只受物理接触作用，动作只控制机器人关节。所有子步骤沿用同一次物理状态和同一个 policy 实例；prompt 变化时清空上一阶段待执行动作。
- `Policy.infer` 只收到官方五个字段：两路 480×640 HWC uint8 腕相机、7 维双手相对位姿（xyzw）、2 维夹爪关节和当前 prompt。中心/主相机、物体真值和接触力只用于录像或评测。
- 返回动作要求 `(N>=16,16)`。局部 SE(3) 增量逐行积分一次，夹爪为绝对关节值，默认采用前 16 行；可用 `--adopt-rows 1` 做逐步闭环诊断。沿用现有 CAD 手根/TCP 变换与夹爪映射，它们仍是近似校准。
- 逐原语评测检查真实接触抓取、规定的手臂、目标空腔内完整物体包络、稳定保持及放置时松开。插接同时检查深度、横向误差与插头方向/截面；失败后的剩余步骤保留在分母中。判定阈值为本地诊断配置，不能替代官方人工裁判。

## 在新机器提供共享资产

本 profile 保留对相邻 `tuned_v1` 的相对符号链接。公开包仅包含场景与源码，不含编译机器人 USD。先按仓库 `simulator_overlay/README.md` 在干净的固定上游版本应用 overlay，并从已许可的 Isaac Sim 安装重建 Panda/YUBI 资产；再把所需机器人资产和 YUBI 公共网格置于本包的 `tuned_v1/yubi_isaac_sim_env/assets/`。保留相邻目录及符号链接结构。新任务的 Toyota STL 与许可证、固定 revision 和哈希已经在本 profile 的 `arena_tasks/assets/` 中。

`UMI_ISAAC_RUNTIME` 指向目标机已配置的 Isaac Sim runtime。源文件、配置和公开 CAD 足以重建任务层；仍须完成目标机 GPU、碰撞和模型接口验收。

## 运行

本机任务页面：[任务 2–5 控制台](http://127.0.0.1:8775/)，杯子控制台也已增加入口。页面可选择任务、随机种子和检查步数，并查看三路录像。

4090 目录：`/home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/arena_tasks_v1`。

相邻 `tuned_v1` 与已配置的 Isaac Sim 5.1 runtime 是依赖。迁移该目录时保留符号链接结构，同时提供对应的已授权 Panda/YUBI 资产；不需要重建机械臂。

```bash
cd /home/claude/Corl_Track_1/umi_workspace_zhangchi/umi-track1-console-4090-20261009/simulator_profiles/arena_tasks_v1
# CPU 目录检查，不启动 GPU
python3 -m yubi_isaac_sim_env.arena_tasks.run --list
# GPU0 空闲且公共 console.lock 可用时，检查场景并生成三路录像
bash run_arena.sh pens /absolute/new/run_directory 42 0 60
# 同样支持 sps、cable、phone。GPU1 及其他模型服务不参与场景检查。
```

单卡或其他安装位置通过 `UMI_ISAAC_RUNTIME` 指定 runtime，直接 CLI 可设置 `CUDA_VISIBLE_DEVICES`/`ISAAC_ACTIVE_GPU`。可选 `--no-randomization`；默认按 seed/index 对整个布局施加可复现的平移和 yaw 扰动，线链与插头同步移动，保持关节连接。

建场景与离线 USD 检查（没有输出时执行；现有文件不会覆盖）：

```bash
bash build_arena.sh
bash build_arena.sh validate
```

策略接入：在已持有独占运行锁、GPU 空闲且加载依赖准备好的环境中运行：

```bash
/path/to/isaac-python -m yubi_isaac_sim_env.arena_tasks.run \
  --task phone --headless --steps 1200 --record-run /absolute/new/policy_run \
  --policy-script /path/to/policy.py --checkpoint-dir /path/to/checkpoint
```

当前杯子微调权重没有在这四个任务上训练或验收，场景检查不会自动调用它们。多任务权重或支持这些指令的策略可由该适配器接入。目标机须预装模型所需依赖；此 Python 类接口不提供独立的模型 websocket 服务或官方提交镜像。

## 验证与限制

67 项 CPU 检查覆盖任务阶段、输入/动作契约和原有相机、姿态、控制逻辑。目标 GPU 的场景录像、物体状态、同步相机审计与接触/空腔 fit probe 见 `../../sim_validation/arena_tasks_20261010/` 以及 `../../deployments/arena_tasks_20261010/validation.json`。

物品尺寸、质量、摩擦、材质和布局部分来自视频估计。线链尚未标定连续材料的弯曲/扭转刚度，USB-C 未实现电子连接检测、卡扣或损坏模型，手机隔板暂用刚性内托近似。当前验收针对可运行的物理场景和接口；任务策略成功率、接触力尺度和真机一致性需要后续验证。
