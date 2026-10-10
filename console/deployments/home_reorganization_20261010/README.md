# Zhangchi UMI 项目目录整理

目标机：4090 `benyun-workstation`，Linux 账号 `claude`。
当前目标目录：`/home/claude/Corl_Track_1/umi_workspace_zhangchi`。
首次整理目录为 `/home/claude/umi_workspace_zhangchi`；之后按用户要求移至与 jiazhen 相同的实际父目录。

迁移范围严格限定为此前本项目部署、修改备份和结果导出记录对应的八个目录：

| 目录 | 本项目用途 |
| --- | --- |
| `dual-franka-yubi-isaac-sim-deploy` | Isaac Sim 运行时、机器人资产和仿真输出 |
| `umi-track1-console-4090-20261009` | 杯子控制台及后续四个任务控制台 |
| `umi_cup_intersection_v2_20261009` | 交集版模型部署副本 |
| `console-plate-radius-45mm-20261009` | 调整杯子成功判定半径的备份 |
| `console-precision-default-20261009-8d5e16b6ec82` | 修改控制台默认参数的备份 |
| `console-shared-camera-render-20261009` | 修改相机渲染的备份 |
| `console_exports` | 本项目运行结果导出 |
| `console_snapshots` | 本项目控制台快照 |

范围根据项目部署记录和用户确认的目录清单确定，不能仅凭 Linux 文件所有者为 `claude` 推定属于某位同学。系统、软件、来源不明目录以及任何 `umi_workspace_jiazhen` 实体目录均不迁移。

`umi_cup_official_prompt_20k_20261009` 已是指向共享位置的链接，本次排除。`Corl_Track_1` 中剩余的其它内容也保留。共享模型继续使用原位置；仅更新其指向本次迁移的 Python 运行时和仿真脚本的引用，不移动权重。

`reorganize_4090.py` 先输出计划，实际迁移时等待正在运行的评测释放 `console.lock`。它用同磁盘 rename 移动八个目录，更新自己的程序、环境符号链接及服务配置，并重启受影响的服务。只保留新目录内的配置修改备份，不建立旧目录兼容链接。

目标机维护记录位于 `umi_workspace_zhangchi/maintenance/home_reorganization_20261010/`。只有取得 `receipt.json` 且通过最终验证后才算完成。核对包括原目录 inode、模型/视频/网格文件的 inode/大小/修改时间、旧路径不存在、共享链接不变、Python 引导及两个控制台的 HTTP 状态。基于路径替换更新的运行准备标记会明确记录；不声称重新进行了模型推理或物理评测。

## 完成结果

2026-10-10 已完成八个目录迁移；原目录 inode 保持一致。536 个模型、视频和网格等二进制文件的 inode、大小及修改时间未改变。八个目录在原 home 位置和 `Corl_Track_1` 位置均不存在，也没有留下旧目录兼容链接。

已更新运行所需路径、八条内部符号链接、服务定义，以及共享环境指向本项目运行时的十五项依赖引用。共享实体目录和模型权重没有迁移。`Corl_Track_1` 中其它人的目录和共享模型链接保留，因此没有删除这个仍含共享内容的容器目录。

迁移是在正在运行的 `live005r3` 评测完成并释放目录锁后执行的。杯子控制台、四任务控制台、主 π0.5 服务及评测用 π0.5 服务均恢复运行；两个控制台目录接口及两个模型 `/health` 接口均返回 HTTP 200。更新的内部链接没有悬空。

本地当前部署脚本也同步到新路径，修改前版本保存在本目录 `local_before/`。完整迁移记录见 `receipt.json`，最终检查见 `final_verification.json`。

## 调整父目录

2026-10-10 根据用户“放到和他一样的父目录下”的要求，将整个已核实的 `umi_workspace_zhangchi` 移至 `/home/claude/Corl_Track_1/umi_workspace_zhangchi`。它与 `/home/claude/Corl_Track_1/umi_workspace_jiazhen` 均为真实目录，父目录相同。原 `/home/claude/umi_workspace_zhangchi` 不存在，未建立兼容软链接。

`relocate_parent_4090.py` 使用同磁盘 rename，保留工作区及八个项目子目录的 inode。536 个模型、视频和网格等二进制文件的 inode、大小及修改时间保持不变；jiazhen 目录和共享链接的 inode/链接目标保持不变。更新自身运行路径、依赖本项目运行时的共享引用及三个用户服务配置，不移动其他人的内容。修改前文件保存在新工作区 `maintenance/parent_relocation_20261010/`，失败时可恢复原配置和位置。

杯子控制台、四任务控制台以及两个 π0.5 服务恢复运行；前端、目录接口和模型健康接口均返回 HTTP 200。只更新路径和对应源文件准备标记，没有重新进行 GPU 物理评测。远端迁移记录为 `maintenance/parent_relocation_20261010/receipt.json`。

本地 56 个当前部署脚本已同步至最终路径，31 个 Python 文件通过语法解析、23 个 shell 文件通过 `bash -n` 检查；两个服务定义同步更新。其修改前版本和记录位于本目录的 `parent_relocation_20261010/`。首次整理的历史审计文件保持原样。
