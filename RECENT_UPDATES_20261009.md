# 仿真环境、模型与部署更新：2026-10-09

这是截至 2026-10-09 的代码和方法快照。此前的筛选结果、展示页面和历史调试说明保留。服务器的实时占用、模型权重、数据、录像、逐 episode 标签与运行报告不属于这份公开代码包。

## 仿真环境

- `tuned_v1` 采用上游 `simulation-replay-259632-259633` 的 `29652dc4e93903514b292b24988fd1a893b10e21` 作为基础。两侧 YUBI 指爪显式驱动，第二爪镜像；夹爪编码角通过 CAD 开度曲线双向转换。
- 使用详细指端接触面、关节阻尼和连续目标控制。保留 0.8 rad/s 速度与 1.5 rad/s² 加速度限制；任务评价器读取物体状态并判断放盘、松手、取回，不控制或移动杯子。
- 腕相机从当前 GPU 夹爪刚体位姿更新；视频录制和模型输入共享当前传感器。视觉对齐 v3 加入固定相机拟合、鱼眼参数、程序化布面、三维背景及夹爪/杯盘外观调整。
- 杯子基准材质采用静摩擦 0.5、动摩擦 0.4，属于未实测初值。指端摩擦实验与柔性杯实验使用独立 profile，不改变基准实验的物理配置。
- 新增薄壳/PVC 刚度、压缩响应和数值精度探针，以及接触、夹爪和杯子提起检查。刚度与外参均保留“未实测”标记；视觉对齐、模型接口检查和任务成功分别记录。

仿真源文件见 [`simulator_overlay/files/`](simulator_overlay/files/)。按 [overlay 使用说明](simulator_overlay/README.md) 在干净的上游 checkout 应用，再使用本地授权的 Isaac Sim Panda 资产重建 USD。本仓库没有发布已编译机器人资产或 NVIDIA 的 stock Panda 文件。

## 模型与动作协议

[`console/`](console/README.md) 更新了 π0.5、LingBot、OpenWAM 的在线适配器、控制台模型登记、模型服务和验证工具。LingBot 官方预训练版有单独的模型标识和入口，不能与杯子任务微调版混淆。

官方阶段指令统一由 [`official_cup_prompts.py`](console/deploy_servers/official_cup_prompts.py) 提供：右手拿杯放到盘子上，完成并松手后切换到左手取回。阶段选择只提供语言条件，不向动作控制器提供杯子位置辅助。

新交集版与历史版使用不同模型 ID 和适配器：

| 交集版 | episode 使用 | 输入 | 动作标签和执行 |
| --- | --- | --- | --- |
| π0.5 30k | 不配对，各 episode 独立；UUID 划分训练/验证 | 当前左右腕图像、手间相对位姿 7D、夹爪 2D、官方阶段 prompt | 16D 双臂动作；未来 100 ms 行 0 积分一次，三个 30 Hz 控制周期保持同一端点 |
| OpenWAM full-pass 5069 | 交集 episode 配对合并；监督不跨 prompt 阶段 | 同上，双腕图像固定左右顺序拼接；内部 state/action padding 80D | 同样采用未来 10 Hz 动作行 0；保存并加载独立归一化与 observation/action contract |

10 Hz 是本轮训练标签设计，不是官网强制频率。三个局部位姿增量按 SE(3) 顺序合成，平移随前面的旋转变换；不能直接逐元素相加。新标签已经对齐未来时间槽，部署不能再次跳过第 0 行或再合成三行。历史模型沿用其各自训练时序，不能套用新适配器。

## 完成的训练

详细汇总见 [`results/latest_training_20261009.json`](results/latest_training_20261009.json)。

- **π0.5：30,000 步完成。** global batch 32、双 A100/FSDP2；2,504 个训练 episode、277 个验证 episode；181,072 个训练窗口、19,664 个验证窗口；10k、20k、30k checkpoint 保存在服务器。固定 256 条验证窗口的 EMA loss 在 10k 为 0.049089、30k 为 0.067746，后期上升；部署仍明确选择用户要求的最终 30k。
- **OpenWAM：5,069 次更新完成。** 双 A100、每卡 batch 16、无梯度累积；全部 160,338 个不同训练窗口和 162,196 个阶段平衡索引实际消费，尾批补齐 12 项，共呈现 162,208 次。读取顺序与完整无放回置乱计划一致，`full_pass_completed=true`；正式训练耗时约 7.72 小时。冻结视频骨干，更新约 10.21 亿 action/proprio 参数。

训练代码分别位于 [`pi05_intersection_20261008`](a100_umi_scripts/pi05_intersection_20261008/)、[`openwam_merged_cup_20261008`](a100_umi_scripts/openwam_merged_cup_20261008/) 和 [`openwam_fullpass_20261009`](a100_umi_scripts/openwam_fullpass_20261009/)。质量筛选与 IK 参考筛选的交集构建方法见 [`src/build_clean_ik_intersection.py`](src/build_clean_ik_intersection.py)。这些脚本需要另外授权的数据与计算资源，公开仓库不提供筛选清单或 checkpoint。

## 5090 / 双 4090 部署

[`intersection_20261009`](console/deployments/intersection_20261009/) 提供两个最终模型的独立服务、适配器、传输工具和因果推理检查。5090 上已完成 275 个文件、34.72 GiB 的迁移及 SHA-256 校验，两个模型在 A100 上通过两种官方 prompt 的真实推理。5090 的新策略仍保留待 GPU 验证状态；登记文件不是服务器实时占用证明。

[`migration_4090_20261009`](console/deployments/migration_4090_20261009/) 提供双卡原生控制台、路径迁移、服务归属检查、推理/仿真 GPU 分离和 OpenWAM CPU 文本编码卸载代码。当前本地迁移记录显示旧七个模型已部署，新交集版传输未完成、仍保持禁用；代码存在不能代替完整权重校验或目标 GPU 验收。模型就绪检查要求验证记录和关键代码哈希匹配。

[`grasp_margin_4090_20261009`](console/deployments/grasp_margin_4090_20261009/) 保存接触/抓取余量分析方法。录像和逐帧输出留在私有运行目录。

## 代码检查

CPU 测试不需要私有权重或 Isaac Sim。依赖见 [`console/requirements-tests.txt`](console/requirements-tests.txt)。发布检查的实际结果记录在 [`results/public_source_validation_20261009.json`](results/public_source_validation_20261009.json)。

```bash
python3 -m unittest discover -s console -p 'test_*.py'
PYTHONPATH=console:console/simulator_profiles/tuned_v1 python3 -m pytest console/simulator_profiles/tuned_v1/tests -q
python3 console/deployments/intersection_20261009/test_contract.py
PYTHONPATH=console python3 -m unittest discover -s console/deployments/migration_4090_20261009 -p 'test_*.py'
python3 -m unittest discover -s console/deployments/grasp_margin_4090_20261009 -p 'test_*.py'
```

涉及私有 practice 数据或本地 Isaac USD 依赖的检查会明确跳过。GPU 加载、实际接触、三路录像和完整放盘放回需要在授权服务器另做；源码测试通过不等于模型任务成功。
