# 训练集筛选：范围、结果与复现

## 杯子质量与参考 IK 交集

杯子原质量范围为 3,975 episode / 827,323 帧，质量保留 3,423 / 746,914。
质量白名单与同事参考 IK 白名单按 **episode_index 精确取交集**，得到
2,781 episode / 600,087 帧（30 Hz，约 5.556 小时）。

| 交集指标 | 数量 |
| --- | ---: |
| 质量范围中因 IK 排除的 episode | 642 |
| IK 范围中未进入质量保留的 episode | 468 |
| 含交集 episode 的 record | 1,550 |
| IK record 的全部 episode 保留 | 1,213 |
| IK record 只保留部分 episode | 337 |
| 每个交集 record 保留两个 / 一个 episode | 1,231 / 319 |

汇总见 [cup_quality_ik_intersection_20261008.json](results/cup_quality_ik_intersection_20261008.json)，
构建代码见 [build_clean_ik_intersection.py](src/build_clean_ik_intersection.py)。
交集以 episode 为单位，不自动补回被筛掉的另一阶段。π0.5 的独立 episode 训练、
OpenWAM 的配对合并与完整一轮遍历是后续不同的数据组织方式，
见 [前一版训练说明](RECENT_UPDATES_20261009.md#完成的训练)。

## 手机装盒与链条版 SPS 的质量筛选

新增 [筛选流水线](a100_umi_scripts/other_tasks_screening_20261010/README.md) 按原始
任务名完全匹配和成功标记选择。没有改写原始 PA prompt，没有配对或 merge episode，
没有开始新任务训练。

扫描 1,075 个分片、29,006 episode、4,738,853 帧；每个 episode 的实际帧数与元数据
一致。保留和隔离完整覆盖输入且不重叠，原始数据与标签未改写。

| 任务 | 输入 episode / 帧 | 保留 episode / 帧 | 隔离 episode / 帧 | 保留率 |
| --- | ---: | ---: | ---: | ---: |
| 手机装盒 | 14,726 / 3,169,018 | 2,978 / 599,292 | 11,748 / 2,569,726 | 20.22% |
| 链条版 SPS | 14,280 / 1,569,835 | 3,273 / 383,446 | 11,007 / 1,186,389 | 22.92% |

严格复用杯子 `episode_metrics`，新任务 `quality_rules.py` 的 SHA-256 是
`7b09e84996cbd829e8731a5713b95c41fe3a99df33c6c5e8a49ae688e86a7a44`。
规则检查时间戳、四元数、双手相对位姿、动作与观测一致性、跳变、运动速度和加速度。
阈值和假设见 [quality method](results/other_tasks_quality_method.json)。

| 隔离原因分类 | 手机装盒 | 链条版 SPS |
| --- | ---: | ---: |
| 仅角速度候选标记 | 10,342 | 9,126 |
| 涉及其它规则 | 1,406 | 1,881 |
| 本轮新增人工确认 | 0 | 0 |

隔离是可恢复的训练质量候选，不能当作数据损坏率或真机不可执行的证明。
角速度候选占多数，各 PA 阶段剩余规模差异较大；
[原语覆盖和复核统计](results/other_tasks_quality_review_summary.json) 保留这些区别。

### 原始三参考 IK 仍待执行

截至本地完成记录，`quality_completed=true`，`ik_completed=false`，
`intersection_completed=false`。已有杯子 IK 名单不能应用到其它任务轨迹。
须取得同事原始生成程序和参考配置，对新的完整记录范围运行，再生成最终交集。

[build_intersection.py](a100_umi_scripts/other_tasks_screening_20261010/build_intersection.py)
会检查 UUID、episode、帧数、五项参考标志、整个输入范围完成情况，以及原始程序 / 配置
哈希。不完整的 IK 结果会被拒绝，不会默认为通过。
参考可解不等于已认证关节连续性、碰撞或真机成功。

## 新场景与数据的对应

| 新场景 | 当前数据对应 |
| --- | --- |
| 手机装盒 | `Pack Smartphone into Box`，八个 PA 对应 |
| YUBI 零件版 SPS | 尚无准确对应；`Set Parts Supply (chain)` 单独保留为相关变体 |
| 笔筒取放 | 尚无准确对应；铅笔盒任务不自动替代 |
| MCU USB-C 插接 | 尚无准确对应；`Phone charge` 不自动替代 |

## 授权复现

代码位于 [other_tasks_screening_20261010](a100_umi_scripts/other_tasks_screening_20261010/)。
公开依赖为 NumPy、SciPy、PyArrow；合成验证入口 `test_pipeline.py` 不需要原始数据。

```bash
python3 -m pip install -r a100_umi_scripts/other_tasks_screening_20261010/requirements.txt
cd a100_umi_scripts/other_tasks_screening_20261010
python3 -m unittest -v test_pipeline.py
```

实际扫描需取得 AIRoA 数据许可。将脚本部署到独立 A100 工作目录的 `scripts/` 后，
按该目录 README 运行 `prepare_selection.py`、`run_quality.sh` 和 `verify_outputs.py`。
本轮私有输出位于 `/mnt/data/benyun/workspace/umi_arena_other_tasks_screening_20261010`。

`data/`、`videos/`、`meta/` 是原始 packed 文件的链接。
**训练加载器必须应用 manifest 的 episode 白名单**；仅换成名为 `clean_dataset`
的路径仍会读取被排除的 episode。逐条白名单、标签、原始数据与 checkpoint
分别留在授权工作区。公开汇总核对见 [verification](results/other_tasks_verification.json)。
