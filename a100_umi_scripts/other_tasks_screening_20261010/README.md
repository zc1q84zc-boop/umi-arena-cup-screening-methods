# 其它任务：质量筛选与原始三参考 IK 交集

## 公开包入口

[筛选结果与复现](../../TRAINING_DATA_SCREENING_20261010.md) 汇总任务对应、阈值和当前状态。公开统计在 [quality summary](../../results/other_tasks_quality_summary.json)、[PA 覆盖](../../results/other_tasks_quality_review_summary.json)、[verification](../../results/other_tasks_verification.json)、[status](../../results/other_tasks_status.json)。下面的 selection/quality/status 路径描述 A100 私有运行目录，逐条数据不随 GitHub 发布。

任务要求：同时使用原杯子数据的质量筛选方法和同事的三参考 IK 方法，再取 episode 交集。原始 Parquet、视频、标签和现有杯子数据视图均保持不变。本次不启动训练，不改写原始 prompt，也不配对或 merge episode。

本地脚本在本目录；A100 运行目录为 `/mnt/data/benyun/workspace/umi_arena_other_tasks_screening_20261010`。`status.json` 区分质量扫描、等待 IK 和完整交集完成，只有 `intersection_completed=true` 表示两套筛选完成。**`quality/` 是中间结果，不能作为“两套方法已经取交集”的最终数据集。**

## 已确认的数据范围

来源为 A100 的 `yubi-corl2026-umi-arena`。按元数据中完全相等的 `short_horizon_task` 和成功标记选择，逐条保留原始 PA 名称。

| 标识 | 原始任务名 | 成功 episode | 原始帧 | 与新仿真场景的对应关系 |
| --- | --- | ---: | ---: | --- |
| `phone` | Pack Smartphone into Box | 14,726 | 3,169,018 | 与官方八步手机装盒一致 |
| `chain_sps` | Set Parts Supply (chain) | 14,280 | 1,569,835 | 链条、链轮、螺栓版本；属于相关任务，不是官方 YUBI 零件版 SPS |

当前发布版元数据没有官方笔筒取放、YUBI 零件版 SPS、YUBI MCU USB-C 插接的对应任务。`Phone charge` 与 `Put pencils into pencil case and take them out` 也不等同于这些官方任务，不能自动合并或重新命名。全部任务元数据统计与匹配说明保存在目标机 `discovery.json`。

## 已完成的质量阶段

已检查 1,075 个数据分片、29,006 个 episode 和 4,738,853 帧。逐 episode 的实际扫描长度均与元数据相等，保留/隔离名单无重叠且完整覆盖输入，原始 PA prompt 和元数据保持一致。`verification.json` 是输出核对记录。

| 任务 | 质量保留 episode | 保留帧 | 隔离 episode | 保留比例 |
| --- | ---: | ---: | ---: | ---: |
| 手机装盒 | 2,978 | 599,292 | 11,748 | 20.22% |
| 链条版 SPS | 3,273 | 383,446 | 11,007 | 22.92% |

这是严格复用杯子阈值产生的中间结果。多数隔离涉及手部角速度候选标记；没有新人工异常确认，不能将隔离比例解释为数据损坏率。`quality_review_summary.json` 保留仅角速度触发/其它规则触发的分类，以及各 PA 阶段的剩余数量，供后续复核和决定是否需要时间缩放使用。未经授权未调整阈值或重新纳入这些 episode。

**同事原始 IK 尚未运行，最终两方法交集尚未生成。** 不应将上表作为最终训练数据规模。

## 质量方法

`quality_rules.py` 是现有杯子 `hard_constraint_prefilter.py` 的逐字节副本，SHA-256 为 `7b09e84996cbd829e8731a5713b95c41fe3a99df33c6c5e8a49ae688e86a7a44`，保留相同的逐帧时间、四元数、双手相对位姿、动作一致性、跳变、速度和加速度规则及 FR3 条件速度参考。`quality_method.json` 记录阈值与来源。

`scan_quality.py` 只更改任务选择、打包文件读取和结果导出，不改动 `episode_metrics`。它按 shard 写入可恢复的进度，并检查相邻文件以修复元数据边界指向偏差。为减少无关 episode 的 Python 解码，Parquet 读取时按当前 episode 编号过滤；每个选中 episode 仍逐帧检查。两名 CPU worker 用低优先级运行，不使用 GPU。

触发机器规则的整个 episode 进入可恢复的隔离视图，不能据此声称真机不可执行。现有人工异常标签只涉及杯子，没有将它们移用到其它任务，也没有声称完成新的人工视频审核。

目标机输出：

- `selection/manifest.json`：各任务成功 episode 的输入名单。
- `selection/full_record_manifest.json`：所选 UUID 的全部 PA episode，包含非成功阶段，供同事的整记录 IK 方法使用。
- `quality/<task>/clean_dataset/`、`quarantined_dataset/`：质量保留与隔离视图。
- `quality/<task>/episode_decisions.jsonl`：每个 episode 的原因及候选帧事件。
- `quality_summary.json`：逐任务统计。
- `quality_review_summary.json`：规则候选和 PA 覆盖统计。
- `verification.json`：完整输出核对与实际帧覆盖记录。

视图用符号链接复用原始数据。训练加载器必须按 `manifest.json` 的 episode 白名单读取；直接遍历链接的原始文件仍会读到被排除的 episode。

## 同事 IK 方法的依赖

目前只找到 `/mnt/data/benyun/workspace/yubi-corl2026-umi-arena/labels/cup_three_reference_intersection_20260924/` 中的杯子通过名单。该目录说明来源是 `platform_labels_start_refit`，但没有提供生成脚本、模型配置或其它任务结果，因此不能将杯子标签套用到新任务，也不能用另外的名义安装位姿替代后声称用了同一方法。

需取得同事原始生成程序、模型/参考起点/工具变换配置，然后对 `selection/full_record_manifest.json` 重新运行 FR3 左右、OpenArm v2 左右、G2 参考双臂的全帧检查。IK 成功仅表示参考可解；连续关节运动、碰撞和真机任务成功没有被认证。

`build_intersection.py` 已准备好接收同事全帧结果。它验证 UUID/episode/帧数与输入相符、五项参考标志的逻辑一致性，以及整个输入范围的完成记录和原始程序/配置哈希，然后输出 `quality_ik_intersection/<task>/manifest.json`。未完成的 IK 不会被默认为通过，也不会被当作已完成的筛选。

```bash
# A100：恢复或执行质量阶段
bash scripts/run_quality.sh /mnt/data/benyun/workspace/umi_arena_other_tasks_screening_20261010

# 取得同事原始程序的全部结果与完成元数据后，再执行交集阶段
/home/benyun/.venvs/umi_arena_pi05/bin/python scripts/build_intersection.py \
  --run-root /mnt/data/benyun/workspace/umi_arena_other_tasks_screening_20261010 \
  --ik-record-labels /path/to/original-method-record-labels.jsonl \
  --ik-completion /path/to/original-method-completion.json \
  --dataset-root /mnt/data/benyun/workspace/yubi-corl2026-umi-arena
```

交集仍以 episode 为单位，可能只剩下完整记录的一部分阶段。`record_membership.jsonl` 会明确记录这一点；完整任务配对、阶段衔接和 merge 是之后独立的训练整理工作。

9 项合成测试验证成功任务范围、原始规则、相邻文件恢复、隔离并集、数据不可变性、IK 输入完整性和未完成结果的拒绝。
