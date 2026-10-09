# Official LingBot-VLA 2.0 on squirrel_5090

## Model identity

- Upstream: https://huggingface.co/robbyant/lingbot-vla-v2-6b
- Pinned revision: `11c703bf6a5c1f45b3b69168482da11fdbba53d7`.
- Six original weight shards, approximately 25.5 GB. Every downloaded file is checked against the independently retrieved upstream size and SHA-256 in `upstream_manifest.json`.
- These are the official pretrained weights. No cup-task fine-tuned checkpoint is read or merged.
- Remote checkpoint: `/home/lrl/workspace/umi_cup_models_5090_20260928/lingbot/official/hf_ckpt`.
- Service: `lingbot-official-squirrel.service`; loopback endpoint: `http://127.0.0.1:18814/infer`.
- Console policy: `lingbot-vla2-official-pretrained`; backend: `lingbot_official_rtx5090`.

## Inference and simulator interface

The deployment uses the existing upstream LingBot inference implementation, in BF16, with the current UMI robot feature mapping, coordinate transforms and dataset normalization statistics. These interface assumptions remain provisional; this is a zero-shot transfer experiment, not an official benchmark result or a verified native UMI deployment recipe.

The loader requires every instantiated inference parameter to be present in the official snapshot, with matching shapes. The verified deployment loaded **all 1,708 parameter entries**, with **zero missing and zero skipped entries**. Coverage is recorded in `lingbot/official/official_load_audit.json`. The source checkpoint files remain unchanged. A compatibility allowlist exists for loss-only projection tensors if an upstream inference architecture omits them; this deployment did not use it. Any other missing or extra parameter aborts loading.

Current wrist RGB and robot state are used. The simulator keeps visual alignment v3, the calibrated initial-pose approximation, fixed CAD jaw mapping, explicit mirrored jaw drives and 30 Hz continuous smoothing. Official right-place / left-return instructions are selected by the task evaluator. No demonstration trajectory, future image, cup-coordinate action correction or inactive-arm hold is added.

The server health response and every inference response identify the official repository, revision, file verification, runtime dtype, robot config hash and normalization hash. The adapter writes this identity into each observation/action audit row.

## Run

Start from the console at http://127.0.0.1:8772 and select **LingBot VLA2 官方预训练版 · 双臂零样本闭环**. The console manages model switching and rejects unrelated GPU compute processes. For a full rollout select **run until success**, retaining the stop button for manual termination.

Alternatively, the console API accepts:

```json
{
  "policy": "lingbot-vla2-official-pretrained",
  "inference_backend": "lingbot_official_rtx5090",
  "setup_index": 0,
  "seed": 42,
  "camera": "head",
  "task_objective": "plate_return",
  "run_until_success": true
}
```

POST to `/api/runs` with `Content-Type: application/json` and `Origin: http://127.0.0.1:8772`. Health: `curl http://127.0.0.1:18814/health` on squirrel. A successful load or finite action chunk does not establish task success.

The temporary authenticated LAN transfer source serves only the public upstream model files. Its token/config and process are removed after transfer; it is not part of runtime inference.
