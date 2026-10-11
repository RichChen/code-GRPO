# 服务器和本地之间传什么

在 4090 上跑 `scripts/server/setup.sh`，仓库用 git 拿。下面这些不在 git 里，要单独传。

**租机时必须选驱动 580 以上的镜像**（`nvidia-smi` 里的 CUDA Version 至少 13.0）。torch 2.13.0 和 vLLM 0.31.0 没有能在 CUDA 12.8 或 12.6 上用的官方组合，`setup.sh` 遇到更低的驱动会停下。详见 PROGRESS 里修订 2C-3 第 5 条。

## 传到服务器

- `runs/sft-v1/adapter/`：SFT-v1 的 fp32 LoRA。服务器上用 `merge_fp16.sh` 把它合并进以 fp16 加载的原始基座。
- `data/tasks/train.jsonl`、`data/tasks/heldout.jsonl`：题面。heldout 只在预注册的那一次评测里用，训练和 bench 都不读它。
- `data/tasks/grpo_v1_ids.txt`、`data/tasks/dev_ids.txt`：GRPO 训练题号和 dev 题号。
- `data/sft/v2/`：只有要在服务器上重放 SFT 数据校验时才需要。正式 GRPO 不读它。

不要传：

- `.env`。这份训练不调用付费 API。
- `runs/sft-preview/merged/`。服务器要自己用 `merge_fp16.sh` 从原始基座合并，不用这份本地预览目录。
- `runs/grpo-calib/` 的 jsonl。题号文件已经是校准的结果。

示例：

```bash
rsync -a runs/sft-v1/adapter/ server:~/agentic-rl/runs/sft-v1/adapter/
rsync -a data/tasks/train.jsonl data/tasks/heldout.jsonl data/tasks/grpo_v1_ids.txt data/tasks/dev_ids.txt \
  server:~/agentic-rl/data/tasks/
```

## 拉回本地

```bash
rsync -a server:~/agentic-rl/runs/grpo-v1/ runs/grpo-v1/
rsync -a server:~/agentic-rl/runs/server-bench/ runs/server-bench/
rsync -a server:~/agentic-rl/runs/sft-v1/merged-fp16/ runs/sft-v1/merged-fp16/
rsync -a server:~/agentic-rl/runs/sft-eval/dev-sft-fp16.jsonl runs/sft-eval/dev-sft-fp16-summary.json \
  runs/sft-eval/
```

留出集评测的 jsonl 也在 `runs/` 下，同一次 rsync 拉回。不要在拉回之前改 `docs/plan/HELDOUT-PREREG.zh-CN.md`。
