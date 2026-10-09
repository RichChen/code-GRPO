你在 WSL Ubuntu 的 `~/agentic-rl` 里工作，这是一个新仓库，目前只有 `docs/plan/`。

先完整读一遍 `docs/plan/PHASE-1-ENV-AND-TASKS.zh-CN.md`，然后严格按它从第 0 步开始执行。

要求：
1. 所有命令都在 WSL 的 bash 里跑，用 uv 管理 Python 环境。不要用 Windows 的 Python，也不要在 `/mnt/` 下面工作。
2. 每一步做完：跑全量测试 `uv run pytest -q`，提交一次 git，并在 `docs/plan/PROGRESS.md` 追加记录。
3. 做到 CHECKPOINT-1A 就停下，写好 `docs/plan/CHECKPOINT-1A.md` 后结束本轮，不要继续第 5 步。
4. `/mnt/d/crq/src/coding-harness` 只读参考，不要 import 它，也不要修改它。
5. 计划里写了"停下报告"的情况，或者某项要求做不到，就停下来写清原因，不要自行降低要求，也不要换方案。
6. 不要调用任何付费模型 API，不要写训练代码。
