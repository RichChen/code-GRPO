# agentic-rl

在单卡 4090 上用 GRPO 训练 1.5B–3B 模型做多轮工具调用的代码修复，检验"跨题行为能否靠训练学会"。

当前状态：阶段 1（环境与题集）进行中，计划见 `docs/plan/PHASE-1-ENV-AND-TASKS.zh-CN.md`，进度见 `docs/plan/PROGRESS.md`。

环境：WSL / Linux，Python 3.12，`uv sync` 安装依赖，`uv run pytest -q` 跑测试。
