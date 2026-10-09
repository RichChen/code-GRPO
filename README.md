# agentic-rl

在单卡 4090 上用 GRPO 训练 1.5B–3B 模型做多轮工具调用的代码修复，检验"跨题行为能否靠训练学会"。

当前状态：阶段 1（环境与题集）进行中，计划见 `docs/plan/PHASE-1-ENV-AND-TASKS.zh-CN.md`，进度见 `docs/plan/PROGRESS.md`。

环境：WSL / Linux，Python 3.12，`uv sync` 安装依赖，`uv run pytest -q` 跑测试。

## Agent 工具（`src/arl/env/tools.py`）

只有五个工具：`read_file`、`search_code`、`list_dir`、`edit_file`、`run_tests`。所有路径都相对仓库根目录解析，绝对路径、`..` 逃逸和指向外部的符号链接一律拒绝。

刻意不提供 `write_file`，也不提供任意 shell：动作空间越小，小模型越容易学，钻空子的面也越小。改代码只能用 `edit_file` 做唯一匹配的替换，跑代码只能通过 `run_tests` 在沙箱里跑 pytest。
