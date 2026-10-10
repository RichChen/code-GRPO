# agentic-rl

在单卡 4090 上用 GRPO 训练 1.5B–3B 模型做多轮工具调用的代码修复，检验"跨题行为能否靠训练学会"。

当前状态：阶段 1（环境与题集）进行中，计划见 `docs/plan/PHASE-1-ENV-AND-TASKS.zh-CN.md`，进度见 `docs/plan/PROGRESS.md`。

环境：WSL / Linux，Python 3.12，`uv sync` 安装依赖，`uv run pytest -q` 跑测试。

## Agent 工具（`src/arl/env/tools.py`）

只有五个工具：`read_file`、`search_code`、`list_dir`、`edit_file`、`run_tests`。所有路径都相对仓库根目录解析，绝对路径、`..` 逃逸和指向外部的符号链接一律拒绝。

刻意不提供 `write_file`，也不提供任意 shell：动作空间越小，小模型越容易学，钻空子的面也越小。改代码只能用 `edit_file` 做唯一匹配的替换，跑代码只能通过 `run_tests` 在沙箱里跑 pytest。

## 生成题集

原始数据在 `data/raw/`，来源、固定的 commit 和 sha256 见 `data/raw/SOURCES.md`。题集用固定 seed 生成，结果是确定的：

```bash
uv run python -m arl.taskgen.build --raw data/raw --out data/tasks   # 约 20 分钟
sha256sum data/tasks/*.jsonl                                          # 和 manifest.json 里的 files 对比
uv run python -m arl.taskgen.report --out runs/report                 # 50 题脚本化检查和抽样
```

输出 `data/tasks/{train,heldout}.jsonl`（不进 git）和 `data/tasks/manifest.json`（进 git，记录配置、统计、丢弃原因和 sha256）。冻结的版本是 tag `phase1-tasks-v1`：

- train `f0038c46eb4b089daabe68ba7246cf086fe521458a24bc2a82e47b5e7840aa82`
- heldout `d28e0709bb81af8196837efaec77ed3833d952ea25fcc1d457aa87f15d7cc3d4`

每行是一道题（`arl.env.task.Task`），包括：仓库文件、题面、隐藏用例（期望值只在宿主机判分时用）、参考答案和 `meta`（bug 类型、来源函数、划分等）。
