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

## 接入一个新的 policy

policy 是任意可调用对象，签名是 `policy(messages, tools) -> assistant message`。`messages` 和返回值都用 OpenAI chat 格式，`tool_calls[].function.arguments` 是 JSON 字符串。episode 循环和判分由环境负责：

```python
from arl.env.task import load_tasks
from arl.env.episode import run_episode

task = load_tasks("data/tasks/heldout.jsonl")[0]
result = run_episode(task, my_policy)          # 默认最多 10 轮，上下文上限 8k token
print(result.reward, result.stop_reason, result.turns)
```

- 每轮调用一次 policy。回复里有工具调用就执行，结果作为 `tool` 消息追加；没有工具调用就视为提交。规则细节写在 `run_episode` 的 docstring 里。
- 服务端报上下文超长时，policy 抛 `arl.env.episode.PolicyContextExceeded`，episode 结束并照常判分。
- OpenAI 兼容的服务（vLLM、付费 API）可以直接用 `arl.rollout.client.OpenAIPolicy(ClientConfig.load("configs/rollout.json"))`。本地测试用 `ScriptedPolicy`。
- 训练框架接入时，可以把它的采样器包成上面这个签名；`EpisodeResult.messages` 就是完整轨迹。

## 启动推理服务（本地，vLLM）

vLLM 装在独立的 venv 里，不进项目依赖。2080 Ti 的算力是 7.5，只能用 fp16，attention 后端是 Triton，需要系统里有 gcc。

```bash
uv venv ~/.venvs/vllm --python 3.12 && VIRTUAL_ENV=~/.venvs/vllm uv pip install vllm
HF_HUB_OFFLINE=1 ~/.venvs/vllm/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.85 --seed 0
```

- Qwen2.5 和 Qwen3 的聊天模板输出 `<tool_call>{json}</tool_call>`，对应的 parser 是 `hermes`。每个模型实际用的 parser 记在 `docs/plan/PROGRESS.md`。
- 客户端配置在 `configs/rollout.json`（`base_url`、模型名、temperature 等）。

然后跑一组题，比如诊断扫描用的 100 题，每题 4 次：

```bash
uv run python -m arl.rollout.calibrate pilot --config configs/rollout.json --name diag-qwen2.5-1.5b
```

每个 episode 判完分就追加到 `runs/calibration/<name>.jsonl`，中断后重跑同一条命令会接着跑；汇总写到 `<name>-summary.json`。

## 付费 API（DeepSeek）

配置从仓库根目录的 `.env` 读取（模板见 `.env.example`，`.env` 不进 git），已经设置的环境变量优先。每次运行都必须给花费上限：

```bash
uv run python -m arl.rollout.calibrate preflight --config configs/deepseek.json --runs runs/deepseek --budget-cny 10
uv run python -m arl.rollout.calibrate pilot --config configs/deepseek.json --runs runs/deepseek --name deepseek --k 2 --budget-cny 10
```

单价写在 `configs/deepseek.json`（元/百万 token）。累计花费写到 `runs/deepseek/cost.json`；达到上限后不再启动新的 episode。
