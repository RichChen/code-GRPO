# 阶段 1：环境与题集

> 执行方：Cursor agent，在 WSL Ubuntu 的 `~/agentic-rl` 里工作。
> 审计方：Claude Code。每到 CHECKPOINT 就停下，写好 `docs/plan/CHECKPOINT-1x.md` 之后等待审计。
> 本阶段只用 CPU，不调用任何付费模型，也不需要 GPU。

## 0. 背景（只读，不要改）

这个仓库是一个练手项目：在单卡 4090 上，用 GRPO 训练 1.5B–3B 的模型做多轮工具调用的代码修复。前置项目 coding-harness 和 coding-agent 的 A5 实验得出的结论是：通过 prompt 注入跨题记忆，没能提升 deepseek-flash 的解题率（−1.8pp，95% 区间 [−8.3, +4.8]）。本项目要换一个问题来问：同样的行为，能不能靠训练学会。

本阶段只做一件事：搭出一个可信的训练和评测环境，包括 agent 能用的工具、隔离沙箱、判分，以及按难度筛选过的题集。阶段 2 和 3 的 SFT、GRPO 都会直接调用这里的接口，所以接口要稳定，并且有测试覆盖。

A5 留下了三条教训，必须当成硬要求：
1. 统计功效主要取决于题量。训练集不少于 500 题，留出集不少于 200 题。
2. 判分不能被钻空子。判分只能用 agent 看不到的隐藏测试；agent 能改的文件不能影响判分结果。
3. 太简单（每次都对）或太难（每次都错）的题不提供学习信号。题集最后要按原模型的解出率筛选，见第 6 步。

## 1. 规则

- 所有代码在 WSL 里写、在 WSL 里跑。Python 3.11 或 3.12，用 `uv` 管理环境（`~/.local/bin/uv`，没有的话就装）。不要用 Windows 的 Python，不要用 `/mnt/c` 或 `/mnt/d` 下的路径。
- 沙箱不用 Docker。将来的服务器（AutoDL）本身就是容器，里面不能再嵌套 Docker。
- 只引入必要的依赖，并锁定版本（`uv.lock`）。允许的依赖：`pytest`、`pydantic`、`pyyaml`、`httpx`、`openai`（客户端，第 6 步用）。其他依赖要先在 PROGRESS 里写明理由。
- coding-harness（`/mnt/d/crq/src/coding-harness`）只读，用来参考。可以照着它的工具语义重新实现，不要 import 它，也不要复制它的运行时、会话存储或预算模块。
- 每一步做完都要：跑全部测试，提交一次 git（提交信息写清这一步），在 `docs/plan/PROGRESS.md` 追加一段，写明做了什么、测试结果、遇到的问题。
- 遇到计划没覆盖的决定，或者某项要求做不到，就停下，写进 PROGRESS，不要自行降低要求。

## 2. 目标目录结构

```text
agentic-rl/
  pyproject.toml  uv.lock  README.md  .gitignore
  src/arl/
    env/        tools.py  sandbox.py  episode.py  grader.py
    taskgen/    sources.py  mutate.py  validate.py  build.py
    rollout/    client.py  calibrate.py
  tests/
  data/         # 生成的题集，.gitignore 掉大文件，只提交 manifest 和样本
  docs/plan/    # 本计划、PROGRESS.md、CHECKPOINT-*.md
```

## 3. 分步计划

### 第 0 步：仓库初始化

- `git init`，写 `pyproject.toml`、`.gitignore`（忽略 `.venv/`、`data/**/workspaces/`、`runs/`）和一个简短的 README（三五行，写项目目标和本阶段状态即可）。
- `uv sync`。写一个占位测试，确认 `uv run pytest -q` 能跑通。
- `PROGRESS.md` 开头记录环境信息：Python 版本、uv 版本、`nproc`、`free -g`、`bwrap --version`。

### 第 1 步：工具层 `env/tools.py`

一共五个工具，名字和参数按 OpenAI function calling 的 JSON schema 写。语义参照 coding-harness 的 `src/tools/files.py`、`search.py`、`patch.py`、`command.py`：

| 工具 | 参数 | 语义 |
| --- | --- | --- |
| `read_file` | `path`, `offset?`, `limit?` | 返回带行号的内容。默认最多返回 200 行，被截断时在末尾说明还剩多少行、下次用多少 offset |
| `search_code` | `pattern`, `path?`, `glob?` | 用 Python 的 `re` 实现，不依赖 ripgrep。返回 `path:line: text`，最多 50 条 |
| `list_dir` | `path?` | 列出一层目录，目录名后加 `/` |
| `edit_file` | `path`, `old_string`, `new_string` | `old_string` 必须恰好出现一次，否则报错并说明出现了几次 |
| `run_tests` | `target?` | 在沙箱里跑可见测试，返回退出码和截断后的输出，最多 4000 字符，保留头和尾 |

要求：
- 所有路径都按工作区根目录解析。绝对路径、`..` 逃逸、指向工作区外的符号链接，一律拒绝。
- 工具不写 `write_file`，也不给任意 shell。原因是动作空间越小，小模型越容易学，钻空子的面也越小。这个决定要写进 README。
- 每个工具都返回 `ToolResult(ok: bool, content: str)`。出错时 `content` 写一句给模型看的错误说明，不抛异常。
- 测试覆盖：每个工具的正常路径和错误路径，以及路径逃逸。

### 第 2 步：沙箱 `env/sandbox.py`

`run_tests` 和判分都通过沙箱执行代码。实现方式：

- 每个 episode 一个临时工作区，放在 `tempfile.mkdtemp` 下面。
- 用 `bwrap`（WSL 里已装在 `/usr/bin/bwrap`）启动子进程：工作区可写，系统目录只读，`--unshare-net` 断网，`--die-with-parent`，环境变量只保留白名单。
- 资源限制：`resource.setrlimit` 限制 CPU 秒数、地址空间和文件大小，外层再加墙钟超时（默认 20 秒）。超时就杀掉整个进程组。
- 不能用 bwrap 的环境要能退化成"只有 rlimit 和超时"，这时在结果里打上 `isolation="rlimit_only"`。在 PROGRESS 里写明 AutoDL 上需要验证 bwrap 能不能用（AutoDL 容器里可能不允许 user namespace，这点只能到服务器上再确认）。
- 测试覆盖：死循环会超时、联网会失败、往工作区外写文件会失败、fork 炸弹会被进程数限制挡住、正常 pytest 能跑通。

### 第 3 步：判分 `env/grader.py`

- 每道题有两组测试：可见测试 `tests/test_visible.py` 放在工作区里，agent 能看能跑；隐藏测试只存在题目数据里，判分时才拷进一个干净的副本。
- 判分流程：拷一份 agent 结束时的工作区，删掉其中所有 `test_*.py`、`conftest.py`、`pytest.ini`、`setup.cfg`、`tox.ini`、`pyproject.toml`、`__pycache__` 和 `.pyc`，再放入原始配置和隐藏测试，在沙箱里跑。
- 奖励只有 0 和 1：隐藏测试全过才是 1。另外在结果里记录 `visible_pass`、`hidden_pass_count`、`hidden_total`、`timed_out`、`files_changed`，供分析用，不进奖励。
- 防作弊测试，每条都要有：
  - 改了可见测试，比如把断言删掉，判分结果不受影响；
  - 写了 `conftest.py` 想劫持 pytest，会被删掉，不生效；
  - 用 `sys.exit(0)` 或 monkeypatch pytest 一类的手段，隐藏测试照样失败；
  - 直接改测试要 import 的目标函数，硬编码返回值，只能骗过可见测试，骗不过隐藏测试（这条要求隐藏测试的输入和可见测试不同）；
  - 什么都不改，奖励是 0。

**审计修订 1**

隐藏测试不再是 pytest 文件，而是 `HiddenCase(call, check)`：`call` 在沙箱里求值，`check` 在宿主机上对解码后的值求值。可见测试仍是 pytest 文件。

### 第 4 步：Episode 循环 `env/episode.py`

- 接口：`run_episode(task, policy, max_turns=8) -> EpisodeResult`。其中 `policy` 是一个可调用对象，输入 messages 和 tools，返回一条 assistant 消息（可以带 tool_calls）。这样阶段 3 可以把训练框架的采样器直接接进来。
- 消息格式用 OpenAI chat 格式。system prompt 放在 `src/arl/env/prompt.md`，不超过 300 词，说明任务、工具，以及"改好就停"。
- 结束条件：模型不再调工具（视为提交），或者用完 `max_turns`。之后统一判分。
- 截断：每条工具结果最多 4000 字符，整个上下文估算不超过 8k token（字符数 ÷ 4）。超了就停止，打上 `truncated=True`，奖励照常判。
- `EpisodeResult` 保存完整的 messages 列表、每步的工具调用、奖励和判分明细，可以序列化成 JSONL，后面训练直接用。
- 用一个脚本化的假 policy（比如按预设顺序读文件、改文件、提交）写端到端测试：一个能把题改对、拿到奖励 1；另一个只改可见测试，拿到 0。

**CHECKPOINT-1A**：第 0–4 步做完后停下，写 `docs/plan/CHECKPOINT-1A.md`。内容包括：各模块的接口签名、测试数量和结果、防作弊测试清单、bwrap 隔离实际验证了哪些项，以及遇到的问题。

### 第 5 步：造题器 `taskgen/`

**来源（`sources.py`）**
- 用 MBPP（sanitized 版）和 HumanEval 的原始 jsonl 文件，从官方仓库直接下载到 `data/raw/`，并在 `data/raw/SOURCES.md` 里记下下载地址、sha256 和许可证。不要用 `datasets` 库。
- 每个来源函数都要能拆出至少 3 条断言：1 条放进可见测试，其余放进隐藏测试。拆不够 3 条的函数丢掉。
- 如果这两个数据集加起来，通过第 5 步筛选的题凑不到 2500 道候选，就停下报告，不要自行换别的数据源。

**组装成小仓库（`build.py`）**
- 每道题是一个小包 `pkg/`，由 3–5 个模块组成，每个模块放 2–4 个来源函数，函数随机抽取（固定 seed）。只往其中一个函数里注入 bug。这样 agent 必须先搜索、阅读，才能找到要改的地方。
- 题面 `issue` 只描述现象，不点明位置，格式类似："`pkg.text.count_vowels('Abc')` 返回 0，期望返回 1。" 题面里可以出现函数名，但不能出现文件名、行号，也不能提示是哪一类 bug。
- 把出 bug 的函数和它所在的文件记下来，作为分析字段，不放进题面。

**注入 bug（`mutate.py`）**，基于 `ast`，至少实现下面 6 种：
1. 算术运算符互换（`+`↔`-`、`*`↔`//`）
2. 比较运算符变化（`<`↔`<=`、`==`↔`!=`）
3. 数字常量差一（`n`→`n±1`），包括 range 的边界
4. 布尔运算互换（`and`↔`or`），或去掉一个 `not`
5. 同一作用域里换错变量名（两个同类型局部变量互换）
6. 删掉一个 `if` 分支，或删掉一个提前 `return`

**验证（`validate.py`）**，每道候选题都要满足下面全部条件，不满足就丢弃，并按原因计数：
- 原代码通过所有可见测试和隐藏测试；
- 改坏后的代码至少有 1 条可见测试失败，并且至少有 1 条隐藏测试失败；
- 改坏后的代码不会超时，也不会在 import 时崩溃（这类太容易发现）；
- 同一个来源函数、同一种 bug 最多保留 2 道题，且注入位置必须不同。MBPP sanitized 加 HumanEval 只有大约 590 个函数，每个 (函数, bug 类型) 只留 1 道题的话，2500 道候选可能凑不够。

**划分**
- 按来源函数划分，同一个来源函数（包括它出现在别的题里当"干扰函数"的情况）不能同时进入训练集和留出集。这条是防泄漏的硬要求，要写测试检查。
- 留出集固定 seed 抽取，候选阶段就先切出来，后面所有筛选对两边用同一套规则。
- 输出 `data/tasks/{train,heldout}.jsonl`，外加一份 `manifest.json`，内容是每类 bug 的数量、丢弃原因统计、来源分布和文件 sha256。

**CHECKPOINT-1B**：生成不少于 2500 道候选题后停下，写 `CHECKPOINT-1B.md`。内容包括：manifest 摘要；随机抽 10 道题，贴出题面和 diff；用第 4 步的脚本化 policy 抽 50 题跑一遍"原样提交"和"只改可见测试"，确认奖励全是 0。

### 第 6 步：本地校准 `rollout/`

前置条件是第 0 步之外的本机准备，由用户自己完成：Windows 的 `.wslconfig` 设 `memory=10GB`、`swap=8GB`（本机共 16GB），NVIDIA 驱动升级到能支持当前 vLLM 的版本，并在 WSL 里确认 `nvidia-smi` 可用。没准备好之前，第 6 步只写代码和单元测试，不跑真实校准。

- `client.py`：一个 OpenAI 兼容客户端，作为 policy 使用。`base_url` 和模型名从配置读，不写死。
- 本地起服务：在单独的 venv（`~/.venvs/vllm`）里装 vLLM，模型用 `Qwen/Qwen2.5-1.5B-Instruct`，参数 `--dtype half --max-model-len 8192`，开启工具调用解析。具体是哪个 parser 和哪些参数，按安装的 vLLM 版本文档来写，写进 PROGRESS。2080 Ti 不支持 bf16，所以只能用 fp16。如果 vLLM 在 2080 Ti 上跑不起来，换 llama.cpp server，也提供 OpenAI 兼容接口，并记录原因。
- `calibrate.py`：对每道候选题跑 K=4 次，temperature=0.8，结果写入 `runs/calibration/*.jsonl`，可以断点续跑。并发数可配置，默认 8。
- 筛选：保留解出率在 [0.25, 0.75] 之间的题，也就是 4 次里对 1 到 3 次。另外统计格式错误率，比如工具参数解析失败、调用了不存在的工具。
- 如果筛选后训练集不到 500 题、留出集不到 200 题，就停下报告，不要放宽区间。到时候由审计方决定：加候选题，或者换 3B 模型做校准。
- 判分要用最新代码再跑一遍，确认结果可复现：每道题拿原始的正确代码判一次，必须全部是 1。

### 第 7 步：收尾

- README 补上：怎么生成题集、怎么跑校准、怎么接入新的 policy。
- 把题集冻结：`data/tasks/` 的 sha256 写进 `manifest.json`，打 tag `phase1-tasks-v1`。
- 写 `CHECKPOINT-1C.md`，内容包括：
  - 校准分布：解出次数为 0、1、2、3、4 的题各有多少；
  - 筛选后的数量，以及各类 bug 的分布；
  - 格式错误率；
  - 每题平均轮数和平均 token 数；
  - 校准总耗时；
  - 推理服务配置；
  - 已知问题；
  - 留给阶段 0 和阶段 2 的事项。

## 4. 不要做的事

- 不要写任何训练代码（SFT、GRPO、LoRA），那是阶段 2 和 3 的事。
- 不要调用 DeepSeek 或其他付费 API。
- 不要用 SWE-bench、Docker，也不要动 coding-harness 和 coding-agent 两个仓库。
- 不要为了凑数量放宽验证条件或筛选区间。
- 不要把奖励改成部分得分。阶段 3 有消融实验时再讨论。
