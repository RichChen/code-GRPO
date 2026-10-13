# 阶段 1 进度记录

## 环境信息（2026-10-09 记录）

| 项 | 值 |
| --- | --- |
| 系统 | WSL2，内核 `6.18.33.2-microsoft-standard-WSL2` |
| Python | 3.12.15（uv 托管；系统自带 `/usr/bin/python3` 是 3.14.4，不满足 3.11/3.12 要求，未使用） |
| uv | 0.12.24（新装到 `~/.local/bin/uv`） |
| `nproc` | 8 |
| `free -g` | Mem 总计 9，可用 8；Swap 8 |
| `bwrap --version` | bubblewrap 0.11.1 |
| git | 2.53.0 |

## 第 0 步：仓库初始化

做了什么：
- `git init`（分支 `main`），仓库级 git 用户设为 `crq <crq@localhost>`（全局没有配置）。
- `pyproject.toml`：包名 `arl`，`src/arl/` 布局，构建后端 `uv_build`，`requires-python = ">=3.11,<3.13"`，`.python-version` 固定 3.12。运行时依赖暂时为空；`pytest` 放在 dev 依赖组里。后续步骤按需加入允许列表里的依赖。
- `.gitignore`：忽略 `.venv/`、`data/**/workspaces/`、`runs/`，另外忽略 `__pycache__/`、`*.pyc`、`.pytest_cache/`。
- README 写了项目目标和本阶段状态。
- 占位测试 `tests/test_smoke.py`。

测试结果：`uv run pytest -q` → 1 passed。

遇到的问题：
- 本机没有 uv，用官方安装脚本装到 `~/.local/bin`。
- 访问 PyPI 很慢（单次请求约 8 秒），第一次 `uv sync` 用了约 5 分钟。没有改用镜像源。

## 第 1 步：工具层 `env/tools.py`

做了什么：
- `ToolResult(ok, content)`（frozen dataclass，只有这两个字段）。`TOOL_SCHEMAS` 是 OpenAI function calling 格式的五个工具，`additionalProperties: false`。
- `Toolbox(root, test_runner=None)`：五个工具方法，外加 `call(name, arguments)` 做分发。`arguments` 可以是 JSON 字符串或 dict，非法输入只返回 `ok=False`，从不抛异常。
- `parse_arguments(name, arguments)`：单独导出，出错时抛 `ToolArgumentError`，`kind` 取 `unknown_tool` / `invalid_json` / `invalid_arguments`。第 4 步的 episode 用它区分格式错误，第 6 步统计格式错误率也用它。整数参数接受纯数字字符串（如 `"10"`），布尔值不算整数。
- 路径：先拒绝绝对路径和 `~`，再对 `root / path` 取 `realpath`，必须落在 root 里面。这样 `..` 逃逸和指向外部的符号链接（目录和文件都算）都被拒绝；指向仓库内部的符号链接允许。`search_code` 遍历目录时不跟随符号链接目录，并逐个文件检查 realpath。
- `read_file`：行号格式 `N| text`，默认最多 200 行，`limit` 也封顶 200；截断时末尾写 `[还剩 N 行; call read_file with offset=K to continue]`（英文）。
- `search_code`：Python `re`，逐行匹配，`path:line: text`，最多 50 条，超出时提示还有多少条；`glob` 同时匹配文件名和相对路径；跳过 `__pycache__`、`.pytest_cache`、`.git`、超过 1MB 的文件和非 UTF-8 文件。
- `list_dir`：一层，目录名加 `/`，隐藏上述缓存目录。
- `edit_file`：`old_string` 必须恰好出现一次，否则报出现次数；不能创建新文件，空 `old_string`、新旧相同都报错。
- `run_tests(target?)`：默认目标 `tests`，支持 `file::node` 形式，拒绝以 `-` 开头的目标（防止注入 pytest 选项）。命令是 `python -B -m pytest -q -p no:cacheprovider --color=no <target>`，输出加 `exit code: N` 头后按头尾保留截断到 4000 字符。测试失败不算工具错误，`ok` 仍为 True。
- test runner 是可注入的 `Callable[[Path, list[str]], RunOutput]`。本步默认用无隔离的 `subprocess_runner`（20 秒墙钟超时），第 2 步换成沙箱。
- README 写明了不提供 `write_file` 和 shell 的原因。

测试结果：`uv run pytest -q` → 47 passed。

遇到的问题：
- 第一次跑时发现 `edit_file` 改完后 `run_tests` 仍然跑旧代码：`a - b` 改成 `a + b` 文件大小不变，又在同一秒内，Python 认为 `__pycache__` 里的 `.pyc` 仍有效。解决办法是 pytest 用 `python -B` 启动，不写字节码。测试里加了"跑完没有 .pyc"的断言。这个问题对 RL 环境很关键，否则模型改对了也看不到效果。

## 第 2 步：沙箱 `env/sandbox.py`

做了什么：
- 接口：`run_in_sandbox(workspace, argv, config=SandboxConfig()) -> SandboxResult`。`SandboxResult` 字段：`exit_code`（超时为 None）、`output`（stdout+stderr 合并）、`timed_out`、`isolation`（`"bwrap"` / `"rlimit_only"`）、`duration_s`、`output_truncated`。另有 `sandbox_runner(config)` 适配成 `Toolbox` 的 test runner，`Toolbox` 默认 runner 已从第 1 步的裸 subprocess 换成沙箱（裸 runner 已删除）。
- `SandboxConfig` 默认值：墙钟 20 秒，CPU 20 秒，地址空间 1024MB，单文件 16MB，进程数 64，`/tmp` tmpfs 64MB，输出最多读 1MB，`isolation="auto"`。
- bwrap 参数：`--die-with-parent --new-session --unshare-user --unshare-pid --unshare-net --unshare-ipc --unshare-uts --unshare-cgroup-try`；`/usr` 只读，`/bin` `/lib` `/lib64` `/sbin` 按宿主机做成符号链接或只读挂载；当前解释器的 venv 和基础安装目录只读挂载（会顺着 `sys.executable` 的符号链接链把中间的链接目录也建出来，uv 的 python 就是 `cpython-3.12-linux-x86_64-gnu -> cpython-3.12.15-...`）；`/proc`、`/dev`、64MB 的 `/tmp`；工作区可写挂载到 `/workspace`；最后 `--remount-ro /` 把 bwrap 的根 tmpfs 也设成只读；`--clearenv` 后只设白名单环境变量。`/etc`、`/home` 下其他内容（包括本仓库的 `src/`、`data/`、`docs/`）在沙箱里都不可见，这样 agent 改的代码在 `run_tests` 里读不到题目数据里的隐藏测试。
- 环境变量：宿主机只透传 `LANG`、`LC_ALL`、`LC_CTYPE`、`TZ`，其余固定为 `PATH`、`HOME=TMPDIR=/tmp`、`PYTHONIOENCODING=utf-8`、`PYTHONDONTWRITEBYTECODE=1`、`PYTHONNOUSERSITE=1`、`PYTHONHASHSEED=0`。
- rlimit（CPU、AS、FSIZE、NPROC、CORE=0）由一个小启动器设置：`python -I -S -c <launcher> <limits> <argv...>`，在沙箱内部 setrlimit（软硬限一起设，非特权进程不能再调高）之后 `execv` 目标命令。没有用 `preexec_fn`，原因见下面"遇到的问题"。
- 墙钟超时：`Popen(start_new_session=True)`，超时或正常结束后都 `killpg(SIGKILL)`。bwrap 模式下杀掉 bwrap 后，`--die-with-parent` 让 PID 命名空间里的 init 退出，命名空间内所有进程随之被内核杀掉。
- 输出写到工作区外的临时文件（受 FSIZE 限制），只读前 1MB。bwrap 模式下输出里的宿主工作区路径替换成 `/workspace`。
- 退化：`isolation="auto"` 时先用 `bwrap_available()` 实测一次（结果缓存），不可用就退化成 `rlimit_only`，并在结果里标出。`isolation="bwrap"` 强制要求 bwrap，不可用时抛 `SandboxUnavailable`。
- 测试：两种模式都测了正常 pytest、死循环超时、超时杀整棵进程树、CPU 限制、内存限制、文件大小限制、fork 炸弹、输出截断、环境变量白名单；仅 bwrap 模式测了断网（连宿主机 127.0.0.1 上的监听端口和 1.1.1.1:53 都失败）、写工作区外失败（宿主 tmp、家目录、`/usr`、`/`、`../`）、看不到本仓库目录、路径归一化成 `/workspace`。另有 `Toolbox` 默认走沙箱和沙箱超时的集成测试。

测试结果：`uv run pytest -q` → 72 passed（本机 bwrap 可用，没有跳过）。沙箱测试连跑两次都是 25 passed，约 13 秒。

遇到的问题：
- `RLIMIT_NPROC` 按真实 uid 统计所有任务（包括线程）。本机 Cursor 自身就有大量线程，在 bwrap 外面用 `preexec_fn` 设 64 会让 bwrap 连命名空间都建不起来（`Creating new namespace failed: Resource temporarily unavailable`）。改成在新 user namespace 里由启动器设置：新内核的 NPROC 计数是按 user namespace 分层的，这样 64 只计沙箱自己的进程。实测 fork 炸弹在 62 个子进程时被挡住。
- bwrap 的根是可写 tmpfs，往没挂载的路径（如 `/home/crq/...`）写会"成功"但只写进沙箱内存。加了 `--remount-ro /` 后这类写入直接报 `Read-only file system`。
- `rlimit_only` 模式的已知限制（只用于 bwrap 不可用的环境）：没有文件系统隔离（能写工作区外、能读本仓库的 `data/`，也就是能读到隐藏测试）、没有网络隔离；NPROC 只能设成"当前用户已有任务数 + 64"，有竞态；进程如果 `setsid` 脱离进程组，超时时杀不到。以 root 运行时 NPROC 不生效，fork 炸弹挡不住。
- **AutoDL 待验证**：AutoDL 实例本身是容器，可能不允许非特权 user namespace，bwrap 能不能用只能到服务器上确认。确认方法：`uv run python -c "from arl.env.sandbox import bwrap_available; print(bwrap_available())"`，再跑 `uv run pytest -q tests/test_sandbox.py` 看 bwrap 相关用例有没有被跳过。如果只能退化成 `rlimit_only`，上面的限制都会影响判分可信度，需要审计方决定怎么处理。

## 第 3 步：判分 `env/grader.py`（另新增 `env/task.py`）

做了什么：
- 新增 `env/task.py`（计划的目录结构里没有，任务数据模型需要被 env、grader、taskgen 共用，所以单独放）：
  - `Task(task_id, issue, files, hidden_setup, hidden_cases, solution, meta)`。`files` 是初始工作区（改坏的代码、`tests/test_visible.py`、原始配置）；`solution` 是覆盖到 `files` 上就得到原始正确代码的文件；`meta` 放 bug 类型、bug 所在文件和函数等分析字段，不进题面。
  - `HiddenCase(call, check)`；`task.materialize(dest, solved=False)`；JSON 序列化、`load_tasks(path)`；构造时检查文件路径必须是仓库内相对路径，且必须有可见测试和至少一条隐藏用例。
- `grade(task, workspace, config=None, check_visible=True) -> GradeResult`，字段：`reward`（0/1）、`hidden_pass_count`、`hidden_total`、`visible_pass`、`timed_out`、`files_changed`、`isolation`、`setup_error`、`case_passed`。不修改 agent 的工作区。
- 判分流程按计划：把 agent 结束时的工作区拷一份（不跟随符号链接），删掉所有 `test_*.py`、`conftest.py`、`pytest.ini`、`setup.cfg`、`tox.ini`、`pyproject.toml`、`__pycache__`、`.pyc`，再写回题目原始的配置文件，放入隐藏测试，在沙箱里跑。隐藏用例全过且没超时，奖励才是 1。
- `visible_pass`：另拷一份干净副本，放回**原始**可见测试，用 pytest 跑 agent 的代码。只做分析，不进奖励。
- `files_changed`：和题目初始文件比，列出新增、修改、删除的路径（忽略 `__pycache__` 和 `.pyc`）。

**设计决定（请审计方确认）：隐藏测试的期望值不进沙箱。** 如果隐藏测试也是 pytest 文件、在沙箱里和 agent 的代码同进程运行，agent 的代码在被 import 时可以 monkeypatch pytest、注册 atexit 伪造报告、或者通过调用栈读到断言里的期望值，进程内无法可靠防住。所以把每条隐藏断言拆成两半：
- `call`：在沙箱里执行的表达式（如 `pkg.nums.add(2, 5)`），由一个小 harness（`python -I -B`，先把 stdout/stderr 重定向到 `/dev/null`，只通过复制出来的 fd 输出）求值，把返回值序列化成带类型标签的纯数据（区分 list/tuple/set/frozenset/dict/bytes/complex；int/float/str/list 等的子类按基类方法取值，自定义 `__eq__` 不会被带出来；其他类型记为 opaque）。抛异常时只记录异常类型名。
- `check`：在宿主机上求值的表达式（如 `result == 7`、`abs(result - 0.5) < 1e-6`、`error == 'ValueError'`），`result` 绑定反序列化后的值。opaque 值和异常标记做任何比较、真值判断、取长度、迭代都会抛错，判为不通过。
- 同一条用例出现多条记录（伪造），该用例判失败；harness 没输出的用例（提前退出、超时）判失败。

这样 agent 的代码在沙箱里能做的最多是"输出一些值"，而不知道期望值就伪造不出正确的值，所以 `sys.exit`、`os._exit`、篡改 pytest、伪造输出、恒等对象都拿不到奖励。判分仍然是"拷干净副本、删测试和配置、放回原始配置和隐藏测试、在沙箱里跑"，只是隐藏测试的格式从 pytest 文件变成了 `call`/`check` 对。代价是第 5 步造题时要把 MBPP/HumanEval 的断言拆成 `call` 和 `check`，拆不出来的断言不能用（计划本来就允许丢掉拆不够 3 条断言的函数）。

防作弊测试（`tests/test_grader.py`，全部通过）：
- 改可见测试（把断言换成 `pass`）：`run_tests` 显示通过，但奖励仍是 0；在此基础上再修对代码，奖励是 1。
- 在根目录或 `tests/` 写劫持用的 `conftest.py`（hookwrapper 把所有报告改成 passed）：`run_tests` 显示通过，判分时被删掉，奖励 0。
- 往 `pytest.ini`、`setup.cfg`、`tox.ini`、`pyproject.toml` 里写 `-p no:python` 一类配置：被删掉并换回原始配置，奖励 0。
- import 时 `sys.exit(0)`、`os._exit(0)`、`raise SystemExit(0)`，以及函数里 `os._exit(0)`：隐藏用例 0 条通过。
- monkeypatch pytest（`_pytest.python.Function.runtest = lambda self: None`）：`run_tests` 显示通过，奖励 0。
- 硬编码返回值（`return 3`）：可见测试通过（`visible_pass=True`），隐藏用例 0 条通过。
- 往 fd 1–19 和 stdout 写伪造的 harness 记录：所有用例因重复记录判失败。
- 返回 `__eq__` 恒为 True 的对象（int 子类和普通类各一种）：`run_tests` 显示通过，奖励 0。
- 在工作区放假的 `_arl_hidden.py` / `_arl_hidden.json` 和 `sitecustomize.py`：被覆盖或不生效，奖励 0。
- 把 bug 文件换成指向工作区外正确答案的符号链接：bwrap 模式下奖励 0。
- 什么都不改：奖励 0。
- 另有：正确修复和参考答案奖励 1、判分不改动工作区、隐藏用例不出现在工作区里、语法错误记 `setup_error`、隐藏用例超时、序列化类型区分（list 和 tuple 不相等、set、dict、nan、异常类型）、opaque 值任何检查都失败、`Task` JSON 往返和非法路径。

测试结果：`uv run pytest -q` → 103 passed（约 22 秒）。

遇到的问题：
- 一开始把 `pkg/conftest.py` 也列为劫持场景，但 pytest 跑 `tests/` 时根本不会加载它，所以测试前提不成立，删掉了这个参数。
- 有一次测试结果是旧代码的：测试进程在编辑器写完文件之前就启动了，重跑即正常，不是代码问题。
- 隐藏用例在同一个进程里顺序执行，没有单条超时；一条死循环会让整次运行超时，`hidden_pass_count` 只统计超时前完成的用例。奖励不受影响（反正是 0）。

## 第 4 步：Episode 循环 `env/episode.py`

做了什么：
- `run_episode(task, policy, max_turns=8, *, sandbox=None, context_max_tokens=8000, workspace_dir=None, keep_workspace=False) -> EpisodeResult`。`policy(messages, tools) -> assistant message`，OpenAI chat 格式；传给 policy 的是消息的浅拷贝，policy 改了也不影响记录。
- system prompt 在 `src/arl/env/prompt.md`（215 词，英文），说明任务、五个工具、工作步骤和"改好就停：不调工具的回复即提交"；还说明了隐藏测试用不同输入、只改 `pkg/`。user 消息是 `Fix this issue:\n\n{issue}`。已确认 `prompt.md` 会被打进 wheel。
- 一次 policy 调用算一轮。结束条件：回复里没有 tool_calls（`stop_reason="submitted"`）；用满 `max_turns`（`"max_turns"`）；每轮调用 policy 之前估算上下文超过 8k token（`"context_limit"`，`truncated=True`）。三种情况都照常判分。用满轮数时如果最终上下文也超过 8k，同样标 `truncated=True`。
- 上下文估算：消息 content、tool call 的函数名和参数、外加工具 schema 的 JSON，总字符数 ÷ 4。工具 schema 也算进去，因为 chat template 会把它放进上下文（约 600 token）。
- 每条工具结果按头尾保留截断到 4000 字符。
- assistant 消息做规范化：只保留 `role`、`content`、`tool_calls`；缺 id 的补 `call_{turn}_{k}`；dict 形式的 arguments 转成 JSON 字符串。tool 消息带 `tool_call_id` 和 `name`。一条回复里可以有多个并行工具调用。
- 每个工具调用记一条 `ToolCallRecord(turn, call_id, name, arguments, ok, error_kind, output_chars, output_truncated)`。`error_kind`：`unknown_tool`、`invalid_json`、`invalid_arguments`、`malformed_call`（缺函数名）算格式错误，`EpisodeResult.format_errors` 统计这四类；`tool_error` 是格式对但执行失败（如文件不存在），不算格式错误。
- `EpisodeResult` 字段：`task_id`、`messages`（完整列表）、`tool_calls`、`reward`、`grade`（`GradeResult` 的 dict）、`turns`、`stop_reason`、`truncated`、`context_tokens_est`、`duration_s`、`meta`（题目的分析字段）。`to_json()` / `from_dict()` 可往返，`write_jsonl(results, path)` 追加写。
- 工作区用 `tempfile.mkdtemp(prefix="arl-ep-")`，结束后删除（`keep_workspace=True` 可保留）。
- `ScriptedPolicy(turns)`：按预设顺序回放工具调用或最终回答，放在 `episode.py` 里，第 5 步的检查点也要用。
- 端到端测试：脚本化 policy 按"列目录、搜索、读文件、改文件、跑测试、提交"修好题目，奖励 1；只改可见测试的 policy 让 `run_tests` 显示通过，但奖励 0；直接提交奖励 0。另测了消息格式、policy 拿到的是拷贝和工具 schema、`max_turns`、一轮多个工具调用、各类格式错误（记录后 episode 继续）、参数规范化、工具输出截断、上下文超限时截断但仍判分（先修好再刷屏读文件，`truncated=True` 且奖励 1）、工作区清理、JSONL 往返、token 估算。

测试结果：`uv run pytest -q` → 119 passed（约 28 秒）。

遇到的问题：
- 编辑器写入的文件偶尔要过一两秒才在 WSL 里可见，立刻跑 pytest 会报文件不存在或跑到旧内容，等一下重跑即可。不是代码问题。

## CHECKPOINT-1A

第 0–4 步完成，`docs/plan/CHECKPOINT-1A.md` 已写好，全量测试 119 passed。按要求停在这里等待审计，没有开始第 5 步。

## CHECKPOINT-1A 审计后的修改

- 按审计结论把"审计修订 1"原样补进计划第 3 步末尾。
- 判分 harness 从 `python -I -B` 改成 `python -S -B`（`grader.harness_argv`）：`-I` 隐含 `-E`，会忽略沙箱里设的 `PYTHONHASHSEED=0`，导致字符串集合、字典的迭代顺序每次运行不同，同一份代码的返回值可能忽对忽错。`-S` 仍然跳过 site（不加载 sitecustomize 和 .pth）。加了测试：连跑 4 次输出相同。

## 网络测速（第 5 步之前）

`curl -o /dev/null -s -w '%{time_total}'` 每个地址 3 次取中位数：

| 地址 | 直连 | 经代理 `http://172.29.64.1:7890` |
| --- | --- | --- |
| `https://pypi.org/simple/pytest/` | 0.42 s | 0.49 s |
| `https://mirrors.aliyun.com/pypi/simple/pytest/` | 0.06 s | 0.35 s |
| `https://pypi.tuna.tsinghua.edu.cn/simple/pytest/` | 0.53 s | 0.48 s |
| `https://raw.githubusercontent.com/` | 2.2 s（301 跳转） | 0.35 s |

- 代理：Windows 上 Clash 开着，端口 7890 是问用户得到的；宿主 IP 用 `ip route show default` 查到 `172.29.64.1`。
- PyPI 直连现在已经不慢（第 0 步时单次约 8 秒），所以**没有配置任何镜像**，`~/.config/uv/uv.toml` 和 `UV_DEFAULT_INDEX` 都没设，`pyproject.toml` 和 `uv.lock` 没改。
- 直连下载 GitHub raw 文件会出现 `Connection reset by peer`，所以数据经代理下载。第 5 步没有新加依赖（下载用 `urllib`）。

## 第 5 步：造题器 `taskgen/`

做了什么：
- `taskgen/sources.py`：下载并校验原始数据（URL 固定到 commit，校验 sha256，来源和许可证见 `data/raw/SOURCES.md`），按补充要求拆断言。
  - MBPP 只接受 `assert f(args) == expected`，HumanEval 只接受 `check()` 函数体里的 `assert candidate(args) == expected`。参数和期望值都必须能 `ast.literal_eval`。其他写法丢弃并按原因计数：`loop`、`helper_definition`、`assignment`、`import`、`other_statement`、`not`、`bool_op`、`not_a_comparison`、`chained_comparison`、`is`、`not_equality`、`reversed_comparison`（`expected == f(x)`）、`call_wrapped_or_other_function`、`star_args`、`args_not_literal`、`expected_not_literal`、`float_inside_container`，以及同一函数里的重复调用 `duplicate_call`。
  - `check` 是 `result == <literal>`；浮点期望值一律用 `math.isclose(result, x, rel_tol=1e-06, abs_tol=1e-09)`。容器里有浮点数的断言没法用一个 `isclose` 表达，丢弃（`float_inside_container`）。
  - 每个函数至少要有 3 条合格断言。来源解里的 import、辅助函数、类和模块级常量一起保留；有其他顶层语句的解丢弃。
  - MBPP 的入口函数取测试里调用次数最多、并且在解里定义的那个；HumanEval 是 `entry_point`。
- `taskgen/mutate.py`：6 类 bug，都是对 AST 的单点修改，用 `ast.walk` 顺序里的节点序号（`position`）加变体名唯一标识，可重放。
  - `arith_swap`：`+↔-`、`*↔//`（含增量赋值）。
  - `compare_swap`：`<↔<=`、`>↔>=`、`==↔!=`。
  - `const_off_by_one`：整数常量 ±1；`range()` 的非常量参数 ±1。
  - `bool_swap`：`and↔or`、去掉 `not`。
  - `var_swap`：把一处变量读取换成另一个同类型局部变量。类型由注解、赋值和测试用例的参数值推断，推不出来或有冲突的变量不参与。替换的变量必须在使用处之前已经绑定，推导式变量和嵌套作用域不参与。
  - `remove_branch`：删掉 `if` 分支（换成 else 部分）、删掉 `else`、删掉非顶层的 `return`。
  - 只改入口函数的函数体，所以不会在 import 时暴露。变体经 `ast.unparse` 后必须能 `compile`，相同代码去重。
- `taskgen/validate.py`：两阶段验证。
  - 筛查：每个来源函数的原始代码和全部变体在沙箱里批量跑，每条用例限时 1 秒。原始代码必须全过；变体必须至少挂 2 条用例（1 条做可见测试，至少 1 条留给隐藏测试），不能超时，不能在 import 时出错。整批卡死或崩溃时，把第一个没有结果的变体判为 timeout/crashed，剩下的重跑。
  - 整题验证：组装好的题用真正的 `grade()` 验证。参考答案必须奖励 1 且可见测试通过；发布版必须可见测试失败、bug 函数至少一条隐藏用例失败、其他函数的隐藏用例全过、奖励 0。另做静态检查：同一函数的隐藏用例里不能有和可见测试相同的 call；题面不能出现文件路径、`pkg.`、行号，也不能出现 operator、swap、off-by-one 一类暴露 bug 类型的词（只检查反引号以外的文字）。
- `taskgen/build.py`：
  - 按来源函数划分 train/heldout（`split-{seed}` 打乱，30% 留出），干扰函数只从同一侧的池子里取，`check_no_leakage` 在写盘前检查，测试也会检查。
  - 每道题是一个小包：3–5 个模块，每个模块 2–4 个函数，模块名从 12 个中性名字里取。bug 函数放在第一个模块（模块内顺序打乱）。顶层名字在全包内唯一，冲突（包括和内置名重名）时按固定规则改成 `name_{k}`，题面、可见测试和隐藏用例一起改；同一模块里同名但不同的 import 冲突时，换一个干扰函数。
  - 可见测试：每个函数一条断言，写成 `from pkg import <modules>` 加 `module.func(args)`，避免名字叫 `test_*` 的来源函数被 pytest 收集。bug 函数的可见用例是第一条失败的用例。
  - 隐藏用例覆盖包里所有函数（各自去掉可见用例），用来发现修 bug 时改坏别的函数。
  - 题面：英文模板，包含调用、实际输出和期望输出（异常时写异常类型），不带模块前缀。
  - 每个（函数，bug 类型）最多 2 道题且位置不同（计划原文要求）。
  - 输出 `data/tasks/{train,heldout}.jsonl` 和 `manifest.json`。构建是确定性的：连续 3 次构建，两个 jsonl 的 sha256 完全相同。
- `taskgen/report.py`：CHECKPOINT 用的报告脚本，见 `CHECKPOINT-1B.md`。
- `.gitignore` 加了 `data/tasks/*.jsonl`（约 27MB，可用固定 seed 重新生成，sha256 记在 manifest）。`data/raw/` 下的原始文件（约 300KB）和 `manifest.json` 提交。

结果：
- **候选题 2146 道（train 1455、heldout 691），少于计划要求的 2500。** 按计划原文"凑不到 2500 道候选，就停下报告，不要自行换别的数据源"，在 CHECKPOINT-1B 停下报告，没有换数据源，也没有放宽筛选或上限。
- 每个（函数，bug 类型）上限为 2 时，筛查后理论最多就是 2146 道，实际产出也是 2146 道，组装和整题验证没有损失（整题验证通过率 100%）。

测试结果：`uv run pytest -q` → 169 passed（约 36 秒）。新增 `tests/test_taskgen.py` 49 个。

遇到的问题：
- 来源函数名叫 `test_duplicate` 时，可见测试 `from pkg.x import test_duplicate` 会让 pytest 把它当成测试收集。改成导入模块、用 `module.func()` 调用。
- MBPP 里有个解定义了 `sum`，和同模块里 HumanEval 函数使用的内置 `sum` 冲突，导致原始代码过不了隐藏用例。改名时把内置名也视为已占用。
- 题面静态检查最初对函数名（`swap_List`）和参数值（`'1/6'`）误报，改成只检查反引号以外的文字。
- `var_swap` 的类型推断对 `c = 0; c = c + 1` 这种自引用推不出类型。改成乐观迭代：迭代时忽略未知，最后一轮要求所有绑定类型一致才算确定。
- HumanEval 的 LICENSE 是在 `d321ec0` 才加进仓库的，固定 commit 从 `463c980` 改成 `d321ec0`，数据文件 sha256 不变。
- `mbpp/123` 的原始代码在给定参数下超过 1 秒（O(n²)，参数 9999），整函数丢弃。

## CHECKPOINT-1B

第 5 步完成，`docs/plan/CHECKPOINT-1B.md` 已写好。候选题 2146 道，少于 2500，按计划停下报告，列出了可选方案，等审计方决定。没有开始第 6 步。

## 第 5b 步（未完成，停下报告）

做了什么：
- 把"审计修订 2"原样补进计划第 5 步末尾，单独提交。
- 第 6 步前置检查：`nvidia-smi` 能跑通（RTX 2080 Ti，11264 MiB，驱动 615.78.05，CUDA 13.4）。
- 下载完整版 MBPP `mbpp.jsonl`（经代理，同一 commit，sha256 已核对，记入 `SOURCES.md`）。
- `sources.py`：sanitized 版收录的题号用 sanitized 的代码和断言，其余题号用完整版；`test_setup_code` 非空的丢弃；`challenge_test_list` 按同样规则拆断言，作为 `kind="challenge"` 只进隐藏用例。`Case` 新增 `kind` 字段（`test` / `challenge` / `generated`），`SourceFunction` 新增 `origin`。来源函数从 540 个增加到 1070 个（sanitized 393、完整版新增 530、HumanEval 147）。
- `validate.py`：抽出 `run_calls`（跑一组调用、返回原始结果），`run_jobs` 改为基于它。筛查规则改成：至少挂 1 条 `test` 用例（当可见测试），并且总共至少挂 2 条。
- 新增 `taskgen/augment.py`（隐藏用例加强，规则按修订 2 第 4 条）和 `taskgen/dedup.py`（近似重复检查，规则按修订 2 第 5 条），都有单元测试。augment 还没接进 `build.py`。

结果：
- 近似重复检查在 seed 0 的划分上，留出集 321 个函数里有 63 个（19.6%）最大 Jaccard 超过 0.8，超过 10% 门槛，**按修订 2 停下报告**。没有移动函数，没有重建题集，没有开始第 6 步。细节和可选方案在 `CHECKPOINT-1C.md`。
- 隐藏用例加强在 60 个函数上试跑：59 个补满 10 条，丢弃的主要原因是原始代码在扰动输入上抛异常（候选的 8%）。

测试结果：`uv run pytest -q` → 181 passed（新增 `tests/test_augment_dedup.py` 12 个）。

遇到的问题：
- 宿主进程的字符串哈希是随机的，`repr(set)` 的元素顺序会变，会让构建不确定。生成用例时改用自己的渲染函数，集合元素排序后输出。
- Python 3.9 以后 `ast.literal_eval("set()")` 是合法的，空集合可以作为期望值；宿主机求值 `check` 时内置函数可用。

## 第 5b 步（完成）

做了什么：
- 把"审计修订 3"原样补进计划第 5 步末尾，单独提交（`1ca78cc`）。
- `dedup.py` 新增 `clusters()`：在全部 1070 个来源函数上两两算 Jaccard，大于 0.8 连边，用并查集求连通分量。
- `build.py` 改为按簇划分：`split_clusters()` 用 `split-{seed}` 打乱簇的顺序，依次放进留出集，直到达到 30%；`check_split()` 划分后再跑一遍 `near_duplicates`，不是 0 或者超过 10% 都会抛 `BuildStopped`；最大簇超过 30 个函数也会抛 `BuildStopped`。
- `augment.py` 新增输入不变量，按修订 3 第 3 条，从 `kind="test"` 用例推断：
  - 整数 ≥0 / >0；
  - 等于某个序列的 `len()`（扰动序列后重新计算，不单独扰动）；
  - 序列非空；
  - 升序 / 降序（要求至少一个种子严格单调）；
  - 两序列等长（要求至少一个种子长度 ≥2）。
  不满足的候选丢弃，按不变量分别计数。
- augment 接进 `build.py`：先丢掉原始代码跑不过的 challenge 用例，每个函数最多补 10 条；bug 函数隐藏用例少于 5 条的题丢弃（`too_few_hidden_cases`）；可见测试只从 `kind="test"` 用例里取。manifest 新增 `clustering`、`augmentation` 两节。
- 新增 `src/arl/rollout/`（`client.py`：OpenAI 兼容客户端；`calibrate.py`：校准、筛选，为第 6 步准备）和 `configs/rollout.json`。`episode.py` 新增 `PolicyContextExceeded`（上下文超长时 episode 结束，记为 `context_limit`，照常判分）和 `replay_workspace()`（重放 edit_file，复原 agent 最终代码）。依赖新增 `openai`，锁文件里的源全部是 pypi.org。

结果：
- 共 4732 道题（train 3232、heldout 1500），整题验证通过率 100%。连续构建两次，sha256 相同：
  - train `f0038c46eb4b089daabe68ba7246cf086fe521458a24bc2a82e47b5e7840aa82`
  - heldout `d28e0709bb81af8196837efaec77ed3833d952ea25fcc1d457aa87f15d7cc3d4`
  - 两份 manifest 除计时字段外完全一致。
- 簇：945 个，大小分布 1:854、2:75、3:7、4:5、5:2、6:1、9:1。最大的簇 9 个函数（逐元素元组运算一族），不超过 30。划分后留出集近似重复为 0。留出集 321 个函数，筛查后可用 320 个；不同 bug 来源函数 train 648、heldout 273。
- 不变量：
  - 命中函数数：int_positive 401、nonempty 673、ascending 105、descending 4、equal_length 47、int_nonnegative 30、len_of 64。
  - 丢弃的候选数（按尝试次数计）：int_positive 14861、ascending 4339、equal_length 4214、nonempty 3038、int_nonnegative 551、descending 93。
- 加强：共生成 10395 条用例。丢弃原因：原始代码抛异常 729、超时 13、期望值过长 27、不能还原为字面量 6、容器里含浮点 4。
- bug 函数的隐藏用例数：
  - 加强前只算 test 用例，3884/4732 道题只有 2 条；
  - 加强后最少 5 条，众数 12 条（3467 道）。
  - 补出来的用例共 45721 条（按题计），其中 33002 条（72.2%）能让 bug 版失败；4654/4732 道题里至少有一条补出的用例失败。
  - 没有题因为 `too_few_hidden_cases` 被丢弃。
- 50 题脚本化检查（`runs/checkpoint-1c-5b/`）：原样提交 0/50、只改可见测试 0/50、应用参考答案 50/50 拿到奖励。
- 构建耗时约 20.5 分钟（加强 74 秒、筛查 146 秒、组装加验证 1005 秒）。

测试结果：`uv run pytest -q` → 198 passed（约 84 秒）。

遇到的问题：
- `openai` 3.27 依赖的是 `httpx2`，不是 `httpx`，测试里改成 `import httpx2`。
- 校准时如果构造 policy 就抛异常，原先不会写记录，改成也记为错误，续跑时重跑。
- 抽查生成的用例时发现，不变量规则覆盖不到"语义约束"。例如 `find_Extra(arr1, arr2, n)` 要求 arr2 只比 arr1 少一个元素，生成的输入少了两个，期望值仍然是原始代码算出的确定结果，但在原题语境下不合理。详见 CHECKPOINT-1C 的抽样表，不再额外处理。

## 第 6 步（试点触发停止条件，停下报告）

做了什么：
- 推理服务：vLLM 0.31.0（torch 2.13.0+cu130），装在独立 venv `~/.venvs/vllm`，不进项目依赖。模型 `Qwen/Qwen2.5-1.5B-Instruct`，HF 版本 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，经代理从 Hugging Face 官方下载。完整命令：

  ```bash
  HF_HUB_OFFLINE=1 ~/.venvs/vllm/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8192 \
    --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.85 --seed 0
  ```

  工具调用 parser 用 `hermes`（Qwen2.5 的模板输出 `<tool_call>{json}</tool_call>`）。attention 后端自动选了 `TRITON_ATTN`，因为算力 7.5 不支持 FlashAttention 2。
- 冒烟测试：2 个 episode 端到端跑通，工具调用被解析成结构化 `tool_calls`，执行和判分都正常。
- 试点：`uv run python -m arl.rollout.calibrate pilot`，train 和 heldout 各 50 题（seed 0，按来源函数分层），K=4，temperature 0.8，并发 8。

结果：
- 解出率 0%（0/400），解出次数在 [0.25, 0.75] 区间里的题 0%，格式错误率（按工具调用）37%。三项都触发 PHASE-1C-PROMPT 规定的停止条件，所以停下报告：没有换模型，没有调参数，没有跑全量校准，也没有做第 7 步。
- 其他指标：平均 1.85 轮；每个 episode 平均生成 220 token，最终上下文 1265 token；停止原因 submitted 398、max_turns 2；吞吐量 5556 episode/小时，估算全量耗时 1.9–3.4 小时。
- 原因分析：
  - 205/400 个 episode 第一轮就回复纯文本、不调工具，按环境规则视为提交；
  - 工具调用的 JSON 经常不合法（比如写 Python 原始字符串 `r"..."`）；
  - `edit_file` 14 次全部失败，400 个 episode 没有一次改动过文件。
  - 链路本身没有问题。详见 `CHECKPOINT-1C.md`。
- 报告写完后关掉了 vLLM 服务。

测试结果：`uv run pytest -q` → 198 passed（约 74 秒，代码和第 5b 步相同）。

遇到的问题：
- vLLM 第一次启动失败：Triton 初始化需要 C 编译器，WSL 里没有 gcc，sudo 又需要密码。用户执行 `sudo apt install -y gcc libc6-dev` 后启动正常。

## CHECKPOINT-1C

`docs/plan/CHECKPOINT-1C.md` 已改写为正式版：第 5b 步完成，第 6 步试点触发停止条件，第 7 步未做。原来的提前停下版移到了附录 A。等审计方决定，没有开始阶段 2。

## 审计修订 4 · 第 1 步（上）：冻结题集

- 把"审计修订 4"原样补进计划第 6 步末尾，单独提交（`5358e63`）。同一个提交里加了 `.env.example`，`.gitignore` 加了 `.env`。提交前按要求跑了 `git grep -n "sk-"`：命中 2 处，都不是密钥，一处是 prompt 原文"用 `git grep -n "sk-"` 检查"，另一处是 `grader.py` 注释里的 "task-validation"。另外用 `.env` 里的真实密钥值（没有打印）搜了全部 git 历史和工作区，0 处命中。之后每次提交前都做这两项检查。
- 题集用 `812d5ce` 构建的版本，没有重建。manifest 新增 `frozen` 字段（tag、构建 commit、两个 jsonl 的 sha256），sha256 和磁盘上的文件核对一致。打 tag `phase1-tasks-v1`。
- README 补上"生成题集"一节。"接入新 policy"和"启动推理服务"两节和环境修改一起提交，免得 tag 上的 README 描述还没提交的代码。

测试结果：`uv run pytest -q` → 210 passed。

## 审计修订 4 · 第 1 步（下）：环境修改 `phase1-env-v1`

做了什么：
- `episode.py`：
  - 提醒一次：模型回复里没有工具调用，并且到这时为止一次 `edit_file` 都没成功过，环境就追加一条固定的 user 消息 `NUDGE_MESSAGE`（原文照修订 4），episode 继续。每个 episode 最多提醒 1 次。第二次没有工具调用视为提交；成功编辑过以后，没有工具调用直接视为提交。
  - "提醒占 1 轮"的实现方式：触发提醒的那条回复照常计入 `turns`，提醒消息本身不另外计数。
  - `EpisodeResult` 新增 `nudged: bool`（默认 False，旧记录照样能读）。`max_turns` 默认值从 8 改成 10（`MAX_TURNS`）。上下文上限 8k 不变。
- `prompt.md`：末尾加了一段 JSON 格式说明，39 个英文词，原有内容不改。parser 不放宽。
- `client.py`：
  - `ClientConfig` 新增 `env_prefix`（从环境变量或 `.env` 读 `{prefix}_BASE_URL/_MODEL/_API_KEY`，环境变量优先；`.env` 用标准库逐行解析）、`extra_body`（Qwen3 的 `chat_template_kwargs`、DeepSeek 的 `thinking`）、`trust_env`、`price_cny_per_mtok`。
  - `Usage` 新增 `cache_hit_tokens`（取 DeepSeek 的 `prompt_cache_hit_tokens`，没有就按 0 算）。不带工具的请求不发 `tools` 和 `tool_choice`。
- `calibrate.py`：
  - 每条记录新增：`nudged`、`nudge_turn`、`first_turn_tool_call`、`tool_call_after_nudge`、`edit_calls`、`edit_ok`、`run_tests_passed`、`cache_hit_tokens`、`cost_cny`；汇总里新增 `diagnostics` 一节。
  - `CostLedger`：累计花费写到 `cost.json`。续跑时从已有记录恢复累计值；"已花费 + 正在跑的 episode 数 × 单个 episode 的最高花费"达到上限时，就不再启动新的 episode。
  - `preflight` 子命令：发 1 条最短请求，验证密钥和模型名。
  - 报错信息先去掉 `Authorization`、`Bearer` 和密钥本身，再写入记录。
- `configs/deepseek.json`：`env_prefix=DEEPSEEK`，`thinking` 关闭，`max_tokens` 1024，单价 0.04 / 2.00 / 8.00 元/百万 token（和 A5 的 `model-price-binding.json` 核对一致）。
- 网络：WSL 里直连 `api.deepseek.com` 是通的（不带密钥请求返回 401，耗时 0.13 秒），所以 `trust_env=false`，不读 `HTTPS_PROXY`。
- README 补上"接入一个新的 policy""启动推理服务""付费 API"三节。

测试结果：`uv run pytest -q` → 210 passed。新增的测试：提醒后继续并修好；第二次没有工具调用时视为提交；提醒计入轮数；编辑成功后不提醒；编辑失败不算编辑过；`nudged` 字段的序列化和旧记录兼容；JSON 说明不超过 60 词；`.env` 解析和环境变量优先；`extra_body` 和缓存命中计价；不带工具的请求；脱敏；花费上限和续跑；诊断指标。

## 审计修订 4 · 第 2 步：本地模型诊断扫描

做了什么：
- 题目是试点那 100 道（seed 0），每题 4 次，temperature 0.8，环境 `phase1-env-v1`。每个模型跑一遍 `scripts/diag_model.sh <model> <name> configs/diag/<name>.json`，步骤如下：
  - 起 vLLM 服务；
  - 记录启动耗时、显存、KV cache 大小和 HF 版本，写入 `runs/diag/<name>-serve.json`；
  - 跑 `calibrate pilot`；
  - 关掉服务；
  - 只在跑完以后删除权重。
- vLLM 命令，所有模型都一样：
  ```
  HF_HUB_OFFLINE=1 ~/.venvs/vllm/bin/vllm serve <model> --dtype half --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.85 --seed 0
  ```
- 下载前用 HF API 确认 5 个模型 ID 都存在。
- parser 全部用 `hermes`：
  - vLLM 文档"Qwen Models"一节写的是 `Qwen/Qwen2.5-*` 用 `--tool-call-parser hermes`；
  - Qwen 官方文档 function_call 一节写的是，Qwen3 的模板内置 Hermes 风格，vLLM 用 `--tool-call-parser hermes`；只有 Qwen3-Coder 用 `qwen3_xml`。
  - 脚本会检查每个模型的聊天模板里是否有 `<tool_call>`，五个都有。
- Qwen3-1.7B 通过 `extra_body.chat_template_kwargs.enable_thinking=false` 关闭思考模式。
- 下载用 `scripts/fetch_models.sh`：
  - 经代理，设置 `HF_HUB_DISABLE_XET=1`；
  - 3 分钟没有写入就重启，最多 10 次。
- 结果：

  | 模型 | 解出率 | 格式错误率 | 平均轮数 | 吞吐量 |
  | --- | --- | --- | --- | --- |
  | Qwen2.5-1.5B | 0% | 37.6% | 3.24 | 3332/小时 |
  | Qwen2.5-Coder-1.5B | 0% | 100% | 4.58 | — |
  | Qwen2.5-3B | 7.2% | 11.4% | 8.62 | 829/小时 |
  | Qwen3-1.7B | 17.7% | 2.1% | 5.37 | 2126/小时 |
  | Qwen3-4B-Instruct-2507 | 显存不够，跳过 | — | — | — |

  - Qwen2.5-Coder-1.5B 的工具调用全部写成 ```json 代码块，从来不用 `<tool_call>`。
  - Qwen3-4B-Instruct-2507：权重 7.64 GiB，KV cache 只剩 0.33 GiB，8k 上下文需要 1.12 GiB。没有改量化方式或显存设置。
  - 完整指标见 CHECKPOINT-1D。

遇到的问题：
- `--gpu-memory-utilization 0.90` 起不来：空闲 9.87 GiB，小于 9.9 GiB。改用 0.85。旧版脚本失败后也删了权重，已改为只在成功后删除。
- `hf` 把大文件放在共享的 `hub/blobs/`，只删模型目录不会释放空间。脚本改为顺着链接删除实际文件。
- xet 下载经代理时多次卡住，改走普通 HTTP，三个模型并行，共约 78 分钟。
- 现在模型缓存已清空，`~/.cache/huggingface` 只剩 2 MB。

## 审计修订 4 · 第 3 步：DeepSeek 参考上限

- 先跑 `calibrate preflight --config configs/deepseek.json --budget-cny 10 --cost-file runs/deepseek/cost.json`：返回 `ok`，花费 0.000024 元。
- 再跑 `calibrate pilot --config configs/deepseek.json --k 2 --runs runs/deepseek --name deepseek --budget-cny 10`：
  - 直连，`trust_env=false`，`thinking` 关闭，`max_tokens` 1024；
  - 累计花费写在 `runs/deepseek/cost.json`。
- 结果：
  - 200 个 episode，没有报错，解出率 89.5%；
  - 解出的 episode 轮数中位数 5、p90 是 6；
  - 格式错误率 0.1%；
  - 总花费 1.2783 元，平均每个 episode 0.0064 元；
  - 估算 1000 条成功轨迹约 7.6 元。
- 密钥检查：`runs/`、`docs/`、`src/`、`configs/`、`scripts/`、`tests/` 里没有密钥，也没有 `Authorization`。

## 审计修订 4 · 第 4 步：判分假阴性

- 新增 `src/arl/rollout/falseneg.py`。命令是 `uv run python -m arl.rollout.falseneg runs/diag/*.jsonl runs/deepseek/deepseek.jsonl --out runs/falseneg`，做三件事：
  - 找出没拿到奖励、隐藏 test 和 challenge 用例全过、只有 generated 用例失败的 episode；
  - 按 seed 每个模型抽 5 个；
  - 重放 `edit_file` 调用，复原模型的 diff，在沙箱里重跑失败的用例，得到实际返回值。
- 额外条件"改过 `pkg/` 下的文件"：因为可见测试用到的那条用例不在隐藏用例里，有 713 道题的 bug 版只挂在生成用例上；模型什么都不改，也满足字面条件。
- 结果（字面条件 / 改过 `pkg/`）：

  | 模型 | 只有生成用例失败 | 其中改过 `pkg/` |
  | --- | --- | --- |
  | Qwen2.5-1.5B | 64 | 0 |
  | Qwen2.5-Coder-1.5B | 64 | 0 |
  | Qwen2.5-3B | 59 | 6 |
  | Qwen3-1.7B | 25 | 12 |
  | DeepSeek | 5 | 5 |

- 人工判断：
  - 小模型的 10 个样例都是模型确实没修对。
  - DeepSeek 有 3 个是假阴性（mbpp/236 两个、mbpp/511 一个），原因是 MBPP 原始代码在合法输入上本身就错；另外 2 个（mbpp/850）是模型修错。

测试结果：`uv run pytest -q` → 211 passed（新增 `test_generated_only_failures`）。

## CHECKPOINT-1D

写好了 `docs/plan/CHECKPOINT-1D.md`。没有写训练代码，没有跑全量校准，没有开始阶段 2。

---

# 阶段 2A 进度记录

计划见 `docs/plan/PHASE-2-SFT.zh-CN.md`。计划文档和启动 prompt 单独提交。

## 第 0 步：判分按用例类型分别记录（不改奖励）

做了什么：
- `GradeResult` 新增两个字段，都只用于分析，默认 `None`：
  - `hidden_by_kind: {kind: [pass, total]}`，`kind` 取自 `task.meta["hidden_case_kinds"]`（`test` / `challenge` / `generated`，只列出这道题里有的类型）；
  - `pass_non_generated`：全部 `test` 和 `challenge` 隐藏用例都过、并且没有超时。
  - 题目没有 `hidden_case_kinds`（或长度对不上）时两个字段都是 `None`。奖励的计算没有改动。
  - 旧 JSONL 里的 `grade` 字典没有这两个键，`GradeResult(**old)` 照样能构造（加了测试）。
- 新增 `arl.rollout.regrade`：按 falseneg 的方法重放成功的 `edit_file` 复原最终工作区，用新判分重新判，逐条比较 `reward`、`hidden_pass_count`、`hidden_total`、`case_passed`、`visible_pass`、`timed_out`、`setup_error`。

结果（`uv run python -m arl.rollout.regrade runs/diag/qwen3-1.7b.jsonl --out runs/regrade`，`runs/diag/qwen3-1.7b*.jsonl` 只匹配这一个文件）：
- 400 个 episode，**奖励不同 0 条**；上面列的其他字段也全部相同。
- 奖励 1 的 71 条（17.75%，和诊断扫描一致）；`pass_non_generated` 为 True 的 96 条，差的 25 条正好是 falseneg 的"字面条件"数。
- 用例类型：400 个 episode 的题都有 `test` 和 `generated`，40 个有 `challenge`。
- 耗时 35 秒（8 线程）。

测试结果：`uv run pytest -q` → 215 passed（新增 4 个：按类型记录且不改奖励、默认为空和旧记录兼容、超时算失败、regrade 能发现不一致）。

提交后打 tag `phase2-grader-v1`。

## 第 1 步：训练集再分成 SFT 池和 RL 池

做了什么：
- 新增 `src/arl/sft/split.py`（`uv run python -m arl.sft.split`）：
  - 簇用 `arl.taskgen.dedup.clusters` 在全部 1070 个来源函数上重新算，和构建题集时一致（945 个簇，和 manifest 相同）；
  - 只取含有训练集 bug 来源函数的簇，按首个成员排序后用 `sftsplit-0` 打乱，依次放进 SFT 池，直到 SFT 池的 bug 来源函数达到训练集的 40%（648 × 0.4 = 259.2），其余归 RL 池；
  - 每道题跟着它的 bug 来源函数走。
- 输出 `data/tasks/train_sft_ids.txt`、`train_rl_ids.txt`（每行一个题号，已排序），sha256 和统计写进 `data/tasks/phase2-split.json`。原来的 jsonl 没动。
- 测试 `tests/test_sft_split.py`：簇不跨池、划分可复现；在真实文件上检查两个池不相交、并集是完整训练集、不含留出集的题、每个簇只落在一个池里、sha256 一致（没有题集文件时跳过）。

结果：

| | SFT 池 | RL 池 |
| --- | --- | --- |
| 题数 | 1279 | 1953 |
| bug 来源函数 | 261（40.3%） | 387 |
| 簇 | 240 | 340 |
| arith_swap / bool_swap / compare_swap / const_off_by_one / remove_branch / var_swap | 255 / 48 / 156 / 335 / 204 / 281 | 367 / 69 / 269 / 487 / 310 / 451 |
| 来源 mbpp / humaneval（按题） | 1046 / 233 | 1600 / 353 |
| 每个函数最多保留 3 条时的理论上限 | **668** | 1027 |

- sha256：`train_sft_ids.txt` `451668fc…a9f1`，`train_rl_ids.txt` `05933ff7…bbf`。
- 干扰函数不受划分约束：RL 池 1953 道题里有 1920 道的干扰函数中含有 SFT 池的 bug 来源函数（干扰函数在构建时就是从整个训练侧取的）。bug 函数本身不重叠。
- **和第 3 步的冲突**：第 3 步要求"同一道题最多 1 条、同一个来源函数最多 3 条"，SFT 池有 261 个函数，就算每道题都解出，筛选后最多也只有 668 条轨迹，低于 CHECKPOINT-2A-1 "少于 800 条就停下"的门槛。见下文。

测试结果：`uv run pytest -q` → 218 passed。

## 第 2 步：用 DeepSeek 采集轨迹（只在 SFT 池上）

做了什么：
- `calibrate.run_calibration` 新增 `with_messages`（默认 False）：记录里带上完整对话。新增 `--ids` 参数（第 5 步用），只跑题号文件里的题。
- 新增 `src/arl/sft/collect.py`：
  - SFT 池按题号排序后用 `sftcollect-0` 打乱，顺序固定；预算中途停下时，已跑的是随机子集。
  - `pilot`：顺序里的前 20 题各跑 1 次；按"花费 ÷ 成功条数 × 1000"估算，超过 12 元就以状态 4 退出。
  - `run`：每题跑 1 次（`k=0`），再给第一次失败的题补跑 1 次（`k=1`）；补跑名单会截断，保证成功轨迹不超过 1500 条。
  - 花费上限用已有的 `CostLedger`（15 元，`runs/deepseek-sft/cost.json`）；结果写进 `runs/deepseek-sft/episodes.jsonl`，可以断点续跑。
- 配置沿用 `configs/deepseek.json`：`.env` 里的 `deepseek-flash`，关闭思考，temperature 0.8，`max_tokens` 1024，环境 `phase1-env-v1`，并发 8。
- 测试 `tests/test_sft_collect.py`：首轮加补跑、记录里带对话、补跑上限。

试点（20 题）：
- 解出 18/20（90%），花费 0.1154 元，每条成功轨迹 0.0064 元，折算 1000 条约 6.41 元，低于 12 元，继续。

**全量开始前停下来问了用户**：第 1 步算出 SFT 池筛选后最多 668 条轨迹，第 3 步"少于 800 条停下"一定会触发。用户选择：照计划采完、构建数据集，到 CHECKPOINT-2A-1 按字面条件停下报告。

全量结果：

| 项 | 值 |
| --- | --- |
| episode | 1443（首轮 1279，补跑 164），报错 0 |
| 首轮解出率 | 87.2%（1115/1279） |
| 补跑解出率 | 34.1%（56/164） |
| 成功轨迹 / 解出的题 | 1171 / 1171（91.6% 的题） |
| 花费 | **9.845 元**（含试点），每个 episode 0.0068 元，每条成功轨迹 0.0084 元；上限 15 元未触发 |
| prompt 缓存命中 | 85.3% |
| 平均轮数 | 5.52；停止原因：提交 1364、跑满轮数 79 |
| 解出的轮数中位数 / p90 | 5 / 7；解出但超过 8 轮的 43 条 |
| 被提醒 | 34 个 episode（2.4%） |
| 有格式错误的 episode | 1 |
| `edit_file` 成功率 | 99.8%（1541 次） |

- 没拿到奖励的 272 个 episode 里，有 78 个（28.7%）隐藏的 test 和 challenge 用例全过，只挂在生成用例上，和 CHECKPOINT-1D 里 DeepSeek 的情况一致（这类里一部分是假阴性）。这些轨迹不进 SFT。
- 单价比 CHECKPOINT-1D 的估算（每条成功轨迹 0.0076 元）高一些，因为补跑的都是难题。

测试结果：`uv run pytest -q` → 229 passed，2 skipped（`tests/train/` 的两个文件在主项目环境里跳过）。

## 第 3 步：构建 SFT 数据集 `src/arl/sft/data.py`

做了什么：
- 运行环境：渲染要用 Qwen3-1.7B 的 tokenizer 和 vLLM 的 hermes parser，所以在 `~/.venvs/vllm` 里跑（`PYTHONPATH=src HF_HUB_OFFLINE=1 ~/.venvs/vllm/bin/python -m arl.sft.data build`），用的 transformers 就是推理时渲染 prompt 的那一份。筛选部分是纯 Python，主项目可以测。
- 为了测试，在 `~/.venvs/vllm` 里加装了 pytest 9.1.1（和主项目同版本），没有动其他包。
- **筛选**，按计划顺序取第一个不满足的原因计数：没解出、格式错误（含 `unparsed_tool_calls`）、被提醒、被截断、轮数超过 8、改了 `pkg/` 以外的文件（或什么都没改）。然后同一道题只留 `k` 最小的一条；同一个来源函数最多 3 条，按 `sftdata-0` 打乱后选取，优先选还没选过的 bug 类型。
- **渲染：每个 assistant 轮单独做一个样本。** 原因：Qwen3 的模板对"历史中、不是最后一条、没有 reasoning"的 assistant 消息不输出 think 块，而 `enable_thinking=False` 的生成前缀以 `<think>\n\n</think>\n\n` 结尾。整条轨迹渲染成一个序列时，除最后一轮外，其余每一轮的 token 都和推理时模型看到的不一样（`tests/train/test_sft_render.py` 里有这个反例）。所以第 t 轮的样本 = 第 t 轮之前全部消息加生成前缀（mask 掉）+ 这一轮的输出 + `<|im_end|>`（算 loss）。空 think 块属于推理时的 prompt，不是模型生成的，因此在 mask 里。
  - 工具调用转成 `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`，参数用 `json.dumps(..., ensure_ascii=False)`；有 content 时，content 和第一个工具调用之间用 `\n` 分隔，多个工具调用之间也用 `\n` 分隔，和模板一致。
  - 传给模板的消息先把 `arguments` 从 JSON 字符串解析成 dict，这一步和 vLLM 渲染前的处理一致；tools 用 `arl.env.tools.TOOL_SCHEMAS`。
- **一致性检查**，每条轨迹的每个 assistant 轮都做，任何一项不过，整条轨迹丢弃：
  - 文本：模板渲染到第 t 轮（含）的结果 == 前缀 + 这一轮的文本 + `<|im_end|>\n`；
  - token：`encode(前缀) + encode(这一轮的文本) + [<|im_end|>]` 和整段文本一起 encode 的结果逐个相同；
  - 解析：把这一轮的文本交给 vLLM 的 `Hermes2ProToolParser.extract_tool_calls`，工具名、参数（解析成 dict 后比较）、content（去掉首尾空白后比较）都要和原轨迹一致。
- **额外做了一项服务端比对**（`arl.sft.data crosscheck`）：用诊断扫描时完全相同的参数起 vLLM（Qwen3-1.7B），对每个样本，把原始 OpenAI 格式的消息、tools 和 `enable_thinking=False` 发给 `/tokenize`，结果和数据里的前缀逐 token 比较。反向检查：去掉 `enable_thinking=False`，或者去掉 tools，都比对不上，说明这项比对是有区分力的。
- 验证集：按来源函数用 `sftval-0` 打乱，取 5%（向上取整，13 个函数）。
- 输出 `data/sft/v1/{train,val}.jsonl`，每行是 `traj_id`、`task_id`、`turn`、`input_ids`、`loss_start`。还有 `stats.json`。`.gitignore` 忽略 `data/sft/**`，只放行 `stats.json`。
- `show` 子命令：随机抽样本，解码后标出 loss 边界，供人工检查。

## CHECKPOINT-2A-1（阶段性结果）

- **花费**：9.845 元（试点 0.1154 元，已包含在内），上限 15 元。
- **成功率**：首轮 87.2%，补跑 34.1%，SFT 池 91.6% 的题有成功轨迹（1171/1279）。
- **筛选**：1443 个 episode → 634 条轨迹（3163 个按轮样本）。

  | 丢弃原因 | 条数 |
  | --- | --- |
  | 没解出 | 272 |
  | 格式错误 | 1 |
  | 被提醒 | 11 |
  | 被截断 | 0 |
  | 轮数超过 8 | 37 |
  | 改了 `pkg/` 以外的文件 | 0 |
  | 同一题重复 | 0（补跑只在首轮失败时发生，所以每道题最多一条成功） |
  | 超过每个来源函数 3 条 | 488 |
  | 一致性检查不过 | 0 |

- **一致性测试**：634 条轨迹、3163 个 assistant 轮，三项检查全部通过，不一致 0%（门槛 2%）。vLLM 服务端 `/tokenize` 比对 3163 个前缀，不一致 0 个。
- **数据集统计**（train / val）：

  | | train | val |
  | --- | --- | --- |
  | 轨迹 | 598 | 36 |
  | 按轮样本 | 2979 | 184 |
  | 来源函数 | 243 | 13 |
  | 样本长度中位数 / p95 / 最大（token） | 1489 / 2332 / 4039 | 1526 / 2424 / 3373 |
  | 整条轨迹最后一轮的长度 中位数 / p95 / 最大 | 1720 / 2584 / 4039 | 1766 / 2612 / 3373 |
  | 总 token / assistant token | 4,527,736 / 149,419 | 282,615 / 10,425 |
  | assistant token 占比 | 3.3% | 3.7% |
  | 超过 8192 的样本 | 0 | 0 |
  | 轮数 4/5/6/7/8 | 190/322/19/41/26 | 5/26/1/4/0 |
  | arith / bool / compare / const / remove_branch / var | 127/15/71/157/101/127 | 8/1/1/12/4/10 |

  每个按轮样本都要重算整个前缀，所以训练的 token 量（每个 epoch 约 453 万）是按整条轨迹算的约 4.2 倍（按最后一轮长度合计约 109 万）。
- **随机抽 3 个样本**（`show --n 3 --seed 0`，system prompt 和过长的工具结果只在展示时缩短）：完整文本见 CHECKPOINT-2A 附录 A。三个样本的 loss 都从生成前缀 `<think>\n\n</think>\n\n` 之后开始，到 `<|im_end|>` 结束；前缀里的历史 assistant 轮都没有 think 块。
- **停止**：筛选后 634 条，少于 800 条，按计划停下报告。这在第 1 步就能确定（SFT 池的理论上限是 668 条），用户确认后照计划采完并构建了数据集。第 4–6 步没有执行，详见 `docs/plan/CHECKPOINT-2A.md`。

测试结果：`uv run pytest -q` → 229 passed，2 skipped；`~/.venvs/vllm` 里跑 `tests/train/test_sft_render.py` 和 `tests/test_sft_data.py` → 9 passed。

## 第 4、5 步：只准备了代码，没有执行

触发停止条件后没有训练，也没有评测。采集期间写好并测过的代码和本报告一起提交：
- `src/arl/sft/train.py`、`configs/sft/preview.json`、`requirements-train.txt`（`~/.venvs/train`：torch 2.13.0、transformers 5.17.0、peft 0.21.2）、`tests/train/test_sft_train.py`（在训练环境里 8 passed，包括 GPU 上 fp16 加 GradScaler 的小模型训练）。
- `data/tasks/dev_ids.txt`（`uv run python -m arl.sft.split dev`，RL 池 100 题，来自 100 个不同函数，seed `dev-0`），没有在上面跑过任何东西。
- `scripts/eval_dev.sh`（启动参数和诊断扫描相同，不删权重），`calibrate --ids`，`src/arl/sft/evaluate.py`（两种口径、配对 bootstrap），`tests/test_sft_evaluate.py`。

## CHECKPOINT-2A（提前停下）

`docs/plan/CHECKPOINT-2A.md` 已写好：第 0–3 步的结论、花费明细（9.845 元）、数据集统计、一致性测试结果、3 个样本，以及需要审计方决定的事（样本门槛和每函数上限的冲突、按轮切样本、fp16 下 inf 的判定口径）。Qwen3-1.7B 权重保留，GPU 空闲。

测试结果：`uv run pytest -q` → 229 passed，2 skipped。

# Phase 2A 第二轮（审计修订 2A-1）

CHECKPOINT-2A-1 审计通过。本轮按修订 2A-1 重建数据集，再做第 4–6 步。

## 修订 2A-1 补进计划

修订原文补在 `PHASE-2-SFT.zh-CN.md` 第 3 步末尾，和本轮 prompt 一起单独提交。

## 重建 SFT 数据集 v2（每个函数最多 5 条）

- `arl.sft.data` 的每函数上限改为参数 `--max-per-function`，默认 5，输出默认 `data/sft/v2/`。同一函数内部的选择规则没变。v1 不动；用 `--max-per-function 3` 重建到临时目录，`train.jsonl`、`val.jsonl` 和 v1 逐字节相同。
- 没有重新采集，用的还是 `runs/deepseek-sft/episodes.jsonl`。
- **筛选结果**：880 条轨迹（门槛 800），来自 256 个函数。弃用原因：没解出 272，格式错误 1，被提醒过 11，超过 8 轮 37，超过每函数上限 242，模板不一致 0。
- **每个函数保留几条**：1 条的 43 个函数，2 条的 48 个，3 条的 30 个，4 条的 24 个，5 条的 111 个。
- **一致性检查**全部重跑：
  - 文本、token 和 hermes 解析三项检查，0 条不一致；
  - vLLM 服务端 `/tokenize` 比对 4410 个前缀，0 处不一致（`runs/sft-data/v2/crosscheck.json`）；
  - 反向对照：抽 20 个已存样本，直接和服务端结果比。请求正确时 20/20 一致；去掉 `enable_thinking=False` 或去掉 tools 后都是 0/20（`runs/sft-data/v2/negative-controls.txt`）。
- **抽 3 个样本**（`show --n 3 --seed 0`，`runs/sft-data/v2/show.md`）：loss 都从 `<think>\n\n</think>\n\n` 之后开始，到 `<|im_end|>` 结束，历史轮没有 think 块。
- **数据集统计**（`data/sft/v2/stats.json`）：

  | | train | val |
  | --- | --- | --- |
  | 轨迹 | 827 | 53 |
  | 按轮样本 | 4145 | 265 |
  | 来源函数 | 243 | 13 |
  | 样本长度 中位数 / p95 / 最大（token） | 1504 / 2325 / 4537 | 1517 / 2064 / 3373 |
  | 整条轨迹最后一轮的长度 中位数 / p95 / 最大 | 1734 / 2587 / 4537 | 1773 / 2556 / 3373 |
  | 总 token / assistant token | 6,342,078 / 212,781 | 402,693 / 14,646 |
  | assistant token 占比 | 3.4% | 3.6% |
  | 超过 8192 的样本 | 0 | 0 |
  | 轮数 4/5/6/7/8 | 237/473/23/58/36 | 9/39/1/4/0 |
  | arith / bool / compare / const / remove_branch / var | 166/26/98/220/133/184 | 12/1/3/17/5/15 |

  每个 epoch 的训练 token 量约 634 万，2 个 epoch 约 1268 万。按最后一轮长度合计，train 只有 152 万，按轮切样本放大了约 4.2 倍。
- **`rl_train_ids`**（`uv run python -m arl.sft.split rl-train`）：RL 池减去 dev 集，共 1853 题，来自 377 个函数，sha256 `65524993…1d9a`。写在 `data/tasks/rl_train_ids.txt`，`phase2-split.json` 里同时记了 `dev_ids` 和 `rl_train_ids` 两项。
  - RL 池原来有 387 个函数，其中 10 个函数只有 1 题，而且都被抽进了 dev 集，所以剩 377 个函数。
  - 重跑 `pools` 和 `dev`，清单和三个 id 文件的 sha256 都不变。
  - 新测试检查：`rl_train_ids` 和 dev 集不相交，等于 RL 池减 dev 集，清单里的题数、函数数和 sha256 都对得上。
- **训练配置**（为第 4 步准备）：数据改为 v2；写明 `adam_betas [0.9, 0.999]`；新增停止条件 `max_skipped_fraction 0.05`，即 GradScaler 跳过的步数超过总步数的 5% 就停止。训练日志新增累计跳过步数，summary 新增跳过比例。新增 GPU 测试：把初始 scale 设成 2^60 强制溢出，第 2 个被跳过的步就会停下。

测试结果：`uv run pytest -q` → 230 passed，2 skipped；`~/.venvs/vllm` 全量 → 234 passed，1 skipped；`~/.venvs/train` 里跑 `tests/train/` → 9 passed，1 skipped（训练环境没有 `openai`，其余测试不在这里跑）。

## 第 4 步：本地 SFT 预览（数据 v2）

- **第一次启动卡在初始验证**：
  - 现象：27 分钟还没算完 40 万 token 的前向。
  - 原因：样本长度不一，缓存分配器产生碎片，保留显存涨过整张卡；WSL 不报 OOM，而是溢出到系统内存，速度慢了上百倍。
  - 处理：训练模块在 import torch 之前设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；配置新增 `cuda_memory_limit_gib` 10.0，超过上限时直接报 OOM。这两项不改模型、精度和超参数。
  - 第一次的日志在 `runs/sft-preview-attempt1/`。
- **第二次正常跑完**：
  - 520 / 520 步，被 GradScaler 跳过 0 步，loss scale 始终是 65536；
  - 吞吐量平均 954 token/秒（第 20 步估算 885 token/秒，预计 3.98 小时）；
  - 显存峰值 9.16 GiB（保留 9.69 GiB）；
  - 纯训练 3.69 小时，墙钟 3.97 小时；
  - val loss 2.306 → 0.4569（第 100 步）→ 0.4213（epoch 1 结束）→ 0.4140（epoch 2 结束）。
- 合并后的 fp16 模型在 `runs/sft-preview/merged/`。tokenizer 文件和 `generation_config.json` 和基座快照逐字节相同。

## 第 5 步：dev 评测（三列）

- 基座和 SFT 模型用 `scripts/eval_dev.sh` 评测；SFT 的客户端配置是 `configs/sft/eval-preview.json`。
- DeepSeek 用 `calibrate run --ids data/tasks/dev_ids.txt --k 4 --config configs/deepseek.json --runs runs/deepseek-dev --budget-cny 5`，花费 2.181 元。
- 三组都是 400 个 episode，报错 0。

| | 基座 | SFT 预览 | DeepSeek |
| --- | --- | --- | --- |
| 严格解出率 | 20.3% | 53.5% | 90.2% |
| 只看 test 和 challenge | 26.0% | 58.0% | 95.8% |
| 解出 1–3 次的题 | 17 | 43 | 9 |
| 被提醒过的 episode | 36.5% | 3.7% | 0.3% |
| 用满 10 轮 | 51 | 100 | 3 |

- 配对差（SFT 减基座）：严格口径 +33.3 个百分点，95% 区间 [+24.8, +41.8]；只看 test 和 challenge 是 +32.0，区间 [+23.3, +40.5]。
- 按 SFT 模型解出 1–3 次的比例（43%）换算，`rl_train_ids` 的 1853 题里约剩 797 题，区间 611–982。
- `arl.sft.evaluate` 新增 `--reference`（第三列，加上"哪些题能解"的统计）和 `--rl-train-ids`（校准估计），并补了测试。

## 第 6 步：CHECKPOINT-2A

`docs/plan/CHECKPOINT-2A.md` 已重写，上一轮的提前停下版挪到附录 B。本阶段付费 API 共花 12.026 元（上限 25 元）。按要求停在这里，没有写 RL 代码，没有跑留出集，没有租服务器。

测试结果：`uv run pytest -q` → 231 passed，2 skipped；`~/.venvs/vllm` 全量 → 235 passed，1 skipped；`~/.venvs/train` 里跑 `tests/train/` → 9 passed，1 skipped。

# Phase 2C：租卡前的本地准备

CHECKPOINT-2A 审计通过。审计修订 2A-2 补在 `PHASE-2-SFT.zh-CN.md` 末尾，替换原阶段 2B，单独提交（`625ba57`）。

## 第 1 步（一）：冻结 SFT-v1

- `runs/sft-preview/adapter-epoch2` 复制到 `runs/sft-v1/adapter/`（fp32 LoRA，不入库）。
- `configs/sft/v1.json` 记录：基座快照 `70d244cc86cc`、适配器三个文件的 sha256、训练数据 `data/sft/v2` 三个文件的 sha256、训练配置路径。
- `tests/test_sft_v1.py` 核对这些 sha256；适配器不在本地时跳过。
- tag `sft-v1`。

测试结果：`uv run pytest -q` → 233 passed，2 skipped。

## 第 1 步（二）：token 级客户端和一致性检查（贪心比对未达标，停下）

### 实现

- `src/arl/rollout/tokenclient.py`：`TokenPolicy`，接口和 `OpenAIPolicy` 相同（`policy(messages, tools) -> assistant message`）。
  - prompt 用 `arl.sft.data.Renderer` 在本地渲染成 token id，调 `/v1/completions`。
  - 每轮记录 `prompt_ids`、`completion_ids`、`completion_logprobs`、`finish_reason`。
  - 输出用自己的 hermes 解析器 `src/arl/rollout/hermes.py` 解析。
  - `verify_tokenize: true` 时，每轮把原始消息加 tools 和 `enable_thinking=False` 发给服务端 `/tokenize`，逐个比对。
- 历史工具调用参数的处理照 vLLM 的 `chat_utils._postprocess_messages`：空串、坏 JSON、非对象一律变成 `{}`（`arl.sft.data.template_arguments`）。
- `calibrate` 按配置的 `endpoint`（`chat` 或 `tokens`）选客户端；新增 `--with-tokens`，在记录里保存每轮 token id。
- 新配置：
  - `configs/sft/v1-eval.json`：评测用，T=0.8，top_k 不设；
  - `configs/grpo/calibrate-v1.json`：校准和 GRPO 用，T=1.0，`top_k: 0`；
  - `configs/check/greedy-{chat,tokens}.json`：贪心比对用。
- `src/arl/rollout/consistency.py`：`ids`、`greedy`、`replay`、`stats` 四个子命令。
- 运行环境：token 客户端要 transformers（本地分词）和 httpx，不要 openai。
  - 它跑在 `~/.venvs/vllm` 里，GRPO 训练端在 `~/.venvs/train` 里也能用。
  - 主 `.venv` 没有 transformers，在主 `.venv` 里跑会报 `ModuleNotFoundError`。
  - 没有新增依赖。

### vLLM 0.31 接口（查源码，并用实际请求核对）

- **`/v1/completions`**（`CompletionRequest`）：
  - `prompt` 可以是 `list[int]`。
  - `return_token_ids: true` 返回 `choices[0].token_ids` 和 `prompt_token_ids`。
  - `logprobs: 0` 返回所选 token 的 `choices[0].logprobs.token_logprobs`。
  - `skip_special_tokens` 默认是 true；我们发 false。
  - `top_k` 不设时用服务端默认值（来自模型的 generation_config）；0 或 -1 表示不截断。
  - logprob 是采样前的原始 logprob（`--logprobs-mode` 默认 `raw_logprobs`）。
- **实测**：
  - `token_ids` 末尾含 `<|im_end|>`（151645），`text` 不含；`token_logprobs` 和 `token_ids` 等长。
  - 超出上下文时返回 400，消息是 "This model's maximum context length is 8192 tokens…"；两个客户端用同一个判定。
  - `/tokenize` 返回 `tokens`、`count`、`max_model_len`、`token_strs`。
  - chat 接口对工具调用轮的 `finish_reason` 是 `tool_calls`，completions 接口一律是 `stop`；calibrate 统计时这个字段两者不可比。
- **默认采样参数**：chat 和 completions 两个接口都从模型 `generation_config` 取默认值（`get_diff_sampling_param`）。Qwen3 的是 temperature 0.6、top_k 20、top_p 0.95。
  - 所以之前所有评测实际都用了 top_k=20；temperature 和 top_p 是请求里显式给的。
  - 决定：
    - 评测配置不设 top_k，和以前可比；
    - 校准和 GRPO 显式发 `top_k: 0`，从模型的完整分布采样（修订 2A-2 第 4 条：T=1.0）。
- **hermes 解析**：`Hermes2ProToolParser.extract_tool_calls`：
  - 正则 `<tool_call>(.*?)</tool_call>|<tool_call>(.*)`（DOTALL），每块 `json.loads`，参数用 `json.dumps(ensure_ascii=False)` 重新序列化。
  - 任何异常都当作没有工具调用，全文作为 content。
  - 有工具调用时，content 取第一个 `<tool_call>` 之前的文本，全是空白就设为 None。
  - 工具解析器的 `adjust_request` 会把 `skip_special_tokens` 设成 false。
  - `tests/test_hermes_vllm.py` 在 vLLM 环境里拿 SFT 数据 v2 的 4145 条助手文本加 20 个边界情况，逐条和 vLLM 的解析结果比对：0 处不同。
- **运行时 LoRA**（第 3 步用）：
  - 环境变量 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`。
  - `POST /v1/load_lora_adapter {lora_name, lora_path, load_inplace}`，`POST /v1/unload_lora_adapter {lora_name}`。
  - 启动参数 `--enable-lora --max-lora-rank`（默认 16）、`--max-loras`（默认 1）、`--max-cpu-loras`。
  - 同一路径原地重载不会让 prefix cache 失效，所以每步换新路径和新名字。
- **sleep mode**（第 3 步用）：
  - 启动参数 `--enable-sleep-mode`，会自动启用 cumem 分配器，平台不支持时直接报错。
  - 接口只在 `VLLM_SERVER_DEV_MODE=1` 时注册：`POST /sleep?level=1|2`、`POST /wake_up?tags=…`、`GET /is_sleeping`。
  - level 1 把权重卸到 CPU，level 2 直接丢掉。
  - WSL 上能不能用还没验证。

### 一致性检查

模型：`runs/sft-preview/merged`（fp16），用诊断扫描的 vLLM 参数（`--dtype half --max-model-len 8192 --gpu-memory-utilization 0.85 --seed 0`）。

| 检查 | 要求 | 结果 | |
| --- | --- | --- | --- |
| 逐轮比对 | `prompt_ids` 等于 `/tokenize` | 2587 + 66 轮，0 处不一致 | 通过 |
| 贪心比对 | 10 题里至少 9 题工具调用序列和奖励完全一致 | **7 / 10**（奖励 10 / 10 一致） | **未通过** |
| 统计比对 | 严格解出率落在 53.5% 的 bootstrap 95% 区间 [45.5%, 61.25%] | 55.25%（400 个 episode，报错 0） | 通过 |

- 统计比对的置信区间：对旧 SFT dev 评测（`runs/sft-eval/dev-sft.jsonl`）的 100 道题的单题解出率做 bootstrap，重采样 10000 次，种子 0。
  - 新客户端：解出 0–4 次的题数是 25 / 15 / 8 / 18 / 34（旧客户端是 25 / 18 / 7 / 18 / 32）；
  - 解出 1–3 次的题 41 道（旧 43）；用满 10 轮的 96 个（旧 100）；被提醒过 4.3%（旧 3.7%）。
- 贪心比对的题目由 `consistency ids` 生成（种子 `greedy-0`，dev 里抽 10 题，存在 `runs/check/greedy_ids.txt`）。两边并发都是 1，K=1。

### 贪心比对的差异分析

- 3 道不一致的题：
  - `train-humaneval-64-arith_swap-14`、`train-mbpp-278-remove_branch-7`、`train-mbpp-481-compare_swap-13`；
  - 两边奖励都是 0，只是工具调用序列在第 5 次或第 8 次调用以后分叉。
- 按渲染后的消息（去掉不进模板的 `tool_call_id`）逐条比：
  - 6 道题第一处不同都是一条 `run_tests` 的工具输出，在这之前的消息完全相同。
  - 不同只有两类：pytest 断言信息里的函数内存地址（`<function vowels_count at 0x7537…>` 对 `0x78b6…>`），以下是这处的 diff；以及耗时（`in 0.03s` 对 `0.04s`）。
    ```
    -E        +  where 0 = <function vowels_count at 0x753791e16f20>('key')
    +E        +  where 0 = <function vowels_count at 0x78b6e69725c0>('key')
    ```
  - 也就是说，只要跑过一次失败的测试，下一轮的 prompt 就不同了，和用哪个客户端无关。上面 3 道题就是 prompt 不同以后，后面的生成分叉了。
- **基线：旧客户端自己跟自己比**，同样的 10 题再跑一遍，也是 **7 / 10**。不一致的 3 道里有 2 道和上面相同，分叉原因也一样（`run_tests` 输出里的地址和耗时）。所以在这个环境下，"至少 9 题一致"连旧客户端自己都达不到。
- **回放比对**（`consistency replay`）：把旧客户端那次运行里每个助手轮之前的完整历史原样交给新客户端做贪心生成，再和旧客户端的回复比。
  - 65 轮里 64 轮完全一致。
  - 不一致的 1 轮在 `humaneval-64` 第 6 条消息，只是解释文字不同。再请求同一个历史时，新客户端和 chat 接口各自得到完全相同的 130 个 token，也和记录一致，复现不出来。归为服务端数值上的非确定性（fp16，同一 prompt 在 prefix cache 命中和未命中时走的计算路径不同）。
- 结论：没有发现渲染或解析上的差异。贪心比对未达标，是因为环境的工具输出本身不确定，不是新客户端的问题。
- 按要求，硬性检查未通过就停在这里，不进入第 2 步。没有改环境：环境不能改，改了 `phase1-env-v1` 就变了。也没有改比对的判定标准。
- 原始结果都在 `runs/check/`：
  - `greedy-report.json`、`greedy-chat-vs-chat.json`、`replay-chat-to-tokens.json`、`stats-report.json`；
  - 运行记录 `greedy-{chat,chat2,tokens}.jsonl`、`dev-sft-tokens.jsonl`。

测试结果：
- `uv run pytest -q` → 257 passed，3 skipped；
- `~/.venvs/vllm` 全量 → 262 passed，1 skipped；
- `~/.venvs/train` 里跑 `tests/train`、`tests/test_tokenclient.py`、`tests/test_hermes.py` → 26 passed，3 skipped。

## 审计修订 2C-1：第 1 步按新判定通过

- 修订 2C-1 原样补在 `PHASE-2C-LOCAL-PREP.zh-CN.md` 第 0 节末尾。
- 贪心比对的硬性判定换成回放比对：
  - 结果是 64 / 65 = 98.5%，不低于 95%；
  - 唯一不一致的一轮重发后复现不出来，属于数值上的非确定性；
  - 没有渲染或解析造成的不一致。
- 所以第 1 步通过。
- 整条轨迹的贪心比对改为参考指标：新客户端对旧客户端 7 / 10，旧客户端对自己 7 / 10。
- 环境保持 `phase1-env-v1` 不变。

## 采样参数配置（修订 2C-1 第 3 条）

- **评测配置** `configs/sft/v1-eval.json`：temperature 0.8、top_p 1.0，显式写上 `top_k: 20`。
  - 效果和以前不设一样：不设时服务端取模型 `generation_config` 里的 top_k，基座快照和 `runs/sft-preview/merged` 的 `generation_config.json` 都是 20。
  - 写死以后，不再受模型目录里 `generation_config` 的影响。
- **校准和 GRPO 配置** `configs/grpo/calibrate-v1.json`：temperature 1.0、top_p 1.0、`top_k: 0`。
- **测试** `tests/test_sampling_configs.py` 核对三件事：
  - 评测配置的四个值；
  - 评测的 top_k 等于快照 `generation_config` 里的值；
  - 校准配置是 (1.0, 1.0, 0)。
- 拒绝启动 GRPO 的检查放在第 3 步的训练代码里。

测试结果：
- `uv run pytest -q` → 260 passed，3 skipped；
- `~/.venvs/vllm` 全量 → 265 passed，1 skipped。

## 第 2 步：试点（修订 2C-1 第 4 条）——触发停止条件，停下

### 做法

- **脚本** `scripts/calibrate_rl.sh pilot|full`：
  - 用诊断扫描的参数起 vLLM，加载 fp16 的 `runs/sft-preview/merged`；
  - 在 `~/.venvs/vllm` 里跑 token 客户端，客户端配置 `configs/grpo/calibrate-v1.json`（T=1.0、top_p 1.0、top_k 0），K=4；
  - 记录写到 `runs/grpo-calib/rl-train.jsonl`。
- **试点** 用 `--ids data/tasks/rl_train_ids.txt --sample 50`，按 bug 来源函数轮流抽，种子 0：50 题、50 个函数，200 个 episode，报错 0，逐轮 `/tokenize` 比对 0 处不一致。
- **全量** 用同一个记录文件续跑，试点的 50 题不重跑。全量还没开始。
- **fp16 和 bf16 的差别**：本地校准用的是 fp16 合并模型，服务器上 GRPO 的起点是 bf16 基座加 SFT-v1 适配器，两者数值上略有不同。服务器上由 `merge_bf16.sh` 在 dev 集上核对。

### 结果（`runs/grpo-calib/pilot-report.json`）

| | 试点：rl_train 50 题，T=1.0，top_k 0 | dev 100 题，T=0.8，top_k 20（新客户端） |
| --- | --- | --- |
| 严格解出率 | 38.5% | 55.25% |
| 只看 test 和 challenge | 44.0% | 59.5% |
| **出现格式错误的 episode** | **18.0%**（36 / 200） | 5.5%（22 / 400） |
| 每次工具调用的格式错误率 | 3.45% | 1.72% |
| 解出 1–3 次的题 | 38%（19 / 50） | 41% |
| 解出 0/1/2/3/4 次的题数 | 21 / 4 / 12 / 3 / 10 | 25 / 15 / 8 / 18 / 34 |
| 平均轮数 | 7.05 | 6.47 |
| 用满 10 轮 | 35% | 24% |
| 被提醒过 | 10% | 4.25% |
| 每个 episode 的输出 token（均值 / p99 / 最大） | 612 / 2781 / 2970 | 508 / 2158 / 3343 |

- **停止条件**：
  - 出现格式错误的 episode 是 18.0%，超过 15%，**触发**；
  - 严格解出率 38.5%，不低于 30%，没触发。
- **按要求停下**：
  - 没有跑全量校准；
  - 没有改温度或 top_k；
  - 没有打 tag `grpo-tasks-v1`；
  - 没有在 GPU 上跑第 3 步的任何东西（包括 sleep mode 探测）。
- **格式错误的构成**：
  - 试点：`invalid_arguments` 26 次、`malformed_call` 17 次、`unknown_tool` 2 次；
  - dev：`invalid_arguments` 23 次、`malformed_call` 17 次。
  - 试点有错的 36 个 episode 里，29 个只错 1 次，5 个错 2 次，2 个错 3 次，其中只有 3 个最后解出。
- **不确定性**：
  - 按题做 bootstrap（重抽 10000 次，种子 0），格式错误 episode 比例的 95% 区间是 [12.5%, 24.0%]。区间跨过了 15%，但点估计在 15% 以上。
  - 50 题里有 25 题至少出过一次格式错误。
- **对比口径**：两组的题集不同（rl_train 抽样对 dev），T 和 top_k 也都不同，所以表里的差别不全是采样参数造成的。

## 审计修订 2C-2：试点按新判定通过

- 修订原文补在 `PHASE-2C-LOCAL-PREP.zh-CN.md` 第 0 节末尾，放在 2C-1 后面。第 4 条同时写进计划第 1 节规则。
- 旧门槛按"至少出现一次格式错误的 episode"算，会随轮数增长，点估计 18% 过了 15% 所以当时停下。新门槛改成每次工具调用的格式错误率，不超过 6%。
- 试点每次调用的格式错误率是 3.45%，低于 6%，**按新判定通过**。出错 episode 的比例 18% 继续报告，不再当门槛。
- 采样参数保持 temperature 1.0、top_p 1.0、top_k 0。接着用同一个记录文件跑全量校准，试点的 50 题不重跑。

## 第 2 步：全量校准（修订 2C-2）

- `scripts/calibrate_rl.sh full` 续跑 `runs/grpo-calib/rl-train.jsonl`，试点的 200 条没有重跑。1853 题 × K=4，7412 个 episode，报错 0，逐轮 tokenize 不一致 0。墙钟约 7.9 小时，吞吐 917 episode/小时。
- 本地仍是 fp16 的 `runs/sft-preview/merged`。服务器上的起点是 bf16 合并，差别留到 `merge_bf16.sh` 的 dev 核对。
- **两条停止线都没踩中**：保留 1018 题（门槛 400）；每次调用的格式错误率 4.27%（门槛 6%）。出错 episode 的比例 17.2%，只作报告。
- 题集：`data/tasks/grpo_v1_ids.txt`、`data/tasks/grpo_v1.json`。1018 题、331 个函数，sha256 `96de604c58692e62be85b1c519ad1947fbc8478d90a5c904cc91eb2436f057d2`。tag `grpo-tasks-v1`。

| | 全量：rl_train 1853 题，T=1.0，top_k 0 |
| --- | --- |
| 严格解出率 | 41.03% |
| 只看 test 和 challenge | 47.95% |
| 每次工具调用的格式错误率 | 4.27%（2114 / 49457） |
| 出现格式错误的 episode | 17.17% |
| 解出 0/1/2/3/4 次的题数 | 559 / 399 / 319 / 300 / 276 |
| 解出 1–3 次 | 1018 / 1853（54.9%） |
| 停止原因 | submitted 4760、max_turns 2651、context_limit 1 |
| 用满 10 轮 | 35.77% |
| 平均轮数 | 7.16 |
| 每个 episode 的输出 token（均值 / p50 / p99 / 最大） | 659.5 / 445 / 2458 / 5136 |

- 格式错误构成：`invalid_arguments` 1248、`malformed_call` 817、`unknown_tool` 49。
- bug 类型（保留的题）：arith_swap 199、bool_swap 35、compare_swap 150、const_off_by_one 275、remove_branch 138、var_swap 221。每个函数 1 到 9 题，1 题的函数 80 个，9 题的函数 1 个。

## 第 3 步的监控（代码已写，GPU 还没跑）

校准占着这张卡。下面这些不占 GPU，测试已过；sleep 探测、过拟合、同步检查等校准结束再做。

- `batch_stats` 增加每次调用的格式错误率，以及出错 episode 的比例。次数仍然报告。
- `loss.token_entropy`：completion 位置上每个 token 一个熵，再取平均。从 detach 后的 logits 算，不进梯度。`train_step` 把均值记成 `completion_entropy_mean`。
- 停止条件在 `grpo.py`：
  - 连续 10 步的每次调用格式错误率均值超过 10% 才停，等于 10% 不停。某一步没有工具调用（比率是空）会打断这 10 步。
  - 平均熵低于**前 5 步**均值的 30% 才停。基线是训练开始的 5 步，不是滑动窗口，这样慢慢坍缩也能被看见。等于 30% 不停。第 6 步起才检查。断点续跑从 `log.jsonl` 把这两列读回来。
- `configs/grpo/v1.json` 的 `stop` 写上了这四个数。过拟合配置继承它们。30 步的过拟合也会检查熵。
- `max_completion_tokens_per_episode` 取全量校准每个 episode 输出 token 的 p99（2458），向上取整到 256 的倍数，是 2560。写在 `configs/grpo/v1.json`。过拟合配置继承这个值。
- **sleep mode 探测**（`runs/grpo-probe/probe.json`，level 2）在 WSL 上能启动：启动 189 秒，醒着 10109 MiB，sleep 0.14 秒后降到 2155 MiB，wake 22.9 秒后回到 10109 MiB。换新适配器名再贪心生成，和睡之前一致。同步检查也过了（贪心一致，logprob 平均绝对差 0.00053）。
- **本地过拟合不能和睡着的 vLLM 同时放**。第一步训练的反向传播要再要 1.77 GiB，当时卡上已经没有空闲显存（训练端已占 8.42 GiB，睡着的 vLLM 还留着大约 2 GiB）。
- **每步重启也放不下**。vLLM 已经退出，训练端单独占 8.88 GiB，反向传播还要 1.53 GiB，卡上只剩 733 MiB，10 GiB 上限同样不够。没有改学习率、算法或显存上限。本地过拟合停在第 1 步之前。训前测量已经跑完：8 题 × K=8，平均奖励 0.375，每次调用的格式错误率 2.58%。正式配置仍是 sleep，留给 4090。

## 第 4 步和第 5 步：脚本和预注册已写，没有上服务器

- `scripts/server/`：`setup.sh`、`selfcheck.sh`、`merge_bf16.sh`、`bench.sh`、`sync.md`。本地只做了 `bash -n`，以及驱动 CUDA 版本比较（13.0 以下拒绝）。没有在服务器上执行。
- `setup.sh` 固定 vLLM 0.31.0 和 `requirements-train.txt` 里的 torch 2.13.0+cu130。驱动 CUDA 低于 13.0 就停下，不换别的 torch。`HF_ENDPOINT` 由环境变量传入，脚本不设镜像。
- `selfcheck.sh` 要求 bwrap 可用，沙箱测试有跳过就失败，然后跑三个环境的测试，并检查 bf16。
- `merge_bf16.sh` 把适配器合并进 bf16 基座（拒绝 `runs/sft-preview/merged`），dev 上用 bfloat16 服务和 token 客户端评测。严格解出率要落在 [0.455, 0.6125]。
- `bench.sh` 先探测 sleep level 1 和 level 2，再试 1 步 colocate。放得下就在同一次运行里续到 3 步；放不下就改用 sleep（两边 sleep 都失败则用 restart）跑 3 步，并按步时估算 200 步。
- AutoDL 的 `source /etc/network_turbo` 写在 README，不写进脚本。
- 留出集预注册：`docs/plan/HELDOUT-PREREG.zh-CN.md`。只读了题号和 `meta`，没有跑模型。

## 审计修订 2C-3

- 修订原文补在 `PHASE-2C-LOCAL-PREP.zh-CN.md` 第 0 节末尾，放在 2C-2 后面。2A-2 原文不改；第 2 条替换其中“合并到 bf16 原始基座”的做法。
- 本轮按修订执行：正式学习率 1e-5、过拟合 5e-5；训练、采样、评测统一 fp16；留出集预注册留修订记录；先测量再以数值等价的方式降低 `train_step` 显存，然后重跑本地过拟合；再查 cu128/cu126 wheel，准备驱动备用环境。然后写 `CHECKPOINT-2C`。

## 修订 2C-3 第 1 条：学习率

- `configs/grpo/v1.json` 的 `lr` 从 1e-6 改为 **1e-5**，常数，没有 warmup。理由：1e-6 是全参数 GRPO 的常用值；LoRA 大约要 10 倍，SFT-v1 用的是 1e-4。AdamW 对 loss 的常数缩放不敏感，`T_norm` 取常数不会把每步更新变大，1e-6 在 30 步里很可能学不动。
- `configs/grpo/overfit-local.json` 覆盖为 **5e-5**。这个测试只看梯度方向和整条链路能不能学，不验证正式超参数；30 步里需要更大的步长，信号才清楚。两个值的理由写在各自配置的 `lr_note` 里。
- 训练端本来就是恒定的 AdamW 学习率，没有调度器，所以没有额外关掉 warmup 的代码。
- 测试：`tests/rl/test_batch.py` 检查正式配置是 1e-5、过拟合覆盖是 5e-5、两边都没有 warmup。`uv run pytest -q` 282 passed、6 skipped；`~/.venvs/train` 的 train/rl/tokenclient/hermes 58 passed、3 skipped；`~/.venvs/vllm` 全量 294 passed、2 skipped。

## 修订 2C-3 第 2 条：合并脚本改为 fp16

- `scripts/server/merge_bf16.sh` 改名为 `merge_fp16.sh`。`arl.sft.merge_fp16` 把原始基座以 fp16 加载，合并 SFT-v1 适配器，写到 `runs/sft-v1/merged-fp16`。仍然拒绝从 `runs/sft-preview/merged` 起步。
- vLLM 用 `--dtype half`。dev 集 K=4、temperature 0.8，严格解出率仍要落在 [0.455, 0.6125]。
- `sync.md` 和 `configs/sft/v1.json` 的说明改成 fp16 合并。本地基线还没跑，下一提交写数字。
- 测试：`bash -n scripts/server/merge_fp16.sh` 通过；`~/.venvs/train` 里 `tests/train/test_merge_fp16.py` 1 passed。三套全量：uv 282 passed、6 skipped（第一次和 vLLM 套件并行时 `test_fork_bomb_is_blocked_by_process_limit[rlimit_only]` 因进程数竞态失败，单独重跑通过）；train 58 passed、3 skipped；vLLM 294 passed、2 skipped。

## 修订 2C-3 第 2 条：本地 fp16 合并的 dev 基线

- 本机跑了 `scripts/server/merge_fp16.sh`。合并结果在 `runs/sft-v1/merged-fp16`（不入库）。vLLM `--dtype half`，dev 100 题 × K=4，token 客户端，temperature 0.8、top_p 1.0、top_k 20。
- **严格解出率 55.5%**，落在 [45.5%, 61.25%] 里，脚本退出码 0。之前 fp16 预览合并模型的 dev 严格解出率是 53.5%。
- 400 个 episode，报错 0。逐轮 tokenize 比对 2577 轮，不一致 0。每次调用的格式错误率 1.80%（42 / 2315）。出错 episode 比例 5.8%。平均轮数 6.44。吞吐 922.6 episode/小时。墙钟约 30.1 分钟（含合并和 vLLM 启动）。
- 没有在留出集上跑。服务器上还要再跑一次对照。

## 修订 2C-3 第 2 条：bf16 转 fp16 再转回

- 检查脚本 `scripts/check_bf16_fp16_roundtrip.py`，逐个张量把基座快照 `70d244cc86cc` 从 bf16 转到 fp16 再转回 bf16。结果在 `runs/sft-v1/bf16-fp16-roundtrip.json`（不入库）。
- 全部 2,031,739,904 个元素都是 bf16。发生变化的有 **266,593** 个（占 0.0131%）。最大绝对误差 **2.98e-8**。
- 这个误差远小于正常权重大小，和“只有绝对值极小的权重在 fp16 里放不下”一致。没有改权重。

## 修订 2C-3 第 3 条：留出集预注册

- `HELDOUT-PREREG.zh-CN.md` 末尾加了“修订记录”（2026-10-12）。被替换的原文引在那一节里，git 历史里也有。
- 三个模型在留出集上都改为 fp16、vLLM `--dtype half`。基座直接以 fp16 加载。SFT-v1 是 `merge_fp16.sh` 的结果。GRPO 选定的检查点先合并成 fp16 再评测，不用运行时 LoRA。
- 已知局限里关于 bf16 起点的那条改成 fp16。另加一条：训练中的 dev 评测仍用运行时 LoRA，只用来选检查点。
- 没有在留出集上跑模型。GRPO 还没开始。

## 修订 2C-3 第 4 条：改代码前的显存测量

过拟合那 8 题的校准记录没有保存 token id，只保存了每轮长度。completion 最长的一轮是 1024 个 token，对应 prompt 2553（`train-mbpp-837-var_swap-41`，`finish_reason` 里有 length）。显存由形状决定，所以用这个长度的固定 token 做了一次前向和反向。脚本 `scripts/measure_train_step_memory.py`，模型按训练步 `model.train()`（梯度检查点只在 train 模式生效）。

第一次没切到 train 模式，检查点没开，前向在已分配 9.25 GiB 时再要 782 MiB，被 10 GiB 上限挡住。那个数不能代表训练步，作废。打开 train 模式之后（`runs/grpo-mem/before.json`）：

| 阶段 | 已分配 GiB |
| --- | --- |
| 模型权重 | 3.335 |
| 前向之后（骨干，峰值 5.699） | 3.767 |
| 整块 fp32 logits（1024×151936，张量本身 0.58 GiB） | 4.347 |
| 熵（图还在） | 5.506 |
| 反向峰值 | 8.165 |
| 反向之后 | 4.616 |

这一条序列的旧路径反向峰值是 8.165 GiB，自己放得进 10 GiB。上次整步 OOM 是批里更长的 prompt 加上整块 logits 的反向，不是这一条单独放不下。

## 修订 2C-3 第 4 条：分块 logp 和熵

- `train_step` 不再把整段 completion 的 fp32 logits 留在图里。骨干只出 hidden state，`lm_head` 按 256 行一块做，每块立刻 `backward`，熵在该块反向之后用 detach 的 logits 算。`completion_logps` 同样分块，给参考模型和同步检查用。上下文长度、`max_turns`、LoRA、`cuda_memory_limit_gib` 都没改，也没有丢弃或截断序列。
- CPU 小模型测试 `test_chunked_step_matches_full_logits_loss_grad_and_entropy`：loss、每个参数的梯度、熵，和整块 logits 的旧实现相差都在 1e-5 以内。分块大小在测试里设成 4，让一条序列跨多块。
- 同一条 2553+1024 的序列，新路径峰值 **7.40 GiB**（旧路径反向峰值 8.165）。两边的熵都是 0.5456380248069763。
- 过拟合配置的 `memory_mode` 改回 `sleep`，先按修订重跑；放不下再改 restart。
- 测试：train 环境 59 passed、3 skipped；uv 282 passed、6 skipped；vLLM 环境 295 passed、2 skipped。

## 修订 2C-3 第 5 条：没有可用的 cu128 / cu126 备用环境

查的是 PyPI 和官方下载页（2026-10-12），没有准备 `~/.venvs/vllm-cu128`，也没有改 `setup.sh` 的安装分支。

- **torch 2.13.0**
  - PyPI `https://pypi.org/pypi/torch/2.13.0/json` 的 Linux x86_64 wheel 是 `torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl`，文件名里没有 cu128 或 cu126。本仓库的 `requirements-train.txt` 用的就是这份，注释写明它是 CUDA 13.0 构建。
  - `https://download.pytorch.org/whl/cu126/torch/` 有 `torch-2.13.0+cu126-cp312-cp312-manylinux_2_28_x86_64.whl`。
  - `https://download.pytorch.org/whl/cu130/torch/` 有 `torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl`。
  - `https://download.pytorch.org/whl/cu128/torch/` 和 `cu124` 的索引里没有 torch 2.13.0。
- **vLLM 0.31.0**
  - PyPI `https://pypi.org/pypi/vllm/0.31.0/json` 只有 `vllm-0.31.0-cp38-abi3-manylinux_2_28_{x86_64,aarch64}.whl` 和源码包。依赖是 `torch==2.13.0` 和 `nvidia-cutlass-dsl[cu13]==4.7.1`，这是 CUDA 13 的构建，不是 cu128 / cu126。
  - GitHub Release `https://github.com/vllm-project/vllm/releases/tag/v0.31.0` 的附件是 cpu、xpu、`vllm-0.31.0+cu129`，以及和 PyPI 同名的那份 wheel。没有 cu128，也没有 cu126。
- cu129 不是修订要求的 cu128 或 cu126，而且驱动只到 CUDA 12.8 时跑不了 cu129，535 / 550 更跑不了。torch 虽有官方 cu126 wheel，vLLM 0.31.0 没有对应的官方 wheel，配不成一套备用环境。
- 因此停止准备备用环境。`scripts/server/sync.md` 写明：租机时必须选驱动 580 以上的镜像。`setup.sh` 仍然是驱动 CUDA 低于 13.0 就停下。

## 第 3 步的代码（未在 GPU 上运行）

试点期间按要求并行写了不占 GPU 的部分，和测试一起提交。

### 文件（`src/arl/rl/`）

- `config.py`：读配置，支持 `base` 加 `overrides`，并做检查。下面任何一条不满足就拒绝启动：
  - 客户端是 token 接口，(temperature, top_p, top_k) 是 (1.0, 1.0, 0)；
  - LoRA dropout 是 0；
  - β 是 0；
  - `memory_mode` 是 sleep、restart、colocate 之一。
- `batch.py`：
  - 优势 `A_i = r_i - mean(r)`，全对或全错的组跳过并计数；
  - 每一轮单独作为一条训练序列，就是 `prompt_ids + completion_ids` 原样拼接，loss 只算 completion 部分；
  - `T_norm = n_tasks · G · max_completion_tokens_per_episode`。
  - 轮与轮不能拼成一条：Qwen3 模板渲染历史助手轮时去掉了空 think 块，下一轮的 prompt 不是上一轮序列的延续。
- `rollout.py`：每题 G 条 episode，种子由 (seed, step, 题号, 序号) 决定。
- `loss.py`：
  - 只为 completion 位置计算 logits；
  - TIS 权重 `min(exp(logp_train - logp_vllm), 2)`，不参与求导；
  - 记录 k3 KL、logprob 差异的均值、p99、最大值，以及被截断的比例；
  - 每批只做 1 次优化器更新。
- `trainer.py`：
  - 在基座上加 SFT-v1 的 LoRA 继续训，另加一个冻结的 `ref` 适配器用来算 KL。
  - sleep 和 restart 模式下，每步训练前把模型建到 GPU 上，训练后释放显存；LoRA 权重、优化器状态、GradScaler 状态留在 CPU。
  - LoRA dropout 设为 0（SFT 时是 0.05），否则训练端的 logp 和采样端对不上。
- `vllm_server.py`：
  - 管理 vLLM 的启停、sleep 和 wake up、运行时 LoRA 加载、贪心生成和 prompt logprob；
  - 带一个 `probe` 子命令，用来探测 sleep mode 能不能用。
- `grpo.py`：训练主循环。每一步依次是：
  1. 采样；
  2. 让出显存给训练端；
  3. 训练；
  4. 保存适配器，用新名字；
  5. 把显存还给 vLLM，热加载新适配器；
  6. 两项同步检查：固定 prompt 的贪心输出一致；本步一条序列上的 logprob 平均绝对差不超过 0.05。

  主循环还负责停止条件、dev 评测、断点续跑，以及过拟合测试训练前后的测量。
- `taskset.py`：由校准记录生成 `grpo_v1_ids.txt` 和 `grpo_v1.json`；保留的题少于 400 道时什么都不写，退出码 4。
- 配置：`configs/grpo/v1.json`，以及过拟合测试用的 `configs/grpo/overfit-local.json`。

### 本地显存调度的计划（还没验证）

- 本机内存 9.7 GiB，vLLM 运行时两个进程加起来约 3.5 GiB。
- sleep level 1 要把 3.4 GiB 权重卸到内存，加上训练端很可能放不下，所以本地默认用 level 2：丢掉权重，wake up 后通过 `POST /collective_rpc {"method": "reload_weights"}` 从磁盘重新加载。
- 这是 vLLM 0.31 源码里 `gpu_worker.sleep / wake_up` 和 dev 路由 `/collective_rpc` 的用法。

### 依赖和测试

- 没有新增依赖：safetensors 已经随 transformers 和 peft 装在 `~/.venvs/train` 里。
- 测试（`tests/rl/`）：
  - 不依赖 torch：优势的三种组、多轮轨迹的 loss mask、训练序列不重新渲染、`T_norm`、配置检查（采样参数不是 (1.0, 1.0, 0) 就拒绝）、任务调度、rollout、题集生成；
  - 依赖 torch（CPU）：completion logp、loss 和 TIS 的数值与梯度、k3、优势为 0 时不更新；
  - 玩具测试：随机初始化的小 Qwen3，平均奖励在 50 步内从约 1/16 升到 0.8 以上；
  - `PolicyTrainer` 用小模型测：两个适配器都正确加载、只训练 `default`、释放后重建能继续训得完全一样、保存的适配器格式和 SFT-v1 相同。
- **还没做**：sleep mode 探测、过拟合测试，以及第 4–6 步。

测试结果：
- `uv run pytest -q` → 277 passed，5 skipped；
- `~/.venvs/vllm` 全量 → 287 passed，2 skipped；
- `~/.venvs/train` 里跑 `tests/train`、`tests/rl`、`tests/test_tokenclient.py`、`tests/test_hermes.py` → 52 passed，3 skipped。

# 阶段 3：A800 上的 GRPO 训练与留出集评测

- 机器：A800-SXM4-40GB，驱动 595.84（CUDA 13.2），16 核，31 GiB 内存。2026-10-12 13:26 开工。
- 所有命令以 `arl` 身份执行（agent 的 shell 是 root，统一用 `su - arl -c` 切换）。git 提交用仓库级身份 `arl (A800 server agent) <arl@ubuntu24>`，只在本地提交，不 push。
- 服务器上没有 tmux，长任务用 `setsid nohup` 放后台，日志落在 `runs/` 下。

## 第 1 步：装环境、自检、核对 SFT-v1

- 预计耗时：`setup.sh` 15–30 分钟，`selfcheck.sh` 约 10 分钟。
- huggingface.co 连不上（10 秒超时），hf-mirror.com 正常，所以 `setup.sh` 用 `HF_ENDPOINT=https://hf-mirror.com` 运行。PyPI 可以直连。
- **超出预期（13:56 记）**：`setup.sh` 开始于 13:27，到 13:56 还在装 vLLM 环境。PyPI 下载约 7 MB/s，训练环境的 wheel 约 4.7 GB。继续等，不换源。
- `setup.sh` 里环境部分全部通过：`train torch 2.13.0+cu130 cuda 13.0`，`vllm 0.31.0 torch 2.13.0+cu130 cuda 13.0`。
- **基座权重下载卡住（14:33 记）**：hf-mirror 把两个 safetensors 302 到 `cas-bridge.xethub.hf.co`，实测约 85 KB/s，读超时反复重试，9 分钟只下了 66 MB。小文件（config、tokenizer 等）已进缓存。停掉了 `hf download`，`setup.sh` 退出码 143。
  - 试过从 ModelScope 分段并行下载（约 4.6 MB/s，`scripts/server/fetch_base_modelscope.sh`，按 hf-mirror 返回的 `X-Linked-ETag` 校验 sha256 后才入缓存）。ModelScope 列出的 sha256 和 HF 快照一致：`model-00001` 为 `169ad53e…30ed5`（3441185608 字节），`model-00002` 为 `912becff…deff9`（622329984 字节）。
  - 14:36 按用户指示停下，等用户手动上传权重。ModelScope 的半截临时文件已删除，没有任何文件进入 HF 缓存的 blobs。
  - **15:00 用户手动上传了权重**到 `~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/`。核对结果：两个 safetensors 的 sha256 等于 HF 快照的 LFS 值（见上）；其余 10 个小文件的 git blob 哈希等于各自的 blob 文件名；`refs/main` 指向 `70d244cc86cc`；`HF_HUB_OFFLINE=1 hf download --revision 70d244cc…` 12/12 个文件都解析到；index 列出的 311 个张量都在，全是 bf16。半截的 `*.incomplete` 已不在。`fetch_base_modelscope.sh` 没有用上，留在仓库里备查。
- **`selfcheck.sh` 第一次失败（15:01–15:05）**：bwrap 和前三组测试都过了，`~/.venvs/train` 里 `test_sft_train.py` 两项 GPU 测试失败。原因是 triton 运行时用 gcc 编译 `cuda_utils.c` 时找不到 `Python.h`：`uv venv --python 3.12` 用的是系统 `/usr/bin/python3.12`，机器上没装 `python3.12-dev`。正式训练的 GradScaler 也走这条路径，必须修。日志在 `runs/phase3/selfcheck-attempt1.log`。
  - 处理：用 root 执行 `apt-get install python3.12-dev`（3.12.3-1ubuntu0.17），只装系统头文件，torch、vLLM 和两个 venv 都没动。
- **`selfcheck.sh` 重跑通过（15:06–15:10，3 分 49 秒）**，日志 `runs/phase3/selfcheck.log`：
  - `bwrap ok`；`tests/test_sandbox.py` **25 passed、0 skipped**，13 项 bwrap 检查全部执行；
  - `uv run pytest -q` 282 passed、6 skipped；
  - `~/.venvs/vllm` 全量 295 passed、2 skipped；
  - `~/.venvs/train` 的 `tests/train tests/rl` 42 passed、1 skipped；
  - GPU：A800-SXM4-40GB，39.5 GiB，bf16 可用。
  - 跳过的原因逐条看过，都是“该环境没装 torch / peft / transformers / vllm”，对应测试在另一个环境里跑过。
- **耗时**：`setup.sh` 的环境部分 13:27–14:23（约 56 分钟，主要是 PyPI 下载）；基座下载卡住加等待上传 14:23–15:00；自检两次共约 8 分钟。第 1 步墙钟合计约 1 小时 45 分钟。
- **磁盘**：根分区已用 42G、可用 145G。`~/.venvs/train` 4.7G，`~/.venvs/vllm` 3.4G，`~/.cache/huggingface` 3.8G。
- **SFT-v1 核对通过**（`scripts/server/check_sft_v1.py`，结果 `runs/phase3/sft-v1-check.json`）：三个文件的 sha256 和 `configs/sft/v1.json` 一致，`adapter_model.safetensors` 69782384 字节；392 个张量全是 fp32，共 17,432,576 个元素，没有 NaN 或 inf，绝对值最大 0.0391。

## 第 2 步：合并 fp16 并核对 dev 基线

- 预计耗时 15–30 分钟。实际：15:14 开始，第一次 vLLM 没起来；修好后 15:25–15:30 跑完，合并加评测 5 分 17 秒。
- **第一次 vLLM 启动失败（15:14–15:15）**：预热时 FlashInfer 要 JIT 编译采样 kernel，调用 `ninja` 时报 `FileNotFoundError: 'ninja'`。`ninja` 1.13.2 装在 `~/.venvs/vllm/bin`，但脚本直接调用 `~/.venvs/vllm/bin/vllm`，没有把这个目录放进 PATH。这次一个 episode 都没跑，评测没开始。日志 `runs/sft-eval/dev-sft-fp16-serve-attempt1-no-ninja.log`。
  - 先单独验证：PATH 里加上 venv 的 bin 后，FlashInfer 的 `sampling` 和 `topk` 两个 JIT 模块编译成功（约 4.5 分钟，缓存在 `~/.cache/flashinfer`，nvcc 用的是 venv 里的 `nvidia-cuda-nvcc` 13.4.92），top_k 20 采样能出结果。
  - 改动（只改启动环境，不改采样参数）：`scripts/server/merge_fp16.sh` export `~/.venvs/vllm/bin` 到 PATH；`VLLMServer.env()` 把 vLLM 所在的 bin 目录放到 PATH 最前面，GRPO 的 vLLM 子进程也要用到。
- **第二次通过**：`runs/sft-v1/merged-fp16`，vLLM `--dtype half`，KV cache 276,688 token。dev 100 题 × K=4，temperature 0.8、top_p 1.0、top_k 20，token 客户端，脚本退出码 0。

| | 服务器 fp16 合并 | 本地 fp16 合并（2C-3） | 本地 fp16 预览合并（2A） |
| --- | --- | --- | --- |
| **严格解出率** | **55.5%**（区间 [45.5%, 61.25%]，通过） | 55.5% | 53.5% |
| 非 generated 口径（只看 test 和 challenge） | 59.8% | — | 58.0% |
| 每次工具调用的格式错误率 | 1.72%（40 / 2320） | 1.80% | 1.9% |
| 出错 episode 比例（只报告） | 6.2% | 5.8% | 5.8% |
| 平均轮数 | 6.45 | 6.44 | 6.53 |
| 停止原因：提交 / 用满轮数 | 308 / 92 | — | 300 / 100 |
| 每题解出 0/1/2/3/4 次 | 22 / 17 / 9 / 21 / 31 | — | 25 / 18 / 7 / 18 / 32 |
| 吞吐 | 5603 episode/小时 | 923 | 1214 |

- 400 个 episode，报错 0；逐轮 `/tokenize` 比对 2581 轮，不一致 0。格式错误：`invalid_arguments` 23、`malformed_call` 17。生成结束原因：stop 2542、length 39。
- 计划里要对比的两个数：本地 53.5% 是 fp16 预览合并的严格口径，55.25% 是 2C 第 1 步新客户端在 dev 上的严格口径。服务器 55.5% 落在这两个数附近，和本地 fp16 合并的 55.5% 相同。
- 按 bug 类型的严格解出率：arith_swap 69.7%、bool_swap 25.0%、compare_swap 65.0%、const_off_by_one 32.4%、remove_branch 47.7%、var_swap 68.5%。完整报告 `runs/phase3/step2-dev-report.json`。
- 吞吐是本地的约 6 倍（单 episode 平均 4.43 秒，并发 8）。按这个速度，留出集每个模型 6000 个 episode 大约要 1.1 小时。
- 测试：uv 282 passed、6 skipped；vLLM 环境 295 passed、2 skipped；train 环境 tests/train tests/rl 42 passed、1 skipped。

## 第 3 步：GRPO 过拟合测试

- 预计耗时：探测约 15 分钟，过拟合 30 步 1–1.5 小时。
- **超出预期（15:48 记）**：过拟合第 1 步用了 239 秒，其中采样 39 秒、训练 197 秒。每步约 85 万个序列 token（每一轮单独一条序列，带完整 prompt），30 步约 2 小时。正式配置每步 128 个 episode，粗算单步约 8 分钟、200 步约 26 小时，可能超过 20 小时门槛。按计划到第 4 步测速后再判断，现在不改任何训练设置。

### 显存模式探测（`runs/server-bench/`）

| 探测 | 结果 |
| --- | --- |
| sleep level 1（`bench.sh` 同款命令，vLLM 0.85） | 通过。启动 117 秒，醒着 34465 MiB，sleep 5.93 秒后 1597 MiB，wake 1.86 秒，换新适配器名和旧名的贪心输出都和睡前一致 |
| sleep level 2 | 通过。启动 60 秒，醒着 34991 MiB，sleep 0.06 秒后 1587 MiB，wake（含 `reload_weights`）6.09 秒，贪心一致 |
| colocate（新脚本 `scripts/server/probe_colocate.py`） | 通过。训练端先建好（3.33 GiB），vLLM 以 0.45 启动，KV cache 120,080 token；在一条 8192 token（prompt 7168 + completion 1024）的合成序列上做一次完整训练步，训练端峰值 **6.03 GiB allocated / 6.33 GiB reserved**，整卡峰值 25,288 MiB；训练步之后 vLLM 贪心输出不变 |

- `bench.sh` 自带的 colocate 判定是跑 1 个正式 GRPO 步，且用的是改之前的 0.85 和 10 GiB，会必然失败。所以 colocate 单独用探测脚本测最坏情况的单条序列（训练端一次只处理一条序列，这就是一步里最大的激活占用）。
- **定下 colocate**，改 `configs/grpo/v1.json`（过拟合配置继承）：
  - `cuda_memory_limit_gib` 10.0 → **16.0**：约为最坏情况实测峰值（6.33 GiB reserved）的 2.5 倍，给多条序列之间的碎片留余量。10.0 是本地 11 GB 卡的值。
  - `vllm.gpu_memory_utilization` 0.85 → **0.45**：vLLM 约 18 GB，KV cache 12 万 token，16 路并发每条平均 7500 token 也放得下；加训练端上限 16 GiB（约 17.2 GB），合计约 35 GB，在 40 GB 以内。
  - `vllm.memory_mode` sleep → **colocate**，过拟合配置里的覆盖也从 sleep 改为 colocate。
  - 依据写在 `v1.json` 的 `memory_note`。学习率、`n_tasks`、`group_size`、温度、top_k、LoRA 都没动。

### 过拟合测试：在第 13 步触发熵坍缩停止条件，停下报告

- `configs/grpo/overfit-local.json`：8 题（seed 0 抽样）、`n_tasks` 8、`group_size` 8、lr 5e-5、T=1.0、top_k 0、colocate。15:41:27 开始，16:27:01 结束，退出码 4。
- **训练前测量**：8 题 × K=8，平均奖励 **0.3281**（本地测的是 0.375），每次调用格式错误率 2.53%。
- **停止原因**：第 13 步 `completion entropy 0.2165 < 0.3 of the first 5 steps' mean 0.7245`。前 5 步均值 0.72449，30% 是 0.21735，第 13 步 0.21654，低了 0.0008。
- 由于停止条件在第 13 步触发，主循环直接退出，**训练后的 K=8 测量没有跑**，所以“训练后比训练前高 0.15”这条通过标准没有正式结果。按计划停下，不调学习率、不关停止条件、不重跑，第 4 步没开始。

| 步 | 奖励 | 非全对全错组比例 | 熵 | 每次调用格式错误率 | logprob 差异均值 | p99 | KL(k3) | 梯度范数 | 同步 MAD | 贪心一致 | 采样 s | 训练 s | 步时 s | 显存峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.391 | 0.88 | 0.866 | 0.063 | 0.0019 | 0.017 | 0.0000 | 0.012 | 0.0004 | 是 | 39 | 197 | 239 | 5.01 |
| 2 | 0.484 | 0.88 | 0.695 | 0.022 | 0.0017 | 0.017 | 0.0006 | 0.013 | 0.0008 | 是 | 37 | 201 | 242 | 5.37 |
| 3 | 0.406 | 0.62 | 0.654 | 0.048 | 0.0018 | 0.017 | 0.0015 | 0.009 | 0.0006 | 是 | 28 | 125 | 156 | 5.14 |
| 4 | 0.500 | 1.00 | 0.764 | 0.046 | 0.0018 | 0.017 | 0.0026 | 0.013 | 0.0005 | 是 | 33 | 223 | 259 | 5.32 |
| 5 | 0.609 | 0.88 | 0.643 | 0.029 | 0.0017 | 0.017 | 0.0039 | 0.013 | 0.0004 | 是 | 31 | 181 | 215 | 5.02 |
| 6 | 0.484 | 0.88 | 0.643 | 0.021 | 0.0017 | 0.017 | 0.0050 | 0.012 | 0.0008 | 是 | 33 | 183 | 219 | 5.09 |
| 7 | 0.562 | 0.88 | 0.697 | 0.022 | 0.0017 | 0.017 | 0.0081 | 0.013 | 0.0006 | 是 | 28 | 164 | 196 | 4.90 |
| 8 | 0.609 | 0.75 | 0.648 | 0.028 | 0.0016 | 0.017 | 0.0077 | 0.011 | 0.0007 | 是 | 25 | 162 | 190 | 5.17 |
| 9 | 0.625 | 0.62 | 0.386 | 0.025 | 0.0012 | 0.016 | 0.0090 | 0.010 | 0.0005 | 是 | 28 | 134 | 165 | 4.96 |
| 10 | 0.547 | 0.88 | 0.422 | 0.040 | 0.0013 | 0.016 | 0.0124 | 0.010 | 0.0002 | 是 | 24 | 182 | 210 | 4.81 |
| 11 | 0.625 | 0.62 | 0.390 | 0.026 | 0.0012 | 0.016 | 0.0116 | 0.008 | 0.0005 | 是 | 26 | 139 | 168 | 5.03 |
| 12 | 0.562 | 1.00 | 0.407 | 0.031 | 0.0011 | 0.015 | 0.0152 | 0.009 | 0.0001 | 是 | 24 | 201 | 228 | 4.90 |
| 13 | 0.750 | 0.50 | **0.217** | 0.018 | 0.0008 | 0.015 | 0.0189 | 0.006 | 0.0007 | 是 | 19 | 113 | 135 | 4.72 |

- **其他停止条件都没触发**：
  - 没有 NaN，GradScaler 0 步跳过，scale 一直 65536；
  - logprob 差异均值每步都在 0.0008–0.0019，13 步平均 0.0015，远低于 0.05；每步的 p99 平均 0.0163；单 token 最大差异 0.391（第 7 步）；TIS 截断比例每步都是 0；
  - 每步同步后贪心输出都和训练端一致，同步 logprob MAD 最大 0.00082；
  - 奖励是上升的（见下）。
- **奖励**：训练 rollout 用的就是这 8 道题、每题 8 条、T=1.0，和测量的协议相同，只是种子不同。第 1–3 步平均 0.427，最后 5 步（9–13）平均 **0.622**，比训练前测量的 0.328 高 0.29。这不是正式的训练后测量，只能作为参考证据。
- **熵**：前 5 步均值 0.724，第 9 步起降到 0.39–0.42，第 13 步 0.217。KL(k3) 对 SFT-v1 从 0 涨到 0.019。只有 8 道题、lr 5e-5（正式值的 5 倍），30 步本来就是为了过拟合，熵快速下降符合预期，但按计划这条停止条件照样生效。
- **格式错误**：每次调用的格式错误率从第 1 步的 6.3% 降到 1.8–3%；出错 episode 比例在 6–16% 之间波动，第 13 步 10.9%。
- **分段耗时（13 步平均）**：采样 28.8 秒，sleep/wake 0（colocate），训练 169.7 秒，保存适配器加训练端参考计算 2.8 秒，同步（加载加两项检查）0.5 秒，步时 201.8 秒。训练占 84%。
- **显存峰值**：训练端 5.37 GiB allocated / 5.51 GiB reserved；整卡 25,372 MiB（nvidia-smi 每 5 秒采样）。
- 最后一个适配器 `runs/grpo-overfit/adapters/step-13` 和 `trainer.pt` 都留着，没有删。

### 第 4 步的时间估算（先记下，等审计方）

- 过拟合每步 64 个 episode，训练 170 秒。正式每步 128 个 episode，训练量约翻倍，粗估训练约 340 秒、采样约 50 秒，单步约 6.5 分钟，200 步约 22 小时，另加 9 次 dev 评测（每次约 5 分钟）。很可能超过 20 小时门槛，最终以 `bench.sh` 3 步实测为准。
- 训练端慢，不是因为显存不够（峰值才 5.4 GiB），而是逐条序列做前向、梯度检查点要重算一遍前向、每条序列还要给 KL 日志再算一次 ref 前向。数值上等价或只影响日志的提速办法（关掉梯度检查点；KL 只抽一部分序列，即 `kl.max_sequences`）都没有做，需要审计方决定。

### 这段时间准备的、不占 GPU 的东西

- **第 0 步 dev 基线**：`grpo.py` 在训练开始前，用运行时 LoRA 加载 SFT-v1，在 dev 上评一次（K=4，T=0.8，top_k 20，和检查点评测同一口径），记为 `eval` 第 0 步。计划第 4 步有一条“第 100 步 dev 严格解出率没超过第 0 步就停”，原代码没有第 0 步评测。
- **第 100 步检查写进代码**：`dev_gain_stop`，配置 `stop.dev_gain_check_step: 100`。第 100 步的 dev 严格解出率不高于第 0 步就写 stop 记录退出（相等也停）。测试 `test_dev_gain_stop_compares_step_100_with_step_0`。
- **第 5 步工具**（没有读 `heldout.jsonl`，用 dev 记录自测）：
  - `scripts/server/eval_heldout.sh <模型> <base|sft|grpo>`：vLLM 参数和 `merge_fp16.sh` 相同，客户端是 `configs/sft/v1-eval.json` 换模型名，脚本里断言 (tokens, 0.8, 1.0, 20)。同名记录已存在就拒绝运行，只有中断后用 `--resume` 续跑同一次运行。并发默认 16。
  - `src/arl/rl/heldout.py`：三模型报告（严格、test+challenge、按 bug 类型、按来源、每次调用格式错误率、出错 episode 比例、平均轮数），以及按 `meta["bug_source"]` 聚类的配对 bootstrap（10000 次，seed 0）。测试 `tests/test_heldout_stats.py`。
- `scripts/server/grpo_report.py`：把 `log.jsonl` 汇总成每步表格和摘要。
- 测试：uv 286 passed、6 skipped；vLLM 环境 299 passed、2 skipped；train 环境 tests/train tests/rl 43 passed、1 skipped。

### 停在这里，需要审计方决定

过拟合测试是第 4 步的前置条件，这次没有正式通过，所以停下。GPU 已空闲（14 MiB），没有任何进程占卡。可选的处理方向（我不自行选择）：

1. 认定“链路能学”已经有证据（奖励 0.33 → 最后 5 步 0.62，其余检查全过），只补一次 `step-13` 适配器在这 8 题上的 K=8 测量（约 3 分钟），看是否比 0.328 高 0.15 以上；
2. 过拟合测试不检查熵条件（熵条件只留给正式训练），按原配置从头重跑 30 步（约 1.7 小时）；
3. 其他。

另外，第 4 步大概率会超过 20 小时，提速与否也请一并决定。

## 阶段 3 · 审计修订 3-1（2026-10-12）

修订原文已经逐字追加到 `PHASE-3-SERVER.zh-CN.md`，单独提交（2b059b4）。

### A：step-13 适配器的训练后测量 —— 通过

- 新脚本 `scripts/server/measure_adapter.py`：只起 vLLM，热加载 `runs/grpo-overfit/adapters/step-13`，调用 `grpo.measure`（和训练前那次是同一个函数、同样的题和种子），K=8、T=1.0，tag `after-step13`，记录追加到 `runs/grpo-overfit/log.jsonl`。耗时 28.6 秒。
- **平均奖励 0.7031**，标准 ≥ 0.478，通过。训练前是 0.3281，高了 0.375。
- 8 道题都提高了：0.375→0.75、0.125→0.375、0.5→0.875、0.125→0.875、0.5→0.875、0.375→0.5、0.0→0.375、0.625→1.0。
- 每次调用的格式错误率从 2.53% 变为 4.05%。

### B：关掉熵停止条件，从第 13 步续跑到第 30 步 —— 两条停止线都没触发

- `runs/grpo-overfit` 整个复制到 `runs/grpo-overfit-cont`，原目录没动。配置 `configs/grpo/overfit-cont.json` 继承 `overfit-local.json`，只改 `out_dir`、`stop.entropy_collapse_ratio: 0` 和测量 tag。
- resume 没有校验配置和保存状态是否一致，所以没有拒绝启动，也就不需要白名单。续跑读的是复制过来的 `state.json`，它指向原目录的 `step-13` 适配器，只读不删（`drop_adapter` 只删本次 `out_dir/adapters` 下的）。
- 18 步用时约 41 分钟（17:51–18:33）。
- **停止线 1**：第 26–30 步 rollout 奖励均值 **0.909**，第 9–13 步是 0.622，没有变差，反而高了 0.29。
- **停止线 2**：`after-step30` = **0.9531**，`after-step13` = 0.7031，高了 0.25。8 道题里 6 道 1.0，另外两道 0.75、0.875。每次调用的格式错误率 1.32%。
- 结论：熵在 0.2–0.35 之间停留了 17 步，奖励一直在涨，没有出现熵低之后奖励变差的情况。
- 进程退出码是 1，但训练和测量都完成了：`after-step30` 写进日志之后，汇总 `reward_gain` 时找 `before` 抛了 KeyError（续跑的这次没有做训练前测量）。已经修好：续跑时从日志里读之前的 `before` 记录（`logged_measure`，有测试）。所以 B 的日志里没有 `end` 记录，日志里最后那条 `end` 是从原目录复制过来的第 13 步停止记录。

逐步数据（第 13 步是原运行的最后一步，作为对照）：

| 步 | 奖励 | 熵 | 有好有坏的组 | 每次调用格式错误率 | KL(k3) | KL 序列数 | logprob 差均值 | 步时 s | 训练 s |
|---|---|---|---|---|---|---|---|---|---|
| 13 | 0.750 | 0.217 | 0.50 | 0.018 | 0.0189 | 241 | 0.00080 | 135 | 113 |
| 14 | 0.609 | 0.292 | 0.50 | 0.050 | 0.0219 | 239 | 0.00092 | 143 | 112 |
| 15 | 0.641 | 0.414 | 0.75 | 0.025 | 0.0229 | 363 | 0.00121 | 200 | 171 |
| 16 | 0.766 | 0.247 | 0.88 | 0.042 | 0.0234 | 423 | 0.00084 | 219 | 192 |
| 17 | 0.781 | 0.203 | 0.50 | 0.035 | 0.0264 | 265 | 0.00078 | 146 | 120 |
| 18 | 0.672 | 0.179 | 0.75 | 0.046 | 0.0272 | 392 | 0.00066 | 204 | 176 |
| 19 | 0.766 | 0.275 | 0.62 | 0.030 | 0.0287 | 327 | 0.00092 | 178 | 150 |
| 20 | 0.766 | 0.283 | 0.75 | 0.036 | 0.0320 | 334 | 0.00087 | 177 | 152 |
| 21 | 0.875 | 0.190 | 0.38 | 0.025 | 0.0272 | 184 | 0.00074 | 111 | 83 |
| 22 | 0.891 | 0.202 | 0.50 | 0.014 | 0.0393 | 195 | 0.00082 | 108 | 86 |
| 23 | 0.859 | 0.274 | 0.38 | 0.011 | 0.0389 | 172 | 0.00093 | 104 | 77 |
| 24 | 0.938 | 0.340 | 0.38 | 0.008 | 0.0321 | 159 | 0.00097 | 97 | 73 |
| 25 | 0.875 | 0.147 | 0.25 | 0.021 | 0.0231 | 114 | 0.00066 | 80 | 53 |
| 26 | 0.891 | 0.228 | 0.62 | 0.003 | 0.0357 | 254 | 0.00085 | 230 | 113 |
| 27 | 0.906 | 0.330 | 0.50 | 0.018 | 0.0324 | 197 | 0.00111 | 113 | 90 |
| 28 | 0.906 | 0.252 | 0.50 | 0.020 | 0.0344 | 202 | 0.00094 | 115 | 91 |
| 29 | 0.938 | 0.200 | 0.25 | 0.006 | 0.0273 | 110 | 0.00082 | 70 | 49 |
| 30 | 0.906 | 0.250 | 0.50 | 0.003 | 0.0383 | 218 | 0.00093 | 122 | 98 |

- 分段均值：第 14–18 步奖励 0.694、有好有坏的组 0.68；第 19–25 步 0.853、0.46；第 26–30 步 0.909、0.48。熵三段都在 0.24–0.27。KL 从 0.024 涨到 0.034。
- 其他检查都正常：GradScaler 0 步跳过，scale 一直 65536；logprob 差均值 0.0009（每步最大 0.0012），单 token 最大差 0.123；TIS 截断 0；每步同步后贪心输出都一致，同步 logprob MAD 最大 0.00134；训练端显存峰值 5.20 GiB allocated / 5.35 GiB reserved。
- 第 14–30 步平均步时 142 秒（采样 27.8 秒，训练 111 秒），比第 1–13 步的 202 秒短，因为题目越解越快，轮数和序列数都变少了。这次续跑的 KL 还是对全部序列计算（进程启动时还没有修订第 4 条的代码）。

### 修订第 3、4、5 条的落实（提交 c9984b9）

- **第 3 条 熵停止**：新函数 `collapse_stop`。熵低于前 5 步均值的 30%，并且最近 10 步 `non_degenerate_fraction`（有好有坏的组的比例）的均值低于 0.25，才停。不足 10 步，或者窗口里有一步没有这个值，不停。
  - 配置：`stop.mixed_group_fraction_min: 0.25`、`stop.mixed_group_window: 10`。
  - `overfit-local.json` 显式设 `mixed_group_fraction_min: null`，所以两份过拟合配置还是只看熵（`overfit-cont` 继承它，并且熵条件本身关掉了）。
  - resume 时 `logged_monitors` 也从日志恢复这个比例。
  - 测试：熵低但组内有差异不停；组内无差异但熵正常不停；两者都满足才停；另外覆盖了窗口不满、窗口里有空值、0.25 正好等于门槛这几种边界。
- **第 4 条 关闭梯度检查点：没有做，保持开着。**
  - 用 `probe_colocate.py` 在同样的最坏序列上测（8192 token，completion 1024，colocate，vLLM 0.45，训练端上限临时设为 19 GiB），结果 **OOM**：PyTorch 已分配 18.79 GiB，还要再分配 96 MiB。所以峰值至少约 18.9 GiB，超过 18 GiB 门槛。
  - 记录在 `runs/server-bench/colocate-probe-nogc/probe.json`。按修订跳过这一项，`gradient_checkpointing: true` 和 `cuda_memory_limit_gib: 16` 不变。
  - 因此没有做开关检查点的等价性检查，也没有省时数据。
- **第 4 条 KL 随机抽样**：`kl.max_sequences: 64`。
  - `batch.kl_sample(n, k, seed, step)` 用 `random.Random(f"kl-{seed}-{step}")` 从这一步全部序列里抽 64 条，排序后传给 `train_step`。序列不足 64 条时全用。
  - `train_step` 改为按索引选序列，每条保留下来的 logp 和它自己的序列配对去算参考 logp（原来是 `zip(kept, seqs)`，只在取前 n 条时才对得上）。
  - 日志里的 `kl_sequences` 是这一步实际参与 KL 的序列数。
  - 测试：同一个 `(seed, step)` 抽到的相同；换 step 或 seed 会变；抽到的不是前 64 条；`train_step` 只对选中的序列调用参考模型，并且每条和自己的参考配对（同一份权重下 KL 为 0）。
  - β=0，这项只改日志，不影响梯度。省下的是每步对其余序列的参考前向：过拟合每步 110–420 条序列，正式训练每步更多。
- **第 5 条 dev K=8**：`eval.k: 8`，第 0 步基线也用这个配置。`HELDOUT-PREREG.zh-CN.md` 的修订记录里补了“2026-10-12 审计修订 3-1”一条，写明在正式 GRPO 开始前提交。留出集仍是 K=4。
- 其他：
  - `measure` 的训练后 tag 可配置（`measure.after_tag`）。
  - `bench.sh` 加了 `SKIP_PROBES=1`，复用第 3 步已经跑过的两个 sleep 探测，只跑 3 步测速。
- 测试：uv 292 passed、6 skipped；vLLM 环境 306 passed、2 skipped；train 环境 tests/train tests/rl 50 passed、1 skipped。

### 配置冻结

- `configs/grpo/v1.json` sha256 `3566ba54767d7f8a1964d785639fb4b8d4b43d2dd91989f27861b0a526f7261c`，git 提交 c9984b9。正式训练期间不改这份配置。

## 阶段 3 · 第 4 步：正式 GRPO

### 3 步测速（bench.sh，SKIP_PROBES=1）

- 18:39–19:00，colocate 模式，配置就是冻结的 `v1.json`（只覆盖 steps、out_dir、`eval.every: 0`）。结果在 `runs/server-bench/colocate-1/`。
- 步时 360.5 / 359.2 / 396.6 秒，平均 **372 秒**，200 步约 **20.7 小时**，低于 40 小时，继续。另外有 9 次 dev 评测（第 0 步和每 25 步一次，K=8，每次 800 个 episode），粗估每次 10–15 分钟，总计约 23 小时。
- 每步 128 个 episode、679–750 条训练序列。采样 57–126 秒，训练 268–301 秒，保存 2.7 秒。
- KL 每步只用 64 条序列（`kl_sequences` = 64）。训练端显存峰值 5.35 GiB allocated / 5.49 GiB reserved。
- 3 步里：奖励 0.500 / 0.406 / 0.570；有好有坏的组 0.81 / 0.75 / 0.88；熵 0.82 / 0.78 / 0.74；每次调用格式错误率 3.4% / 5.6% / 2.6%；logprob 差均值约 0.0018；同步全部一致；没有跳过的步。
- 提速前后的步时：
  - 提速前，正式规模的步时没有实测过。第 3 步按过拟合（每步 64 个 episode，KL 用全部序列）外推，单步约 6.5 分钟（约 390 秒）。
  - 提速后实测 372 秒。只做了 KL 抽样这一项，梯度检查点没关。
  - 两者批次不同，不能精确拆出 KL 抽样省了多少。

### 正式训练启动

- 19:01:20 用 nohup 启动：`python -m arl.rl.grpo --config configs/grpo/v1.json`，输出目录 `runs/grpo-v1/`。启动时核对 `v1.json` 的 sha256 为 3566ba54…，和冻结记录一致。中断后可以用 `--resume` 续跑。
- vLLM 启动 54 秒，同步检查通过（贪心一致，logprob MAD 0.00026）。卡上训练端约 4.2 GB、vLLM 约 19.2 GB，合计 23.4 GB。
- **第 0 步 dev 基线**（SFT-v1，运行时 LoRA，K=8，T=0.8，top_k 20，100 题 800 个 episode）：严格解出率 **56.4%**，test+challenge 61.0%，0 个错误，耗时 599 秒。第 2 步合并模型 K=4 是 55.5%，两者一致。
- 按测速，200 步约 20.7 小时，加 8 次 dev 评测（每次约 10 分钟），预计明天 16:30 左右结束。

### 简报 1（10-12 21:11，已跑 2 小时 10 分）

- 进度：17 / 200 步，没有停止记录。
- 第 1–17 步均值：
  - 奖励 0.459，在 0.375–0.547 之间波动，还看不出趋势；
  - 有好有坏的组 0.87（最近 10 步 0.88）；
  - 熵 0.789（前 5 步均值 0.826，第 17 步 0.645）；
  - 每次调用格式错误率 4.0%（最近 10 步 4.0%），出错 episode 比例 15.5%；
  - logprob 差均值 0.0018（每步最大 0.0020）；KL(k3) 涨到 0.0009，每步 64 条序列；平均 7.1 轮。
- 其他：没有跳过的步；每步同步都一致，同步 logprob MAD 最大 0.00127；训练端显存峰值 5.73 GiB。
- 耗时：平均步时 403 秒（采样 79 秒，训练 320 秒），比测速的 372 秒慢约 8%，按这个速度 200 步约 22.4 小时。

### 简报 2（10-12 23:08，已跑 4 小时 7 分）

- 进度：34 / 200 步，没有停止记录。
- **第 25 步 dev**（K=8）：严格解出率 **59.4%**，test+challenge 63.6%。第 0 步是 56.4% / 61.0%。评测耗时 566 秒。
- 第 18–34 步均值：
  - 奖励 0.494，第 1–17 步是 0.459；
  - 有好有坏的组 0.85（最近 10 步 0.85）；
  - 熵 0.645，第 34 步 0.558，前 5 步均值 0.826，还在 30% 门槛（0.248）以上很多；
  - 每次调用格式错误率 3.5%（最近 10 步 3.4%），出错 episode 比例 15.1%；
  - logprob 差均值 0.0016（每步最大 0.0019）；KL(k3) 均值 0.0019。
- 其他：没有跳过的步；同步全部一致，MAD 最大 0.00073；显存峰值 5.46 GiB。
- 耗时：平均步时 391 秒（采样 83 秒，训练 305 秒）。

### 简报 3（10-13 01:05，已跑 6 小时 4 分）

- 进度：51 / 200 步，没有停止记录。
- **第 50 步 dev**（K=8）：严格解出率 **62.6%**，test+challenge 68.1%，评测耗时 511 秒。dev 严格解出率依次为：第 0 步 56.4%，第 25 步 59.4%，第 50 步 62.6%。
- 第 35–51 步均值：
  - 奖励 0.565（前两段是 0.459、0.494）；
  - 有好有坏的组 0.85（最近 10 步 0.85）；
  - 熵 0.524，第 51 步 0.447，30% 门槛是 0.248；
  - 每次调用格式错误率 2.9%（最近 10 步 3.0%），出错 episode 比例 12.3%；
  - logprob 差均值 0.0014（每步最大 0.0017）；KL(k3) 均值 0.0038。
- 其他：没有跳过的步；同步全部一致，MAD 最大 0.0013；显存峰值 5.86 GiB。
- 耗时：平均步时 373 秒（采样 68 秒，训练 302 秒）。

### 简报 4（10-13 03:02，已跑 8 小时）

- 进度：71 / 200 步，没有停止记录。这一段没有 dev 评测，下一次在第 75 步。
- 第 52–71 步均值：
  - 奖励 0.599（前三段是 0.459、0.494、0.565）；
  - 有好有坏的组 0.80（最近 10 步 0.81）；
  - 熵 0.536，第 71 步 0.634，前 5 步均值 0.826，门槛 0.248；
  - 每次调用格式错误率 2.3%（最近 10 步 2.4%），出错 episode 比例 10.4%；
  - logprob 差均值 0.0015（每步最大 0.0018）；KL(k3) 均值 0.0055；平均 6.9 轮。
- 其他：没有跳过的步；同步全部一致，MAD 最大 0.00136；显存峰值 5.54 GiB。
- 耗时：平均步时 363 秒（采样 75 秒，训练 285 秒）。按目前的速度，预计 10-13 下午 4 点前后结束。

### 简报 5（10-13 04:59，已跑 10 小时）

- 进度：88 / 200 步，没有停止记录。
- **第 75 步 dev**（K=8）：严格解出率 **62.5%**，test+challenge 68.0%，评测耗时 547 秒。和第 50 步的 62.6% 基本持平。第 100 步会按 `dev_gain_check_step` 和第 0 步的 56.4% 比较。
- 第 72–88 步均值：
  - 奖励 0.586（上一段 0.599）；
  - 有好有坏的组 0.80（最近 10 步 0.81）；
  - 熵 0.541，第 88 步 0.532；
  - 每次调用格式错误率 2.3%（最近 10 步 1.9%），出错 episode 比例 10.3%；
  - logprob 差均值 0.0015（每步最大 0.0018）；KL(k3) 均值 0.0077。
- 其他：没有跳过的步；同步全部一致，MAD 最大 0.00102；显存峰值 5.50 GiB。
- 耗时：平均步时 380 秒（采样 84 秒，训练 292 秒）。

### 简报 6（10-13 06:56，已跑 11 小时 55 分）

- 进度：106 / 200 步，没有停止记录。
- **第 100 步 dev**（K=8）：严格解出率 **64.5%**，test+challenge 69.0%，评测耗时 579 秒。
- **第 100 步检查**（`dev_gain_check_step`）：64.5% 高于第 0 步的 56.4%，通过，训练继续。
- dev 严格解出率依次为：第 0 步 56.4%，第 25 步 59.4%，第 50 步 62.6%，第 75 步 62.5%，第 100 步 64.5%。
- 第 89–106 步均值：
  - 奖励 0.610；
  - 有好有坏的组 0.78（最近 10 步 0.80）；
  - 熵 0.557，第 106 步 0.441；
  - 每次调用格式错误率 1.8%（最近 10 步 1.75%），出错 episode 比例 6.5%；
  - logprob 差均值 0.0016（每步最大 0.0019）；KL(k3) 均值 0.010；平均 6.8 轮。
- 其他：没有跳过的步；同步全部一致，MAD 最大 0.00112；显存峰值 5.72 GiB。
- 耗时：平均步时 356 秒（采样 75 秒，训练 278 秒）。

### 简报 7（10-13 08:53，已跑 13 小时 52 分）

- 进度：125 / 200 步，没有停止记录。第 125 步的 dev 评测正在跑。
- 第 107–125 步均值：
  - 奖励 0.656；
  - 有好有坏的组 0.76（最近 10 步 0.74）；
  - 熵 0.562，第 125 步 0.498；
  - 每次调用格式错误率 1.2%（最近 10 步 1.1%），出错 episode 比例 4.7%；
  - logprob 差均值 0.0016（每步最大 0.0018）；KL(k3) 均值 0.0127；平均 6.5 轮。
- 其他：没有跳过的步；同步全部一致，MAD 最大 0.00101；显存峰值 5.48 GiB。
- 耗时：平均步时 353 秒（采样 80 秒，训练 269 秒）。

### 简报 8（10-13 10:50，已跑 15 小时 49 分）

- 进度：144 / 200 步，没有停止记录。
- **第 125 步 dev**（K=8）：严格解出率 **67.9%**，test+challenge 72.1%，评测耗时 545 秒。是目前的最高值。
- dev 严格解出率依次为：第 0 步 56.4%，第 25 步 59.4%，第 50 步 62.6%，第 75 步 62.5%，第 100 步 64.5%，第 125 步 67.9%。
- 第 126–144 步均值：
  - 奖励 0.660；
  - 有好有坏的组 0.76（最近 10 步 0.71）；
  - 熵 0.538，第 144 步 0.451；
  - 每次调用格式错误率 1.1%（最近 10 步 1.3%），出错 episode 比例 4.5%；
  - logprob 差均值 0.0016（每步最大 0.0018）；KL(k3) 均值 0.0145。
- 其他：没有跳过的步；同步全部一致，MAD 最大 0.00079；显存峰值 5.57 GiB。
- 耗时：平均步时 351 秒（采样 72 秒，训练 275 秒）。预计 16:30 前后结束。

### 简报 9（10-13 12:47，已跑 17 小时 46 分）

- 进度：164 / 200 步，没有停止记录。
- **第 150 步 dev**（K=8）：严格解出率 **68.9%**，test+challenge 72.6%，评测耗时 503 秒。仍是最高值，比第 125 步的 67.9% 高 1.0 个百分点。
- 第 145–164 步均值：
  - 奖励 0.704；
  - 有好有坏的组 0.70（最近 10 步 0.71）；
  - 熵 0.418；
  - 每次调用格式错误率 1.1%（最近 10 步 1.0%），出错 episode 比例 4.0%；
  - logprob 差均值 0.0013（每步最大 0.0016）；KL(k3) 均值 0.0163。
- **熵在下降**：第 154 步起在 0.30–0.45 之间，第 164 步 0.296，离 30% 门槛 0.248 已经不远。按修订 3-1，熵低于门槛还要求最近 10 步有好有坏的组的比例均值低于 0.25 才停。现在这个比例是 0.71，所以就算熵跌破门槛，训练也会继续。
- 其他：没有跳过的步；同步全部一致，MAD 最大 0.00114；显存峰值 5.51 GiB。
- 耗时：平均步时 324 秒（采样 74 秒，训练 247 秒），步时变短是因为平均轮数在减少。预计 16:00 前后结束。

### 简报 10（10-13 15:04，已跑 20 小时 3 分）

- 进度：190 / 200 步，没有停止记录。
- **第 175 步 dev**（K=8）：严格解出率 **69.6%**，test+challenge 73.9%。比第 150 步的 68.9% 高 0.7 个百分点。
- 第 165–190 步均值：
  - 奖励 0.709；
  - 有好有坏的组 0.65（最近 10 步 0.64）；
  - 熵 0.369，最低一步是第 188 步的 0.278，还在 0.248 以上；
  - 每次调用格式错误率 1.0%（最近 10 步 0.9%），出错 episode 比例 3.3%；
  - logprob 差均值 0.0012（每步最大 0.0015）；KL(k3) 均值 0.019。
- 其他：没有跳过的步；同步全部一致，MAD 最大 0.00078；显存峰值 5.45 GiB。
- 耗时：平均步时 297 秒（采样 62 秒，训练 231 秒）。剩 10 步加第 200 步的 dev 评测，预计 16:05 前后结束。

### 正式训练结束（10-13 15:58）

- 200 步跑完，exit=0，`end` 记录 status done，没有任何 stop 记录。总用时 20 小时 57 分（75,414 秒），其中 9 次 dev 评测合计约 85 分钟。
- 每 25 步一段的均值：

| 步 | 奖励 | 有好有坏的组 | 熵 | 每次调用格式错误率 | logprob 差均值 | KL(k3) | 平均轮数 | 步时 s |
|---|---|---|---|---|---|---|---|---|
| 1–25 | 0.474 | 0.87 | 0.753 | 3.9% | 0.0018 | 0.0007 | 7.08 | 398 |
| 26–50 | 0.534 | 0.85 | 0.562 | 3.1% | 0.0015 | 0.0033 | 7.17 | 376 |
| 51–75 | 0.591 | 0.80 | 0.541 | 2.4% | 0.0015 | 0.0056 | 6.96 | 364 |
| 76–100 | 0.595 | 0.81 | 0.546 | 2.1% | 0.0015 | 0.0089 | 6.92 | 381 |
| 101–125 | 0.653 | 0.76 | 0.556 | 1.2% | 0.0016 | 0.0122 | 6.57 | 347 |
| 126–150 | 0.668 | 0.75 | 0.522 | 1.1% | 0.0015 | 0.0146 | 6.61 | 345 |
| 151–175 | 0.701 | 0.68 | 0.382 | 1.1% | 0.0012 | 0.0173 | 6.54 | 313 |
| 176–200 | 0.724 | 0.66 | 0.371 | 0.85% | 0.0012 | 0.0217 | 6.32 | 295 |

- 全程：
  - logprob 差均值 0.0015，单步最大 0.0020，p99 均值 0.0165，单 token 最大 0.573；TIS 截断 0；
  - GradScaler 0 步跳过；每步同步后贪心都一致，同步 MAD 最大 0.00136；
  - 训练端显存峰值 5.86 GiB allocated / 6.04 GiB reserved；
  - 每次调用格式错误率最高一步是第 6 步的 6.7%。
- 平均步时 353 秒：采样 74 秒，训练 275 秒，保存 3.0 秒，同步 0.5 秒。
- **熵**：最低一步是第 195 步的 0.228，低于 30% 门槛 0.248。当时最近 10 步有好有坏的组的比例均值约 0.66，远高于 0.25。按修订 3-1 的规则不停，代码也确实没停。
- 有好有坏的组的比例最低一步是第 186 步的 0.375。
- 勘误见 CHECKPOINT-3 末尾。

### 选检查点

dev 集 100 题，K=8，800 个 episode，运行时 LoRA：

| 步 | 严格解出数 | 严格解出率 | test+challenge |
|---|---|---|---|
| 0（SFT-v1） | 451 | 56.375% | 61.0% |
| 25 | 475 | 59.375% | 63.6% |
| 50 | 501 | 62.625% | 68.1% |
| 75 | 500 | 62.5% | 68.0% |
| 100 | 516 | 64.5% | 69.0% |
| 125 | 543 | 67.875% | 72.1% |
| 150 | 551 | 68.875% | 72.6% |
| 175 | 557 | **69.625%** | 73.9% |
| 200 | 551 | 68.875% | 74.1% |

- 规则（HELDOUT-PREREG 及修订 3-1）：选严格解出率最高的；相差不到 1 个百分点时，选步数更早的。
- 最高的是第 175 步，69.625%。
  - 第 150 步 68.875%，低 0.75 个百分点，不到 1 个百分点，而且步数更早。
  - 第 200 步同样低 0.75，但步数更晚。
  - 第 125 步低 1.75 个百分点，不在 1 个百分点之内。
- **选第 150 步**（`runs/grpo-v1/adapters/step-150`）。
- 说明：
  - 第 125 步和第 150 步正好差 1.0 个百分点（8 个 episode），不算“不到 1 个百分点”。而且比较的基准是最高值第 175 步，不是第 150 步，所以不会再往前链式比较。
  - 汇总记录里的严格解出率保留三位小数（例如 0.696），上表用逐条记录重新数了解出数。
- 下一步：把第 150 步的适配器合并成 fp16（`runs/grpo-v1/merged-fp16-step-150`），然后进入第 5 步。

## 阶段 3 · 第 5 步：留出集评测

### GRPO 检查点合并

- `python -m arl.sft.merge_fp16 --adapter runs/grpo-v1/adapters/step-150 --out runs/grpo-v1/merged-fp16-step-150`。基座快照 70d244cc 以 fp16 加载，用时 15 秒。`config.json` 和 `runs/sft-v1/merged-fp16` 的完全相同。
- **合并核对**（`runs/phase3/merge-check/`）：在 3 段文本上比较合并模型和“fp16 基座 + step-150 适配器”的逐 token logprob。
  - 平均差 0.0055，最大 0.055。
  - 作为对比，适配器本身相对基座的平均差是 0.50。
  - 合并后的差异只是 fp16 权重舍入，属于预注册已知局限里“运行时 LoRA 和合并权重之间的数值差别”。

### 开始前逐条确认预注册（`HELDOUT-PREREG.zh-CN.md`，含修订 2C-3 和 3-1）

1. **三个模型**，都用 token 客户端，环境 `phase1-env-v1`，fp16，vLLM `--dtype half`：
   - 基座 `Qwen/Qwen3-1.7B` 快照 70d244cc，直接以 fp16 加载（`eval_heldout.sh` 对基座传 `--revision`）；
   - SFT-v1：第 2 步 `merge_fp16.sh` 的结果 `runs/sft-v1/merged-fp16`，不用 `runs/sft-preview/merged`；
   - GRPO：第 150 步的适配器合并成 fp16（`runs/grpo-v1/merged-fp16-step-150`），不用运行时 LoRA。确认。
2. **检查点选法**：只看 dev 严格解出率（K=8，修订 3-1），取最高；相差不到 1 个百分点取更早的步，结果是第 150 步（见第 4 步）。dev 集和留出集不相交。确认。
3. **留出集**：`data/tasks/heldout.jsonl` 的 sha256 实测 `d28e0709bb81af8196837efaec77ed3833d952ea25fcc1d457aa87f15d7cc3d4`，和预注册一致。1500 题，第 1–4 步没有读它做任何评测。确认。
4. **每题 K=4，只跑一次**（修订 3-1 明确留出集不改为 K=8）。`eval_heldout.sh` 在 `runs/heldout/<name>.jsonl` 已存在时拒绝运行，只有中断后才用 `--resume` 续跑同一次运行，已完成的 episode 不会重采。确认。
5. **采样**：temperature 0.8、top_p 1.0、top_k 20，配置里写明 `top_k: 20`。客户端是 `configs/sft/v1-eval.json` 换模型名，脚本里断言 (tokens, 0.8, 1.0, 20)。确认。
6. **主指标**是严格解出率（隐藏用例全过）。次要指标：test+challenge 解出率；按 bug 类型和数据来源分开的严格解出率；每次工具调用的格式错误率（另报至少出错一次的 episode 比例，不作门槛）；平均轮数。由 `arl.rl.heldout` 输出。确认。
7. **统计**：按 `bug_source` 来源函数做聚类 bootstrap，有放回重抽簇，10000 次，seed 0，簇内题全部带上。每题先算 K=4 的解出率，再在抽中的簇上取平均。报告 GRPO−SFT、SFT−基座两组同题配对差值和 95% 置信区间。确认。
8. **已知局限**：会在 CHECKPOINT-3 逐条说明实际影响。确认。
9. 运行方式：三个模型依次各跑一次（基座、SFT-v1、GRPO），GPU 一次只跑一个，vLLM `--gpu-memory-utilization 0.85`，并发 16。按第 2 步每小时约 5600 个 episode 估算，每个模型 6000 个 episode 约 1.1 小时，三个共约 3.5 小时。

### 三个模型的留出集结果（每个模型只跑一次）

- 用 `runs/phase3/heldout-chain.sh` 依次运行 `eval_heldout.sh`，三个模型的退出码都是 0，每次之间卡上显存都回到 14 MiB：
  - 基座 15:59–16:19（19 分钟）；
  - SFT-v1 16:19–17:00（41 分钟）；
  - GRPO 17:00–17:38（38 分钟）。
- 每个模型 6000 个 episode（1500 题 × K=4，每个 k 都是 1500 条），报错 0。逐轮 `/tokenize` 比对：基座 31,470、SFT 41,073、GRPO 40,609 轮，不一致都是 0。
- 统计：`python -m arl.rl.heldout`，输出在 `runs/heldout/{base,sft,grpo}-summary.json` 和 `runs/heldout/compare.json`，完整文本在 `runs/phase3/heldout-report.txt`。

| | 基座 | SFT-v1 | GRPO（第 150 步） |
|---|---|---|---|
| **严格解出率** | **13.9%** | **47.1%** | **60.3%** |
| test+challenge 解出率 | 21.5% | 53.0% | 65.6% |
| arith_swap（1252） | 16.7% | 55.7% | 69.1% |
| bool_swap（224） | 4.0% | 31.2% | 51.3% |
| compare_swap（728） | 8.5% | 38.0% | 49.2% |
| const_off_by_one（1472） | 18.7% | 47.8% | 63.2% |
| remove_branch（944） | 2.5% | 33.2% | 44.1% |
| var_swap（1380） | 18.6% | 55.5% | 67.7% |
| MBPP（5160） | 14.8% | 48.1% | 61.0% |
| HumanEval（840） | 8.3% | 41.2% | 56.1% |
| 每次调用格式错误率 | 2.7% | 1.9% | 0.7% |
| 出错 episode 比例（只报告） | 3.5% | 7.3% | 2.1% |
| 平均轮数 | 5.24 | 6.84 | 6.77 |
| 停止：提交 / 用满轮数 / 超上下文 | 5360 / 627 / 13 | 4229 / 1766 / 5 | 4323 / 1677 / 0 |
| 每题解出 0/1/2/3/4 次 | 1157 / 124 / 54 / 56 / 109 | 413 / 256 / 243 / 268 / 320 | 299 / 187 / 192 / 241 / 581 |

（括号里是 episode 数。）

**配对聚类 bootstrap**：按 `bug_source` 分成 273 簇，10000 次，seed 0，同题配对。

| 对比 | 严格解出率差值 | 95% 置信区间 | 变好 / 变差的题 | test+challenge 差值 [95% CI] |
|---|---|---|---|---|
| **GRPO − SFT** | **+13.2 个百分点** | **[11.7, 14.7]** | 667 / 178 | +12.6 [11.1, 14.1] |
| **SFT − 基座** | **+33.2 个百分点** | **[30.4, 36.1]** | 926 / 82 | +31.5 [28.6, 34.4] |

- 两组差值的置信区间都远离 0。
- 留出集上 SFT-v1 是 47.1%，比它在 dev 上的 55.5% 低 8 个百分点，GRPO 也同样比 dev 低（60.3% 对 68.9%）。原因是两个集合的题目构成不同，例如留出集 HumanEval 占 14%，并且 HumanEval 题更难。两个模型的相对差在 dev 上（+13.4，K 不同）和留出集上（+13.2）基本一致。

## 阶段 3 · 第 6 步：CHECKPOINT-3

- 写了 `docs/plan/CHECKPOINT-3.md`：第 1–5 步的结论；修订 3-1 的 A、B 结果，提速前后的步时，梯度检查点因峰值超过 18 GiB 没有关闭（所以没有等价性差值）；训练曲线；各检查点的 dev 结果和选择理由；留出集三方对比和配对置信区间；已知局限的实际影响；机器时间（约 28.5 小时，付费 API 0 元）；遇到的问题。
- 到此停下。没有删除任何 `runs/` 数据，没有 `git push`。GPU 已空闲。

# 阶段 4：收尾

计划见 `docs/plan/PHASE-4-CLOSEOUT.zh-CN.md`，和启动 prompt 一起单独提交（`e4acc0b`）。本阶段在本地 WSL 里做，不跑模型、不调用付费 API、不租服务器。

## 第 1 步：本地复核（有两处对不上，停下等审计方）

### 做了什么

- `scripts/closeout/verify_heldout.py`：只读 `runs/heldout/{base,sft,grpo}.jsonl`、`runs/heldout/compare.json` 和 `data/tasks/heldout.jsonl`。
  - 各项比率直接从 episode 记录重新数；两组配对差值和区间调用 `arl.rl.heldout.paired`（273 簇，10000 次，seed 0），另外独立算一遍配对差的点估计。
  - 每个数和 `compare.json` 比，也和 CHECKPOINT-3"结论""留出集三方对比"里印出来的数比。少数几个数的出处是 PROGRESS（`/tokenize` 比对轮数、停止原因、SFT − 基座的变好/变差题数）和 HELDOUT-PREREG（题集构成）。
  - 最后用 `arl.rl.heldout.compare` 在原始记录上重新生成一遍，和 `compare.json` 整个比较。
- `scripts/closeout/verify_training.py`：只读 `runs/grpo-v1/log.jsonl`、`runs/grpo-v1/eval/step-*.jsonl`、`data/tasks/dev_ids.txt` 和 `configs/grpo/v1.json`。
  - 训练曲线表每 25 步一段，取逐步数值的平均。每步奖励是 128 个 0/1 episode 的均值，日志里只存了 4 位小数，脚本先按整数个数还原再平均。
  - dev 结果表从逐条 eval 记录重新数（每个检查点 100 题 × K=8 = 800 个 episode），也和 `log.jsonl` 里的 eval 记录比。
  - 按预注册规则重新选检查点：dev 严格解出率最高的；和最高值相差不到 1 个百分点的里面取步数最早的，不做链式比较。用整数判断：`100 × (最高解出数 − 解出数) < 800`。
  - 核对 `configs/grpo/v1.json` 的 sha256：工作区和提交 c9984b9 里的版本都要等于冻结值。
- 判定：四舍五入到报告里的位数以后相同，就算一致。恰好差半个单位的值（例如精确值 3.55，报告写 3.5）也算一致，但单独列在输出的 `half_unit_ties` 里。
- 输出 `runs/closeout/verify-heldout.json`、`runs/closeout/verify-training.json`（`runs/` 不进 git，关键数字摘在下面）。
- 新增 `tests/test_closeout.py`：选检查点的规则（阶段 3 的实际数据选出第 150 步；恰好差 1 个百分点不算"不到 1 个百分点"；不做链式比较；最高值并列时取更早的），以及取整判定。
- 顺带改了 `tests/test_heldout_stats.py` 两行注释的措辞。阶段 3 在服务器上提交的这两行注释里有英文连字符复合词（"task" 加 "weighted" / "level"），让工作区的密钥前缀检查多出一处命中。不是密钥，只改注释，测试逻辑没动。改完以后，工作区前缀命中回到 CHECKPOINT-1D 第 6 节第 4 条登记的三处。
  - 这次提交的暂存区前缀检查命中 2 处，就是被删掉的这两行旧注释（diff 里的 `-` 行），新增行 0 处；真实密钥值在工作区、暂存区、全部历史和 `runs/` 等目录里都是 0。删除这两行的提交一定会在 diff 里出现它们，确认以后提交。

### 留出集：150 项全部一致

| | 基座 | SFT-v1 | GRPO（第 150 步） |
|---|---|---|---|
| episode / 题 / 每题次数 / 报错 | 6000 / 1500 / 4 / 0 | 6000 / 1500 / 4 / 0 | 6000 / 1500 / 4 / 0 |
| 严格解出率（精确值） | 13.933% | 47.100% | 60.300% |
| test+challenge | 21.45% | 52.95% | 65.567% |
| 每题 4 次全解出 / 都没解出 | 109 / 1157 | 320 / 413 | 581 / 299 |

| 配对差值（273 簇，10000 次，seed 0） | 严格 | 95% CI | test+challenge | 95% CI | 变好 / 变差 |
|---|---|---|---|---|---|
| GRPO − SFT | +13.20 | [11.70, 14.68] | +12.62 | [11.12, 14.07] | 667 / 178 |
| SFT − 基座 | +33.17 | [30.36, 36.06] | +31.50 | [28.59, 34.43] | 926 / 82 |

- 按 bug 类型、按来源的严格解出率，格式错误率，出错 episode 比例，平均轮数，停止原因，`/tokenize` 比对轮数，题集 sha256 和构成，全部一致。
- GRPO − SFT 在六种 bug 类型和两个来源上都为正，最小 +10.91（remove_branch），最大 +20.09（bool_swap），HumanEval +14.88，和 CHECKPOINT-3 的"10.9–20.1""+20.1""+14.9"一致。
- `compare()` 在原始记录上重新生成的结果和 `compare.json` 完全相同。
- 恰好差半个单位的 10 处，都是二进制浮点取整的边界值，两种写法离精确值都是半个单位：基座 test+challenge 21.45→21.5，基座出错 episode 比例 3.55→3.5，SFT test+challenge 52.95→53.0，SFT bool_swap 31.25→31.2，SFT 平均轮数 6.845→6.84，GRPO 出错 episode 比例 2.05→2.1，其余 4 处是 `compare.json` 里同样的值。

### 训练和检查点：160 项里 158 项一致，2 项对不上

- **检查点选择：结论是第 150 步，和 CHECKPOINT-3 一致。** 最高是第 175 步（557/800）；不到 1 个百分点的是第 150、175、200 步，取最早的第 150 步。第 125 步比最高值低 1.75 个百分点。第 100 步 516 高于第 0 步 451。
- dev 表 9 个检查点的解出数（451、475、501、500、516、543、551、557、551）、严格解出率、test+challenge 解出率全部一致；每个检查点都是 dev 集 100 题、每题 8 次、报错 0。
- `configs/grpo/v1.json` 的 sha256 是 `3566ba54767d7f8a1964d785639fb4b8d4b43d2dd91989f27861b0a526f7261c`，工作区和 c9984b9 里都一样，等于冻结值。
- 训练曲线 8 段 × 8 列里 63 格一致。表下各行（logprob 差均值 0.0015、单步最大 0.0020、p99 均值 0.0165、TIS 截断 0、跳过 0 步、同步全部一致、MAD 最大 0.00136、显存峰值 5.86 / 6.04 GiB、平均步时 353 秒）、总用时、熵门槛 0.248、第 195 步熵 0.228 都一致。恰好差半个单位的 4 处：126–150 段奖励（精确值 0.6675，报告 0.668）、26–50 和 51–75 段有好有坏的组（0.845、0.795）、总用时 75413.5 秒。
- **对不上 1：训练曲线 76–100 段的每次调用格式错误率。** CHECKPOINT-3 写 2.1%，逐步比率的平均是 2.049%，四舍五入是 2.0%。
  - 原因很可能是取整了两次：先把平均值取成 4 位小数 0.0205，再换算成 2.05% 印成 2.1%。按这种两次取整的做法，其他 7 段也都和表里一致。
  - 换成"这一段的格式错误总数 ÷ 工具调用总数"是 2.054%，能印成 2.1%；但这样 51–75 段会是 2.455%（印成 2.5%，表里是 2.4%），所以这不是表用的口径。
  - 各段三种口径的数字都在 `verify-training.json` 的 `diagnostics` 里。
- **对不上 2：修订 3-1 第 3 条"第 195 步……最近 10 步有好有坏的组的比例约 0.66"**（PROGRESS 正式训练结束一节写的是"均值约 0.66"）。第 186–195 步的实际均值是 **0.625**。
  - 0.66 和 176–200 段的均值 0.6575 对得上，很可能是把这一段的均值当成了 10 步窗口的值。第 190 步的 10 步窗口是 0.644，简报 10 写的 0.64 是对的。
  - 不影响结论：0.625 仍远高于停止门槛 0.25，按修订 3-1 的规则第 195 步不该停，代码也没停。
- 两处都只是监控指标的文字描述，不涉及留出集数字、检查点选择和任何停止判断。按规则没有改 CHECKPOINT-3 和 PROGRESS 里原来的句子，也没有改 `runs/` 数据。

### 停在这里

按计划第 1 步"任何一项对不上就停下报告"，第 2–5 步都没开始：没有写最终报告、没有改 README、没有改服务器脚本、没有打 tag。等审计方决定这两处怎么处理。

测试结果：
- `uv run pytest -q` → 295 passed，6 skipped；
- `~/.venvs/vllm` 全量（`PYTHONPATH=src`）→ 309 passed，2 skipped；
- `~/.venvs/train` 里跑 `tests/train tests/rl` → 50 passed，1 skipped。
