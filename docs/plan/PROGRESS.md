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
