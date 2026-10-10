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
