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
