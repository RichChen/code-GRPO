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
