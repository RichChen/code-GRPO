# CHECKPOINT-1A：第 0–4 步完成

日期：2026-10-09。环境：WSL2（内核 6.18.33.2），Python 3.12.15（uv 托管），uv 0.12.24，bubblewrap 0.11.1，8 核，内存 9GB + swap 8GB。

全量测试：`uv run pytest -q` → **119 passed，0 skipped，0 failed**，约 28 秒。本机 bwrap 可用，bwrap 相关用例全部实际运行，没有跳过。

提交记录：

| 提交 | 内容 |
| --- | --- |
| `dc45ac3` | 第 0 步：仓库初始化 |
| `730b426` | 第 1 步：工具层 |
| `a28447c` | 第 2 步：沙箱 |
| `e457f75` | 第 3 步：判分 + 任务模型 |
| `e851e31` | 第 4 步：Episode 循环 |

运行时依赖为空，只用标准库；dev 依赖只有 `pytest`。计划允许的 `pydantic`、`pyyaml`、`httpx`、`openai` 都还没引入。

## 1. 接口签名

### `arl.env.task`（新增文件，计划目录结构里没有）

```python
@dataclass(frozen=True)
class HiddenCase:
    call: str    # 在沙箱里求值的表达式，如 "pkg.nums.add(2, 5)"
    check: str   # 在宿主机上求值的表达式，绑定 result / error，如 "result == 7"

@dataclass
class Task:
    task_id: str
    issue: str
    files: dict[str, str]          # 初始工作区：改坏的代码 + tests/test_visible.py + 原始配置
    hidden_setup: str              # 隐藏用例前在沙箱里执行的代码，如 "import pkg.nums"
    hidden_cases: list[HiddenCase]
    solution: dict[str, str]       # 覆盖到 files 上即为原始正确代码
    meta: dict[str, Any]           # bug 类型、bug 文件和函数等分析字段，不进题面
    def materialize(self, dest, *, solved=False) -> Path
    def to_json(self) -> str
    @classmethod
    def from_json(cls, line: str) -> Task

def load_tasks(path) -> list[Task]
VISIBLE_TEST_PATH = "tests/test_visible.py"
```

### `arl.env.tools`

```python
@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str

TOOL_SCHEMAS: list[dict]     # OpenAI function calling 格式，五个工具
TOOL_NAMES = ("read_file", "search_code", "list_dir", "edit_file", "run_tests")

class Toolbox:
    def __init__(self, root, test_runner: TestRunner | None = None)   # 默认 runner 是沙箱
    def call(self, name: str, arguments: str | dict | None) -> ToolResult
    def read_file(self, path, offset=None, limit=None) -> ToolResult
    def search_code(self, pattern, path=None, glob=None) -> ToolResult
    def list_dir(self, path=None) -> ToolResult
    def edit_file(self, path, old_string, new_string) -> ToolResult
    def run_tests(self, target=None) -> ToolResult

def parse_arguments(name, arguments) -> dict   # 抛 ToolArgumentError(kind=unknown_tool|invalid_json|invalid_arguments)
def truncate_middle(text: str, max_chars: int) -> str
def pytest_argv(target: str) -> list[str]      # python -B -m pytest -q -p no:cacheprovider --color=no <target>
TestRunner = Callable[[Path, list[str]], RunOutput]
```

### `arl.env.sandbox`

```python
@dataclass(frozen=True)
class SandboxConfig:
    wall_timeout_s: float = 20.0
    cpu_seconds: int = 20
    address_space_mb: int = 1024
    file_size_mb: int = 16
    max_processes: int = 64
    tmpfs_mb: int = 64
    max_output_bytes: int = 1_000_000
    isolation: str = "auto"          # "auto" | "bwrap" | "rlimit_only"

@dataclass
class SandboxResult:
    exit_code: int | None            # 超时为 None
    output: str                      # stdout + stderr
    timed_out: bool
    isolation: str                   # "bwrap" | "rlimit_only"
    duration_s: float
    output_truncated: bool

def run_in_sandbox(workspace, argv: list[str], config: SandboxConfig | None = None) -> SandboxResult
def sandbox_runner(config=None) -> TestRunner
def bwrap_available() -> bool
def resolve_isolation(requested="auto") -> str
```

### `arl.env.grader`

```python
@dataclass
class GradeResult:
    reward: int                  # 0 或 1
    hidden_pass_count: int
    hidden_total: int
    visible_pass: bool | None    # 原始可见测试在 agent 代码上是否通过，仅供分析
    timed_out: bool
    files_changed: list[str]
    isolation: str
    setup_error: str | None
    case_passed: list[bool] | None

def grade(task, workspace, config: SandboxConfig | None = None, *, check_visible=True) -> GradeResult
def changed_files(task, workspace) -> list[str]
```

### `arl.env.episode`

```python
Policy = Callable[[list[Message], list[dict]], Message]   # OpenAI chat 格式

def run_episode(task, policy, max_turns=8, *, sandbox=None, context_max_tokens=8000,
                workspace_dir=None, keep_workspace=False) -> EpisodeResult

@dataclass
class EpisodeResult:
    task_id: str
    messages: list[dict]           # 完整对话，含 system / user / assistant / tool
    tool_calls: list[ToolCallRecord]
    reward: int
    grade: dict                    # GradeResult.to_dict()
    turns: int
    stop_reason: str               # "submitted" | "max_turns" | "context_limit"
    truncated: bool
    context_tokens_est: int
    duration_s: float
    meta: dict
    format_errors: int             # property
    def to_json(self) -> str
    @classmethod
    def from_dict(cls, d) -> EpisodeResult

@dataclass
class ToolCallRecord:
    turn: int; call_id: str; name: str; arguments: str
    ok: bool; error_kind: str | None; output_chars: int; output_truncated: bool

class ScriptedPolicy:              # 测试和第 5 步检查点用
    def __init__(self, turns: list[list[tuple[str, dict | str]] | str])

def write_jsonl(results, path) -> None
def estimate_tokens(messages, tools=TOOL_SCHEMAS) -> int    # 字符数 ÷ 4
SYSTEM_PROMPT                       # 来自 src/arl/env/prompt.md，215 词
```

## 2. 测试数量和结果

| 文件 | 用例数 | 结果 | 覆盖内容 |
| --- | --- | --- | --- |
| `tests/test_tools.py` | 46 | 全过 | 五个工具的正常和错误路径；路径逃逸（绝对路径、`..`、`~`、指向外部的目录和文件符号链接）对五个工具逐一验证；参数校验和分发 |
| `tests/test_sandbox.py` | 25 | 全过 | 两种隔离模式下的正常 pytest、超时、杀进程树、CPU/内存/文件大小限制、fork 炸弹、输出截断、环境变量白名单；仅 bwrap 的断网、写外部、看不到仓库、路径归一化 |
| `tests/test_grader.py` | 31 | 全过 | 正确修复和参考答案得 1、不改得 0、防作弊（见下节）、序列化和检查、任务模型 |
| `tests/test_episode.py` | 16 | 全过 | 脚本化 policy 端到端（修对得 1、只改可见测试得 0）、结束条件、格式错误、截断、JSONL |
| `tests/test_smoke.py` | 1 | 全过 | 占位 |
| **合计** | **119** | **全过** | |

## 3. 防作弊测试清单

先说明一个需要审计方确认的设计决定：**隐藏测试的期望值不进沙箱。** 每条隐藏断言拆成 `call`（沙箱里只求值并输出带类型标签的纯数据）和 `check`（宿主机上用解码后的值求值）。原因是如果隐藏测试是 pytest 文件、和 agent 代码同进程跑，agent 代码在 import 时可以篡改 pytest 或伪造报告，进程内防不住。判分流程本身仍按计划：拷工作区、删测试和配置文件、放回原始配置和隐藏测试、在沙箱里跑。详细说明见 `PROGRESS.md` 第 3 步。

计划要求的五条，全部有测试：

| 计划要求 | 测试 | 结果 |
| --- | --- | --- |
| 改可见测试（删断言）不影响判分 | `test_cheat_edit_visible_test_does_not_change_reward`：`run_tests` 显示通过，奖励仍 0；再修对代码奖励 1。端到端版本：`test_scripted_visible_test_cheater_gets_0` | 通过 |
| 写 `conftest.py` 劫持 pytest 会被删掉 | `test_cheat_conftest_hijack_is_removed`（根目录和 `tests/` 两处）：hookwrapper 把报告全改成 passed，`run_tests` 被骗过，判分奖励 0 | 通过 |
| `sys.exit(0)` 或 monkeypatch pytest，隐藏测试照样失败 | `test_cheat_exit_at_import`（`sys.exit(0)`、`os._exit(0)`、`raise SystemExit(0)`）、`test_cheat_exit_inside_function`、`test_cheat_monkeypatch_pytest`（`run_tests` 被骗过，判分奖励 0） | 通过 |
| 硬编码目标函数返回值只能骗过可见测试 | `test_cheat_hardcoded_return_value`：`visible_pass=True`，隐藏用例 0/3 | 通过 |
| 什么都不改奖励 0 | `test_unchanged_workspace_gets_0`、`test_submit_immediately_gets_0` | 通过 |

额外加的：

| 场景 | 测试 | 结果 |
| --- | --- | --- |
| 往 `pytest.ini`、`setup.cfg`、`tox.ini`、`pyproject.toml` 写恶意配置 | `test_cheat_config_files_are_replaced_by_originals` | 奖励 0 |
| 往 fd 1–19 和 stdout 写伪造的 harness 记录 | `test_cheat_forged_harness_output` | 重复记录全部判失败 |
| 返回 `__eq__` 恒为 True 的对象（int 子类、普通类） | `test_cheat_always_equal_object` | `run_tests` 被骗过，奖励 0 |
| 放假的 harness 文件和 `sitecustomize.py` | `test_cheat_replace_harness_and_sitecustomize` | 奖励 0 |
| 把 bug 文件换成指向外部正确答案的符号链接 | `test_cheat_symlink_to_solution_is_not_followed` | bwrap 下奖励 0 |
| opaque 值和异常标记做任何检查都失败 | `test_undecidable_values_fail_every_check` | 通过 |
| 判分不改动 agent 工作区；隐藏用例不出现在工作区 | `test_grading_does_not_touch_workspace`、`test_hidden_cases_are_not_in_the_workspace` | 通过 |

另外，工具层本身就不提供 `write_file` 和 shell，agent 在真实 episode 里只能改已有文件，上面多数"写新文件"的场景是直接在测试里模拟的最坏情况。

## 4. bwrap 隔离实际验证了哪些项

本机 WSL2 + bubblewrap 0.11.1，非特权 user namespace 可用。以下都是实际运行验证的，不是推断：

| 项 | 怎么验证 | 结果 |
| --- | --- | --- |
| 断网（`--unshare-net`） | 宿主机在 127.0.0.1 开监听端口，沙箱内连接；另连 1.1.1.1:53 | 两个都失败，宿主机端没收到连接 |
| 工作区可写 | 写 `/workspace/inside.txt` | 成功，宿主机工作区里能看到 |
| 工作区外不可写 | 写宿主 tmp 路径、家目录、`/usr`、`/`、`../` | 全部失败，宿主机上没有出现文件（加了 `--remount-ro /`） |
| 看不到本仓库 | 检查仓库的 `src/`、`docs/` 是否存在 | 都不存在，所以 agent 代码读不到 `data/` 里的隐藏测试 |
| 环境变量白名单（`--clearenv`） | 宿主机设 `ARL_SECRET_TOKEN`、`PYTHONPATH` | 沙箱内都看不到 |
| 墙钟超时 | 死循环，超时 2 秒 | 2 秒左右返回，`timed_out=True` |
| 超时杀整棵进程树（`--unshare-pid` + `--die-with-parent`） | 沙箱内起子进程和 fork 出的孙进程各睡 60 秒 | 超时后宿主机 `/proc` 里找不到它们 |
| CPU 限制 | `RLIMIT_CPU=1`，死循环 | 被信号杀掉，没等到墙钟超时 |
| 内存限制 | `RLIMIT_AS=512MB`，申请 2GB | `MemoryError` |
| 文件大小限制 | `RLIMIT_FSIZE=1MB`，写 3MB | 进程被杀，文件不超过 1MB |
| fork 炸弹（`RLIMIT_NPROC`） | 进程上限 32，无限 fork 且子进程常驻 | 不到 32 个就 `EAGAIN`，没有超时 |
| 正常 pytest | 在沙箱里跑一个通过的测试 | `1 passed`，退出码 0 |
| 路径归一化 | 打印 cwd、错误输出 | 都是 `/workspace`，不暴露宿主路径 |

`rlimit_only` 模式（同一套用例强制退化运行）验证了：正常 pytest、超时、杀进程树、CPU/内存/文件大小限制、fork 炸弹、输出截断、环境变量白名单。它**没有**文件系统和网络隔离。

## 5. 遇到的问题和待确认事项

1. **需要审计确认的设计决定**：隐藏测试改成 `call`/`check` 对，期望值留在宿主机（见第 3 节）。影响第 5 步：造题时要把 MBPP/HumanEval 的断言拆成 `call` 和 `check`，拆不出来的断言不能用。另外 `check` 应尽量用相等比较；像 `result is not None` 这种弱检查对 opaque 值会成立，第 5 步应避免生成。
2. **`RLIMIT_NPROC` 必须在 user namespace 里面设**：它按 uid 统计所有任务（含线程），本机 Cursor 自身线程很多，在 bwrap 外设 64 会让 bwrap 起不来。改成沙箱内的启动器先 setrlimit 再 exec。
3. **字节码缓存导致改了代码不生效**：同一秒内改动且文件大小不变时，Python 会复用旧 `.pyc`。所有在沙箱里跑的 Python 都加了 `-B`，并设 `PYTHONDONTWRITEBYTECODE=1`。
4. **AutoDL 上 bwrap 能不能用待验证**：容器里可能不允许非特权 user namespace。验证方法：`uv run python -c "from arl.env.sandbox import bwrap_available; print(bwrap_available())"`，再跑 `uv run pytest -q tests/test_sandbox.py` 看有没有跳过的用例。如果只能用 `rlimit_only`：agent 代码能读到 `data/` 里的隐藏测试（期望值就在里面）、能联网、能写工作区外，以 root 运行时 fork 炸弹也挡不住。到时需要审计方决定怎么处理（比如把题目数据放到沙箱用户不可读的位置）。
5. **隐藏用例没有单条超时**：一条死循环会让整次运行超时，`hidden_pass_count` 只统计超时前完成的用例；奖励不受影响。
6. **上下文估算把工具 schema 算进去了**（约 600 token），比只算消息更保守。
7. 系统自带 Python 是 3.14，不满足 3.11/3.12，改用 uv 托管的 3.12.15。本机原来没有 uv，已装到 `~/.local/bin/uv`。git 全局没有用户名，用仓库级配置 `crq <crq@localhost>`。
8. 访问 PyPI 很慢（单次请求约 8 秒），第一次 `uv sync` 用了 5 分钟。第 5 步下载 MBPP/HumanEval 时可能也会慢。

按要求在此停下，没有开始第 5 步。
