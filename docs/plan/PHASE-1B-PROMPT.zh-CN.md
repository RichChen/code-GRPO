CHECKPOINT-1A 审计已通过。接下来执行 `docs/plan/PHASE-1-ENV-AND-TASKS.zh-CN.md` 的第 5 步，做到 CHECKPOINT-1B 停下。其他规则和之前一样。

## 审计结论

1. 隐藏测试的设计改为 `call`/`check` 对，期望值留在宿主机，这一改动确认通过。把下面这段原样补进计划第 3 步的末尾，标题为"审计修订 1"：
   隐藏测试不再是 pytest 文件，而是 `HiddenCase(call, check)`：`call` 在沙箱里求值，`check` 在宿主机上对解码后的值求值。可见测试仍是 pytest 文件。
2. `RLIMIT_NPROC` 放到沙箱内设置、全部 Python 加 `-B`、新增 `env/task.py`，这三处都确认，不用再改。
3. AutoDL 上隔离方式退化的问题先不处理，留到阶段 0 在服务器上实测后再定。

## 第 5 步的补充要求（优先于计划原文）

- **拆断言**
  - MBPP 只接受 `assert f(args) == expected` 这种形式，HumanEval 只接受 `assert candidate(args) == expected` 这种形式。其他写法一律丢弃并计数，包括循环、辅助函数、`not`、`is`、多个比较串在一起等。
  - `check` 必须引用 `result`。可以用 `result == <literal>`；浮点数一律用 `math.isclose(result, x, rel_tol=1e-6, abs_tol=1e-9)`。禁止 `result is not None`、`bool(result)` 这类弱检查。
  - `expected` 必须能用 `ast.literal_eval` 求值，不能的丢弃。
- **可见测试和隐藏测试的输入不能重复**：同一个函数的隐藏用例里，不能出现和可见测试完全相同的 `call`。要写测试检查。
- **拼小包时**
  - 来源函数依赖的 import 和辅助函数要一起带上，比如 `import re`、`heapq`，或者同一个解里定义的 helper。
  - 函数重名时，用固定规则改名，题面、可见测试和隐藏用例要跟着一起改。
  - 改完以后，要用原始代码验证一遍全部用例都能通过（计划里本来就有这一条，这里强调一下）。
- **注入 bug 不能把函数改成语法错误**，也不能让 bug 只在 import 时就暴露出来。
- **CHECKPOINT-1B 额外要报告这些数据**
  - 抽 50 题，统计单题判分耗时的中位数和 p95，以及一次 episode 判分的总耗时。RL 训练的吞吐量要用这个来估算。
  - 每类 bug 的验证通过率。
  - 拆断言时各种丢弃原因的计数。
  - 题面里各种模板各占多少。

## 网络（PyPI 和下载慢）

先测一下，再选方案，测试结果写进 PROGRESS：

1. **测速**：分别对 `https://pypi.org/simple/pytest/`、`https://mirrors.aliyun.com/pypi/simple/pytest/`、`https://pypi.tuna.tsinghua.edu.cn/simple/pytest/`、`https://raw.githubusercontent.com/`，用 `curl -o /dev/null -s -w '%{time_total}\n'` 测三次，取中位数。
2. **代理**：如果 Windows 上开着代理，先在 WSL 里设 `HTTPS_PROXY`/`HTTP_PROXY` 再测一遍。
   - WSL2 默认是 NAT 网络，代理地址要用 `ip route show default | awk '{print $3}'` 查到的宿主 IP，不能用 127.0.0.1；Windows 代理软件还得打开"允许局域网连接"。
   - 端口问用户，不要猜。连不上就跳过这一项。
3. **镜像**：PyPI 如果走镜像，只能写在用户级配置 `~/.config/uv/uv.toml`（`[[index]]` 加 `default = true`），或者用环境变量 `UV_DEFAULT_INDEX`。不要写进仓库的 `pyproject.toml`。如果 `uv.lock` 因此要改写源地址，先停下报告，不要提交。
4. **数据下载**：MBPP 和 HumanEval 从 GitHub 下。直连太慢的话就走代理。下完必须核对 sha256 和许可证，记到 `data/raw/SOURCES.md`。不要从来路不明的镜像站下数据。

第 5 步本身不需要新加依赖，下载用标准库 `urllib` 就够了。

做完第 5 步，写好 `docs/plan/CHECKPOINT-1B.md` 就停下，不要开始第 6 步。
