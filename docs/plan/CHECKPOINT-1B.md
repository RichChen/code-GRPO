# CHECKPOINT-1B：第 5 步（造题器）

## 结论：候选题 2146 道，少于 2500，按计划停下报告

- 生成并通过全部验证的候选题共 **2146 道**（train 1455、heldout 691），少于计划要求的 2500。
- 计划第 5 步原文："如果这两个数据集加起来，通过第 5 步筛选的题凑不到 2500 道候选，就停下报告，不要自行换别的数据源。"所以停在这里，**没有换数据源，没有放宽筛选条件，也没有提高每个（函数，bug 类型）最多 2 道的上限**。
- 缺口不是组装或验证造成的。上限为 2 时，筛查后理论最多就是 2146 道，实际产出也是 2146 道，整题验证通过率 100%。缺口来自两个数据集里合格来源函数的数量（540 个，筛查后有合格变体的 461 个）。
- 第 5 步的其余要求都已完成：防泄漏测试、留出集固定 seed、manifest、补充要求里的各项检查和统计、脚本化 policy 检查。全量测试 169 passed。

### 不同上限下的最大候选数（只作参考，未采用）

按筛查结果统计：每个（函数，bug 类型）取 min(上限, 合格的不同注入位置数)，再求和。

| 每个（函数，bug 类型）上限 | 最大候选数 |
| --- | --- |
| 1 | 1339 |
| 2（计划要求，当前） | 2146 |
| 3 | 2648 |
| 4 | 2982 |
| 不设上限 | 3852 |

### 需要审计方决定的事

以下方案我都没有采用，等审计方选择：

1. **接受 2146 道作为候选池**，修改计划里的 2500。阶段 1 第 6 步之后的筛选区间会在这个池子上做。
2. **把上限从 2 提到 3**：最多 2648 道，超过 2500。需要改计划第 5 步"同一个来源函数、同一种 bug 最多保留 2 道题"这一条。注入位置多的函数（主要是 `const_off_by_one` 和 `var_swap`）占比会变高，同一个函数出现在更多题里。
3. **加用完整版 MBPP**（同一仓库的 `mbpp/mbpp.jsonl`，974 题，sanitized 是从中人工挑出的 427 题）。许可证相同，但计划写的是"MBPP sanitized 加 HumanEval"，这算不算"换数据源"需要审计方定。没有下载，也没有估算能多出多少。

放宽拆断言规则（比如接受容器里的浮点数）或筛查规则（比如接受只挂 1 条用例的变体），都和补充要求或计划第 4 节"不为凑数量放宽验证条件"冲突，没有考虑。

## manifest 摘要

`data/tasks/manifest.json`，seed 0。两个 jsonl 不提交（约 27MB），重新生成命令：`uv run python -m arl.taskgen.build --raw data/raw --out data/tasks`（8 线程约 5 分 20 秒）。连续 3 次构建的 sha256 相同：

- `train.jsonl`：`86ca767c0b898da3486e4fd2902c683933be1265cb8cdcf88d8a2e77e6a37743`
- `heldout.jsonl`：`3b8c2e036dc9e67fa48ddc7e909f2d09879369c841cc91eaff7cacc6a881327b`

原始数据：MBPP sanitized（google-research `f82046b`，Apache-2.0）和 HumanEval（openai/human-eval `d321ec0`，MIT），sha256 和许可证见 `data/raw/SOURCES.md`。

| 项 | 值 |
| --- | --- |
| 来源函数（拆断言后） | 540 个（MBPP 393、HumanEval 147） |
| 筛查后可用（原始代码通过） | 539 个（`mbpp/123` 原始代码超时） |
| 划分 | train 377 个函数，heldout 162 个（30%） |
| 实际作为 bug 函数出现的不同函数 | train 321 个，heldout 140 个 |
| 题目里 bug 来源 | train：MBPP 961、HumanEval 494；heldout：MBPP 416、HumanEval 275 |
| 每题模块数 | 3 / 4 / 5 个：727 / 700 / 719 道 |
| 每题函数数 | 6–19 个，中位数 12 |
| 每题隐藏用例 | 中位数 37 条（含干扰函数）；其中 bug 函数中位数 2 条 |
| 包代码长度 | 中位数 3431 字符 |
| 改名 | 70 道题各改名 1 个函数（其中 9 道改的是 bug 函数） |

每类 bug 的题数：

| bug 类型 | train | heldout | 合计 |
| --- | --- | --- | --- |
| `arith_swap` | 301 | 132 | 433 |
| `compare_swap` | 178 | 101 | 279 |
| `const_off_by_one` | 389 | 185 | 574 |
| `bool_swap` | 55 | 26 | 81 |
| `var_swap` | 316 | 135 | 451 |
| `remove_branch` | 216 | 112 | 328 |
| 合计 | 1455 | 691 | 2146 |

防泄漏：train 和 heldout 的 `source_ids`（包括干扰函数）没有交集。`check_no_leakage` 在写盘前检查，`tests/test_taskgen.py` 对合成构建和真实题集都有测试。

## 补充要求里的统计

### 判分耗时（50 题）

`uv run python -m arl.taskgen.report`，固定 seed 从 train+heldout 里抽 50 题，顺序执行（不并发）：

| 项 | 中位数 | p95 | 最大 |
| --- | --- | --- | --- |
| 单题 `grade()`（未修改的工作区，含隐藏用例和原始可见测试两次沙箱运行） | 0.283 s | 0.333 s | 0.342 s |
| 一次 episode 总耗时：直接提交（只有最终判分） | 0.283 s | 0.285 s | |
| 一次 episode 总耗时：改可见测试 + 1 次 `run_tests` + 判分 | 0.500 s | 0.504 s | |
| 一次 episode 总耗时：应用参考答案 + 判分 | 0.284 s | 0.294 s | |

- 一次 episode 的判分开销约 0.28 秒，每次 `run_tests` 另加约 0.22 秒。模型推理时间不在其中。
- 造题时 8 个线程并发判分，单次 `grade()` 中位数是 0.496 秒（manifest 里的 `grade_median`），说明并发时有争用。估算 RL 吞吐时建议按并发下约 0.5 秒/次判分计。
- `grade()` 的大部分时间花在 `visible_pass`（用 pytest 重跑原始可见测试，只做分析，不进奖励）。另抽 20 题顺序测：完整 `grade()` 中位数 0.283 秒，`check_visible=False` 时只有 0.067 秒。如果训练时吞吐不够，可以跳过这一项。

### 每类 bug 的验证通过率

| bug 类型 | 生成变体 | 通过筛查 | 筛查通过率 | 没有用例失败 | 只挂 1 条用例 | 超时 | 崩溃 | 整题验证 | 整题通过率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `arith_swap` | 848 | 708 | 83.5% | 64 | 54 | 21 | 1 | 433 | 100% |
| `compare_swap` | 525 | 344 | 65.5% | 120 | 53 | 8 | 0 | 279 | 100% |
| `const_off_by_one` | 3344 | 2146 | 64.2% | 724 | 411 | 63 | 0 | 574 | 100% |
| `bool_swap` | 121 | 92 | 76.0% | 11 | 16 | 2 | 0 | 81 | 100% |
| `var_swap` | 2403 | 1752 | 72.9% | 307 | 289 | 55 | 0 | 451 | 100% |
| `remove_branch` | 716 | 432 | 60.3% | 127 | 145 | 12 | 0 | 328 | 100% |

- "没有用例失败"的变体可能是等价变异，也可能是现有用例没覆盖到，没有逐个区分。
- 没有 import 时出错的变体（只修改入口函数的函数体）。唯一的崩溃是 `//` 改成 `*` 后数值爆炸，沙箱进程被杀。
- 整题验证只对每个（函数，bug 类型）依次尝试候选，凑满 2 道就停，所以"整题验证"一列等于最终题数。

### 拆断言的丢弃原因

函数：MBPP 427 个里保留 393 个，HumanEval 164 个里保留 147 个。丢弃原因都是"合格断言不足 3 条"（MBPP 34 个，HumanEval 17 个）。

断言：MBPP 1324 条里保留 1217 条，HumanEval 1196 条里保留 1037 条。

| 丢弃原因 | MBPP | HumanEval | 说明 |
| --- | --- | --- | --- |
| `not_a_comparison` | 51 | 60 | `assert f(x)`、`assert True` 等，不是比较 |
| `float_inside_container` | 27 | 18 | 期望值是含浮点数的容器，没法用一个 `isclose` 表达 |
| `not` | 10 | 18 | `assert not f(x)` |
| `call_wrapped_or_other_function` | 15 | 10 | `set(f(x)) == ...`、`abs(f(x) - y) < ...`、调用了别的函数 |
| `args_not_literal` | 0 | 17 | 参数不能 `literal_eval`（如含表达式、变量） |
| `import` | 0 | 9 | `check()` 里的 import 语句 |
| `is` | 0 | 6 | `assert f(x) is True` |
| `assignment` | 0 | 5 | `check()` 里的赋值 |
| `loop` | 0 | 5 | `check()` 里的循环 |
| `not_equality` | 0 | 5 | `<`、`!=` 等 |
| `duplicate_call` | 1 | 4 | 同一函数里重复出现的调用 |
| `expected_not_literal` | 3 | 1 | 期望值不能 `literal_eval` |
| `other_statement` | 0 | 1 | 其他语句 |
| 合计 | 107 | 159 | |

`helper_definition`、`chained_comparison`、`reversed_comparison`、`star_args`、`bool_op` 在这两份数据里计数为 0。

### 题面模板占比

| 模板 | 写法 | 题数 | 占比 |
| --- | --- | --- | --- |
| `returns_a` | `` `f(x)` returns `A`, but it should return `E`. `` | 615 | 28.7% |
| `returns_b` | `` Calling `f(x)` gives `A` instead of `E`. `` | 594 | 27.7% |
| `returns_c` | `` `f(x)` should return `E`, but it returns `A`. `` | 589 | 27.4% |
| `raises` | `` `f(x)` raises `IndexError`, but it should return `E`. `` | 336 | 15.7% |
| `long_input` | `` `f` returns a wrong result for the input used in its visible test. `` | 9 | 0.4% |
| `wrong_no_actual` | `` `f(x)` returns a wrong result; the expected result is `E`. `` | 3 | 0.1% |

前三种在实际输出可以正常显示时随机选。`raises` 用于 bug 版抛异常的情况；`long_input` 用于调用或期望值太长的情况；`wrong_no_actual` 用于实际输出太长或无法显示的情况。题面中位数长度 73 字符，全部英文。

## 脚本化 policy 检查（50 题）

同一批 50 题，用第 4 步的 `ScriptedPolicy` 跑完整 episode（`run_episode`，bwrap 沙箱）：

| policy | 做法 | 奖励 0 | 奖励 1 |
| --- | --- | --- | --- |
| 原样提交 | 直接回复 "Done." | 50 | 0 |
| 只改可见测试 | `edit_file` 把 bug 函数的可见断言改成 `assert True`，`run_tests`，提交 | 50 | 0 |
| 应用参考答案（正例） | `edit_file` 把 bug 文件整体换成原始代码，提交 | 0 | 50 |

只改可见测试的 50 个 episode 里 `files_changed` 都是 `tests/test_visible.py`，`visible_pass`（用原始可见测试重跑）都是 False。原始结果在 `runs/checkpoint-1b/`（不提交）。

## 随机抽 10 题：题面和 diff

固定 seed 抽取，diff 方向是"发布版 → 参考答案"，也就是正确的修复。

#### `train-humaneval-54-compare_swap-7`（train，compare_swap `0:==->!=`，来源 humaneval/54，4 个模块 / 12 个函数，模板 returns_b）

> Calling `same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')` gives `False` instead of `True`.

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -15,5 +15,5 @@
     False
     """
-    return set(s0) != set(s1)
+    return set(s0) == set(s1)
```

#### `train-mbpp-168-var_swap-12`（train，var_swap `count->x`，来源 mbpp/168，3 个模块 / 8 个函数，模板 returns_a）

> `frequency([1, 2, 3], 4)` returns `4`, but it should return `0`.

```diff
--- a/pkg/misc.py
+++ b/pkg/misc.py
@@ -17,5 +17,5 @@
         if i == x:
             count += 1
-    return x
+    return count
```

#### `heldout-mbpp-766-var_swap-42`（heldout，var_swap `l1->temp`，来源 mbpp/766，4 个模块 / 14 个函数，模板 returns_b）

> Calling `pair_wise([1, 1, 2, 3, 3, 4, 4, 5])` gives `[]` instead of `[(1, 1), (1, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5)]`.

```diff
--- a/pkg/misc.py
+++ b/pkg/misc.py
@@ -39,5 +39,5 @@
 def pair_wise(l1):
     temp = []
-    for i in range(len(temp) - 1):
+    for i in range(len(l1) - 1):
         current_element, next_element = (l1[i], l1[i + 1])
         x = (current_element, next_element)
```

#### `train-mbpp-230-var_swap-12`（train，var_swap `char->str1`，来源 mbpp/230，4 个模块 / 12 个函数，模板 returns_a）

> `replace_blank('hello people', '@')` returns `'hellohello peoplepeople'`, but it should return `'hello@people'`.

```diff
--- a/pkg/common.py
+++ b/pkg/common.py
@@ -20,5 +20,5 @@
 
 def replace_blank(str1, char):
-    str2 = str1.replace(' ', str1)
+    str2 = str1.replace(' ', char)
     return str2
```

#### `train-humaneval-107-const_off_by_one-59`（train，const_off_by_one `+1`，来源 humaneval/107，3 个模块 / 7 个函数，模板 returns_c）

> `even_odd_palindrome(123)` should return `(8, 13)`, but it returns `(8, 0)`.

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -28,5 +28,5 @@
     odd_palindrome_count = 0
     for i in range(1, n + 1):
-        if i % 2 == 2 and is_palindrome(i):
+        if i % 2 == 1 and is_palindrome(i):
             odd_palindrome_count += 1
         elif i % 2 == 0 and is_palindrome(i):
```

#### `train-humaneval-47-arith_swap-50`（train，arith_swap `//->*`，来源 humaneval/47，5 个模块 / 14 个函数，模板 raises）

> `median([-10, 4, 6, 1000, 10, 20])` raises `IndexError`, but it should return `8.0`.

```diff
--- a/pkg/misc.py
+++ b/pkg/misc.py
@@ -10,5 +10,5 @@
         return l[len(l) // 2]
     else:
-        return (l[len(l) // 2 - 1] + l[len(l) * 2]) / 2.0
+        return (l[len(l) // 2 - 1] + l[len(l) // 2]) / 2.0
```

#### `train-humaneval-74-remove_branch-7`（train，remove_branch `drop_else`，来源 humaneval/74，3 个模块 / 8 个函数，模板 returns_a）

> `total_match(['hi', 'admin'], ['hi', 'hi'])` returns `None`, but it should return `['hi', 'hi']`.

```diff
--- a/pkg/funcs.py
+++ b/pkg/funcs.py
@@ -28,2 +28,4 @@
     if l1 <= l2:
         return lst1
+    else:
+        return lst2
```

#### `heldout-mbpp-306-const_off_by_one-114`（heldout，const_off_by_one `+1`，来源 mbpp/306，4 个模块 / 12 个函数，模板 returns_b）

> Calling `max_sum_increasing_subseq([1, 101, 2, 3, 100, 4, 5], 7, 4, 6)` gives `111` instead of `11`.

```diff
--- a/pkg/routines.py
+++ b/pkg/routines.py
@@ -14,5 +14,5 @@
     for i in range(n):
         if a[i] > a[0]:
-            dp[0][i] = a[i] + a[1]
+            dp[0][i] = a[i] + a[0]
         else:
             dp[0][i] = a[i]
```

#### `train-mbpp-619-arith_swap-25`（train，arith_swap `+->-`，来源 mbpp/619，3 个模块 / 7 个函数，模板 raises）

> `move_num('I1love143you55three3000thousand')` raises `TypeError`, but it should return `'Iloveyouthreethousand1143553000'`.

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -64,5 +64,5 @@
             dig += ele
         else:
-            res -= ele
+            res += ele
     res += dig
     return res
```

#### `heldout-mbpp-227-var_swap-41`（heldout，var_swap `a->smallest`，来源 mbpp/227，5 个模块 / 15 个函数，模板 raises）

> `min_of_three(10, 20, 0)` raises `UnboundLocalError`, but it should return `0`.

```diff
--- a/pkg/common.py
+++ b/pkg/common.py
@@ -31,5 +31,5 @@
     if a <= b and a <= c:
         smallest = a
-    elif b <= smallest and b <= c:
+    elif b <= a and b <= c:
         smallest = b
     else:
```

## 补充要求的落实情况

| 要求 | 怎么做的 | 测试 |
| --- | --- | --- |
| MBPP 只收 `assert f(args) == expected`，HumanEval 只收 `assert candidate(args) == expected`，其他丢弃并计数 | `sources.parse_assert`，原因见上表 | 接受和各类拒绝的单元测试 |
| `check` 必须引用 `result`，浮点用 `math.isclose(..., rel_tol=1e-6, abs_tol=1e-9)`，没有弱检查 | `Case.check()` 只生成这两种形式 | 从真实数据拆出的每条 `check` 都以 `result == ` 或 `math.isclose(result, ` 开头 |
| `expected` 能 `literal_eval` | 拆断言时求值，失败丢弃 | 同上 |
| 隐藏用例里没有和可见测试相同的 call | 拆断言时同一函数内去重；组装时可见用例从该函数的隐藏用例里去掉；`check_task_static` 再查一遍 | 合成构建和真实题集都检查 |
| 带上 import 和 helper；重名按固定规则改名，题面、可见、隐藏一起改；改完用原始代码验证 | `normalise_module` 保留 import/helper/常量；`PackageBuilder.try_place` 改成 `name_{k}`（内置名也算占用）；整题验证要求参考答案奖励 1 | 改名、内置名遮蔽、import 冲突的单元测试；合成数据里的 `def sum(` 被改名 |
| bug 不造成语法错误、不在 import 时暴露 | 变体必须能 `compile`；只改入口函数体；筛查时 import 出错判为 `import_error` 丢弃（实际 0 个） | 6 类 bug 的变体都能编译、import 不受影响 |

## 观察到的问题和风险（请审计方看一下）

1. **bug 函数的隐藏用例偏少**：中位数只有 2 条（MBPP sanitized 大多数题只有 3 条断言，1 条用作可见测试）。不完全正确的修复（比如只对这几个输入碰巧对）可能拿到奖励。硬编码可见用例的输出拿不到奖励，因为隐藏用例的输入一定和可见测试不同。
2. **HumanEval 的 docstring 里有示例**（比如上面第 1 题 `same_chars` 的 docstring 写了输入和输出），这些示例可能和隐藏用例重合。这是题目本身的规格说明，不是泄漏判分代码，但模型能从中看到部分隐藏输入的期望值。没有删 docstring。
3. **可见测试写成 `module.func(args)`**，从测试文件能看出 bug 函数在哪个模块。题面里本来就有函数名，用 `search_code` 一次就能找到，所以多给的信息有限。改成这样是为了不让名为 `test_*` 的来源函数被 pytest 收集。
4. **只按 `source_id` 防泄漏**：MBPP 和 HumanEval 之间、MBPP 内部可能有功能几乎一样的函数（比如都在求最大公约数），没有做近似去重检查。
5. **`bool_swap` 只有 81 道**（3.8%），因为两个数据集的解里 `and`/`or`/`not` 本来就少。
6. **`var_swap` 会造成 `UnboundLocalError`**（上面最后一题）：这是运行时错误，不是 import 时错误，符合要求，但这类 bug 不太像"人写错的变量名"。
7. **判分吞吐**：顺序执行时最终判分约 0.28 秒，每次 `run_tests` 约 0.22 秒；8 线程并发时单次判分升到约 0.5 秒。这些只是环境侧的开销，没有测过和模型推理同时运行时的情况。

## 本步其他改动

- 判分 harness 从 `python -I -B` 改成 `python -S -B`（`-I` 会忽略 `PYTHONHASHSEED=0`，导致集合迭代顺序每次不同），加了测试。
- `.gitignore` 加了 `data/tasks/*.jsonl`；`data/raw/` 的两个原始文件、`SOURCES.md` 和 `data/tasks/manifest.json` 提交。
- 网络：PyPI 直连已经不慢，没有配置镜像，`uv.lock` 没改；GitHub 直连会被重置，数据经代理下载。测速结果在 PROGRESS。

按要求停在这里，没有开始第 6 步。
