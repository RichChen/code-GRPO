# CHECKPOINT-1C：第 6 步试点触发停止条件

## 结论

- **第 5b 步已完成**（`812d5ce`）：
  - 按簇划分，最大的簇有 9 个函数，划分后留出集里近似重复为 0。
  - 生成用例时保持输入不变量，augment 已接进构建。
  - 共 4732 道题（train 3232、heldout 1500），连续构建两次 sha256 相同。
  - 50 题脚本化检查结果符合预期，全量测试 198 passed。
- **第 6 步的试点触发了停止条件，所以停在这里**：
  - 100 题 × 4 次，`Qwen2.5-1.5B-Instruct` 一次都没解出（解出率 0%），解出次数落在 [0.25, 0.75] 区间里的题为 0%；
  - 按工具调用算，格式错误率 37%，超过 30%；
  - 估算全量耗时约 2–3.4 小时，这一项没有触发。
- 按 PHASE-1C-PROMPT 的规定，**没有换模型，没有调参数，没有跑全量校准**；**第 7 步（README、冻结题集、打 tag `phase1-tasks-v1`）也没做**。
- 我检查过，0% 不是 harness 的 bug（证据见下面"试点"一节）。
  - 链路是通的：工具调用能被正常解析、执行，episode 结束后照常判分；
  - 问题出在模型本身：一半的 episode 第一轮直接回复纯文本，不调任何工具；其余 episode 写出的工具调用 JSON 经常不合法；400 个 episode 里没有一次成功改动文件。

## 第 5b 步：按簇划分和隐藏用例加强

### 按簇划分（审计修订 3 第 1 条）

- 在全部 1070 个来源函数上两两算 Jaccard，大于 0.8 连边，求连通分量。用 `split-{seed}` 打乱簇的顺序，依次放进留出集，直到达到 30%。划分后再跑一遍 `near_duplicates`，结果为 **0**。
- 簇：共 945 个。大小分布：1 个函数的 854 个，2 个 75 个，3 个 7 个，4 个 5 个，5 个 2 个，6 个 1 个，9 个 1 个。最大簇 9 个函数，没有超过 30。
- 最大的 5 个簇：

| 大小 | 划分 | 成员 |
| --- | --- | --- |
| 9 | heldout | mbpp/197 find_exponentio、213 concatenate_strings、261 division_elements、399 bitwise_xor、401 add_nested_tuples、429 and_tuples、445 index_multiplication、616 tuple_modulo、888 substract_elements |
| 6 | heldout | mbpp/43 text_match、44 text_match_string、719 text_match、879 text_match、917 text_uppercase_lowercase、930 text_match |
| 5 | train | mbpp/16 text_lowercase_underscore、285 text_match_two_three、434 text_match_one、454 text_match_wordz、756 text_match_zero_one |
| 5 | train | mbpp/282 sub_list、358 moddiv_list、682 mul_list、729 add_list、865 ntimes_list |
| 4 | train | humaneval/55 fib、humaneval/63 fibfib、mbpp/389 find_lucas、mbpp/960 get_noOfways |

- 留出集共 321 个函数，筛查后可用 320 个。按来源分：MBPP sanitized 122、MBPP 完整版新增 161、HumanEval 37。
- 新增的测试：
  - 同一簇的函数不跨两边；
  - 合成数据里，近似重复的两个函数会被分到同一边；
  - 真实数据按簇划分后，近似重复为 0；
  - 原有"干扰函数只从同一边取"的测试保留。

### 输入不变量（审计修订 3 第 3 条）

从每个来源函数全部 `kind="test"` 用例的参数里推断不变量。生成的候选输入必须全部满足，不满足的丢弃，按不变量分别计数。丢弃数按尝试次数计（一个函数最多尝试 1000 次，凑够 25 个合格候选为止），所以会大于最终的候选数。

| 不变量 | 命中的函数数 | 丢弃的候选数 |
| --- | --- | --- |
| 整数 > 0 | 401 | 14861 |
| 整数 ≥ 0（但不全 > 0） | 30 | 551 |
| 整数 = 另一个参数的 `len()`（不单独扰动，扰动序列后重新计算） | 64 | 不适用（不会产生违反） |
| 序列非空 | 673 | 3038 |
| 升序 | 105 | 4339 |
| 降序 | 4 | 93 |
| 两序列等长 | 47 | 4214 |

升序或降序要求至少一个种子是严格单调的；等长要求至少一个种子的长度 ≥ 2。这样做是为了避免全部种子都只有一个元素时，推断出没有意义的约束。

### 加强前后的分布

| 项 | 数值 |
| --- | --- |
| 合格的候选输入 | 24300 |
| 保留（每函数最多 10 条） | 10395 |
| 合格但超出 10 条上限 | 13126 |
| 原始代码抛异常 / 超时 | 729 / 13 |
| 期望值过长 / 不能还原为字面量 / 容器里含浮点 | 27 / 6 / 4 |
| 原始代码跑不过的 challenge 用例（丢弃） | 1 |
| 因 bug 函数隐藏用例不足 5 条而丢弃的题 | 0 |

bug 函数的隐藏用例数（按题统计，4732 道）：

| 隐藏用例条数 | 只算 test | test + challenge | 加强后 |
| --- | --- | --- | --- |
| 2 | 3884 | 3827 | 0 |
| 3–4 | 294 | 334 | 0 |
| 5–11 | 496 | 513 | 394 |
| 12 | 14 | 14 | **3467**（众数） |
| 13 及以上 | 44 | 44 | 871 |

- 按题计，补出来的用例共 45721 条，其中 **33002 条（72.2%）能让 bug 版失败**；4732 道题里有 4654 道至少有一条补出的用例失败。
- 整题验证通过率 100%。

### 随机抽 20 条生成的用例（供人工判断）

在全部题目的 `generated` 用例里按来源函数去重，用 seed 2026 随机抽 20 个函数，每个函数抽 1 条：

| 函数 | 输入 | 期望 |
| --- | --- | --- |
| mbpp/191 | `check_monthnumber('beFruary')` | `result == False` |
| mbpp/568 | `empty_list(3)` | `result == [{}, {}, {}]` |
| mbpp/911 | `maximum_product([12, 74, 9, 61, 41])` | `result == 185074` |
| mbpp/93 | `power(3, 7)` | `result == 2187` |
| mbpp/16 | `text_lowercase_underscore('aa_bcbbbc')` | `result == True` |
| mbpp/393 | `max_length_list([[3, 4, 5], [6, 7, 8, 9], []])` | `result == (4, [6, 7, 8, 9])` |
| mbpp/758 | `unique_sublists_2([['green', 'orange'], ['black'], ['green', 'orange'], [], ['white']])` | `result == {('green', 'orange'): 2, ('black',): 1, (): 1, ('white',): 1}` |
| mbpp/890 | `find_Extra([1, 3, 5, 7, 9, 11], [1, 3, 7, 9], 4)` | `result == 2` |
| mbpp/798 | `_sum([12, 13, 10])` | `result == 35` |
| mbpp/422 | `find_Average_Of_Cube(6)` | `math.isclose(result, 73.5, rel_tol=1e-06, abs_tol=1e-09)` |
| humaneval/102 | `choose_num(547, 546)` | `result == -1` |
| mbpp/112 | `perimeter(3, 2)` | `result == 10` |
| mbpp/175 | `is_valid_parenthese('()[[{)}')` | `result == False` |
| mbpp/507 | `remove_words(['red', 'green', 'blue', 'white', 'black', 'wh', 'orange'], ['black', 'orange'])` | `result == ['red', 'green', 'blue', 'white', 'wh']` |
| mbpp/149 | `longest_subseq_with_diff_one([1, 2, 3, 2, 5, 3, 4], 7)` | `result == 6` |
| mbpp/814 | `rombus_area(8, 5)` | `math.isclose(result, 20.0, rel_tol=1e-06, abs_tol=1e-09)` |
| humaneval/119 | `match_parens(['(()())((', '(())))'])` | `result == 'Yes'` |
| mbpp/89 | `closest_num(10)` | `result == 9` |
| mbpp/558 | `digit_distance_nums(21, 56)` | `result == 8` |
| mbpp/368 | `repeat_tuples((1, 5), 4)` | `result == ((1, 5), (1, 5), (1, 5), (1, 5))` |

我自己看下来，多数在原题语境下是合理的，至少有 1 条不合理：

- **`find_Extra([1, 3, 5, 7, 9, 11], [1, 3, 7, 9], 4)`**：原题是两个有序数组，第二个只比第一个少一个元素，求多出来的那个元素的下标。生成的输入少了两个元素，等长和有序这两条不变量都管不到这种语义约束。期望值 2 是原始代码在这个输入上的确定行为，把 bug 改回原样一定能通过。但如果模型换一种写法，对合法输入都正确，在这条上可能拿不到奖励。
- 其余几条（如 `match_parens` 的括号串、`is_valid_parenthese('()[[{)}')`）属于正常的边界输入。

### 其他检查

- 连续构建两次，sha256 相同：train `f0038c46eb4b089daabe68ba7246cf086fe521458a24bc2a82e47b5e7840aa82`，heldout `d28e0709bb81af8196837efaec77ed3833d952ea25fcc1d457aa87f15d7cc3d4`。两份 manifest 除计时字段外完全一致。
- 50 题脚本化检查：原样提交 0/50 拿到奖励，只改可见测试 0/50，应用参考答案 50/50。
- 各类 bug 的题数：

| bug 类型 | train | heldout |
| --- | --- | --- |
| arith_swap | 622 | 313 |
| compare_swap | 425 | 182 |
| const_off_by_one | 822 | 368 |
| bool_swap | 117 | 56 |
| var_swap | 732 | 345 |
| remove_branch | 514 | 236 |

- 不同 bug 来源函数：train 648 个，heldout 273 个。
- 题面模板：returns_b 1386、returns_a 1322、returns_c 1311、raises 695、long_input 11、wrong_no_actual 7。
- 构建耗时 20.5 分钟。

## 第 6 步：推理服务和试点

### 推理服务

- 模型：`Qwen/Qwen2.5-1.5B-Instruct`，HF 版本 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，`model.safetensors` 3.09 GB，经代理从 Hugging Face 官方下载。
- vLLM 0.31.0（torch 2.13.0+cu130），装在独立的 venv `~/.venvs/vllm`，不在项目依赖里。
- 完整命令：

```bash
HF_HUB_OFFLINE=1 ~/.venvs/vllm/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.85 --seed 0
```

- 工具调用 parser 用 `hermes`：Qwen2.5 的聊天模板输出的是 `<tool_call>{json}</tool_call>`，对应 vLLM 的 `Hermes2ProToolParser`，vLLM 文档对 Qwen2.5 写的也是这个。
- 2080 Ti 的算力是 7.5，不支持 FlashAttention 2，vLLM 自动选了 `TRITON_ATTN`。Triton 需要 C 编译器，第一次启动因为缺 gcc 失败了，用户执行 `sudo apt install gcc libc6-dev` 后正常。启动约 150 秒。KV cache 5.3 GiB（198k token），8192 上下文下最多并发 24.2 个请求。
- 客户端参数（`configs/rollout.json`）：temperature 0.8，top_p 1.0，max_tokens 1024，并发 8，每个 episode 最多 8 轮。

### 试点结果

train 和 heldout 各取 50 题（seed 0，按来源函数分层），每题跑 4 次，共 400 个 episode，0 个出错，用时 4 分 19 秒。

| 指标 | 数值 | 停止条件 |
| --- | --- | --- |
| 总体解出率 | **0.0%**（0/400；train、heldout、6 类 bug、各题面模板都是 0） | < 10% → **触发** |
| 解出次数在 [0.25, 0.75] 区间的题 | **0%**（100 题全是 0/4） | < 15% → **触发** |
| 格式错误率（按工具调用） | **37%**（131/354） | > 30% → **触发** |
| 有格式错误的 episode | 20% | — |
| 平均轮数 | 1.85 | — |
| 每个 episode 的 token 数 | 生成 219.7；最终上下文 1265；各轮 prompt 累计 2068 | — |
| 停止原因 | submitted 398，max_turns 2 | — |
| 吞吐量 | 5556 episode/小时 | — |
| 估算全量耗时 | heldout 1500 题 + train 抽 1200 题：10800 个 episode，约 1.9 小时；train 全跑：18928 个 episode，约 3.4 小时 | > 24 小时 → 没有触发 |

格式错误的构成：没被解析出来的工具调用（`malformed_call`）124 次，参数不合法 6 次，未知工具 1 次。

### 为什么是 0%（不是 harness 的问题）

- **一半的 episode 一次工具都没调**：400 个里有 205 个第一轮就回复纯文本（解释"问题可能在哪里"，或者直接贴一段代码），按环境规则，不调工具就视为提交，episode 结束。轮数分布：1 轮 205、2 轮 119、3 轮 40、4 轮 20、5 轮及以上 16。
- **工具调用的 JSON 经常不合法**：模型在 `<tool_call>` 里用 Python 的原始字符串写法 `r"..."`，`<tool_call>` 后面又接自然语言或第二个没闭合的 `<tool_call>`。例子：

```text
<tool_call>
{"name": "search_code", "arguments": {"pattern": r"check\(\s*'SeQuOiA'\s*=\s*'accepteD'\s*\)", "path": "./checks/"}}
</tool_call>
```

  hermes parser 拒绝这些是对的。客户端会把它们记成格式错误、反馈给模型，episode 继续。
- **从来没有成功改动过文件**：`edit_file` 只调用了 14 次，全部失败。路径或 `old_string` 是编出来的，比如 `/pkg/slope.py`，`old_string` 还写成正则。400 个 episode 里 `files_changed` 全为空，`visible_pass` 全为 False。合法的工具调用里最多的是 `search_code`（194 次），但常常搜不存在的路径，工具报错 124 次。
- **链路本身是通的**：
  - 冒烟测试和试点里，合法的工具调用都被 vLLM 解析成结构化的 `tool_calls`（`finish_reason=tool_calls` 223 次），并正常执行；
  - 判分正常；
  - 脚本化策略在同一 harness 上应用参考答案，50/50 拿到奖励。

## 第 7 步：未做

没有补 README，没有冻结 sha256，没有打 tag `phase1-tasks-v1`，也没有生成 `data/tasks/calibrated/`。第 7 步要求的报告项都依赖全量校准的结果，暂时没有数据：

- 解出 0 到 4 次的题各有多少；
- 筛选后的数量；
- 每类 bug 的解出率（`UnboundLocalError` 类单独一行）；
- 题面模板和解出率的关系；
- 留出集里不同 bug 来源函数的数量；
- 抽 50 个解出的 episode，在加强前的用例上重新判。

只能报告试点里的情况：试点 100 题全部 0/4，`UnboundLocalError` 类在试点里没有抽到。

## 需要审计方决定的事

以下方案我都没有采用，只列出来：

1. **换更强的模型**，参数不变。11 GB 显存放得下的候选有：
   - `Qwen2.5-Coder-1.5B-Instruct`；
   - `Qwen2.5-3B-Instruct` / `Qwen2.5-Coder-3B-Instruct`，fp16 约 6 GB；
   - 7B 的 AWQ 4-bit 量化版。算力 7.5 上 vLLM 能不能跑 AWQ 需要先验证。

   代价是阶段 2 的训练对象也要跟着变，或者校准用的模型和训练用的模型不是同一个。
2. **改 system prompt 或工具说明**（比如强调"必须用工具"、给一个工具调用示例），模型不变。这算调参数，按规定需要审计方同意。
3. **把"不调工具就算提交"改成提醒一次再继续**。这会改环境语义，影响阶段 1A 已定的规则。
4. **降低停止条件的门槛**。不推荐：1.5B 模型在 400 个 episode 里一次都没成功改过文件，放宽门槛也筛不出 1–3 次解出的题。

## 当前状态

- 代码和题集在 `812d5ce`（第 5b 步），之后只加了本报告和 PROGRESS 记录。
- 试点记录在 `runs/calibration/pilot.jsonl` 和 `pilot-summary.json`（`runs/` 不进 git），vLLM 日志在 `runs/vllm/serve.log`。
- 写完报告后关掉了 vLLM 服务，要重启就用上面的命令。

---

## 附录 A：提前停下版（第 5b 步近似重复检查超过 10%，已由审计修订 3 解决）

以下是原来的 `CHECKPOINT-1C.md`（提前停下版）全文，标题降了两级，内容没有改动。

### CHECKPOINT-1C（提前停下）：第 5b 步近似重复检查超过 10%

#### 结论

- 按审计修订 2 第 5 条做近似重复检查：留出集 **321 个来源函数里有 63 个（19.6%）** 和训练集函数的最大 Jaccard 超过 0.8，超过了"移动的超过留出集函数总数的 10% 就停下报告"的门槛。
- 所以停在第 5b 步：**没有移动任何函数，没有重建题集，没有开始第 6 步和第 7 步**。没有自行改阈值、改规范化方式或改划分方法。
- 抽查结果显示这些大多是真重复，不是规范化过头造成的误报（例子见下）。其中不少对只差一个运算符或常量，比如 `remove_even` 和 `remove_odd` 只差 `!=` 和 `==`，而这正是一道 `compare_swap` 题的标准答案。

#### 已完成的部分

| 项 | 状态 |
| --- | --- |
| "审计修订 2"原样补进计划第 5 步末尾，单独提交 | 完成（`8dbbf1b`） |
| `nvidia-smi` 前置检查 | 能跑通：RTX 2080 Ti，11264 MiB 显存，驱动 615.78.05，支持 CUDA 13.4。第 6 步的前置条件满足，但还没装 vLLM |
| 下载完整版 MBPP，校验 sha256，写进 `SOURCES.md` | 完成：`mbpp.jsonl`，同一仓库同一 commit（`f82046b`），sha256 `ccf64cea…a92a9f`，Apache-2.0 |
| `sources.py` 合并 sanitized 版和完整版，加入 `challenge_test_list` | 完成，有测试 |
| 隐藏用例加强 `taskgen/augment.py` | 代码和测试完成，在 60 个函数上试跑过；**还没接进 `build.py`** |
| 近似重复检查 `taskgen/dedup.py` | 代码和测试完成；在 seed 0 的划分上运行后触发停止条件 |
| 接进构建、重建题集、两次构建对比 sha256、50 题脚本化检查 | 未做（停在近似重复检查） |
| 第 6 步、第 7 步 | 未做 |

全量测试 181 passed（新增 `tests/test_augment_dedup.py` 12 个）。

##### 来源函数（合并完整版 MBPP 之后）

| 来源 | 原始题数 | 保留 | 丢弃原因 |
| --- | --- | --- | --- |
| MBPP sanitized（沿用 sanitized 版的代码和断言） | 427 | 393 | 合格断言不足 3 条 34 |
| MBPP 完整版新增（不在 sanitized 里的题号） | 547 | 530 | 合格断言不足 3 条 15，`test_setup_code` 非空 2 |
| HumanEval | 164 | 147 | 合格断言不足 3 条 17 |
| 合计 | 1138 | **1070** | |

- 完整版新增部分的断言 1635 条，保留 1601 条。丢弃原因：容器里有浮点数 16、参数不是字面量 9、重复调用 6、调用被包装 3。
- `challenge_test_list` 只有 11 题有，拆出 16 条，全部合格（sanitized 题号 6 条，完整版新增 10 条），作为 `kind="challenge"` 只进隐藏用例。原始代码跑不过的 challenge 用例在加强阶段丢掉并计数。
- seed 0 划分（30% 留出）：train 749 个函数，heldout 321 个（完整版新增 156、sanitized 121、HumanEval 44）。

#### 近似重复检查的结果

方法按修订 2：对每个来源函数的入口函数，去掉 docstring，把参数名和所有非内置的名字都换成同一个占位符，`ast.unparse` 后用 `tokenize` 切成 token，取 3-gram 集合，计算每个留出集函数和全部训练集函数的最大 Jaccard。检查对象是加载后的全部 1070 个来源函数（在变异筛查之前）。

##### 移动数量和阈值的关系

| 阈值（最大 Jaccard >） | 留出集函数数 | 占 321 的比例 |
| --- | --- | --- |
| 0.6 | 125 | 38.9% |
| 0.7 | 89 | 27.7% |
| **0.8（修订 2 规定）** | **63** | **19.6%** |
| 0.9 | 20 | 6.2% |
| 0.95 | 17 | 5.3% |
| 规范化后完全相同（1.0） | 17 | 5.3% |

##### 重复来自哪里（阈值 0.8 的 63 个）

| 留出集函数来源 → 最相似的训练集函数来源 | 数量 |
| --- | --- |
| MBPP 完整版新增 → MBPP 完整版新增 | 24 |
| MBPP sanitized → MBPP 完整版新增 | 18 |
| MBPP 完整版新增 → MBPP sanitized | 13 |
| MBPP sanitized → MBPP sanitized | 7 |
| HumanEval → MBPP 完整版新增 | 1 |

- 按留出集函数的来源算：MBPP 完整版新增 37/156（23.7%），MBPP sanitized 25/121（20.7%），HumanEval 1/44（2.3%）。
- 对照：只用之前的 540 个函数（sanitized 加 HumanEval），同样的 seed 和方法，是 10/162（6.2%），在门槛以内。所以超标主要是加入完整版 MBPP 带来的。MBPP 本身就有不少重复题（同名的 `max_occurrences`、`remove_empty`、`word_len`、`snake_to_camel`、`change_date_format` 等各出现两次），完整版把它们带进来了。
- 在全部 1070 个函数之间两两比较：Jaccard 超过 0.8 的函数对有 173 对，按"相似就连边"连起来，有 216 个函数落在 91 个多成员的簇里，最大的簇有 9 个函数。

##### 例子（0.8–0.9 之间随机抽的）

```python
# mbpp/202（heldout）                         # mbpp/437（train），Jaccard 0.889
def remove_even(str1):                        def remove_odd(str1):
    str2 = ''                                     str2 = ''
    for i in range(1, len(str1) + 1):             for i in range(1, len(str1) + 1):
        if i % 2 != 0:                                if i % 2 == 0:
            str2 = str2 + str1[i - 1]                     str2 = str2 + str1[i - 1]
    return str2                                   return str2

# mbpp/831（heldout）                         # mbpp/841（train），Jaccard 0.885
def count_Pairs(arr, n):                      def get_inv_count(arr, n):
    cnt = 0                                       inv_count = 0
    for i in range(n):                            for i in range(n):
        for j in range(i + 1, n):                     for j in range(i + 1, n):
            if arr[i] == arr[j]:                          if arr[i] > arr[j]:
                cnt += 1                                      inv_count += 1
    return cnt                                    return inv_count

# mbpp/429（heldout）                         # mbpp/197（train），Jaccard 0.818
def and_tuples(test_tup1, test_tup2):         def find_exponentio(test_tup1, test_tup2):
    res = tuple((ele1 & ele2 for ...))            res = tuple((ele1 ** ele2 for ...))
    return res                                    return res
```

完整清单（63 个）见本文末尾的附录。

#### 需要审计方决定的事

以下方案我都没有采用：

1. **按簇划分**：先在全部来源函数上把 Jaccard > 0.8 的函数连成簇（上面统计的 91 个簇、216 个函数），以簇为单位做 train/heldout 划分，然后照常跑一遍检查（按构造，移动数应该接近 0）。这是处理数据泄漏的常见做法，留出集比例不变。但这改变了修订 2 规定的"先划分、再把重复的移到训练集"的流程。
2. **照常移动 63 个，忽略 10% 门槛**：留出集剩 258 个函数，训练集 812 个。
3. **提高阈值**：比如只移动规范化后完全相同的 17 个（5.3%），或者阈值取 0.9（20 个，6.2%）。0.8–0.9 之间的对子，从上面的例子看多数是真正的"同一模板换一个运算符"，提高阈值会把这类泄漏留在留出集里。
4. **不用完整版 MBPP 新增的部分**：回到 540 个函数，近似重复 6.2%，在门槛内。但这样又回到 CHECKPOINT-1B 的题量，修订 3 的新门槛（校准后留出集至少 200 题、120 个不同 bug 来源函数）能不能满足要等校准结果。

#### 隐藏用例加强：已实现，试跑结果

还没接进构建，但代码和测试已经完成，等近似重复的问题定下来再接。规则按修订 2 第 4 条：

- 以每个已有用例的参数为种子，每个候选只扰动 1 个参数（30% 的概率扰动 2 个）。整数：加减 1–3、取负、置 0、变成负数；浮点：加减小量、取负、置 0；布尔：取反；字符串、列表、元组：清空、截断、删一个元素、插入一个元素（字符串插入它本身已有的字符，列表插入一个已有元素的扰动版）、交换两个元素、扰动其中一个元素（保持嵌套结构）；字典：删一个键、扰动一个值、清空；集合：删除或插入元素。容器最多比种子长 5。
- 每个函数用固定 seed（`augment-{source_id}`）生成最多 25 个不重复的候选，并排除和已有用例（包括可见测试的候选）完全相同的参数。
- 每个候选在沙箱里用原始代码跑两次（两个独立进程，每条限时 1 秒）。两次都正常返回、结果相同、能用 `repr` 加 `ast.literal_eval` 还原且类型逐层一致，才保留。抛异常、超时、不确定、不能还原的都丢弃并计数。浮点结果用 `math.isclose`；容器里含浮点、复数结果、期望值超过 500 字符的也丢弃。每个函数最多保留 10 条。
- 集合按排序后的元素输出，宿主机的哈希随机化不会让构建结果不确定。
- 生成的用例记为 `kind="generated"`，只进隐藏用例；可见测试只能从 `kind="test"` 里选。筛查规则相应调整：变体必须至少挂 1 条 `test` 用例（当可见测试），并且总共至少挂 2 条。
- 还没做的：bug 函数隐藏用例不足 5 条的题丢弃并计数、整题重新验证、加强前后的分布统计、补出的用例让 bug 版失败的比例。这些都在接进构建时一起做。

试跑（随机 60 个来源函数，8 线程，23 秒）：

| 项 | 数量 |
| --- | --- |
| 候选输入（当时上限是 40，之后为了速度改成 25） | 2153 |
| 保留（每函数上限 10） | 595（59 个函数补满 10 条，1 个函数补了 5 条） |
| 原始代码抛异常 | 182 |
| 超时 | 11 |
| 合格但超出 10 条上限 | 1365 |

例子（`mbpp/88 freq_count`，原测试是 13 个元素的列表）：`[]` → `{}`；`[10, 10, 10, 0, 20, ...]` → `{10: 3, 0: 1, 20: 4, ...}`。

**观察到的风险**：扰动后的输入可能超出原题的预期用法，比如 `max_sub_array_sum([-1, -2, 3], -7)`，第二个参数本来应该是列表长度。期望值来自原始代码的实际行为，所以把 bug 改回原样一定能通过；但如果模型重写了整个函数，写法不同但对正常输入都正确，在这类输入上可能拿不到奖励。

#### 附录：阈值 0.8 下的 63 个留出集函数

| 留出集函数 | 来源 | 最大 Jaccard | 最相似的训练集函数 |
| --- | --- | --- | --- |
| `mbpp/10` small_nnum | mbpp_full | 1.0 | `mbpp/496` heap_queue_smallest |
| `mbpp/17` square_perimeter | mbpp_sanitized | 1.0 | `mbpp/716` rombus_perimeter |
| `mbpp/45` get_gcd | mbpp_full | 1.0 | `mbpp/520` get_lcm |
| `mbpp/116` tuple_to_int | mbpp_sanitized | 1.0 | `mbpp/439` multiple_to_single |
| `mbpp/178` string_literals | mbpp_full | 1.0 | `mbpp/186` check_literals |
| `mbpp/338` count_Substring_With_Equal_Ends | mbpp_full | 1.0 | `mbpp/30` count_Substring_With_Equal_Ends |
| `mbpp/413` extract_nth_element | mbpp_sanitized | 1.0 | `mbpp/49` specified_element |
| `mbpp/458` rectangle_area | mbpp_sanitized | 1.0 | `mbpp/52` parallelogram_area |
| `mbpp/542` fill_spaces | mbpp_full | 1.0 | `mbpp/732` replace_specialchar |
| `mbpp/649` sum_Range_list | mbpp_full | 1.0 | `mbpp/170` sum_range_list |
| `mbpp/658` max_occurrences | mbpp_full | 1.0 | `mbpp/362` max_occurrences |
| `mbpp/693` remove_multiple_spaces | mbpp_full | 1.0 | `mbpp/892` remove_spaces |
| `mbpp/703` is_key_present | mbpp_full | 1.0 | `mbpp/405` check_tuplex |
| `mbpp/800` remove_all_spaces | mbpp_sanitized | 1.0 | `mbpp/451` remove_whitespaces |
| `mbpp/825` access_elements | mbpp_full | 1.0 | `mbpp/49` specified_element |
| `mbpp/842` get_odd_occurence | mbpp_full | 1.0 | `mbpp/29` get_Odd_Occurrence |
| `mbpp/966` remove_empty | mbpp_full | 1.0 | `mbpp/361` remove_empty |
| `mbpp/752` jacobsthal_num | mbpp_sanitized | 0.912 | `mbpp/529` jacobsthal_lucas |
| `mbpp/383` even_bit_toggle_number | mbpp_full | 0.911 | `mbpp/155` even_bit_toggle_number |
| `mbpp/855` check_Even_Parity | mbpp_full | 0.902 | `mbpp/768` check_Odd_Parity |
| `mbpp/373` volume_cuboid | mbpp_full | 0.9 | `mbpp/52` parallelogram_area |
| `mbpp/942` check_element | mbpp_full | 0.895 | `mbpp/808` check_K |
| `mbpp/442` positive_count | mbpp_sanitized | 0.893 | `mbpp/530` negative_count |
| `mbpp/168` frequency | mbpp_sanitized | 0.889 | `mbpp/446` count_Occurrence |
| `mbpp/202` remove_even | mbpp_full | 0.889 | `mbpp/437` remove_odd |
| `mbpp/564` count_Pairs | mbpp_sanitized | 0.885 | `mbpp/841` get_inv_count |
| `mbpp/831` count_Pairs | mbpp_full | 0.885 | `mbpp/841` get_inv_count |
| `mbpp/964` word_len | mbpp_full | 0.872 | `mbpp/79` word_len |
| `mbpp/327` check_isosceles | mbpp_full | 0.871 | `mbpp/51` check_equilateral |
| `mbpp/682` mul_list | mbpp_full | 0.867 | `mbpp/865` ntimes_list |
| `mbpp/440` find_adverb_position | mbpp_sanitized | 0.86 | `mbpp/584` find_adverbs |
| `mbpp/345` diff_consecutivenums | mbpp_full | 0.854 | `mbpp/503` add_consecutive_nums |
| `mbpp/389` find_lucas | mbpp_sanitized | 0.854 | `mbpp/960` get_noOfways |
| `mbpp/690` mul_consecutive_nums | mbpp_full | 0.854 | `mbpp/503` add_consecutive_nums |
| `mbpp/517` largest_pos | mbpp_full | 0.846 | `mbpp/443` largest_neg |
| `mbpp/726` multiply_elements | mbpp_sanitized | 0.842 | `mbpp/206` concatenate_elements |
| `mbpp/290` max_length | mbpp_sanitized | 0.838 | `mbpp/528` min_length |
| `mbpp/412` remove_odd | mbpp_sanitized | 0.838 | `mbpp/824` remove_even |
| `mbpp/888` substract_elements | mbpp_full | 0.838 | `mbpp/401` add_nested_tuples |
| `mbpp/452` loss_amount | mbpp_sanitized | 0.833 | `mbpp/954` profit_amount |
| `mbpp/875` min_difference | mbpp_full | 0.833 | `mbpp/63` max_difference |
| `mbpp/913` end_num | mbpp_full | 0.833 | `mbpp/900` match_num |
| `mbpp/258` count_odd | mbpp_full | 0.829 | `mbpp/697` count_even |
| `mbpp/807` first_odd | mbpp_sanitized | 0.829 | `mbpp/221` first_even |
| `mbpp/928` change_date_format | mbpp_full | 0.826 | `mbpp/427` change_date_format |
| `mbpp/932` remove_duplic_list | mbpp_full | 0.825 | `mbpp/436` neg_nos |
| `mbpp/285` text_match_two_three | mbpp_sanitized | 0.824 | `mbpp/16` text_lowercase_underscore |
| `mbpp/917` text_uppercase_lowercase | mbpp_full | 0.824 | `mbpp/43` text_match |
| `humaneval/63` fibfib | humaneval | 0.822 | `mbpp/960` get_noOfways |
| `mbpp/294` max_val | mbpp_sanitized | 0.818 | `mbpp/410` min_val |
| `mbpp/399` bitwise_xor | mbpp_sanitized | 0.818 | `mbpp/197` find_exponentio |
| `mbpp/429` and_tuples | mbpp_sanitized | 0.818 | `mbpp/197` find_exponentio |
| `mbpp/731` lateralsurface_cone | mbpp_sanitized | 0.816 | `mbpp/497` surfacearea_cone |
| `mbpp/402` ncr_modp | mbpp_full | 0.813 | `mbpp/952` nCr_mod_p |
| `mbpp/41` filter_evennumbers | mbpp_full | 0.812 | `mbpp/426` filter_oddnumbers |
| `mbpp/282` sub_list | mbpp_sanitized | 0.812 | `mbpp/358` moddiv_list |
| `mbpp/919` multiply_list | mbpp_full | 0.812 | `mbpp/925` mutiple_tuple |
| `mbpp/411` snake_to_camel | mbpp_sanitized | 0.811 | `mbpp/102` snake_to_camel |
| `mbpp/573` unique_product | mbpp_sanitized | 0.811 | `mbpp/925` mutiple_tuple |
| `mbpp/654` rectangle_perimeter | mbpp_full | 0.808 | `mbpp/369` lateralsurface_cuboid |
| `mbpp/645` find_k_product | mbpp_full | 0.806 | `mbpp/611` max_of_nth |
| `mbpp/549` odd_Num_Sum | mbpp_full | 0.804 | `mbpp/655` fifth_Power_Sum |
| `mbpp/770` odd_num_sum | mbpp_sanitized | 0.804 | `mbpp/655` fifth_Power_Sum |
