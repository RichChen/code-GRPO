# CHECKPOINT-1D：冻结题集、诊断扫描、DeepSeek 参考上限

## 结论

- **题集和环境都已冻结。**
  - 题集打了 tag `phase1-tasks-v1`（`436d8db`），用的是 `812d5ce` 构建的 4732 道题，没有重建。
  - 环境改动（提醒一次、`max_turns` 改为 10、system prompt 加一段 JSON 格式说明）提交后打了 tag `phase1-env-v1`（`14c155e`）。
  - 打 `phase1-env-v1` 时全量测试 210 passed；加上假阴性分析的测试后，本次提交 211 passed。
- **本地诊断扫描**：5 个模型跑完 4 个；`Qwen3-4B-Instruct-2507` 在 fp16 下显存不够，服务起不来，按规定跳过。四个模型里：
  - **`Qwen3-1.7B`（关闭思考模式）明显最好**：解出率 17.7%，格式错误率 2.1%，74% 的 episode 至少成功编辑过一次。
  - `Qwen2.5-3B-Instruct` 次之：解出率 7.2%。但它很慢（每小时 829 个 episode），而且一半的 episode 跑满了 10 轮。
  - 两个 1.5B 模型（Qwen2.5-1.5B 和 Qwen2.5-Coder-1.5B）仍然是 0%，`edit_file` 一次都没成功过。
- **DeepSeek 参考上限**：
  - 100 题 × 2 次，解出率 89.5%。解出的 episode 轮数中位数 5，p90 是 6，没有达到 9。
  - 格式错误率 0.1%。
  - 花费 1.28 元，平均每个 episode 0.0064 元。在训练题上拿到 1000 条成功轨迹，估计约 7.6 元。
  - 这说明这些题在 10 轮内是解得出来的，用来采 SFT 轨迹的链路也通了。
- **判分假阴性**：
  - 小模型抽查的 10 个都是模型真的没修对。
  - DeepSeek 没拿到奖励的 21 个 episode 里，有 5 个只挂在生成用例上；人工判断其中 **3 个是假阴性**，占全部 200 个 episode 的 1.5%。
  - 原因不是生成的输入越界，而是 **MBPP 原始代码本身有潜在 bug**：模型修好了我们注入的 bug，又顺手把原代码的毛病也改了，于是在生成的边界输入上和原始代码对不上。
  - 这是第 5b 步没预料到的一类风险，详见第 4 节。

## 第 1 步：冻结题集和环境 v1

| 项 | 结果 |
| --- | --- |
| 审计修订 4 原样补进计划第 6 步末尾，单独提交 | `5358e63`（同时提交了 `.env.example`，`.gitignore` 加了 `.env`） |
| manifest 写入 `frozen`（tag、构建 commit、sha256），打 tag `phase1-tasks-v1` | `436d8db`。train `f0038c46…aa82`，heldout `d28e0709…c3d4`，和磁盘文件核对一致 |
| README 补上生成题集、接入新 policy、启动推理服务三节 | 分两次提交，每个 tag 上的 README 只描述那个 tag 里已有的代码 |
| 环境修改，打 tag `phase1-env-v1` | `14c155e` |

环境修改的细节：

- **提醒一次**：回复里没有工具调用、并且还没成功编辑过文件时，环境追加原文照录的 `NUDGE_MESSAGE`，每个 episode 最多一次。第二次没有工具调用视为提交；成功编辑过以后，没有工具调用直接视为提交。
  - "提醒占 1 轮"的实现：触发提醒的那条回复计入 `turns`，提醒消息本身不另外计数。
  - `EpisodeResult.nudged` 默认 False，旧记录照样能读。
- **JSON 说明**：39 个英文词，追加在 prompt 末尾，原有内容不改，parser 不放宽。
- **轮数和上下文**：`max_turns` 默认 10；上下文上限 8k 不变。
- **新增测试**：
  - 提醒后继续并修好；
  - 第二次没有工具调用时视为提交；
  - 提醒计入轮数；
  - 编辑成功后不提醒；
  - 编辑失败不算编辑过；
  - `nudged` 的序列化，以及旧记录兼容；
  - JSON 说明不超过 60 词。

## 第 2、3 步：诊断扫描和 DeepSeek 汇总

题目是试点那 100 道（seed 0，train 和 heldout 各 50 道），temperature 0.8，环境 `phase1-env-v1`，客户端并发 8。本地模型每题 4 次，DeepSeek 每题 2 次。

| 指标 | Qwen2.5-1.5B | Qwen2.5-Coder-1.5B | Qwen2.5-3B | Qwen3-1.7B（不思考） | Qwen3-4B-2507 | DeepSeek flash |
| --- | --- | --- | --- | --- | --- | --- |
| episode 数 | 400 | 400 | 400 | 400 | 跳过（显存不够） | 200 |
| **解出率** | 0% | 0% | 7.2% | **17.7%** | — | **89.5%** |
| 按解出次数的题数（0/1/2/3/4） | 100/0/0/0/0 | 100/0/0/0/0 | 78/16/5/1/0 | 75/5/4/6/10 | — | 7/7/86（K=2：0/1/2） |
| 解出 1–3 次的题 | 0 | 0 | 22 | 15 | — | — |
| 格式错误率（按工具调用） | 37.6% | 100% | 11.4% | 2.1% | — | 0.1% |
| 有格式错误的 episode | 29.2% | 54.0% | 37.0% | 3.0% | — | 0.5% |
| 第一轮就不调工具 | 42.2% | 46.8% | 0% | 2.8% | — | 0% |
| 被提醒 | 99.8% | 74.5% | 56.5% | 25.5% | — | 3.0% |
| 提醒后又开始调工具（占被提醒的） | 34.3% | 5.0% | 85.4% | 77.5% | — | 100% |
| `edit_file` 成功率 | 0/41 | 0/0 | 34.8%（446 次） | 52.6%（591 次） | — | 100%（211 次） |
| 至少成功编辑一次的 episode | 0% | 0% | 33.8% | 73.8% | — | 97.5% |
| 至少一次 `visible_pass` 的 episode | 0% | 0% | 8.5% | 21.0% | — | 93.5% |
| 平均轮数 | 3.24 | 4.58 | 8.62 | 5.37 | — | 5.26 |
| 停止原因（提交/跑满轮数/上下文超长） | 399/1/0 | 294/106/0 | 205/194/1 | 352/48/0 | — | 190/10/0 |
| 解出的 episode 轮数中位数 / p90 | — | — | 8 / 10 | 4 / 5 | — | 5 / 6 |
| 平均生成 token / 最终上下文 token | 379 / 1537 | 424 / 1635 | 924 / 2487 | 286 / 1779 | — | 430 / 2100 |
| 吞吐量（episode/小时） | 3332 | 2718 | 829 | 2126 | — | 5229 |
| 显存占用（vLLM 预分配后，含 Windows 约 0.7 GiB） | 10382 MiB | 10065 MiB | 9980 MiB | 9854 MiB | 起不来 | — |
| 权重 / KV cache | 2.98 GiB / 210k token | 2.98 GiB / 198k token | 5.79 GiB / 65k token | 3.22 GiB / 46k token | 7.64 GiB / 只剩 0.33 GiB | — |
| 服务启动耗时 | 81 秒 | 115 秒 | 110 秒 | 135 秒 | — | — |
| 花费 | — | — | — | — | — | 1.28 元（每个 episode 0.0064 元） |

补充说明：

- "至少一次 `visible_pass`"的统计口径：最终判分时可见测试通过，或者 episode 里有过一次退出码为 0 的 `run_tests`。两种口径算出来几乎一样，只有 Qwen2.5-3B 的最终通过率略低，为 8.0%。
- 按 bug 类型的解出率：
  - Qwen3-1.7B：var_swap 28.9%、arith_swap 21.7%、compare_swap 16.7%、const_off_by_one 15.0%、remove_branch 0%；bool_swap 只有 4 个 episode，0%。
  - Qwen2.5-3B：compare_swap 13.9%、arith_swap 13.3%、var_swap 10.2%，其余 ≤2%。
- 按划分的解出率：Qwen3-1.7B 训练集 20.0%、留出集 15.5%；DeepSeek 训练集 87%、留出集 92%。
- Qwen3-1.7B 有 10 道题 4 次全解出、75 道 4 次全没解出，分布两头集中。
- 四个本地模型都没有出现乱码、NaN 或输出异常。fp16 下 Qwen3 也正常。

### 各模型的推理服务

所有模型都用同一条命令，只换模型名：

```bash
HF_HUB_OFFLINE=1 ~/.venvs/vllm/bin/vllm serve <model> --dtype half --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.85 --seed 0
```

| 模型 | HF 版本 | parser | 依据 |
| --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4c` | hermes | vLLM 文档"Qwen Models"一节：`Qwen/Qwen2.5-*` 用 `--tool-call-parser hermes` |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | `2e1fd397ee46` | hermes | 同上；聊天模板里有 `<tool_call>` 格式说明（已核对） |
| Qwen/Qwen2.5-3B-Instruct | `aa8e72537993` | hermes | 同上 |
| Qwen/Qwen3-1.7B | `70d244cc86cc` | hermes | Qwen 官方文档 function_call 一节：Qwen3 的模板内置 Hermes 风格，`vllm serve ... --tool-call-parser hermes`。请求里传 `chat_template_kwargs={"enable_thinking": false}` |
| Qwen/Qwen3-4B-Instruct-2507 | `cdbee75f17c0` | hermes | 同上（Qwen3-Coder 才用 `qwen3_xml`） |

- 五个模型 ID 下载前都在 Hugging Face API 上确认过。
- 每个模型跑完后，权重缓存都删掉了，包括 `hub/blobs/` 下的共享文件，现在 `~/.cache/huggingface` 只剩 2 MB。
- `--gpu-memory-utilization` 本来想统一用 0.90，但 Windows 自己占了约 0.7 GiB，0.90 起不来，所以统一用 0.85，也就是第 6 步文档里的值。

**Qwen3-4B-Instruct-2507 跳过的原因**：fp16 权重 7.64 GiB，在 0.85 的显存上限里只给 KV cache 留下 0.33 GiB，而 8192 上下文至少需要 1.12 GiB。vLLM 报错 `estimated maximum model length is 2368`。按规定记下原因、跳过，没有改量化方式，也没有改 `--max-model-len`。
- 我估算过，如果同时开 `--enforce-eager`（省掉 CUDA graph 约 0.42 GiB）并把显存上限提到能用的最大值 0.89（多出约 0.44 GiB），KV 大约有 1.19 GiB，刚刚够一个 8k 请求，几乎没有并发余量。这个方案没有试，由审计方决定。

**Qwen2.5-Coder-1.5B 格式错误率 100% 的原因**：1238 次工具调用全部写成 Markdown 的 ```json 代码块，一次都没用 `<tool_call>` 标签。它的聊天模板明确要求 `<tool_call>` 格式，vLLM 文档对 Qwen2.5-* 指定的也是 hermes，所以问题在模型，不在 parser。按修订 4，parser 保持严格。

### DeepSeek 参考上限（第 3 步）

- 配置：`.env` 里的 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL=deepseek-flash`，请求体带 `"thinking": {"type": "disabled"}`，`max_tokens` 1024。WSL 里直连可以访问 DeepSeek，所以 `trust_env=false`，没读 `HTTPS_PROXY`。
- 第一次调用前，用 1 条最短请求（"Reply with ok."）验证了密钥和模型名：返回 `ok`，花费 0.000024 元。
- 解出率 89.5%（179/200）：训练集 87%，留出集 92%。按解出次数：2 次 86 题，1 次 7 题，0 次 7 题。
- 解出的 episode 轮数分布：4 轮 56、5 轮 99、6 轮 8、7 轮 8、8 轮 4、9 轮 1、10 轮 3。**中位数 5，p90 是 6**，没有达到 9 轮。另外有 10 个 episode 跑满 10 轮，其中 2 个解出。
- 格式错误率 0.1%（898 次工具调用里有 1 次 `invalid_json`）。
- 花费：
  - 总计 1.2783 元（另有预检 0.000024 元），平均每个 episode 0.0064 元。
  - prompt token 里约 85% 命中缓存。每个 episode 平均 8844 prompt token，其中命中 7518，生成 430 token。
  - 累计花费记在 `runs/deepseek/cost.json`，上限 10 元，没有触发。
- **估算 1000 条成功轨迹的花费**：训练集上每个 episode 0.0066 元，解出率 87%，即每条成功轨迹约 0.0076 元。1000 条约需 1150 个 episode，**约 7.6 元**。这个估算假设训练题的整体难度和这 50 道训练题相当。
- 密钥安全：
  - 密钥只在 `.env` 里；
  - 搜了 `runs/`、`docs/`、`src/`、`configs/`、`scripts/`、`tests/`，都没有密钥，也没有 `Authorization` 字样；
  - 每次提交前都按规定对工作区和暂存区做密钥前缀检查，并用真实密钥值再搜一遍。真实密钥值的命中一直是 0；前缀检查只命中第 6 节第 4 条列出的几处已入库的非密钥文字。

## 第 4 步：判分假阴性

条件：没拿到奖励，所有 `test` 和 `challenge` 隐藏用例都过了，只有 `generated` 用例失败。

我额外加了一个条件："改过 `pkg/` 下的文件"。原因是：可见测试用到的那条用例不在隐藏用例里，有 713 道题（15%）的 bug 版本来就能通过全部隐藏 test 用例，只有生成用例能抓住 bug。这类题如果模型什么都没改，也满足字面条件，但显然不算假阴性。两个数都列出来：

| 模型 | episode | 没拿到奖励 | 只有生成用例失败（字面条件） | 其中改过 `pkg/` | 占没拿到奖励的比例 |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | 400 | 400 | 64 | 0 | 0% |
| Qwen2.5-Coder-1.5B | 400 | 400 | 64 | 0 | 0% |
| Qwen2.5-3B | 400 | 371 | 59 | 6 | 1.6% |
| Qwen3-1.7B | 400 | 329 | 25 | 12 | 3.6% |
| DeepSeek | 200 | 21 | 5 | 5 | 23.8% |

每个模型抽 5 个（不足 5 个全列，seed 0）。模型的 diff 是重放成功的 `edit_file` 调用复原出来的，失败用例的实际返回值是在沙箱里重新跑出来的。完整 diff 见附录 A。人工判断如下：

| 模型 | 题目 | 判断 |
| --- | --- | --- |
| Qwen2.5-3B | mbpp/914 `is_Two_Alter` | 模型修错：改成了切片比较，`'aba'` 应该返回 True |
| Qwen2.5-3B | mbpp/84 `sequence` | 模型修错：改了递归式，真正的 bug（`n == 2 or n == 2`）没动，`sequence(1)` 无限递归 |
| Qwen2.5-3B | mbpp/16 `text_lowercase_underscore` | 模型修错：`re.search(text.lower(), text)`，`'a'` 应该返回 False |
| Qwen2.5-3B | mbpp/482 `match` | 模型改错了函数（改的是干扰函数 `check_Equality`），bug 没修 |
| Qwen2.5-3B | mbpp/589 `perfect_squares` | 模型改错了函数（改的是 `get_perrin`），bug 没修 |
| Qwen3-1.7B | mbpp/309 `maximum` | 模型修错：`else: pass` 没动 |
| Qwen3-1.7B | mbpp/589 `perfect_squares`（2 个 episode） | 模型修错：加了无关的一行，`range(a, b - 1)` 没动 |
| Qwen3-1.7B | mbpp/236 `No_of_Triangle` | 模型修错：`// 1` 没改 |
| Qwen3-1.7B | mbpp/838 `min_Swaps` | 模型修错：`% 3` 没改 |
| DeepSeek | mbpp/850 `is_triangleexists` #1 | 模型修错：删掉了 `a + b + c == 180` 这个条件，`(90, 46, 45)` 应该返回 False |
| DeepSeek | mbpp/850 `is_triangleexists` #0 | 模型修错：`179 → 180` 修对了，但又把 `or` 改成了 `and`。按 MBPP 850 的题意，a、b、c 是角度，三个正角之和为 180 就能构成三角形，`(93, 42, 45)` 应该返回 True。原始代码里那句 `or` 判断对正数总是成立，是多余的，但不算错；模型改成 `and` 以后，把三边不等式套到了角度上，这才错了 |
| DeepSeek | mbpp/236 `No_of_Triangle`（2 个 episode） | **假阴性**：`// 1 → // 2` 修对了，又加了 `N - 2K + 1 > 0` 的保护。`(4, 4)` 时原始代码算的是 `(-3)(-2)/2 = 3`，加上 1，返回 4；按题意边长 4 的大三角形里只有 1 个边长 4 的三角形，模型返回 1 |
| DeepSeek | mbpp/511 `find_Min_Sum` | **假阴性**：`-= → +=` 修对了，又改成整除并加了 `if num > 1`。`find_Min_Sum(9)` 的最小因子和是 3 + 3 = 6；原始代码对完全平方数会多加一个 1.0，返回 7.0 |

结论：

- **小模型的 10 个全部是模型确实没修对**（改错了地方或者根本没修）。以这些模型现在的能力，生成用例带来的假阴性可以忽略：Qwen3-1.7B 的上限是 12/329，抽查的 5 个都不是假阴性。
- **DeepSeek 的 5 个里有 3 个是假阴性**（mbpp/236 两个、mbpp/511 一个），占它 200 个 episode 的 1.5%；如果算上，解出率是 91.0%。另外 2 个都在 mbpp/850 上，是模型自己修错了。这些都**不是**"生成的输入越界"（`find_Extra` 那类），而是 MBPP 原始代码在某些合法输入上本身就是错的，`test` 用例没覆盖到，生成用例覆盖到了。模型越强，越可能顺手把这些潜在 bug 也"修正"掉，然后被判失败。
- 对后续的影响：
  - 阶段 2 用 DeepSeek 采 SFT 轨迹时，只取拿到奖励的轨迹，这类题会被少采一些，不会采进错误的轨迹，影响不大。
  - 阶段 3 训练到策略变强之后，这类假阴性会惩罚"写得更对"的修复，需要审计方决定是否处理（见第 7 节）。

## 第 5 节：环境修改前后 Qwen2.5-1.5B 的对比

| 指标 | 旧环境（CHECKPOINT-1C 试点，8 轮，不提醒） | 环境 v1（10 轮，提醒一次，JSON 说明） |
| --- | --- | --- |
| 解出率 | 0% | 0% |
| 至少调用过一次工具的 episode | 48.7% | 75.5% |
| 第一轮就不调工具 | 51.2% | 42.2% |
| 被提醒 / 提醒后又开始调工具 | — | 99.8% / 34.3% |
| 工具调用次数 | 354 | 505 |
| 格式错误率（按工具调用） | 37.0% | 37.6% |
| 没被解析出来的工具调用 | 118 | 175 |
| `edit_file` 成功 | 0/14 | 0/41 |
| 平均轮数 | 1.85 | 3.24 |
| 平均生成 token | 220 | 379 |
| 吞吐量（episode/小时） | 5556 | 3332 |

结论：

- 提醒确实让这个模型多调了工具：至少调过一次工具的 episode 从 48.7% 升到 75.5%，编辑尝试从 14 次增加到 41 次。
- JSON 说明对格式错误率没有作用（37.0% → 37.6%）。
- 编辑仍然全部失败，解出率仍是 0%。
- 环境修改不能弥补这个模型的能力不足。

## 第 6 节：遇到的问题

1. **第一次跑扫描时，把 `--gpu-memory-utilization` 设成了 0.90**，vLLM 因为空闲显存只有 9.87 GiB 而拒绝启动，脚本又在失败后删掉了权重。之后改成：只有扫描跑完才删权重；显存上限统一用 0.85。
2. **删不干净的权重缓存**：这个版本的 `hf` 把大文件放在共享的 `hub/blobs/` 下，模型目录里只有符号链接。只删模型目录不会释放空间，所以 Qwen2.5-1.5B 重新"下载"只用了 5 秒。脚本改成顺着链接删除实际文件。这几个模型没有共用的大文件，所以删除不会误伤其他模型。Qwen3-4B 的服务没起来，脚本没有删它的权重，是我手动删掉的。
3. **下载卡住**：`hf` 默认的 xet 传输经过代理时多次停住（进程在，但写入 0 字节），直连 Hugging Face 不通。
   - 改用 `HF_HUB_DISABLE_XET=1` 走普通 HTTP（会从 `.incomplete` 断点续传），并写了一个"3 分钟没写入就重启"的下载脚本 `scripts/fetch_models.sh`。
   - 改完后三个模型并行，合计约 3–4 MB/s，三个模型共下载了约 78 分钟。
   - 模型都是 Hugging Face 官方仓库，经代理下载。
4. **密钥前缀检查的误报**：按规定对工作区做的前缀检查（`git grep -n`）会命中三处已入库的非密钥文字：
   - PHASE-1E prompt 里描述这条检查的原文；
   - PROGRESS 里对这件事的记录；
   - `grader.py` 第 34 行注释里的一个英文连字符复合词（"task" 加 "validation"）。

   修订 4 那次提交是在确认这些不是密钥之后提交的。之后每次提交都加做真实密钥值搜索，结果都是 0；暂存区的前缀检查也保持 0 命中。为此，测试里的假密钥改成了 `fake-secret-…`，报告里也不再写出这个前缀。
5. **判分假阴性的新来源**：MBPP 原始代码的潜在 bug（见第 4 节），第 5b 步的分析没有预料到。

## 第 7 节：需要审计方决定的事

以下我都没有执行：

1. **选基座模型**：
   - 能在这块卡上跑起来的四个模型里，`Qwen3-1.7B`（不思考）的各项指标都最好：
     - 解出率 17.7%；
     - 格式错误率 2.1%；
     - 编辑成功率 52.6%；
     - 解出的 episode 中位数只要 4 轮；
     - 吞吐量 2126 episode/小时。
   - `Qwen2.5-3B` 解出率 7.2%，但慢 2.6 倍，一半的 episode 跑满轮数。
   - 两个 1.5B 模型不可用。
   - `Qwen3-4B-Instruct-2507` 需要审计方决定：是否允许 `--enforce-eager` 加更高的显存上限（估算刚好够一个 8k 请求），或者其他做法；也可以就此放弃。
2. **生成用例里的"原始代码潜在 bug"**：可选做法：
   - (a) 不处理；
   - (b) 用 DeepSeek 在全部题上各跑 1–2 次，把"只挂在生成用例上、并且修复人工确认更对"的来源函数找出来，删掉它们的生成用例或者整题；
   - (c) 只在阶段 3 评估时，把"只挂在生成用例上"的情况单独统计。
3. **阶段 2 的 SFT 轨迹**：按本次的单价，1000 条成功轨迹约 7.6 元，花费上限需要写进阶段 2 的计划。

## 当前状态

- 代码在 `phase1-env-v1`（`14c155e`）之后，又加了诊断脚本（`scripts/diag_model.sh`、`scripts/fetch_models.sh`）、每个模型的客户端配置（`configs/diag/`）和判分假阴性分析（`arl.rollout.falseneg`），和本报告一起提交。环境代码没有再改。
- 原始结果在 `runs/diag/`、`runs/deepseek/`、`runs/falseneg/`，`runs/` 不进 git。
- 没有写训练代码，没有跑全量校准，没有开始阶段 2。vLLM 服务已关，GPU 空闲，模型权重缓存已清空。

---

## 附录 A：判分假阴性抽样的完整 diff

由 `uv run python -m arl.rollout.falseneg runs/diag/*.jsonl runs/deepseek/deepseek.jsonl --out runs/falseneg` 生成。每个样例包括：模型的 diff、参考答案的 diff、失败的生成用例（最多 3 条），以及模型代码的实际返回值。

#### qwen2.5-3b: `heldout-mbpp-914-const_off_by_one-52` #3 (mbpp/914, const_off_by_one; 2/130 generated cases fail)

Model diff:

```diff
--- a/pkg/misc.py
+++ b/pkg/misc.py
@@ -1,5 +1,5 @@
 def is_Two_Alter(s):
     for i in range(len(s) - 2):
-        if s[i] != s[i + 3]:
+        if s[i:i+2] != s[i+2:i+4]:
             return False
     if s[0] == s[1]:
```

Reference fix:

```diff
--- a/pkg/misc.py
+++ b/pkg/misc.py
@@ -1,5 +1,5 @@
 def is_Two_Alter(s):
     for i in range(len(s) - 2):
-        if s[i] != s[i + 3]:
+        if s[i] != s[i + 2]:
             return False
     if s[0] == s[1]:
```

- `pkg.misc.is_Two_Alter('aba')` expects `result == True`, got `False`
- `pkg.misc.is_Two_Alter('ababa')` expects `result == True`, got `False`

#### qwen2.5-3b: `heldout-mbpp-84-const_off_by_one-14` #2 (mbpp/84, const_off_by_one; 10/130 generated cases fail)

Model diff:

```diff
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -40,5 +40,5 @@
         return 1
     else:
-        return sequence(sequence(n - 1)) + sequence(n - sequence(n - 1))
+        return sequence(n - 1) + sequence(n - sequence(n - 1))
 
 
```

Reference fix:

```diff
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -37,5 +37,5 @@
 
 def sequence(n):
-    if n == 2 or n == 2:
+    if n == 1 or n == 2:
         return 1
     else:
```

- `pkg.core.sequence(1)` expects `result == 1`, got `raises RecursionError`
- `pkg.core.sequence(13)` expects `result == 8`, got `12`
- `pkg.core.sequence(5)` expects `result == 3`, got `4`

#### qwen2.5-3b: `train-mbpp-16-var_swap-12` #1 (mbpp/16, var_swap; 1/139 generated cases fail)

Model diff:

```diff
--- a/pkg/routines.py
+++ b/pkg/routines.py
@@ -19,5 +19,5 @@
 def text_lowercase_underscore(text):
     patterns = '^[a-z]+_[a-z]+$'
-    if re.search(text, text):
+    if re.search(text.lower(), text):
         return True
     else:
```

Reference fix:

```diff
--- a/pkg/routines.py
+++ b/pkg/routines.py
@@ -19,5 +19,5 @@
 def text_lowercase_underscore(text):
     patterns = '^[a-z]+_[a-z]+$'
-    if re.search(text, text):
+    if re.search(patterns, text):
         return True
     else:
```

- `pkg.routines.text_lowercase_underscore('a')` expects `result == False`, got `True`

#### qwen2.5-3b: `heldout-mbpp-482-var_swap-12` #2 (mbpp/482, var_swap; 8/160 generated cases fail)

Model diff:

```diff
--- a/pkg/misc.py
+++ b/pkg/misc.py
@@ -4,5 +4,5 @@
 def check_Equality(str):
     if str[0] == str[-1]:
-        return 'Equal'
+        return 'No'
     else:
         return 'Not Equal'
```

Reference fix:

```diff
--- a/pkg/common.py
+++ b/pkg/common.py
@@ -6,5 +6,5 @@
 def match(text):
     pattern = '[A-Z]+[a-z]+$'
-    if re.search(text, text):
+    if re.search(pattern, text):
         return 'Yes'
     else:
```

- `pkg.common.match('gekes')` expects `result == 'No'`, got `'Yes'`
- `pkg.common.match('geeksfo')` expects `result == 'No'`, got `'Yes'`
- `pkg.common.match('geks')` expects `result == 'No'`, got `'Yes'`

#### qwen2.5-3b: `heldout-mbpp-589-arith_swap-20` #3 (mbpp/589, arith_swap; 2/129 generated cases fail)

Model diff:

```diff
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -6,6 +6,5 @@
     if n == 2:
         return 2
-    return get_perrin(n - 2) + get_perrin(n - 3)
-
+    return get_perrin(n - 2) + get_perrin(n - 3) if n >= 2 else n
 
 def multi_list(rownum, colnum):
```

Reference fix:

```diff
--- a/pkg/funcs.py
+++ b/pkg/funcs.py
@@ -6,5 +6,5 @@
 def perfect_squares(a, b):
     lists = []
-    for i in range(a, b - 1):
+    for i in range(a, b + 1):
         j = 1
         while j * j <= i:
```

- `pkg.funcs.perfect_squares(98, 197)` expects `result == [100, 121, 144, 169, 196]`, got `[100, 121, 144, 169]`
- `pkg.funcs.perfect_squares(51, 100)` expects `result == [64, 81, 100]`, got `[64, 81]`

#### qwen3-1.7b: `heldout-mbpp-309-remove_branch-7` #1 (mbpp/309, remove_branch; 6/130 generated cases fail)

Model diff:

```diff
--- a/pkg/helpers.py
+++ b/pkg/helpers.py
@@ -31,5 +31,5 @@
 def maximum(a, b):
     if a >= b:
-        return a
+        return max(a, b)
     else:
         pass
```

Reference fix:

```diff
--- a/pkg/helpers.py
+++ b/pkg/helpers.py
@@ -33,3 +33,3 @@
         return a
     else:
-        pass
+        return b
```

- `pkg.helpers.maximum(-10, 7)` expects `result == 7`, got `None`
- `pkg.helpers.maximum(-1, 2)` expects `result == 2`, got `None`
- `pkg.helpers.maximum(-9, 4)` expects `result == 4`, got `None`

#### qwen3-1.7b: `heldout-mbpp-589-arith_swap-20` #3 (mbpp/589, arith_swap; 2/129 generated cases fail)

Model diff:

```diff
--- a/pkg/funcs.py
+++ b/pkg/funcs.py
@@ -13,4 +13,5 @@
             j = j + 1
         i = i + 1
+        j = j + 1
     return lists
 
```

Reference fix:

```diff
--- a/pkg/funcs.py
+++ b/pkg/funcs.py
@@ -6,5 +6,5 @@
 def perfect_squares(a, b):
     lists = []
-    for i in range(a, b - 1):
+    for i in range(a, b + 1):
         j = 1
         while j * j <= i:
```

- `pkg.funcs.perfect_squares(98, 197)` expects `result == [100, 121, 144, 169, 196]`, got `[100, 121, 144, 169]`
- `pkg.funcs.perfect_squares(51, 100)` expects `result == [64, 81, 100]`, got `[64, 81]`

#### qwen3-1.7b: `heldout-mbpp-236-const_off_by_one-38` #1 (mbpp/236, const_off_by_one; 2/90 generated cases fail)

Model diff:

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -13,5 +13,5 @@
         Tri_up = 0
         Tri_up = (N - K + 1) * (N - K + 2) // 2
-        Tri_down = 0
+        Tri_down = 0 if K == 0 else (N - 2 * K + 1) * (N - 2 * K + 2) // 1
         Tri_down = (N - 2 * K + 1) * (N - 2 * K + 2) // 1
         return Tri_up + Tri_down
```

Reference fix:

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -14,5 +14,5 @@
         Tri_up = (N - K + 1) * (N - K + 2) // 2
         Tri_down = 0
-        Tri_down = (N - 2 * K + 1) * (N - 2 * K + 2) // 1
+        Tri_down = (N - 2 * K + 1) * (N - 2 * K + 2) // 2
         return Tri_up + Tri_down
 
```

- `pkg.extras.No_of_Triangle(4, 4)` expects `result == 4`, got `7`
- `pkg.extras.No_of_Triangle(4, 1)` expects `result == 16`, got `22`

#### qwen3-1.7b: `train-mbpp-838-const_off_by_one-94` #0 (mbpp/838, const_off_by_one; 4/120 generated cases fail)

Model diff:

```diff
--- a/pkg/tools.py
+++ b/pkg/tools.py
@@ -15,5 +15,5 @@
         return result
     elif (c0 + c1) % 3 == 0:
-        return result + 2
+        return result + 2 if (c0 + c1) % 3 == 0 else -1
     else:
         return -1
```

Reference fix:

```diff
--- a/pkg/tools.py
+++ b/pkg/tools.py
@@ -14,5 +14,5 @@
     if c0 % 2 == 0 and c1 % 2 == 0:
         return result
-    elif (c0 + c1) % 3 == 0:
+    elif (c0 + c1) % 2 == 0:
         return result + 2
     else:
```

- `pkg.tools.min_Swaps('0001', '0100')` expects `result == 2`, got `-1`
- `pkg.tools.min_Swaps('10001', '01001')` expects `result == 2`, got `-1`
- `pkg.tools.min_Swaps('01010', '01001')` expects `result == 2`, got `-1`

#### qwen3-1.7b: `heldout-mbpp-589-arith_swap-20` #0 (mbpp/589, arith_swap; 2/129 generated cases fail)

Model diff:

```diff
--- a/pkg/funcs.py
+++ b/pkg/funcs.py
@@ -13,4 +13,5 @@
             j = j + 1
         i = i + 1
+        j = 1
     return lists
 
```

Reference fix:

```diff
--- a/pkg/funcs.py
+++ b/pkg/funcs.py
@@ -6,5 +6,5 @@
 def perfect_squares(a, b):
     lists = []
-    for i in range(a, b - 1):
+    for i in range(a, b + 1):
         j = 1
         while j * j <= i:
```

- `pkg.funcs.perfect_squares(98, 197)` expects `result == [100, 121, 144, 169, 196]`, got `[100, 121, 144, 169]`
- `pkg.funcs.perfect_squares(51, 100)` expects `result == [64, 81, 100]`, got `[64, 81]`

#### deepseek: `train-mbpp-850-const_off_by_one-29` #1 (mbpp/850, const_off_by_one; 4/120 generated cases fail)

Model diff:

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -1,5 +1,5 @@
 def is_triangleexists(a, b, c):
-    if a != 0 and b != 0 and (c != 0) and (a + b + c == 179):
-        if a + b >= c or b + c >= a or a + c >= b:
+    if a != 0 and b != 0 and (c != 0):
+        if a + b >= c and b + c >= a and a + c >= b:
             return True
         else:
```

Reference fix:

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -1,4 +1,4 @@
 def is_triangleexists(a, b, c):
-    if a != 0 and b != 0 and (c != 0) and (a + b + c == 179):
+    if a != 0 and b != 0 and (c != 0) and (a + b + c == 180):
         if a + b >= c or b + c >= a or a + c >= b:
             return True
```

- `pkg.extras.is_triangleexists(90, 46, 45)` expects `result == False`, got `True`
- `pkg.extras.is_triangleexists(93, 42, 45)` expects `result == True`, got `False`
- `pkg.extras.is_triangleexists(90, 47, 45)` expects `result == False`, got `True`

#### deepseek: `heldout-mbpp-236-const_off_by_one-38` #1 (mbpp/236, const_off_by_one; 1/90 generated cases fail)

Model diff:

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -14,5 +14,7 @@
         Tri_up = (N - K + 1) * (N - K + 2) // 2
         Tri_down = 0
-        Tri_down = (N - 2 * K + 1) * (N - 2 * K + 2) // 1
+        t = N - 2 * K + 1
+        if t > 0:
+            Tri_down = t * (t + 1) // 2
         return Tri_up + Tri_down
 
```

Reference fix:

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -14,5 +14,5 @@
         Tri_up = (N - K + 1) * (N - K + 2) // 2
         Tri_down = 0
-        Tri_down = (N - 2 * K + 1) * (N - 2 * K + 2) // 1
+        Tri_down = (N - 2 * K + 1) * (N - 2 * K + 2) // 2
         return Tri_up + Tri_down
 
```

- `pkg.extras.No_of_Triangle(4, 4)` expects `result == 4`, got `1`

#### deepseek: `train-mbpp-850-const_off_by_one-29` #0 (mbpp/850, const_off_by_one; 1/120 generated cases fail)

Model diff:

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -1,5 +1,5 @@
 def is_triangleexists(a, b, c):
-    if a != 0 and b != 0 and (c != 0) and (a + b + c == 179):
-        if a + b >= c or b + c >= a or a + c >= b:
+    if a != 0 and b != 0 and (c != 0) and (a + b + c == 180):
+        if a + b >= c and b + c >= a and a + c >= b:
             return True
         else:
```

Reference fix:

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -1,4 +1,4 @@
 def is_triangleexists(a, b, c):
-    if a != 0 and b != 0 and (c != 0) and (a + b + c == 179):
+    if a != 0 and b != 0 and (c != 0) and (a + b + c == 180):
         if a + b >= c or b + c >= a or a + c >= b:
             return True
```

- `pkg.extras.is_triangleexists(93, 42, 45)` expects `result == True`, got `False`

#### deepseek: `heldout-mbpp-236-const_off_by_one-38` #0 (mbpp/236, const_off_by_one; 1/90 generated cases fail)

Model diff:

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -14,5 +14,6 @@
         Tri_up = (N - K + 1) * (N - K + 2) // 2
         Tri_down = 0
-        Tri_down = (N - 2 * K + 1) * (N - 2 * K + 2) // 1
+        if N - 2 * K + 1 > 0:
+            Tri_down = (N - 2 * K + 1) * (N - 2 * K + 2) // 2
         return Tri_up + Tri_down
 
```

Reference fix:

```diff
--- a/pkg/extras.py
+++ b/pkg/extras.py
@@ -14,5 +14,5 @@
         Tri_up = (N - K + 1) * (N - K + 2) // 2
         Tri_down = 0
-        Tri_down = (N - 2 * K + 1) * (N - 2 * K + 2) // 1
+        Tri_down = (N - 2 * K + 1) * (N - 2 * K + 2) // 2
         return Tri_up + Tri_down
 
```

- `pkg.extras.No_of_Triangle(4, 4)` expects `result == 4`, got `1`

#### deepseek: `train-mbpp-511-arith_swap-25` #0 (mbpp/511, arith_swap; 2/72 generated cases fail)

Model diff:

```diff
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -11,8 +11,9 @@
     while i * i <= num:
         while num % i == 0:
-            sum -= i
-            num /= i
+            sum += i
+            num //= i
         i += 1
-    sum += num
+    if num > 1:
+        sum += num
     return sum
 
```

Reference fix:

```diff
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -11,5 +11,5 @@
     while i * i <= num:
         while num % i == 0:
-            sum -= i
+            sum += i
             num /= i
         i += 1
```

- `pkg.core.find_Min_Sum(9)` expects `math.isclose(result, 7.0, rel_tol=1e-06, abs_tol=1e-09)`, got `6`
- `pkg.core.find_Min_Sum(108)` expects `math.isclose(result, 14.0, rel_tol=1e-06, abs_tol=1e-09)`, got `13`
