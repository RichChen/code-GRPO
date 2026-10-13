# agentic-rl 最终报告：用 SFT 和 GRPO 训练 1.7B 模型做多轮工具调用的代码修复

日期：2026-10-13。留出集和训练的数字都来自阶段 4 的复核脚本 `scripts/closeout/verify_heldout.py` 和 `scripts/closeout/verify_training.py`。它们读阶段 3 的原始记录，输出写在 `runs/closeout/`：`verify-heldout.json`、`verify-training.json`、`heldout-tables.md`、`training-tables.md`。其余数字的出处在每处单独写明，路径都相对仓库根目录。`runs/` 不进 git，复核脚本可以从原始记录重新生成这些文件（见第 8 节）。

## 1. 摘要

研究问题：A5 实验里，往 prompt 注入记忆没有提高解题率。那么同类的"读代码、改代码、跑测试"行为，能不能靠训练让一个小模型学会？
做法：用 MBPP 和 HumanEval 生成 4732 道单 bug 修复题，配五个工具、bwrap 沙箱和隐藏测试判分。Qwen3-1.7B 先用 880 条 DeepSeek 轨迹做 LoRA SFT，再自己实现多轮 GRPO 训练 200 步。
留出集 1500 题，每题 4 次。严格解出率：基座 13.9%，SFT-v1 47.1%，GRPO（第 150 步）60.3%。
GRPO 比 SFT 高 13.2 个百分点，95% 区间 [11.7, 14.7]；SFT 比基座高 33.2 个百分点，区间 [30.4, 36.1]。
统计方法是按 273 个 bug 来源函数做聚类配对 bootstrap，重抽 10000 次，seed 0。
结论：在这类合成的单 bug 修复题上，SFT 带来的提升最大，GRPO 在 SFT 基础上又稳定提高了一截，两组区间都远离 0。这个结果只适用于 1.7B 模型和合成题，没有在真实仓库任务上验证。

出处：`runs/closeout/heldout-tables.md`；`verify-heldout.json` 的 `items`（`solve_rate_strict`、`paired_*`）；题集规模见第 3 节。

## 2. 背景和动机

这个项目接在 coding-harness 和 coding-agent 两个项目后面。coding-harness 是单 agent 的运行时和评测框架（工具、策略、预算），coding-agent 负责记忆卡片的提取和检索。两者合起来做了 A5 实验。

A5 的结论（`D:\crq\src\coding-agent\docs\A5-FINAL-REPORT.zh-CN.md` 第 1 节）：在 56 道 SWE-bench Verified 题上，模型是 deepseek-flash，注入同仓库已验证修复的记忆卡片（B2）对比不注入（B0）。B0 解出率 68.5%，B2 − B0 平均配对差 −1.8 个百分点，95% 区间 [−8.3, +4.8]。没有检出记忆带来的效应。

A5 是零结果，这个项目由此开始。研究问题换成：靠 prompt 注入改不了的行为，能不能靠训练学会（`docs/plan/PHASE-1-ENV-AND-TASKS.zh-CN.md` 第 0 节）。A5 留下三条教训，阶段 1 把它们当成硬要求：

- 统计功效主要取决于题量。训练集不少于 500 题，留出集不少于 200 题。
- 判分不能被钻空子。
- 太难或太简单的题没有学习信号，要按模型的解出率筛题。

需要说明的是，这里训练的是"修 bug"这个通用行为，不是 A5 里"利用跨题记忆"的行为。题目和模型规模也都和 A5 不同。所以本项目的结果不能直接回答"A5 的记忆行为能不能训练出来"，见第 7 节。

## 3. 环境和题集

### 3.1 工具和沙箱

- **五个工具**：`read_file`、`search_code`、`list_dir`、`edit_file`、`run_tests`（`src/arl/env/tools.py`）。
  - 不提供 `write_file`，也不提供任意 shell。动作空间小，小模型更容易学，可钻的空子也少。
  - 改代码只能用 `edit_file` 做唯一匹配的替换。跑代码只能通过 `run_tests` 在沙箱里跑 pytest（`README.md`"Agent 工具"一节）。
- **沙箱**是 bwrap（`src/arl/env/sandbox.py`）。
  - 隔离：独立的 user、pid、net、ipc 命名空间；`/usr` 只读；工作区挂到 `/workspace`；根目录重新挂成只读；环境变量用白名单。
  - 资源限制：墙钟 20 秒、CPU 20 秒、地址空间 1024 MB、进程数 64。
  - 仓库的 `src/`、`data/` 在沙箱里看不到。
  - 出处：`docs/plan/PROGRESS.md`"第 2 步：沙箱"。
- **episode 规则**（`docs/plan/CHECKPOINT-1D.md`"第 1 步"）：
  - 最多 10 轮，上下文上限 8k token；
  - 回复里没有工具调用就算提交；
  - 还没成功编辑过文件时，环境提醒一次。

### 3.2 判分

- 隐藏测试拆成两半（`docs/plan/PROGRESS.md`"第 3 步：判分"）：
  - `call` 在沙箱里求值；
  - `check` 带着期望值，只在宿主机上求值。
- 判分前把 agent 的工作区拷一份，删掉所有测试文件和 pytest 配置，换回原始配置后再跑。
- 隐藏用例全部通过才给奖励 1，这就是**严格解出率**。
  - 隐藏用例分三类：MBPP/HumanEval 原有的 `test`、MBPP 的 `challenge`、按规则生成的 `generated`。
  - 只看前两类的口径叫 **test+challenge**。
- 防作弊测试在 `tests/test_grader.py`。以下手段都拿不到奖励：
  - 改可见测试、写劫持用的 `conftest.py`、改 pytest 配置；
  - import 时 `sys.exit`、monkeypatch pytest、往输出里伪造判分记录；
  - 返回 `__eq__` 恒为真的对象、硬编码返回值、用符号链接指向答案。

### 3.3 题集

- **来源**：MBPP（sanitized 393 个函数，完整版补 530 个）和 HumanEval（147 个），共 1070 个来源函数（`docs/plan/PROGRESS.md`"第 5b 步（未完成）"）。断言要能拆成 `call` 和 `check`。
- **6 类 AST 单点变异**（`src/arl/taskgen/mutate.py`）：
  - `arith_swap`、`compare_swap`、`const_off_by_one`、`bool_swap`、`var_swap`、`remove_branch`；
  - 只改入口函数的函数体。
- **组装**：
  - 每道题是一个 3–5 个模块的小包，bug 函数和若干干扰函数混在一起。
  - 题面只给调用、实际输出和期望输出，不给文件路径。
  - 隐藏用例覆盖包里所有函数，用来发现改坏别的函数的情况。
- **按簇划分**：
  - 入口函数体先归一化（去掉 docstring，名字换成占位符），再算 token 3-gram 集合的 Jaccard（`src/arl/taskgen/dedup.py`）。大于 0.8 的连边，得到 945 个簇。
  - 按簇随机切出 30% 作留出集。
  - 划分后留出集的近似重复是 0（`docs/plan/PROGRESS.md`"第 5b 步（完成）"）。
- **隐藏用例加强**：
  - 在原有用例的输入附近扰动，用原始代码算出期望值，每个函数最多补 10 条。
  - 加强后 bug 函数最少 5 条隐藏用例，众数 12 条（出处同上）。
- **规模**：4732 题，其中 train 3232、heldout 1500（出处同上）。
  - 留出集来源：MBPP 1290 题、HumanEval 210 题；273 个 bug 来源函数。
  - 留出集 bug 类型：const_off_by_one 368、var_swap 345、arith_swap 313、remove_branch 236、compare_swap 182、bool_swap 56。
  - 出处：`docs/plan/HELDOUT-PREREG.zh-CN.md`"评测设置"，`verify-heldout.json` 的 `heldout composition` 各项。
- **冻结**：题集 tag `phase1-tasks-v1`，环境 tag `phase1-env-v1`。
  - heldout 的 sha256 是 `d28e0709bb81af8196837efaec77ed3833d952ea25fcc1d457aa87f15d7cc3d4`（`data/tasks/manifest.json` 的 `frozen`）。

## 4. 训练流程

### 4.1 选基座

诊断扫描：100 题，本地模型每题 4 次，DeepSeek 每题 2 次，temperature 0.8，环境 `phase1-env-v1`（`docs/plan/CHECKPOINT-1D.md`"第 2、3 步"）。Qwen3-4B-Instruct-2507 在 11 GB 的卡上用 fp16 起不来，跳过。

| 模型 | 解出率 | 每次工具调用的格式错误率 | `edit_file` 成功率 | 平均轮数 | episode/小时 |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0% | 37.6% | 0/41 | 3.24 | 3332 |
| Qwen2.5-Coder-1.5B | 0% | 100% | 0/0 | 4.58 | 2718 |
| Qwen2.5-3B | 7.2% | 11.4% | 34.8% | 8.62 | 829 |
| **Qwen3-1.7B（不思考）** | **17.7%** | **2.1%** | **52.6%** | 5.37 | 2126 |
| DeepSeek flash（参考上限） | 89.5% | 0.1% | 100% | 5.26 | 5229 |

选了 Qwen3-1.7B，快照 `70d244cc86cc`（`configs/grpo/v1.json` 的 `base_revision`）。理由如下：

- 它在能跑起来的模型里各项都最好；
- 解出率不是 0，GRPO 有信号可学；
- 也不太高，还有提升空间。

DeepSeek 的 89.5% 说明这些题在 10 轮内是解得出来的。

### 4.2 SFT

- **轨迹**：DeepSeek flash 在 SFT 池上采了 1443 个 episode，其中成功 1171 条（`docs/plan/CHECKPOINT-2A.md` 附录 B"第 2 步"）。
- **筛选**：
  - 只要拿到奖励的轨迹；
  - 不要有格式错误、被提醒过、超过 8 轮的轨迹；
  - 每个来源函数最多 5 条。
  - 结果是 880 条轨迹、4410 个按轮样本（`docs/plan/CHECKPOINT-2A.md`"结论"）。
- **按轮切样本**：Qwen3 的聊天模板渲染历史 assistant 轮时会去掉 think 块。所以第 t 轮的样本是"前 t−1 轮加生成前缀"（mask 掉）接这一轮的输出（算 loss），原因见 6.3。
- **一致性检查**：以下几项全部 0 处不一致（`docs/plan/CHECKPOINT-2A.md`"数据集 v2"）：
  - 模板文本比对；
  - 分段 encode 和整段 encode 的比对；
  - hermes parser 的解析结果；
  - vLLM `/tokenize` 对 4410 个前缀的逐 token 比对。
- **训练**：本地 2080 Ti（11 GB），fp16 权重、fp32 LoRA，r 16，alpha 32。学习率 1e-4，cosine，2 个 epoch，520 步。
  - 墙钟 3.97 小时，val loss 从 2.306 降到 0.414（`docs/plan/CHECKPOINT-2A.md`"第 4 步"）。
  - 这份适配器冻结为 SFT-v1，tag `sft-v1`（`configs/sft/v1.json`）。
- **dev 结果**：100 题 × 4 次，严格解出率从基座的 20.3% 升到 53.5%。按题配对差 +33.3 个百分点，95% 区间 [+24.8, +41.8]（`docs/plan/CHECKPOINT-2A.md`"第 5 步"）。

### 4.3 GRPO 题集

- 用 SFT 模型在 RL 池的 1853 题上校准：T=1.0，top_k 0，每题 4 次。
- 只保留解出 1–3 次的题，共 1018 题、331 个函数。全对或全错的题组内没有差异，提供不了梯度。
- 出处：`docs/plan/PROGRESS.md`"第 2 步：全量校准（修订 2C-2）"，tag `grpo-tasks-v1`。

### 4.4 GRPO

**为什么自己实现**（`docs/plan/PHASE-2C-LOCAL-PREP.zh-CN.md` 第 0 节第 2 条）：

- 这个项目最大的坑是 token 一致性（见 6.3）。SFT 已经有一套验证过的按轮渲染，自己写可以让采样、训练、评测共用它。
- 用现成框架，得先弄清它怎么处理 Qwen3 去掉 think 块这件事，而且换个版本就可能变。
- 环境和判分可以原样复用。

**算法要点**（`src/arl/rl/batch.py`、`src/arl/rl/loss.py`、`configs/grpo/v1.json`）：

- **批次和采样**：每步 16 题，每题 8 条 episode，T=1.0，top_p 1.0，top_k 0。配置检查不满足这组采样参数就拒绝启动。
- **优势按 Dr. GRPO 算**：A_i = r_i − mean(r)，不除以组内标准差。全对或全错的组跳过。
- **每一轮单独作为一条训练序列**：就是采样时记录的 `prompt_ids + completion_ids` 原样拼接，loss 只算 completion 部分。
  - loss 除以固定的 `T_norm = n_tasks × G × 2560`，不除以本批实际 token 数，长轨迹不会被稀释。
  - 2560 是校准时每个 episode 输出 token 的 p99 向上取整。
- **严格 on-policy**：每批只做一次优化器更新，批次由当前权重采样，所以没有 PPO 的比率裁剪。
- **TIS**：截断重要性采样，权重 min(exp(logp_train − logp_vllm), 2)，不参与求导。它用来吸收 vLLM 和训练端之间的数值差别。
- **KL**：β = 0，不进 loss。每步随机抽 64 条序列，对 SFT-v1 计算 k3 KL，只记日志。
- **其他超参数**：LoRA r 16，dropout 0；学习率 1e-5 常数，AdamW；梯度裁剪 1.0；共 200 步。

**token 级客户端**（`src/arl/rollout/tokenclient.py`）：

- prompt 在本地用和 SFT 相同的渲染器转成 token id，调 vLLM 的 `/v1/completions`。
- 每轮记录 `prompt_ids`、`completion_ids` 和采样时的 logprob。训练直接用这些 id，不重新渲染。
- 评测用同一个客户端。每轮还把原始消息发给 `/tokenize` 逐个比对。
- 留出集评测时，三个模型分别比对了 31,470、41,073、40,609 轮，不一致都是 0（`verify-heldout.json` 的 `tokenize checked` 和 `tokenize mismatches`）。

**colocate 模式**（`docs/plan/PROGRESS.md`"阶段 3 · 第 3 步"，`configs/grpo/v1.json` 的 `memory_note`）：

- 训练端和 vLLM 同时放在一张 A800 40GB 上：vLLM 占 0.45 的显存（KV cache 120,080 token），训练端上限 16 GiB。
- 在最坏情况的序列上探测过：8192 token，completion 1024。训练端峰值 6.03 GiB allocated / 6.33 GiB reserved。
- 每步训练完保存适配器，vLLM 热加载新适配器。之后做两项同步检查：
  - 固定 prompt 的贪心输出一致；
  - logprob 平均绝对差不超过 0.05。

**训练过程**（`verify-training.json` 的 `items`）：

- 1018 题，200 步，每步 128 个 episode，没有任何 stop 记录。总用时 75,413.5 秒，约 20 小时 57 分。
- 平均步时 353 秒，其中采样 74 秒、训练 275 秒。
- 训练端和 vLLM 的 logprob 差：逐步均值的平均 0.0015，单步最大 0.0020。TIS 截断比例 0。
- 每步同步后贪心输出都一致。训练端显存峰值 5.86 GiB allocated / 6.04 GiB reserved。
- 选检查点用 dev 集 100 题，每题 8 次（K=8），每 25 步评一次。规则是预先写好的：选 dev 严格解出率最高的；相差不到 1 个百分点时选步数更早的（`docs/plan/HELDOUT-PREREG.zh-CN.md`"检查点怎么选"）。

## 5. 结果

### 5.1 留出集三方对比

留出集评测设置：1500 题，每题 K=4，每个模型只跑一次。temperature 0.8，top_k 20。三个模型都是 fp16 合并权重，vLLM `--dtype half`。评测方案在 GRPO 开始前预注册（`docs/plan/HELDOUT-PREREG.zh-CN.md`）。下表由 `scripts/closeout/verify_heldout.py` 生成（`runs/closeout/heldout-tables.md`）。

| | 基座 | SFT-v1 | GRPO（第 150 步） |
|---|---|---|---|
| **严格解出率** | **13.9%** | **47.1%** | **60.3%** |
| test+challenge | 21.5% | 53.0% | 65.6% |
| 每次调用格式错误率 | 2.7% | 1.9% | 0.7% |
| 出错 episode 比例（只报告） | 3.6% | 7.3% | 2.1% |
| 平均轮数 | 5.24 | 6.85 | 6.77 |
| 每题 4 次全部解出 | 109 | 320 | 581 |
| 每题 4 次都没解出 | 1157 | 413 | 299 |

| 配对差值（273 簇，10000 次，seed 0） | 严格 | 95% CI | test+challenge | 95% CI | 变好 / 变差的题 |
|---|---|---|---|---|---|
| GRPO − SFT | +13.2 | [11.7, 14.7] | +12.6 | [11.1, 14.1] | 667 / 178 |
| SFT − 基座 | +33.2 | [30.4, 36.1] | +31.5 | [28.6, 34.4] | 926 / 82 |

- 差值的单位是百分点。配对是同一道题上两个模型相减。每题先算 K=4 的解出率，再按来源函数聚类重抽。
- 表中有三格和 CHECKPOINT-3 正文的末位不同：基座出错 episode 比例 3.6%（CHECKPOINT-3 写 3.5%）、SFT 平均轮数 6.85（6.84）。下一节 SFT 的 bool_swap 是 31.3%（31.2%）。这三处的精确值恰好落在两种写法中间，说明见第 8 节。
- 两种口径的差值几乎一样，GRPO − SFT 分别是 +13.2 和 +12.6。所以生成用例带来的假阴性不影响结论。
- 平均轮数：SFT 后从 5.24 涨到 6.85，GRPO 后回落到 6.77。每题 4 次都没解出的题从 413 降到 299。

### 5.2 按 bug 类型和来源

| 严格解出率（留出集题数） | 基座 | SFT-v1 | GRPO（第 150 步） |
|---|---|---|---|
| arith_swap（313） | 16.7% | 55.7% | 69.1% |
| bool_swap（56） | 4.0% | 31.3% | 51.3% |
| compare_swap（182） | 8.5% | 38.0% | 49.2% |
| const_off_by_one（368） | 18.7% | 47.8% | 63.2% |
| remove_branch（236） | 2.5% | 33.2% | 44.1% |
| var_swap（345） | 18.6% | 55.5% | 67.7% |
| MBPP（1290） | 14.8% | 48.1% | 61.0% |
| HumanEval（210） | 8.3% | 41.2% | 56.1% |

- 出处：`runs/closeout/heldout-tables.md`；题数见 `docs/plan/HELDOUT-PREREG.zh-CN.md`"评测设置"。
- 六种 bug 类型和两个来源上，GRPO 都比 SFT 高。提升最小的是 remove_branch，+10.9 个百分点；最大的是 bool_swap，+20.1。HumanEval 上 +14.9（`verify-heldout.json` 的 `GRPO - SFT gain` 各项）。
- bool_swap 只有 56 题，这一行的不确定性比其他行大。上面的区间只针对整体，没有按子类做区间，按预注册也不做新的子群分析。

### 5.3 GRPO 训练曲线

每 25 步一段，取逐步数值的平均（`runs/grpo-v1/log.jsonl`）。

| 步 | 奖励 | 有好有坏的组 | 熵 | 每次调用格式错误率 | logprob 差均值 | KL(k3) | 平均轮数 | 步时 s |
|---|---|---|---|---|---|---|---|---|
| 1–25 | 0.474 | 0.87 | 0.753 | 3.9% | 0.0018 | 0.0007 | 7.08 | 398 |
| 26–50 | 0.534 | 0.85 | 0.562 | 3.1% | 0.0015 | 0.0033 | 7.17 | 376 |
| 51–75 | 0.591 | 0.80 | 0.541 | 2.4% | 0.0015 | 0.0056 | 6.96 | 364 |
| 76–100 | 0.595 | 0.81 | 0.546 | 2.0% | 0.0015 | 0.0089 | 6.92 | 381 |
| 101–125 | 0.653 | 0.76 | 0.556 | 1.2% | 0.0016 | 0.0122 | 6.57 | 347 |
| 126–150 | 0.668 | 0.75 | 0.522 | 1.1% | 0.0015 | 0.0146 | 6.61 | 345 |
| 151–175 | 0.701 | 0.68 | 0.382 | 1.1% | 0.0012 | 0.0173 | 6.54 | 313 |
| 176–200 | 0.724 | 0.66 | 0.371 | 0.85% | 0.0012 | 0.0217 | 6.32 | 295 |

数值来自 `scripts/closeout/verify_training.py`，只取整一次，和 CHECKPOINT-3 有一处 0.1 个百分点的差别，见其勘误。

- 各列含义：
  - "奖励"是训练 rollout 的平均奖励，题目是校准时解出 1–3 次的那 1018 题，不是留出集。
  - "有好有坏的组"是组内奖励不全相同的组所占比例，只有这些组产生梯度。
  - "熵"是 completion token 的平均熵。
- 奖励从 0.474 升到 0.724。有好有坏的组从 0.87 降到 0.66：题变得容易了，可用的信号在变少。
- 熵在第 151 步以后明显下降。最低一步是第 195 步的 0.228，低于停止门槛 0.248；那时最近 10 步有好有坏的组的均值是 0.625，所以按规则不停。第 200 步熵回到 0.42（`verify-training.json` 的 `items`；规则见 6.6）。

### 5.4 各检查点的 dev 结果和选择

dev 集 100 题，K=8，共 800 个 episode。T=0.8，top_k 20，用运行时 LoRA（`runs/closeout/training-tables.md`）。

| 步 | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| 严格解出数 | 451 | 475 | 501 | 500 | 516 | 543 | **551** | 557 | 551 |
| 严格解出率 | 56.4% | 59.4% | 62.6% | 62.5% | 64.5% | 67.9% | **68.9%** | 69.6% | 68.9% |
| test+challenge | 61.0% | 63.6% | 68.1% | 68.0% | 69.0% | 72.1% | 72.6% | 73.9% | 74.1% |

- 第 0 步是 SFT-v1。最高的是第 175 步，557/800。
- 和最高值相差不到 1 个百分点的有第 150、175、200 步，按规则选最早的第 150 步。第 125 步低 1.75 个百分点，不在范围内。比较的基准是最高值，不做链式比较（`verify-training.json` 的 `choice`）。
- 复核脚本按同一规则独立重选，结果也是第 150 步。
- dev 上第 150 步比第 0 步高 12.5 个百分点，和留出集上的 +13.2 接近（`verify-training.json` 的 `items`）。
- 三个模型在留出集上都比在 dev 上低。两个集合的题目构成不同，留出集里 HumanEval 占 14%，而且 HumanEval 题更难（`docs/plan/CHECKPOINT-3.md`"留出集三方对比"表下第 2 条）。

## 6. 过程中发现并修好的问题

### 6.1 期望值不进沙箱

- **现象**：设计判分时发现一个风险。如果隐藏测试是 pytest 文件，在沙箱里和 agent 的代码同进程运行，agent 的代码在 import 时就能 monkeypatch pytest、注册 atexit 伪造报告，或者顺着调用栈读到断言里的期望值。
- **原因**：同一个进程里没法可靠地防住这些手段。
- **修法**：每条隐藏断言拆成 `call` 和 `check`。
  - 沙箱里只对 `call` 求值，返回值序列化成带类型标签的纯数据；自定义 `__eq__` 带不出来，异常只记类型名。
  - `check` 带着期望值，在宿主机上求值。
  - agent 的代码最多只能"输出一些值"，不知道期望值就伪造不出正确结果。
- **验证**：`tests/test_grader.py` 覆盖了 3.2 节列出的各种作弊方式，奖励都是 0；参考答案奖励 1。
- 出处：`docs/plan/PROGRESS.md`"第 3 步：判分"。

### 6.2 按簇划分，防近似重复泄漏

- **现象**：最初按来源函数随机划分。留出集 321 个函数里有 63 个（19.6%）和训练侧某个函数的最大 Jaccard 超过 0.8，超过 10% 的门槛。
- **原因**：MBPP 和 HumanEval 里有大量写法几乎相同的函数，例如逐元素元组运算这一族。
- **修法**：在全部 1070 个函数上两两算归一化后的 token 3-gram Jaccard，大于 0.8 的连边，用并查集得到 945 个簇。按簇划分，最大的簇有 9 个函数。
- **验证**：划分后再跑一遍近似重复检查，留出集是 0。构建流程里也加了检查：不为 0 就停止构建。
- 出处：`docs/plan/PROGRESS.md`"第 5b 步（未完成）"和"第 5b 步（完成）"。

### 6.3 Qwen3 模板去掉历史 think 块，所以按轮切样本

- **现象**：Qwen3 的模板渲染历史 assistant 轮时不输出 think 块。但推理时生成前缀以空 think 块结尾。
  - 所以把整条轨迹渲染成一个序列时，第 1 到 t−1 轮的上下文和它们被生成时不一样。
  - 用这样的序列训练，学的不是推理时实际看到的 token。
- **原因**：模板只在最后一个真正的 user 消息之后的最后一条 assistant 消息上输出 think 块。
- **修法**：
  - SFT：每个 assistant 轮单独做一个样本，训练 token 量变成按整条轨迹算的约 4.2 倍。
  - GRPO：同理，每轮的 `prompt_ids + completion_ids` 原样作为一条序列，不重新渲染。
- **验证**：
  - SFT 数据的 4410 个前缀和 vLLM `/tokenize` 逐 token 比对，0 处不一致。
  - 反向对照：去掉 `enable_thinking=False` 或者去掉 tools 后，0/20 一致。说明这项比对确实能区分对错。
  - 测试 `tests/train/test_sft_render.py::test_whole_trajectory_as_one_sequence_would_not_match` 演示了整条渲染为什么不行。
- 出处：`docs/plan/CHECKPOINT-2A.md` 附录 B"渲染：每个 assistant 轮一个样本"和"数据集 v2"。

### 6.4 贪心比对因为环境不确定而失效，改用回放比对

- **现象**：新写的 token 客户端和原来的 chat 客户端做整条轨迹的贪心比对。10 题里只有 7 题工具调用序列完全一致，要求至少 9 题。
- **原因**：
  - `run_tests` 的失败输出里有函数的内存地址和耗时，例如 `0x7537…` 对 `0x78b6…`、`0.03s` 对 `0.04s`。
  - 只要跑过一次失败的测试，下一轮的 prompt 就不同了。
  - 原来的 chat 客户端自己和自己比，也是 7/10。
- **修法**：硬性判定改成回放比对。把参考运行里每个助手轮之前的完整历史原样交给新客户端做贪心生成，逐轮比较。整条轨迹的比对降为参考指标。
- **验证**：回放 65 轮，64 轮完全一致（98.5%，门槛 95%）。不一致的那一轮重发后复现不出来，归为 fp16 下服务端的数值非确定性。
- 出处：`docs/plan/PROGRESS.md`"第 1 步（二）"和"审计修订 2C-1"。

### 6.5 门槛按每次调用算，不用会随轮数增长的口径

- **现象**：GRPO 校准的试点里，"至少出现一次格式错误的 episode"占 18.0%，超过 15% 的门槛，停下。
- **原因**：这个比例随 episode 的轮数增长。T=1.0 时平均 7.05 轮，比 dev 上的 6.47 轮多，工具调用也更多，出错 episode 的比例自然更高。同一批数据的每次调用错误率只有 3.45%。
- **修法**：审计修订 2C-2 把门槛改成每次工具调用的格式错误率，不超过 6%。出错 episode 的比例照样报告，不再当门槛。之后新定的门槛都先换算成每次调用或每个 token 的口径（`docs/plan/PHASE-3-SERVER.zh-CN.md` 第 1 节）。
- **验证**：全量校准的每次调用错误率是 4.27%，出错 episode 比例 17.17%；保留 1018 题。正式训练中，每次调用错误率从第 1–25 步的 3.9% 降到第 176–200 步的 0.85%（见 5.3）。
- 出处：`docs/plan/PROGRESS.md`"第 2 步：试点"、"审计修订 2C-2"和"第 2 步：全量校准"。

### 6.6 熵停止条件被误触发，改成"熵低并且有好有坏的组少"才停

- **现象**：服务器上的过拟合测试用 8 题、lr 5e-5。第 13 步熵 0.2165，低于前 5 步均值 0.7245 的 30%（0.21735），停止条件触发，训练后的测量没有跑。
- **原因**：只有 8 道题、学习率是正式值的 5 倍，熵快速下降是预期内的。这时奖励在涨，熵低本身不代表坍缩。
- **修法**（审计修订 3-1）：
  - 先补测第 13 步的适配器：平均奖励 0.7031，训练前是 0.3281，8 道题全部提高。
  - 再关掉熵条件续跑到第 30 步：第 26–30 步的 rollout 奖励 0.909，训练后测量 0.9531。熵在低位停留了 17 步，奖励一直在涨。
  - 正式规则改成：熵低于前 5 步均值的 30%，**并且**最近 10 步有好有坏的组的均值低于 0.25，才停。
- **验证**：
  - 单元测试覆盖了只满足一条时不停、两条都满足才停，以及窗口不满、窗口里有空值、正好等于门槛这几种边界。
  - 正式训练第 195 步熵 0.228，低于门槛 0.248。最近 10 步有好有坏的组的均值是 0.625，所以没停。之后熵回到 0.42，奖励继续上升。
- 出处：`docs/plan/PROGRESS.md`"第 3 步：GRPO 过拟合测试"和"阶段 3 · 审计修订 3-1"；`verify-training.json` 的 `items`；`docs/plan/CHECKPOINT-3.md` 末尾的勘误第 2 条。

### 6.7 WSL 下显存溢出不报错，而是悄悄变慢

- **现象**：本地 SFT 启动后 27 分钟，初始验证（40 万 token，只做前向）还没算完，整卡显存占到 10.8 / 11 GiB。
- **原因**：
  - 实际在用的显存只有 3.3–4.4 GiB。PyTorch 的缓存分配器因为碎片，保留的显存一路涨到 8 GiB 以上。
  - WSL 上超出显存时，驱动不报 OOM，而是把超出的部分放到系统内存里，进程慢了上百倍。
- **修法**：
  - import torch 之前设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
  - 配置新增 `cuda_memory_limit_gib`，超过上限时 PyTorch 直接报 OOM。OOM 本身是停止条件，不会再悄悄溢出。
- **验证**：初始验证 132 秒跑完，峰值 5.1 GiB。正式训练 520 步跑完，显存峰值 9.16 GiB。
- 出处：`docs/plan/CHECKPOINT-2A.md`"遇到的问题"第 1 条。

### 6.8 root 会绕过进程数限制

- **现象**：沙箱用 `RLIMIT_NPROC` 挡 fork 炸弹。在 bwrap 外面用 `preexec_fn` 设成 64 时，bwrap 连命名空间都建不起来。另外以 root 运行时，这个限制完全不生效。
- **原因**：
  - `RLIMIT_NPROC` 按真实 uid 统计所有任务。本机桌面进程的线程很多，早就超过 64。
  - root 不受这个限制约束。
- **修法**：
  - rlimit 改在新的 user namespace 里由一个小启动器设置，64 只计沙箱自己的进程。
  - 服务器上一律用非 root 用户 `arl`，root 只用来装系统包。
  - 阶段 4 起，`scripts/server/setup.sh` 遇到 root 就停下（见 README"服务器上怎么跑"）。
- **验证**：
  - 本地 fork 炸弹在 62 个子进程时被挡住。
  - 服务器上以 `arl` 身份跑 `tests/test_sandbox.py`，25 passed、0 skipped，13 项 bwrap 检查全部执行。
- 出处：`docs/plan/PROGRESS.md`"第 2 步：沙箱"和"阶段 3 · 第 1 步"；`docs/plan/PHASE-3-SERVER.zh-CN.md` 第 0 节。

## 7. 局限

### 7.1 预注册里的已知局限，以及实际影响

下面每条的引文照抄 `docs/plan/HELDOUT-PREREG.zh-CN.md`"已知局限"，"实际影响"摘自 `docs/plan/CHECKPOINT-3.md`"预注册里的已知局限，逐条说明实际影响"。

1. > 干扰函数跨池。RL 池绝大多数题的干扰函数里有 SFT 池的来源函数，dev 上的 SFT 增益可能略偏乐观。留出集的来源函数和训练侧分开，这条以留出集为准。

   实际影响：SFT − 基座在 dev 上是 +33.3（K=4），在留出集上是 +33.2，几乎一样，这次没有看到明显的偏乐观。

2. > 生成用例的假阴性大约 1.5%（CHECKPOINT-1D：DeepSeek 的 200 个 episode 里 3 个是假阴性）。强模型把 MBPP 原题里的潜在 bug 一并改掉时，会被判失败。

   实际影响：对三个模型基本是同一批题上的系统性偏差，对配对差值影响很小。绝对解出率可能都低估了约 1 个百分点。test+challenge 口径的差值（+12.6、+31.5）和严格口径（+13.2、+33.2）接近，也说明这一点。

3. > SFT-v1 是本地 fp16 预览训练的适配器，不是在 4090 上重训的。

   实际影响：GRPO 从它出发，留出集上 SFT-v1 也是用它评测的，两者一致。这条只限制"SFT 能做到多好"的结论，不影响 GRPO − SFT 的比较。

4. > 贪心评测无法逐 token 复现。`run_tests` 的失败输出里有函数内存地址和耗时，同一条轨迹重跑，prompt 会从第一次失败的测试开始分叉。所有结论都按 K=4 采样的统计量来下，不用单条贪心轨迹做比较。

   实际影响：所有结论都用 K=4 采样的统计量，没有影响。

5. > 三个模型都在 fp16 下评测。基座直接以 fp16 加载；SFT-v1 和选定的 GRPO 检查点都是合并后的 fp16 权重，vLLM 用 `--dtype half`。服务器上的 SFT-v1 合并结果在 dev 集上核对（严格解出率要落在 53.5% 的 bootstrap 95% 区间 [45.5%, 61.25%] 里），不在留出集上核对。

   实际影响：都满足。服务器上 SFT-v1 的合并结果在 dev 上是 55.5%，在区间内。

6. > GRPO 训练过程中的 dev 评测仍用 vLLM 运行时 LoRA，只用来选检查点。留出集不用运行时 LoRA。运行时 LoRA kernel 和合并后的 fp16 权重之间可能有数值差别。

   实际影响：合并模型和"基座 + 适配器"的逐 token logprob 平均差是 0.0055，只有适配器作用量（0.50）的约 1%。三个候选检查点（150、175、200）在 dev 上只差 6 个 episode，这点数值差别不影响"选哪个"以外的结论。

### 7.2 其他局限

- **只测了一个模型规模和一种任务**：模型是 Qwen3-1.7B；任务是单点注入的 bug，每题只有一处改动。更大的模型、多处 bug、需要新增代码的任务都没有测。
- **题目是合成的**：bug 由 AST 变异产生，来源是 MBPP 和 HumanEval 的小函数，代码量和真实仓库差得很远。题面直接给出了失败的调用和期望值，真实 issue 通常没有这么明确。
- **没有验证迁移**：没有在 SWE-bench 这类真实仓库任务上测过，不知道这里学到的行为能不能迁移过去。所以本项目没有回答"A5 里 prompt 注入做不到的事，训练能不能做到"这个原始问题，只说明了训练能让小模型在这类合成题上明显变好。
- **GRPO 只跑了一组超参数**：16 题 × 8 条，lr 1e-5，200 步，没有做消融。不知道 Dr. GRPO 的优势、TIS、固定 `T_norm` 各自贡献多少，也不知道 GRPO 的增益里有多少来自"在校准过的 1018 题上多见了题"。
- **没有和"继续 SFT"对照**：GRPO − SFT 的 +13.2 是和 SFT-v1 比的。没有做"同样算力继续 SFT"的对照组。
- **训练后期信号变少**：有好有坏的组从 0.87 降到 0.66，熵也在下降。再往后训，收益可能会变小。dev 上第 175 步以后已经不再上升（见 5.4）。

## 8. 复现

### 8.1 数字怎么来的

报告里的数字从精确值四舍五入一次得到；精确值恰好落在两种写法中间的地方，`verify-*.json` 的 `half_unit_ties` 里列出了全部 14 处。

- 这 14 处里，留出集 10 处（其中 4 处是 `compare.json` 里同一个值的重复），训练 4 处。
- 报告统一按"恰好一半时进位"处理。所以 5.1 和 5.2 节有三格比 CHECKPOINT-3 正文的末位大 1：
  - 基座出错 episode 比例：精确值 3.55%，报告 3.6%，CHECKPOINT-3 写 3.5%；
  - SFT bool_swap：精确值 31.25%，报告 31.3%，CHECKPOINT-3 写 31.2%；
  - SFT 平均轮数：精确值 6.845，报告 6.85，CHECKPOINT-3 写 6.84。
- 这三处都不是错误。两种写法离精确值都是半个单位。
- CHECKPOINT-3 有两处真正的错误，都只是监控指标的文字描述，已经在它末尾加了勘误，正文没改：
  - 训练曲线 76–100 段的格式错误率，2.1% 应为 2.0%；
  - 第 195 步时有好有坏的组的 10 步均值，约 0.66 应为 0.625。

重新生成复核结果和本报告的表格，不需要 GPU，几十秒：

```bash
uv run python scripts/closeout/verify_heldout.py     # runs/closeout/verify-heldout.json, heldout-tables.md
uv run python scripts/closeout/verify_training.py    # runs/closeout/verify-training.json, training-tables.md
```

前提是 `runs/heldout/`、`runs/grpo-v1/` 和 `data/tasks/` 都在本地。`runs/` 不进 git。

### 8.2 从零到留出集评测

下面是这次实际用过的命令。所有命令都在仓库根目录运行。"GPU"一列写的是这次用的卡，时间是这次的实测墙钟。

| 步骤 | 命令 | GPU | 时间（出处） |
|---|---|---|---|
| 装依赖、跑测试 | `uv sync`；`uv run pytest -q` | 否 | 几分钟 |
| 生成题集 | `uv run python -m arl.taskgen.build --raw data/raw --out data/tasks`，用 `sha256sum` 和 `manifest.json` 核对 | 否 | 约 20.5 分钟（PROGRESS"第 5b 步（完成）"） |
| 诊断扫描（选基座） | `scripts/diag_model.sh <模型> <名字> configs/diag/<模型>.json` | 本地 2080 Ti | 每个模型 400 个 episode，吞吐见 4.1 节 |
| 划分 SFT 池、RL 池和 dev | `uv run python -m arl.sft.split`；`… split dev`；`… split rl-train` | 否 | 几秒 |
| 采 DeepSeek 轨迹（付费） | `uv run python -m arl.sft.collect run --budget-cny 15`，需要 `.env` | 否 | 花费见第 9 节 |
| 构建 SFT 数据 | `PYTHONPATH=src HF_HUB_OFFLINE=1 ~/.venvs/vllm/bin/python -m arl.sft.data build` | 否 | 几分钟 |
| SFT | `PYTHONPATH=src ~/.venvs/train/bin/python -m arl.sft.train --config configs/sft/preview.json` | 本地 2080 Ti | 3.97 小时（CHECKPOINT-2A"第 4 步"） |
| dev 评测 | `scripts/eval_dev.sh` | 本地 2080 Ti | 400 个 episode；基座每小时 2256 个，SFT 每小时 1214 个（CHECKPOINT-2A"第 5 步"） |
| GRPO 校准、出题集 | `scripts/calibrate_rl.sh full`；`uv run python -m arl.rl.taskset` | 本地 2080 Ti | 约 7.9 小时（PROGRESS"第 2 步：全量校准"） |
| 服务器装环境、自检 | `scripts/server/setup.sh`；`scripts/server/selfcheck.sh` | A800 | 约 1 小时 45 分，含下载等待（CHECKPOINT-3"机器时间和费用"） |
| 合并 SFT-v1 并核对 dev | `scripts/server/merge_fp16.sh` | A800 | 约 20 分钟（同上） |
| 测速 | `SKIP_PROBES=1 scripts/server/bench.sh` | A800 | 21 分钟（同上） |
| 正式 GRPO | `PYTHONPATH=src ~/.venvs/train/bin/python -m arl.rl.grpo --config configs/grpo/v1.json` | A800 | 20 小时 57 分（`verify-training.json`） |
| 合并选定检查点 | `PYTHONPATH=src ~/.venvs/train/bin/python -m arl.sft.merge_fp16 --adapter runs/grpo-v1/adapters/step-150 --out runs/grpo-v1/merged-fp16-step-150` | A800 | 15 秒（PROGRESS"阶段 3 · 第 5 步"） |
| 留出集评测（每个模型一次） | `scripts/server/eval_heldout.sh <模型> base\|sft\|grpo` | A800 | 三个共 1 小时 39 分（CHECKPOINT-3"机器时间和费用"） |
| 留出集统计 | `uv run python -m arl.rl.heldout --base … --sft … --grpo …` | 否 | 几分钟 |

需要注意的地方：

- 服务器上的环境坑见 README"服务器上怎么跑"。
- 两个 fp16 合并模型没有保留。要用的话，用适配器加原始基座重新合并（`docs/plan/PHASE-4-CLOSEOUT.zh-CN.md` 第 0 节第 3 条）。
- 留出集评测由脚本保证每个模型只跑一次：记录已存在就拒绝运行。

## 9. 花费

### 9.1 付费 API（DeepSeek flash）

| 阶段 | 用途 | 花费（元） | 出处 |
|---|---|---|---|
| 1 | 参考上限：100 题 × 2 次 | 1.2783（另有预检 0.000024） | `docs/plan/CHECKPOINT-1D.md`"DeepSeek 参考上限" |
| 2A | SFT 轨迹采集：1443 个 episode | 9.845 | `docs/plan/CHECKPOINT-2A.md`"花费" |
| 2A | dev 参考评测：400 个 episode | 2.181 | 同上 |
| 2C | 无 | 0 | `docs/plan/PHASE-2C-LOCAL-PREP.zh-CN.md`（本阶段不调用付费 API） |
| 3 | 无 | 0 | `docs/plan/CHECKPOINT-3.md`"机器时间和费用" |
| 4 | 无 | 0 | 本阶段计划第 3 节 |
| **合计** | | **约 13.30** | 上面各项相加 |

阶段 1 的 CHECKPOINT-1A、1B、1C 没有付费 API 花费。

### 9.2 GPU 时间

- **本地**：RTX 2080 Ti 11 GB，WSL2（`docs/plan/PROGRESS.md`"第 5b 步（未完成）"）。有明确墙钟记录的几项：
  - SFT 3.97 小时；
  - GRPO 全量校准约 7.9 小时；
  - 本地 fp16 合并加 dev 基线约 30 分钟；
  - 合计约 12.4 小时。

  诊断扫描、几次 dev 评测、sleep 探测没有逐项记墙钟，所以本地的实际总时间比 12.4 小时多。
- **A800-SXM4-40GB**：约 28.5 小时，10-12 13:26 到 10-13 约 17:50（`docs/plan/CHECKPOINT-3.md`"机器时间和费用"）。
  - 其中正式训练 20 小时 57 分，留出集评测 1 小时 39 分。
  - 等审计方决定期间 GPU 空闲约 1 小时 20 分。
  - 服务器上没有单价记录，费用按 28.5 小时乘以单价计算。

## 10. 文档索引

| 内容 | 位置 |
|---|---|
| 阶段 1 计划（环境和题集） | `docs/plan/PHASE-1-ENV-AND-TASKS.zh-CN.md`，各轮 prompt 在 `docs/plan/PHASE-1*-PROMPT.zh-CN.md` |
| 阶段 1 检查点 | `docs/plan/CHECKPOINT-1A.md`（工具、沙箱、判分），`1B`（造题），`1C`（簇划分、试点），`1D`（冻结、诊断扫描、DeepSeek 上限、假阴性） |
| 阶段 2 计划和检查点（SFT） | `docs/plan/PHASE-2-SFT.zh-CN.md`，`docs/plan/CHECKPOINT-2A.md` |
| 阶段 2C（租卡前的本地准备） | `docs/plan/PHASE-2C-LOCAL-PREP.zh-CN.md`；结果记在 PROGRESS"Phase 2C"一节，没有单独的 CHECKPOINT 文件 |
| 留出集预注册 | `docs/plan/HELDOUT-PREREG.zh-CN.md` |
| 阶段 3 计划和检查点（A800 上的 GRPO） | `docs/plan/PHASE-3-SERVER.zh-CN.md`，`docs/plan/CHECKPOINT-3.md`（末尾有阶段 4 的勘误） |
| 阶段 4 计划和检查点（收尾） | `docs/plan/PHASE-4-CLOSEOUT.zh-CN.md`，`docs/plan/CHECKPOINT-4.md` |
| 全过程记录 | `docs/plan/PROGRESS.md` |
| 环境 | `src/arl/env/`：`tools.py`、`sandbox.py`、`grader.py`、`episode.py`、`prompt.md` |
| 造题 | `src/arl/taskgen/`：`sources.py`、`mutate.py`、`augment.py`、`dedup.py`、`validate.py`、`build.py`；题集清单 `data/tasks/manifest.json` |
| 采样客户端 | `src/arl/rollout/`：`tokenclient.py`、`hermes.py`、`calibrate.py`、`consistency.py` |
| SFT | `src/arl/sft/`：`collect.py`、`data.py`、`train.py`、`merge_fp16.py`；配置 `configs/sft/` |
| GRPO | `src/arl/rl/`：`grpo.py`、`batch.py`、`loss.py`、`trainer.py`、`vllm_server.py`、`heldout.py`；配置 `configs/grpo/v1.json` |
| 服务器脚本 | `scripts/server/`，同步方法见 `scripts/server/sync.md` |
| 复核脚本 | `scripts/closeout/verify_heldout.py`、`scripts/closeout/verify_training.py` |
| 前置项目 | `D:\crq\src\coding-harness`、`D:\crq\src\coding-agent`（A5 报告：`docs/A5-FINAL-REPORT.zh-CN.md`） |
