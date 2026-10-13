# 阶段 4：agentic-rl 收尾

> 执行方：本地 Cursor agent，在 WSL 的 `~/agentic-rl` 里工作。审计方：Claude Code。
> 这一阶段不训练、不跑模型、不调用付费 API、不租服务器，只做核对、整理文档和打 tag。

## 0. 审计结论（不要改这些句子）

1. **阶段 3 审计通过。** 审计方用 `runs/heldout/{base,sft,grpo}.jsonl` 的原始记录复算了三个模型的留出集严格解出率：0.1393、0.471、0.603。每个模型都是 6000 个 episode，覆盖 1500 题，每题 4 次，和 CHECKPOINT-3 一致。检查点的选择、各项停止条件、预注册的执行情况，审计方也逐项核对过，没有问题。
2. **服务器上的 165 个文件**（`runs/` 和 `data/`，不含两个 fp16 合并模型）已经拉回本地，逐个核对了 sha256，全部一致。服务器上的 23 个提交已合并到本地 `cf3ebc5`。服务器已经关机。
3. **两个 fp16 合并模型有意不保留。** 它们是 `runs/sft-v1/merged-fp16` 和 `runs/grpo-v1/merged-fp16-step-150`，用适配器加原始基座就能重新合并出来（`scripts/server/merge_fp16.sh`）。
4. **主结论**：留出集 1500 题，每题 K=4。严格解出率：基座 13.9%，SFT-v1 47.1%，GRPO 第 150 步 60.3%。GRPO 比 SFT 高 13.2 个百分点，95% 区间 [11.7, 14.7]；SFT 比基座高 33.2 个百分点，[30.4, 36.1]。统计方法是按 273 个 bug 来源函数做聚类配对 bootstrap，重抽 10000 次，seed 0。
5. 本地 `runs/sft-v1/merged-fp16` 目录也不存在了，和第 3 条一致。本地预览阶段的 fp16 合并模型 `runs/sft-preview/merged` 还在，保留不动。

## 1. 规则

- 之前的规则全部有效：在 WSL 里工作，用 uv；每一步做完跑全量测试、提交、写 PROGRESS；每次提交前做密钥检查，方法和 CHECKPOINT-1D 第 6 节第 4 条相同。
- **不改实验结论和数字。** `CHECKPOINT-3.md`、`HELDOUT-PREREG.zh-CN.md` 以及所有 `runs/` 数据，内容都不能改。发现错误就写进 PROGRESS，等审计方处理。
- 新写的文档里，每个数字都要能追到出处：文件路径加上节名或字段名。
- 不删除任何 `runs/` 数据，不重写 git 历史。
- 写作风格：中文，平实直接，不要宣传腔。不用"革命性""显著突破"这类词。用数字说话，局限要写清楚。

## 2. 分步计划

### 第 1 步：本地复核

只读，不跑模型。

1. 写 `scripts/closeout/verify_heldout.py`：读 `runs/heldout/{base,sft,grpo}.jsonl`，在本地重新算出 CHECKPOINT-3"留出集三方对比"里的数字，和 `runs/heldout/compare.json` 以及 CHECKPOINT-3 正文逐项比对：
   - 三个模型的严格解出率和 test+challenge 解出率；
   - 两组配对差值和 95% 置信区间（用 `arl.rl.heldout`，273 簇，10000 次，seed 0）；
   - 按 bug 类型、按来源分开的严格解出率；
   - 每题 4 次全部解出、都没解出的题数。

   任何一项对不上（四舍五入以后仍然不同）就停下报告，不要改 CHECKPOINT。
2. 写 `scripts/closeout/verify_training.py`：读 `runs/grpo-v1/log.jsonl` 和 `runs/grpo-v1/eval/`，复算 CHECKPOINT-3 里的训练曲线表（每 25 步一段的均值）和各检查点的 dev 结果表，再按预注册的规则重新判断一遍应该选哪个检查点：只看 dev 严格解出率，相差不到 1 个百分点时选步数更早的。结论必须是第 150 步。
3. 核对 `configs/grpo/v1.json` 的 sha256 等于 CHECKPOINT-3 记录的冻结值（`3566ba54…7261c`）。
4. 把两个脚本和它们的输出 `runs/closeout/verify-*.json` 都提交。`runs/` 不进 git，所以把输出里的关键数字摘要写进 PROGRESS。

### 第 2 步：最终报告 `docs/FINAL-REPORT.zh-CN.md`

给面试官和技术读者看，要求一个人第一次读就能看懂。结构如下：

1. **摘要**（不超过 8 行）：研究问题、做法、留出集的三个数字和两组配对区间、一句话结论。
2. **背景和动机**：
   - 和 coding-harness、coding-agent 两个项目的关系，A5 的零结果是这个项目的起点。引用 `D:\crq\src\coding-agent\docs\A5-FINAL-REPORT.zh-CN.md` 的结论，只引用、不复述细节。
   - 研究问题：靠 prompt 注入改不了的行为，能不能靠训练学会。
3. **环境和题集**：
   - 五个工具，以及为什么不给 shell；bwrap 沙箱；隐藏测试判分；防作弊测试。
   - 题集怎么造：MBPP 和 HumanEval，6 类 AST 变异，按簇划分，隐藏用例加强。
   - 数字：4732 题，train 3232 题、heldout 1500 题，heldout 273 个来源函数。
4. **训练流程**：
   - 基座怎么选：诊断扫描的对比表，只保留关键列。
   - SFT：DeepSeek 轨迹，880 条，按轮切样本的原因，模板一致性检查。
   - GRPO：自己实现的原因；算法要点，包括 Dr. GRPO 的优势、严格 on-policy、TIS、β=0 但记录 KL；token 级客户端怎么保证采样、训练、评测用的是同一份 token；colocate 模式。
5. **结果**：
   - 留出集三方对比表和配对区间；
   - 按 bug 类型、按来源的结果；
   - 训练曲线（用表格就行）；
   - 各检查点的 dev 结果和选择规则。
6. **过程中发现并修好的问题**：从 PROGRESS 和各 CHECKPOINT 里挑最有代表性的 6 到 8 个，每个写清楚：现象、原因、怎么修的、修完以后怎么验证的。至少要包括：
   - 判分里期望值放在宿主机，不进沙箱；
   - 按簇划分防近似重复泄漏；
   - Qwen3 模板会去掉历史 think 块，所以要按轮切样本；
   - 贪心比对因为环境不确定而失效，改用回放比对；
   - 门槛要用每次调用的口径，不能用会随轮数增长的口径；
   - 熵停止条件被误触发，改成"熵低并且有好有坏的组少"才停；
   - WSL 下显存溢出不报错、悄悄变慢；
   - root 会绕过进程数限制。
7. **局限**：逐条照抄预注册的已知局限，并写出 CHECKPOINT-3 里实际评估出的影响。另外补上：
   - 只测了一个模型规模（1.7B）、一种任务类型（单点注入的 bug）；
   - 题目是合成的，不是真实仓库里的 bug；
   - 没有在 SWE-bench 这类真实仓库任务上验证能否迁移；
   - GRPO 只跑了一组超参数，没有做消融。
8. **复现**：从零到留出集评测的命令清单，引用已有的脚本，标出哪些步骤要用 GPU、大概多久。
9. **花费**：付费 API 的累计花费（阶段 1 到 3 各部分，从各 CHECKPOINT 里汇总），以及 GPU 时间（本地若干小时，A800 约 28.5 小时）。
10. **文档索引**：各阶段的计划、CHECKPOINT、PROGRESS 和关键脚本在哪里。

报告里不贴大段日志，表格只保留读者需要的列。

### 第 3 步：README 重写

- README 开头第一屏要有：
  - 一句话说明项目做什么；
  - 留出集的三个数字和两组配对区间；
  - 一张流程图（mermaid）：题集 → 环境 → SFT → GRPO → 留出集评测；
  - 指向 `docs/FINAL-REPORT.zh-CN.md` 的链接。
- 删掉"当前状态：阶段 1 进行中"这类过时的话。现有的工具说明、生成题集、接入 policy、启动推理服务这几节保留，往后放。
- 新增一节"服务器上怎么跑"，把 PROGRESS 里第 3 阶段踩过的环境坑整理成清单：
  - 要用非 root 用户；
  - bwrap 的 AppArmor 规则；
  - `python3.12-dev`；
  - venv 的 bin 目录要放进 PATH，否则 FlashInfer 找不到 ninja；
  - hf-mirror 下大文件会卡住，怎么处理；
  - colocate 模式下的两个显存参数。

  下次租机直接照着做。
- 加一节"项目历史"，列出阶段 1 到 4 的 tag 和每个阶段一句话的结论。

### 第 4 步：服务器脚本补全

把 CHECKPOINT-3"遇到的问题"里改过的东西补进脚本或文档，避免下次再踩：

- `scripts/server/setup.sh`：
  - 开头检查当前用户不是 root，是 root 就停下，并提示新建用户；
  - 检查 `python3.12-dev` 有没有装（看 `Python.h` 在不在），没装就停下并提示安装命令；
  - 检查 bwrap 能不能为非 root 用户创建命名空间，不能就停下，并提示 AppArmor 规则的写法。规则原文写进 `scripts/server/bwrap-userns.apparmor`，由用户用 root 手动安装，脚本本身不改系统配置。
- `scripts/server/sync.md`：根据这次的实际做法，改成用 git bundle 同步代码，因为仓库是私有的，服务器上没有凭据；拉回数据时用 tar 加 sha256 校验。
- 这些脚本改动只能在本地做 `bash -n` 和单元测试，在 PROGRESS 里写明"没有在服务器上验证"。

### 第 5 步：打 tag，写 CHECKPOINT-4

1. 全量测试：`uv run pytest -q`，`~/.venvs/vllm` 全量，`~/.venvs/train` 里跑 `tests/train tests/rl`。
2. 打 tag `phase3-grpo-v1`，指向 CHECKPOINT-3 的提交 `cf3ebc5`；打 tag `final-v1`，指向本阶段的最后一个提交。
3. 写 `docs/plan/CHECKPOINT-4.md`，内容包括：
   - 第 1 到 4 步的结论；
   - 复核结果，有没有对不上的地方；
   - 新写和修改了哪些文件；
   - 测试结果；
   - 两个 tag；
   - 留给后续的事项：A6-R 在 coding-harness 和 coding-agent 里进行，A6-M 等 A6-R 做完以后再考虑，需要重新租 GPU，约 20 小时。
4. **不要 push。** 由用户决定什么时候推到 GitHub。在 CHECKPOINT-4 里列出 push 前要检查的项：密钥扫描结果；`.gitignore` 是否覆盖 `runs/` 和 `.env`；仓库大小。

## 3. 不要做的事

- 不跑任何模型，不调用付费 API，不租服务器。
- 不改 `CHECKPOINT-3.md`、`HELDOUT-PREREG.zh-CN.md` 和任何 `runs/` 数据。
- 不在留出集上做任何新的分析，比如换一种统计方法、找新的子群。复核只能复算已经报告过的数字。
- 不修改 coding-harness 和 coding-agent 两个仓库。
- 不 `git push`。
