# CHECKPOINT-4：阶段 4 收尾完成（复核通过，加了两处勘误，最终报告和 README 写完）

## 结论

- **第 1–5 步和审计修订 4-1 全部执行完。** 本阶段没有跑模型，没有调用付费 API，没有租服务器，没有在留出集上做新分析，没有 push。
- **复核**：
  - 留出集的数字 150 项全部和原始记录一致。
  - 训练和检查点 160 项里，158 项一致。另外 2 项是 CHECKPOINT-3 里监控指标的文字描述不对，已在 CHECKPOINT-3 末尾加勘误。
  - 按预注册规则重选检查点，仍是第 150 步。
  - `configs/grpo/v1.json` 的 sha256 等于冻结值。
- **新文档**：`docs/FINAL-REPORT.zh-CN.md`（最终报告），重写的 `README.md`。表格由复核脚本从精确值只取整一次生成。
- **服务器脚本**：`setup.sh` 加了三项前置检查，新增 AppArmor 规则文件，`sync.md` 改成 git bundle 加 tar 和 sha256。没有在服务器上验证。
- **tag**：`phase3-grpo-v1` 指向 `cf3ebc5`；`final-v1` 指向提交本文件的那个提交。

## 第 1–4 步的结论

| 步 | 结论 | 提交 |
|---|---|---|
| 1. 本地复核 | 留出集 150 项全部一致。训练 160 项里有 2 项对不上，按规则停下报告，审计方出了修订 4-1 | `6e2fd1e` |
| 修订 4-1 | 修订原文追加到计划末尾，单独提交。CHECKPOINT-3 末尾追加勘误，正文 171 行没动。PROGRESS 只在"正式训练结束"一节末尾加了一行指向勘误。规则放宽的范围写进了 PROGRESS | `0a2f398`、`5280f05` |
| 2. 最终报告 | 按计划写了十节，每个数字都写了出处。两个复核脚本改成同时生成报告用的表，统一只取整一次 | `0de4acd` |
| 3. README | 第一屏放结果、两组区间、mermaid 流程图和报告链接。删掉"阶段 1 进行中"。新增"服务器上怎么跑"和"项目历史"两节。原有各节保留，放在后面 | `88165d3`，第 5 步又补了一处 |
| 4. 服务器脚本 | `setup.sh` 在装任何东西之前先做三项检查：不是 root、有 `Python.h`、bwrap 能建 user namespace。新增 `bwrap-userns.apparmor`，由用户用 root 手动安装。`sync.md` 改成 git bundle 加 tar 和 sha256。本地只做了 `bash -n` 和单元测试 | `397257f` |

## 复核结果

脚本是 `scripts/closeout/verify_heldout.py` 和 `scripts/closeout/verify_training.py`。输出在 `runs/closeout/`：`verify-heldout.json`、`verify-training.json`、`heldout-tables.md`、`training-tables.md`。`runs/` 不进 git，关键数字摘在 PROGRESS"阶段 4"一节。判定标准是四舍五入到报告里的位数以后相同。

### 留出集：150 项，0 项失败

- 以下各项全部和 `compare.json`、CHECKPOINT-3 正文一致：
  - 三个模型的严格解出率 13.9%、47.1%、60.3%，以及 test+challenge 解出率；
  - 按 bug 类型、按来源的解出率；
  - 格式错误率、出错 episode 比例、平均轮数、停止原因、`/tokenize` 比对轮数；
  - 每题 4 次全解出和都没解出的题数；
  - 题集的 sha256 和构成。
- 两组配对差值和区间：GRPO − SFT +13.2 [11.7, 14.7]，SFT − 基座 +33.2 [30.4, 36.1]。用 `arl.rl.heldout`，273 簇，10000 次，seed 0。
- `compare()` 在原始记录上重新生成的结果和 `compare.json` 完全相同。

### 训练和检查点：160 项，0 项失败，其中 2 项按勘误比对

- 检查点选择：最高的是第 175 步（557/800）。不到 1 个百分点的有第 150、175、200 步，取最早的**第 150 步**，和 CHECKPOINT-3 一致。
- dev 表 9 个检查点的数字全部一致。
- `configs/grpo/v1.json` 的 sha256 是 `3566ba54767d7f8a1964d785639fb4b8d4b43d2dd91989f27861b0a526f7261c`，工作区和 `c9984b9` 里的版本都一样。
- **两处勘误**（`docs/plan/CHECKPOINT-3.md` 末尾"勘误（阶段 4 复核，2026-10-13）"）：
  1. 训练曲线 76–100 段的每次调用格式错误率：原文 2.1%，正确值 **2.0%**。逐步比率的平均是 2.049%，原文先取 4 位小数再换算成百分比，取整了两次。
  2. 第 195 步时最近 10 步（186–195）有好有坏的组的均值：原文"约 0.66"，正确值 **0.625**。0.66 是 176–200 这一段的均值。0.625 仍远高于门槛 0.25，第 195 步不停止的判断不变。
  - 两处都只是监控指标的文字描述，不影响留出集的数字、检查点的选择和任何停止判断。
  - 复核脚本对这两项用勘误后的值比对，原来印的值留在 `verify-training.json` 的 `corrected_by_erratum` 里。

### 恰好差半个单位的 14 处

这 14 处的精确值恰好落在两种写法中间，两种写法离精确值都是半个单位。按修订 4-1 第 4 条，它们不算错误，不加勘误。全部列在 `verify-heldout.json` 和 `verify-training.json` 的 `half_unit_ties` 里。

| 来源 | 项 | 精确值 | CHECKPOINT-3 写的 |
|---|---|---|---|
| 留出集 | 基座 test+challenge | 21.45% | 21.5% |
| 留出集 | 基座出错 episode 比例 | 3.55% | 3.5% |
| 留出集 | SFT test+challenge | 52.95% | 53.0% |
| 留出集 | SFT bool_swap | 31.25% | 31.2% |
| 留出集 | SFT 平均轮数 | 6.845 | 6.84 |
| 留出集 | GRPO 出错 episode 比例 | 2.05% | 2.1% |
| 留出集 | 以上第 2、4、5、6 项在 `compare.json` 里的同一个值 | （4 处） | |
| 训练 | 126–150 段奖励 | 0.6675 | 0.668 |
| 训练 | 26–50 段有好有坏的组 | 0.845 | 0.85 |
| 训练 | 51–75 段有好有坏的组 | 0.795 | 0.80 |
| 训练 | 总用时 | 75413.5 秒 | 75414 秒 |

- 最终报告统一按"恰好一半时进位"写，所以有三格比 CHECKPOINT-3 的末位大 1：基座出错 episode 比例 3.6%、SFT bool_swap 31.3%、SFT 平均轮数 6.85。报告第 8.1 节逐个说明了。
- 其余 11 处，报告和 CHECKPOINT-3 写法相同。

## 新写和修改的文件

| 文件 | 新写 / 修改 | 内容 |
|---|---|---|
| `docs/plan/PHASE-4-CLOSEOUT.zh-CN.md`、`PHASE-4-PROMPT.zh-CN.md`、`PHASE-4-PROMPT-2.zh-CN.md` | 新写 | 计划和两轮 prompt，修订 4-1 逐字追加在计划末尾 |
| `scripts/closeout/verify_heldout.py` | 新写 | 留出集复核，并生成 `heldout-tables.md` |
| `scripts/closeout/verify_training.py` | 新写 | 训练曲线、dev 表、检查点重选、配置 sha256 的复核，并生成 `training-tables.md` |
| `tests/test_closeout.py` | 新写 | 选检查点规则、取整判定、只取整一次的函数 |
| `tests/test_server_scripts.py` | 新写 | 全部 `.sh` 过 `bash -n`；`setup.sh` 三项检查的失败分支；AppArmor 文件 |
| `docs/FINAL-REPORT.zh-CN.md` | 新写 | 最终报告 |
| `scripts/server/bwrap-userns.apparmor` | 新写 | bwrap 的 AppArmor 规则 |
| `docs/plan/CHECKPOINT-4.md` | 新写 | 本文件 |
| `docs/plan/CHECKPOINT-3.md` | 只在末尾追加 | 勘误一节，正文没动 |
| `docs/plan/PROGRESS.md` | 追加 | 阶段 4 各步；"正式训练结束"一节末尾加一行指向勘误 |
| `README.md` | 重写 | 见第 3 步 |
| `scripts/server/setup.sh` | 修改 | 开头三项检查；第一行注释的机器描述 |
| `scripts/server/sync.md` | 重写 | git bundle、tar 加 sha256 |
| `tests/test_heldout_stats.py` | 修改两行注释 | 去掉会被密钥前缀检查误判的英文连字符复合词，测试逻辑没动 |

没有改 `HELDOUT-PREREG.zh-CN.md` 和任何 `runs/` 数据，没有删除 `runs/` 数据，没有重写 git 历史，没有改 coding-harness 和 coding-agent。

## 测试结果

最后一次提交前跑的三套全量测试，依次跑，没有并行：

| 环境 | 命令 | 结果 |
|---|---|---|
| uv | `uv run pytest -q` | 310 passed，6 skipped |
| `~/.venvs/vllm` | `PYTHONPATH=src HF_HUB_OFFLINE=1 ~/.venvs/vllm/bin/python -m pytest -q` | 324 passed，2 skipped |
| `~/.venvs/train` | `PYTHONPATH=src ~/.venvs/train/bin/python -m pytest -q tests/train tests/rl` | 50 passed，1 skipped |

- 阶段 4 开始前是 292 / 306 / 50。
- uv 环境多出的 18 项是 `test_closeout.py`（4 项）和 `test_server_scripts.py`（14 项）；vLLM 环境同样多 18 项。
- 跳过的都是该环境没装 torch、peft、vllm 的测试，在另一个环境里跑过。

## tag

| tag | 指向 | 说明 |
|---|---|---|
| `phase3-grpo-v1` | `cf3ebc5` | CHECKPOINT-3：GRPO 第 150 步，留出集结果 |
| `final-v1` | 提交本文件的那个提交 | 阶段 4 收尾 |

之前的 tag 没有动：`phase1-tasks-v1`（`436d8db`）、`phase1-env-v1`（`14c155e`）、`phase2-grader-v1`（`a73e4e0`）、`sft-v1`（`576d202`）、`grpo-tasks-v1`（`2f847cb`）。

## 留给后续的事项

1. **A6-R**：在 coding-harness 和 coding-agent 两个仓库里进行，不在本仓库。
2. **A6-M**：等 A6-R 做完以后再考虑。需要重新租 GPU，约 20 小时。
3. **服务器脚本的改动没有在服务器上跑过**：`setup.sh` 的三项检查、AppArmor 规则、`sync.md` 的命令。
   - AppArmor 规则是 Ubuntu 23.10 以后的标准写法。阶段 3 审计方在服务器上装的那份原文我没看到，下次租机时请核对。
   - 下次租机跑 `setup.sh` 时，顺带确认三项检查在正常环境下能通过。

## push 前要检查的项（本阶段没有 push）

由用户决定什么时候推到 GitHub。`origin` 是 `https://github.com/RichChen/code-GRPO.git`。

1. **密钥扫描**：每次提交前都按 CHECKPOINT-1D 第 6 节第 4 条做了检查，最后一次的结果如下。
   - 前缀检查：工作区 3 处，就是登记过的三处非密钥文字（`PHASE-1E-PROMPT.zh-CN.md`、`PROGRESS.md`、`src/arl/env/grader.py`）。暂存区 0 处。
   - 用 `.env` 里的真实密钥值搜索：工作区、暂存区、全部 git 历史，以及 `runs/`、`docs/`、`src/`、`configs/`、`scripts/`、`tests/`、`README.md`，都是 0 处。
   - push 前建议再跑一遍真实密钥值的全历史搜索：`git log -p --all` 加上 `git grep` 所有提交。push 以后历史就公开了，私有仓库也一样要查。
   - 已入库的前缀文字会随仓库一起上传，它们不是密钥。
2. **`.gitignore`**：`git check-ignore -v` 确认以下路径都被忽略：
   - `runs/`（第 6 行）；
   - `.env`（第 10 行）和 `.env.*`（第 11 行，`.env.example` 例外）；
   - `data/tasks/*.jsonl`（第 7 行）。
   - 唯一被跟踪的 `runs/` 文件是 `runs/phase3/heldout-chain.sh`，13 行运行脚本，阶段 3 第 5 步提交时连带加入（CHECKPOINT-3"遇到的问题"第 8 条）。
   - 被跟踪的 `.env*` 只有模板 `.env.example`。
3. **仓库大小**：`.git` 目录约 12 MB。`git count-objects -vH`：松散对象 4.71 MiB，pack 710.62 KiB。最大的文件是 `data/raw/mbpp.jsonl`（约 551 KB）和 `data/raw/sanitized-mbpp.json`（约 249 KB）。模型权重、适配器、题集 jsonl、`runs/` 都不在 git 里。
4. **tag 也要推**：`git push origin main --tags` 才会带上 7 个 tag。
