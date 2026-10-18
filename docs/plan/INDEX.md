# docs/plan 导读

这个目录保存实验设计、预注册、阶段记录和进度流水。当前分支只保留能解释"做了什么、结论是什么、怎么复算"的材料；当时逐轮发给执行 agent 的派工提示词不在当前分支（见文末）。

## 从这里开始

1. [FINAL-REPORT.zh-CN.md](../FINAL-REPORT.zh-CN.md)：背景、做法、结果、修过的问题、局限、花费。
2. [HELDOUT-PREREG.zh-CN.md](HELDOUT-PREREG.zh-CN.md)：留出集上要跑的模型、指标和判定规则，在任何真实运行之前冻结。
3. [CHECKPOINT-3.md](CHECKPOINT-3.md)：A800 上 200 步 GRPO 的训练记录和结果。

想复现运行环境，看 README 的"服务器上怎么跑"；想看每一阶段的取舍，按下面的表查。

## 文档分类

| 文件 | 是什么 |
| --- | --- |
| `PHASE-1-ENV-AND-TASKS.zh-CN.md` | 阶段 1 计划：题集生成、环境、判分。含审计修订 1–4 |
| `PHASE-2-SFT.zh-CN.md` | 阶段 2 计划：SFT 数据、按轮切样本、模板一致性。含审计修订 2A-1、2A-2 |
| `PHASE-2C-LOCAL-PREP.zh-CN.md` | 阶段 2C 计划：GRPO 实现、校准、本地过拟合测试、显存。含审计修订 2C-1 到 2C-3 |
| `PHASE-3-SERVER.zh-CN.md` | 阶段 3 计划：A800 上的训练、评测、停止条件。含审计修订 3-1 |
| `PHASE-4-CLOSEOUT.zh-CN.md` | 阶段 4 计划：复算、最终报告、README、服务器脚本。含审计修订 4-1 |
| `HELDOUT-PREREG.zh-CN.md` | 留出集预注册，附 fp16 评测的修订记录 |
| `CHECKPOINT-1A` … `CHECKPOINT-1D` | 阶段 1 的逐步记录：题集构建、判分假阴性、模型诊断扫描、DeepSeek 参考上限 |
| `CHECKPOINT-2A.md` | SFT 记录：880 条轨迹、4410 个按轮样本、前缀逐 token 对齐、dev 结果 |
| `CHECKPOINT-3.md` | GRPO 训练记录：200 步、显存与耗时、停止条件、遇到的问题 |
| `CHECKPOINT-4.md` | 收尾记录：数字复算、勘误、最终报告与 README |
| `PROGRESS.md` | 按时间追加的执行流水，约 160 KB。用于追查某一步的细节，不是入口文档 |

## 不在当前分支的内容

15 份逐轮派工提示词（`PHASE-*-PROMPT*.zh-CN.md`、`PHASE-1-START-PROMPT.zh-CN.md`）已在一次文档整理提交中移出当前分支。它们的内容是"读哪个计划、做到第几步停下、提交后等审计"，对项目使用者没有帮助。

其中夹带的技术决定都已落到阶段计划里，逐条核对过：

| 原文件 | 决定现在在哪 |
| --- | --- |
| `PHASE-1B-PROMPT` | 审计修订 1 在 `PHASE-1-ENV-AND-TASKS` 第 3 步末；拆断言规则在 `CHECKPOINT-1B` 的验收表 |
| `PHASE-1C-PROMPT` | 审计修订 2，同计划 |
| `PHASE-1D-PROMPT` | 审计修订 3，同计划 |
| `PHASE-1E-PROMPT` | 审计修订 4（提醒消息原文、`max_turns` 10、JSON 说明）在计划第 6 步末 |
| `PHASE-2A2-PROMPT` | 审计修订 2A-1 在 `PHASE-2-SFT` |
| `PHASE-2C-PROMPT` | 审计修订 2A-2，同上 |
| `PHASE-2C-PROMPT-2/3/4` | 审计修订 2C-1/2C-2/2C-3 在 `PHASE-2C-LOCAL-PREP` 第 0 节 |
| `PHASE-3-PROMPT-2` | 审计修订 3-1 在 `PHASE-3-SERVER` |
| `PHASE-4-PROMPT-2` | 审计修订 4-1 在 `PHASE-4-CLOSEOUT` |
| `PHASE-1-START`、`PHASE-2A`、`PHASE-3`、`PHASE-4` | 纯派工，没有技术内容 |

原文仍可从 git 历史取出，例如：

```bash
git show e5ddf28:docs/plan/PHASE-2C-PROMPT-4.zh-CN.md
```
