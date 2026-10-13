阶段 3 审计已通过，服务器已经关机。审计方用原始记录复算了留出集上三个模型的严格解出率（0.1393、0.471、0.603），和 CHECKPOINT-3 一致。服务器上的 165 个文件和 23 个提交已经拉回本地，逐个核对过 sha256。

接下来进入阶段 4 收尾。先完整读一遍 `docs/plan/PHASE-4-CLOSEOUT.zh-CN.md`，再从第 1 步开始，严格按它执行到第 5 步，写好 `docs/plan/CHECKPOINT-4.md` 就停下。

要求：
1. 之前的规则全部有效：在 WSL 里工作，用 uv；每一步做完跑全量测试、提交 git、写 PROGRESS；每次提交前做密钥检查，方法和 CHECKPOINT-1D 第 6 节第 4 条相同。
2. 本阶段不跑任何模型，不调用付费 API，不租服务器，也不在留出集上做任何新分析。第 1 步的复核只能复算 CHECKPOINT-3 已经报告过的数字。
3. `CHECKPOINT-3.md`、`HELDOUT-PREREG.zh-CN.md` 和所有 `runs/` 数据都不能改。复核时发现对不上，就停下写进 PROGRESS，等审计方处理。
4. `docs/FINAL-REPORT.zh-CN.md` 和新的 README 是给面试官和技术读者看的。写之前先把 `CHECKPOINT-1D.md`、`CHECKPOINT-2A.md`、`CHECKPOINT-3.md`、`HELDOUT-PREREG.zh-CN.md` 和 PROGRESS 读一遍。每个数字都要写出处，语气平实，局限要写清楚。
5. 不修改 coding-harness 和 coding-agent 两个仓库，只读引用它们的文档。
6. 不要 `git push`。
