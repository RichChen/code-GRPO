你在一台按小时计费的 A800 服务器上继续 agentic RL 项目。环境已由审计方（本地的 Claude Code）准备好，先完整读一遍 `docs/plan/PHASE-3-SERVER.zh-CN.md`，再从第 1 步开始严格执行到第 6 步，写好 `docs/plan/CHECKPOINT-3.md` 就停下。

背景：`docs/plan/` 下有阶段 1、2 的完整计划和 CHECKPOINT，开工前至少读 `PHASE-2C-LOCAL-PREP.zh-CN.md`（含审计修订 2C-1/2C-2/2C-3）、`CHECKPOINT-2A.md` 和 `HELDOUT-PREREG.zh-CN.md`。

要求：
1. **用 `arl` 用户工作，不要用 root。** 项目在 `/home/arl/work/agentic-rl`。root 会绕过进程数限制，防 fork 炸弹的 2 项测试会失败。
2. 之前的规则全部有效：每一步做完跑全量测试、在本地提交 git、写 PROGRESS。**不要 `git push`**，仓库是私有的、服务器上没有凭据，代码同步由审计方负责。
3. **GPU 一次只跑一件事。** 每一步结束后确认显存已释放再开始下一步。长任务放 tmux。
4. **留出集只在第 5 步按预注册跑一次。** 第 1 到 4 步不得读 `heldout.jsonl` 做评测。每个模型只评一次，不重跑。
5. 遇到计划里写了"停下报告"的情况就停下，写清原因和证据，不要自行改学习率、`n_tasks`、`group_size`、温度、top_k、LoRA 配置，也不要换方案。只有第 3 步明确允许调 `cuda_memory_limit_gib` 和 `gpu_memory_utilization` 这两个显存参数。
6. 第 3 步的过拟合测试是正式训练的前置条件。它在本地因显存不足没能跑起来，这里是第一次真跑。不通过就停下报告，不要跳过它直接开始第 4 步。
7. 第 4 步正式训练期间，**每约 2 小时在 PROGRESS 里追加一次简报**，内容按计划第 4 步第 2 小点。这样审计方中途也能判断要不要继续。
8. 不调用付费 API，服务器上没有 `.env`。
9. vLLM 0.31 的接口名，包括运行时 LoRA 加载、sleep mode、`/v1/completions` 的 token id 输入输出，一律以文档或源码为准，不要凭记忆写。某个功能不存在就停下报告，不要升降 vLLM 版本。
10. 机器按小时计费。每一步开始前先说明预计耗时；发现某步明显超出预期，先在 PROGRESS 里记一笔再继续。
