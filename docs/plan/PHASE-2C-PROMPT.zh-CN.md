CHECKPOINT-2A 审计已通过，阶段 2A 完成。审计方用原始 jsonl 独立复算过：基座 20.25%、SFT 53.5%，解出 1–3 次的题分别是 17 和 43；dev 集和 SFT 池、`rl_train_ids` 都不相交。另外两点做得对：WSL 下显存溢出的问题，你查到了碎片和静默换到系统内存这个根因；`cuda_memory_limit_gib` 让溢出直接报 OOM，没有改动任何超参数。

接下来进入阶段 2C，也就是租卡前的本地准备。先完整读一遍 `docs/plan/PHASE-2C-LOCAL-PREP.zh-CN.md`，从第 1 步开始严格按它执行到第 6 步，写好 `docs/plan/CHECKPOINT-2C.md` 就停下。

要求：
1. 之前的规则全部有效：在 WSL 里工作、用 uv；每一步做完跑全量测试、提交、写 PROGRESS；每次提交前做密钥检查，方法和 CHECKPOINT-1D 第 6 节第 4 条相同。
2. 先把计划第 0 节"审计修订 2A-2"写进 `PHASE-2-SFT.zh-CN.md` 末尾，作为对原 2B 的替换说明，单独提交一次。
3. 本阶段不调用付费 API，不租服务器，不在留出集上跑任何模型。
4. GRPO 自己实现，不引入 TRL、verl、OpenRLHF 等框架。新依赖要先在 PROGRESS 里写明理由。
5. 第 1 步的三项一致性检查和第 3 步的本地过拟合测试是硬要求。不通过就停下报告，不要绕过，也不要自行调学习率或改算法。
6. 本地 GPU 一次只跑一个进程。所有 PyTorch 进程都设 `expandable_segments` 和显存上限。
7. vLLM 0.31 的接口名，包括 `/v1/completions` 的 token id 输入和输出参数、运行时 LoRA 加载、sleep mode，一律以文档或源码为准。查到的写进 PROGRESS，不要凭记忆写。某个功能在 0.31 里不存在，就停下报告，不要升级或降级 vLLM。
8. 第 2 步校准大约要跑 6–7 小时，要能断点续跑。跑的时候可以并行写第 3 步里不占 GPU 的代码和测试。
