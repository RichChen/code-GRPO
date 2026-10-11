# 留出集评测预注册

在服务器上开始 GRPO 之前提交。提交之后不能改。留出集不参与选模、不参与训练、不参与校准。本文只统计题号，没有在留出集上跑模型。

## 评测哪些模型

三个，都用 token 客户端，环境 `phase1-env-v1`，一律以 fp16、vLLM `--dtype half` 评测：

1. 基座 Qwen3-1.7B，快照 `70d244cc86cc`，直接以 fp16 加载。
2. SFT-v1：`scripts/server/merge_fp16.sh` 的结果，适配器合并进以 fp16 加载的原始基座。不用 `runs/sft-preview/merged`。
3. GRPO 最终选定的检查点。先把该检查点的适配器合并成 fp16 再评测，不用 vLLM 的运行时 LoRA。

## 检查点怎么选

只看 dev 集的严格解出率，选最高的。两处相差不到 1 个百分点时，选步数更早的那个。dev 集 100 题（`data/tasks/dev_ids.txt`），和留出集、`rl_train_ids`、SFT 池都不相交。

## 评测设置

- 留出集 1500 题（`data/tasks/heldout.jsonl`，sha256 `d28e0709bb81af8196837efaec77ed3833d952ea25fcc1d457aa87f15d7cc3d4`），273 个来源函数。来源：MBPP 1290 题，HumanEval 210 题。
- bug 类型：const_off_by_one 368、var_swap 345、arith_swap 313、remove_branch 236、compare_swap 182、bool_swap 56。
- 每题 K=4，只跑一次。
- 采样：temperature 0.8、top_p 1.0、top_k 20。配置里写明 `top_k: 20`。以前的评测不写这个字段，服务端会用模型 `generation_config` 里的默认值，基座和本地合并模型都是 20，所以和以前的 dev 结果是同一套采样。写死是为了以后不受默认值变化的影响。
- 和 `rl_train_ids`（1853）、`train_rl_ids`（1953）、`train_sft_ids`（1279）、dev（100）的题号交集都是 0。

## 指标

- **主指标**：严格解出率。隐藏用例全过才算解出，和训练奖励一致。
- **次要指标**：
  - 只看 test 和 challenge 的解出率；
  - 按 bug 类型、按数据来源（MBPP / HumanEval）分开的严格解出率；
  - 每次工具调用的格式错误率，另外报告至少出错一次的 episode 比例，后者不作为门槛；
  - 平均轮数。

## 统计方法

按 bug 来源函数做聚类 bootstrap：以来源函数为簇，有放回地重抽簇，10000 次，seed 0。簇里的题全部带上。对每道题先算 K=4 的解出率，再在抽中的簇上取平均。报告两组配对差值和 95% 置信区间：GRPO 减 SFT、SFT 减基座。配对是同一道题上的两个模型相减。

## 已知局限

- 干扰函数跨池。RL 池绝大多数题的干扰函数里有 SFT 池的来源函数，dev 上的 SFT 增益可能略偏乐观。留出集的来源函数和训练侧分开，这条以留出集为准。
- 生成用例的假阴性大约 1.5%（CHECKPOINT-1D：DeepSeek 的 200 个 episode 里 3 个是假阴性）。强模型把 MBPP 原题里的潜在 bug 一并改掉时，会被判失败。
- SFT-v1 是本地 fp16 预览训练的适配器，不是在 4090 上重训的。
- 贪心评测无法逐 token 复现。`run_tests` 的失败输出里有函数内存地址和耗时，同一条轨迹重跑，prompt 会从第一次失败的测试开始分叉。所有结论都按 K=4 采样的统计量来下，不用单条贪心轨迹做比较。
- 三个模型都在 fp16 下评测。基座直接以 fp16 加载；SFT-v1 和选定的 GRPO 检查点都是合并后的 fp16 权重，vLLM 用 `--dtype half`。服务器上的 SFT-v1 合并结果在 dev 集上核对（严格解出率要落在 53.5% 的 bootstrap 95% 区间 [45.5%, 61.25%] 里），不在留出集上核对。
- GRPO 训练过程中的 dev 评测仍用 vLLM 运行时 LoRA，只用来选检查点。留出集不用运行时 LoRA。运行时 LoRA kernel 和合并后的 fp16 权重之间可能有数值差别。

## 修订记录

### 2026-10-12 审计修订 2C-3

GRPO 还没有开始，按“开始 GRPO 之前提交”修改。被替换的原文留在下面，git 历史里也有。

改了什么：

- 三个模型在留出集上一律以 fp16 评测，vLLM `--dtype half`。基座直接以 fp16 加载。SFT-v1 改为 `merge_fp16.sh` 的结果。GRPO 选定的检查点先合并成 fp16 再评测，不用 vLLM 运行时 LoRA。
- 已知局限里关于 bf16 起点的那条，改成上面的 fp16 说法。
- 新增一条已知局限：训练中的 dev 评测仍用运行时 LoRA，只用来选检查点。

为什么改：SFT-v1、本地校准、dev 评测和 `configs/grpo/v1.json` 都是 fp16。若服务器只把起点改成 bf16，校准就不再对应起点。bf16 转 fp16 只损失绝对值极小的权重；反过来把 fp16 合并结果转成 bf16 会丢 3 位尾数。留出集不用运行时 LoRA，是为了避免 LoRA kernel 和合并权重之间的数值差别。

替换前的原文：

> 2. SFT-v1：`runs/sft-v1/adapter` 合并进 **bf16 原始基座** 的结果。不用本地 fp16 的 `runs/sft-preview/merged`。
> 3. GRPO 最终选定的检查点。
>
> - SFT-v1 是本地 fp16 预览训练的适配器，不是在 4090 上用 bf16 重训的。
> - 本地校准用的是 fp16 合并模型。服务器上的起点是 bf16 合并。两者的差别用 dev 集核对（严格解出率要落在 53.5% 的 bootstrap 95% 区间 [45.5%, 61.25%] 里），不在留出集上核对。
