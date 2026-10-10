# 阶段 2C：租卡前的本地准备

> 执行方：Cursor agent，在 WSL 的 `~/agentic-rl` 里工作。审计方：Claude Code。
> 目标：租到 4090 之前，在本机把 GRPO 需要的东西都准备好——统一的 token 级 rollout 客户端、GRPO 题集、GRPO 训练代码（本地验证能学）、服务器启动脚本、留出集评测的预注册。上了服务器以后只做三件事：跑通、测吞吐、正式训练。
> 原阶段 2B 的"正式 bf16 SFT"和原阶段 0 的"框架冒烟测试"按审计修订 2A-2 合并改写，见第 0 节。

## 0. 审计修订 2A-2（不要改）

1. **SFT 预览直接升级为 SFT-v1，不在 4090 上用 bf16 重训。**
   - 预览没有出现数值问题：0 步被跳过，loss scale 一直是 65536，val loss 在第 2 个 epoch 已经收平。
   - 用 bf16 重训，得到的基本是同一个模型，而后面的 GRPO 题集是用这个模型校准的。重训以后校准就得重做，换来的只有精度上的一点差别。
   - 正式的 SFT-v1 是 `runs/sft-preview/adapter-epoch2`（fp32 LoRA）。复制到 `runs/sft-v1/adapter/`，记下 sha256。
   - 上服务器以后，把这个适配器合并到 **bf16 的原始基座**上，得到 GRPO 的起点。不要用本地的 fp16 `merged/`：fp16 转 bf16 会丢 3 位尾数。
   - 服务器上要先在 dev 集上做一次一致性检查，见第 4 步的启动脚本。
2. **GRPO 不用现成框架，自己写一个最小实现。** 理由：
   - 本项目踩过的最大的坑是 token 一致性（修订 2A-1 第 2 条），而 SFT 这边已经有一套验证过的按轮渲染方式（`arl.sft.data`，4410 个前缀和 vLLM 逐 token 比对，0 处不一致）。自己写的话，rollout、训练、评测可以共用这一套。用框架，要先弄清它怎么处理 Qwen3 去掉 think 块这件事，而且换个版本就可能变。
   - 环境、判分和 episode 循环可以原样复用，不用改成框架要求的接口。
   - 算法本身很简单：每批采样只更新 1 次梯度（严格 on-policy，重要性比值恒为 1），不需要 critic，也不需要 PPO 裁剪。
   - 代价是正确性全靠自己的测试来保证。所以第 3 步要求单元测试，还要在本地做一个"能学会"的过拟合测试。
3. **rollout 改用 token 级接口。** 我们自己用已验证的渲染器生成 prompt 的 token id，调用 vLLM 的 `/v1/completions`（prompt 直接传 token id 列表），拿回生成的 token id 和 logprob，再用自己的 hermes 解析器解析工具调用。这样采样用的 token 和训练用的 token 从构造上就是同一份。从这一步起，校准、GRPO 和最后的留出集评测都只用这一个客户端。
4. **采样温度**：GRPO 的 rollout 和校准都用 temperature 1.0，这样采样分布就是模型本身的分布，训练算 logprob 时不需要额外做温度缩放。评测沿用 0.8，和之前所有 dev 结果的口径一致。
5. **GRPO 题集**：用 SFT-v1 在 `rl_train_ids` 上校准，每题 4 次，保留解出 1–3 次的题。dev 和留出集不进 GRPO 训练。

## 1. 规则

- 之前的规则全部有效：在 WSL 里工作；每一步做完跑全量测试、提交、写 PROGRESS；每次提交前做密钥检查；遇到"停下报告"不得自行降低要求。
- 本阶段不调用付费 API。
- 训练代码放 `src/arl/rl/`，测试放 `tests/rl/`，依赖 torch 的部分沿用 `~/.venvs/train` 的约定。新依赖要先在 PROGRESS 里写明理由。
- 本地 GPU 一次只跑一个进程，vLLM 和训练不能同时占显存。所有 PyTorch 进程都设 `expandable_segments` 和显存上限（CHECKPOINT-2A"遇到的问题"第 1 条）。
- 留出集只在第 5 步写预注册时读取题号和统计信息，不跑任何模型。

## 2. 分步计划

### 第 1 步：冻结 SFT-v1，实现 token 级 rollout 客户端

- **冻结 SFT-v1。** 把 `adapter-epoch2` 复制到 `runs/sft-v1/adapter/`。在新文件 `configs/sft/v1.json` 里记录这些信息：
  - 基座快照 `70d244cc86cc`；
  - 适配器各文件的 sha256；
  - 训练数据 `data/sft/v2` 的 sha256；
  - 训练配置文件的路径。
  
  适配器本身不入库，只提交这份 json。然后打 tag `sft-v1`。
- **实现 `src/arl/rollout/tokenclient.py`。**
  - 每轮的流程：
    1. 用 `arl.sft.data` 的渲染器，对当前消息加 `add_generation_prompt=True` 渲染出 prompt token id；
    2. 调用 `/v1/completions`，`prompt` 直接传 token id 列表，并带上 `return_token_ids`、`logprobs=0`。参数名以 vLLM 0.31 的文档为准，查到的写进 PROGRESS；
    3. 生成以 `<|im_end|>` 结束。用我们自己的 hermes 解析器解析工具调用，解析规则和现在 parser 的严格程度一致。
  - 每轮都要保存：`prompt_ids`、`completion_ids`、`completion_logprobs`、`finish_reason`。保存的就是采样时实际用的 token，训练时原样使用，不再重新渲染。
  - 接口要和现有客户端一致，这样 `run_episode` 不用改就能接上。
- **一致性检查（必须有）：**
  - **逐轮比对：** 对每一轮，`prompt_ids` 等于 vLLM 服务端 `/tokenize` 的结果，用原始消息加 tools 和 `enable_thinking=False`。
  - **贪心比对：** 从 dev 集抽 10 题，temperature 0，旧客户端（chat 接口）和新客户端各跑一遍，逐 episode 比较工具调用序列和奖励。至少 9 题完全一致。不一致的要贴出差异，并分析是 batch 的非确定性，还是渲染或解析不同。
  - **统计比对：** 用新客户端在 dev 集上评测 SFT 模型，K=4、temperature 0.8。严格解出率要落在 53.5% 的 bootstrap 95% 置信区间里。不在区间里就停下报告。
- 测试、提交。从这里开始，校准、GRPO 和评测都只用新客户端。

### 第 2 步：用 SFT-v1 在 `rl_train_ids` 上校准

- 本地 vLLM 加载 fp16 的 `merged/` 来跑。服务器上用的是 bf16 合并的模型，和本地有一点数值差别。这一点要写进 PROGRESS，服务器上会在 dev 集上核对（见第 4 步）。
- 1853 题，每题 K=4，temperature 1.0（修订 2A-2 第 4 条），环境 `phase1-env-v1`，要能断点续跑。按现在的吞吐量估计约 6–7 小时。
- 输出：
  - `data/tasks/grpo_v1_ids.txt`：解出 1–3 次的题；
  - `data/tasks/grpo_v1.json`：题数、函数数、sha256、bug 类型分布、每个函数有几题的分布、解出次数分布（0/1/2/3/4）。
- 另外报告：
  - 两种口径的解出率；
  - 停止原因分布；
  - 用满 10 轮的 episode 占多少；
  - 每个 episode 的 token 数分布，GRPO 估算吞吐量要用。
- **停止条件**：保留的题少于 400 道就停下报告，不要放宽区间。
- 提交，打 tag `grpo-tasks-v1`。

### 第 3 步：GRPO 最小实现 `src/arl/rl/`

**算法**（写进 `configs/grpo/v1.json`，本地测试和服务器正式训练共用这套代码）：

- 每一步从 `grpo_v1_ids` 取 `n_tasks` 道题，每道题采 `G` 条 rollout。默认 `n_tasks` 16、`G` 8，temperature 1.0。
- **优势**：每组内 `A_i = r_i - mean(r)`，不除以标准差，也就是 Dr. GRPO 的做法：除以标准差会放大接近全对、全错那些组的噪声。全对或全错的组 A 全为 0，直接跳过，跳过的比例要计数。
- **loss**：只在每轮的 `completion_ids` 上计算，工具结果、system 和 user 部分不参与。`loss = -Σ_i A_i · Σ_{t∈i} logp_t / T_norm`，其中 `T_norm` 取固定常数 `n_tasks · G · max_completion_tokens_per_episode`，不按实际长度归一，避免长轨迹被稀释。
- **严格 on-policy**：每批 rollout 只做 1 次优化器更新，不用 PPO 裁剪，不用 critic。KL 系数 β 设为 0，但每一步都要记录对 SFT-v1 参考模型的 KL 估计（k3 估计量，用 rollout 的 token 算）。
- **训练和推理的 logprob 差异**：每一步统计训练端算出的 logp 和 vLLM 返回的 logprob 差多少，记录均值、p99 和最大值。另外实现截断重要性采样（TIS，`ratio = exp(logp_train - logp_vllm)`，截断到不超过 2），由配置开关控制，默认打开。
- **模型和权重同步**：
  - 训练端是 SFT-v1 的 LoRA，继续训练同一个适配器，r 16，学习率 1e-6 起，具体值写进配置；
  - vLLM 用 `--enable-lora` 启动，每一步结束后保存适配器，用运行时 LoRA 加载接口热加载，请求里指定新的适配器名。接口名以 vLLM 0.31 文档为准。
  - 每次加载以后，先用一条固定 prompt 做贪心生成，确认输出和训练端一致，再继续下一步。
- **显存调度**：
  - 本地 11 GB 放不下 vLLM 和训练同时运行，用 vLLM 的 sleep mode 轮流用显存：采样完让 vLLM sleep，训练完再 wake up。
  - 4090 上先尝试两者同时驻留，放不下再退回 sleep mode。两种模式都写进配置，可以切换。
- **日志**（`runs/grpo-*/log.jsonl`）：每一步都要记：
  - 平均奖励，以及非全对全错组所占比例；
  - 损失、梯度范数、学习率；
  - KL 估计，logprob 差异的均值、p99、最大值，TIS 截断的比例；
  - 平均轮数、输出 token 数、停止原因分布；
  - 采样、训练、同步各用多少秒。
- **检查点**：每 `eval_every` 步保存一次适配器，并在 dev 集上评测：K=4，temperature 0.8，用第 1 步的新客户端，两种口径都记。

**测试**（在 `tests/rl/` 里，不需要 GPU 的部分放进主测试）：

- 优势计算：全对、全错、混合三种组。
- loss 只落在 `completion_ids` 上：构造一条多轮轨迹，检查 mask 的位置。
- 训练端的 token 序列：用 rollout 保存的 `prompt_ids` 和 `completion_ids` 拼起来，**不重新渲染**。这一点也要写测试。
- 一个很小的玩具测试：用一个随机初始化的小 transformer，奖励规则是"生成某个固定 token 得 1 分"。跑 50 步，平均奖励要从约 0 升到 0.8 以上。

**本地过拟合测试**（这一步的验收标准）：

- 从 `grpo_v1_ids` 里固定 seed 取 8 道题，只在这 8 道题上训练 30 步，`n_tasks` 8、`G` 8，用 sleep mode。
- 训练前后，在这 8 道题上各跑 K=8、temperature 1.0。训练后的平均奖励要比训练前高 0.15 以上。
- 报告每一步的耗时，按阶段分开（采样、sleep 和 wake、训练、同步），以及显存峰值。
- **停止条件**，满足任何一条就停下报告，不要自行改学习率或算法：
  - 奖励没有上升；
  - 出现 NaN；
  - logprob 差异的均值大于 0.05；
  - 权重同步以后贪心生成和训练端不一致。

### 第 4 步：服务器脚本 `scripts/server/`

脚本只写好、在本地能做的检查跑一遍，不在服务器上运行。

- **`setup.sh`**：
  - 安装 uv，按 lock 文件装环境；
  - 按 `requirements-train.txt` 装训练环境，vLLM 版本固定为 0.31.0。CUDA 版本要和服务器驱动匹配，不匹配就停下；
  - 下载 Qwen3-1.7B，指定快照 `70d244cc86cc`；
  - 用 `HF_ENDPOINT` 支持镜像。AutoDL 常用的学术加速做法写进 README，但不写进脚本。
- **`selfcheck.sh`**，失败任何一项就退出，不往下走：
  - 沙箱隔离自检，沿用阶段 1 的 13 项 bwrap 检查。这是阶段 0 一直没在服务器上验证过的事。bwrap 在容器里不能用的话，就停下报告，不能退回到不隔离的模式；
  - 全量测试；
  - GPU 型号、显存、bf16 支持。
- **`merge_bf16.sh`**：把 `runs/sft-v1/adapter` 合并到 bf16 基座，然后在 dev 集上用新客户端评测，K=4、temperature 0.8。严格解出率要落在 53.5% 的 bootstrap 95% 置信区间里，否则停下报告。
- **`bench.sh`**：跑 3 个 GRPO 步，报告每一步的耗时和显存，并按这个速度估算 200 步要多久。
- **`sync.md`**：写清哪些文件要传到服务器（适配器、题集的 jsonl、`.env` 不传），以及怎么用 rsync 把 `runs/` 拉回本地。

### 第 5 步：留出集评测预注册 `docs/plan/HELDOUT-PREREG.zh-CN.md`

在服务器上开始 GRPO 之前提交，之后不能改。内容：

- **评测哪几个模型**：基座 Qwen3-1.7B、SFT-v1、GRPO 最终选定的检查点。
- **检查点怎么选**：只看 dev 集的严格解出率，选最高的。dev 集上相差不到 1 个百分点时，选步数更早的那个。
- **评测设置**：留出集 1500 题，每题 K=4，temperature 0.8，新客户端，环境 `phase1-env-v1`，只跑一次。
- **主指标**：严格解出率。次要指标：只看 test 和 challenge 的解出率、按 bug 类型和数据来源分开的解出率、格式错误率、平均轮数。
- **统计方法**：按 bug 来源函数做聚类 bootstrap，重抽 10000 次，seed 0。报告两组配对差值和 95% 置信区间：GRPO 减 SFT、SFT 减基座。
- **已知局限**：干扰函数跨池；生成用例的假阴性约 1.5%；SFT 用的是本地 fp16 预览训练的模型。

### 第 6 步：CHECKPOINT-2C

写 `docs/plan/CHECKPOINT-2C.md`，内容包括：

- 第 1–5 步各自的结论；
- 新旧两个客户端的一致性检查结果；
- 校准结果和 GRPO 题集的统计；
- 过拟合测试的奖励曲线和每一步的耗时；
- 估算 4090 上 200 个 GRPO 步要多久；
- 服务器上要做的步骤清单；
- 遇到的问题。

## 3. 不要做的事

- 不租服务器，不调用付费 API。
- 不在留出集上跑任何模型。
- 不改冻结的题集，不改环境行为，也不改奖励。
- 不在 dev 集或留出集上做 GRPO 训练。
- 遇到停止条件时，不自行改算法或超参数。
