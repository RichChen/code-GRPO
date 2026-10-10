CHECKPOINT-1C 审计已通过。试点触发停止条件后停下是对的；没有换模型、没有调参数，这点也做得对。另外，"0% 不是 harness 的问题"这个判断有证据支撑，我认可。

本轮不再按"每题解出 1 到 3 次"来筛题。先把题集冻结，然后用诊断扫描决定后面用哪个模型。做完写好 `docs/plan/CHECKPOINT-1D.md` 就停下。其他规则不变。

开工前，先把下面"审计修订 4"一节原样补进 `PHASE-1-ENV-AND-TASKS.zh-CN.md` 第 6 步的末尾，单独提交一次。

## 审计修订 4

1. **原计划的顺序有问题。** 筛选区间 [0.25, 0.75] 应该用"开始 GRPO 时的那个策略"来校准，也就是 SFT 之后的模型，而不是原始模型。所以按区间筛题这一步挪到阶段 2 末尾：SFT 做完后，用 SFT 模型来校准。阶段 1 的第 6 步改为"诊断扫描"，用来选定基座模型。阶段 1 结束时交付冻结的全量题集，不做筛选。
2. **环境修改（定下来以后冻结，对所有模型都一样）：**
   - **提醒一次：** 模型回复里没有工具调用，并且到这时为止一次 `edit_file` 都没成功过，环境就回一条固定的 user 消息，原文是："You have not changed any file yet. Use the tools to find and fix the bug, then reply without a tool call to submit." 每个 episode 最多提醒 1 次，这条提醒占 1 轮。第二次再出现没有工具调用的回复，就视为提交。如果已经成功编辑过文件，没有工具调用的回复直接视为提交，和原来一样。`EpisodeResult` 里新增 `nudged: bool` 字段。
   - **system prompt 加一段 JSON 说明，不超过 60 个英文词：** 工具参数必须是合法的 JSON；反斜杠要转义；不能用 Python 的 `r"..."` 写法；一条回复里每个 `<tool_call>` 都要闭合。只加这一段，不加完整示例，原有内容不改。
   - **parser 保持严格。** 不要为了兼容 `r"..."` 之类的写法去放宽解析。格式本身就是要学的东西。
   - 测试要覆盖：提醒后继续；第二次没有工具调用时视为提交；编辑成功后没有工具调用直接提交；`nudged` 字段能正确序列化。
3. **上下文和轮数：** `max_turns` 从 8 提到 10，因为提醒会多占 1 轮。上下文上限 8k 不变。
4. **付费 API 的配置约定：**
   - DeepSeek 的配置一律从仓库根目录的 `.env` 读取，包括 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`，已存在的环境变量优先。`.env` 不入库，`.env.example` 入库。
   - 模型用 `deepseek-flash`，非思考模式。
   - 计价沿用 A5 的人民币单价表。
   - 每次调用付费 API 都要设花费上限，上限写在计划或 prompt 里。阶段 2 采 SFT 轨迹时也照这条约定执行。

## 本轮步骤

### 1. 冻结题集（相当于原第 7 步，去掉校准相关的部分）

- 在 README 里补上：怎么生成题集、怎么接入一个新的 policy、怎么启动推理服务（vLLM 命令和 parser）。
- 把 `data/tasks/{train,heldout}.jsonl` 的 sha256 写进 manifest，打 tag `phase1-tasks-v1`。题集用现在 `812d5ce` 的版本，不重建。
- 实现修订 4 第 2、3 条的环境修改，全量测试通过后提交，再打 tag `phase1-env-v1`。

### 2. 本地模型诊断扫描

- **题目：** 用试点那 100 道（seed 0，train 和 heldout 各 50 道）。每题跑 4 次，temperature 0.8，环境用 `phase1-env-v1`。
- **模型：** 依次跑下面 5 个，一次只开一个 vLLM 服务，全部用 `--dtype half`、`--max-model-len 8192`：
  1. `Qwen/Qwen2.5-1.5B-Instruct`（在新环境下重跑，用来对比）
  2. `Qwen/Qwen2.5-Coder-1.5B-Instruct`
  3. `Qwen/Qwen2.5-3B-Instruct`
  4. `Qwen/Qwen3-1.7B`，关闭思考模式（请求里传 `chat_template_kwargs={"enable_thinking": false}`）
  5. `Qwen/Qwen3-4B-Instruct-2507`
- **规则：**
  - 下载之前先确认 Hugging Face 上确实有这个模型 ID。没有就跳过并记下来，不要换成别的模型。
  - 每个模型的工具调用 parser 都要按 vLLM 文档确认，把实际用的 parser 写进 PROGRESS。
  - 如果某个模型在 fp16 下输出乱码、出现 NaN，或者服务起不来，比如显存不够，就记下原因、跳过这个模型，不要改量化方式或硬件设置。
  - 跑完一个模型就删掉它的权重缓存，免得把 WSL 磁盘占满。
- **每个模型都要报告：**
  - 解出率，以及 100 道题按解出次数（0、1、2、3、4）的分布；
  - 格式错误率；
  - 第一轮就不调工具的比例，被提醒的比例，提醒之后又开始调工具的比例；
  - `edit_file` 的成功率，以及至少成功编辑过一次的 episode 占多少；
  - 至少有一次 `visible_pass` 的 episode 占多少；
  - 平均轮数，停止原因的分布；
  - 吞吐量，显存占用，服务启动耗时。

### 3. DeepSeek 参考上限（获准使用付费 API，有花费上限）

目的有两个：一是确认这些题在 10 轮之内是解得出来的，二是试一下阶段 2 用来采 SFT 轨迹的那条链路。

- **题目：** 同样的 100 道，每题跑 2 次，temperature 0.8，环境用 `phase1-env-v1`。
- **配置（已由审计方准备好，不用问用户）：**
  - 仓库根目录的 `.env` 已经写好 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL=https://api.deepseek.com/v1`、`DEEPSEEK_MODEL=deepseek-flash`，权限 600。`.gitignore` 已加入 `.env`；`.env.example` 是不含密钥的模板，可以提交。
  - 代码从 `.env` 读配置：用标准库写一个几行的解析函数，按 `KEY=VALUE` 逐行读，不新增依赖。已经存在的环境变量优先。
  - 调用方式照 coding-harness 的 `configs/deepseek.yaml`：OpenAI 兼容的 `POST {base_url}/chat/completions`；请求体加 `"thinking": {"type": "disabled"}`（非思考模式）；`max_tokens` 用 1024，和本地模型一致。
  - WSL 里如果设了代理、直连 DeepSeek 又不通，就让 HTTP 客户端读取 `HTTPS_PROXY`，否则不读。实际用了哪种写进 PROGRESS。
  - 第一次调用前，先用 1 条最短请求验证密钥和模型名，例如"回复 ok"。返回 401、404 或"模型不存在"就停下报告，不要换成别的模型名。
- **密钥安全：**
  - 密钥只能出现在 `.env` 里，不能写进代码、日志、`runs/`、PROGRESS 或 CHECKPOINT。
  - 报错信息里如果带了请求头，打印前先去掉 `Authorization`。
  - 每次提交前，用 `git grep -n "sk-"` 和 `git diff --cached | grep -n "sk-"` 检查一遍，命中就不提交。
- **花费：**
  - 用接口返回的 usage 算累计花费。单价沿用 A5 的人民币计价表（`/mnt/d/crq/src/coding-harness/configs/a5/model-price-binding.json`，元/百万 token）：缓存命中 0.04，未命中 2.00，输出 8.00。缓存命中数取 usage 里的 `prompt_cache_hit_tokens`，没有这个字段就按全部未命中算。
  - 累计花费写进 `runs/deepseek/cost.json`。累计到 10 元就停止，按已完成的部分报告。
- **要报告的：**
  - 解出率和按解出次数的分布；
  - 解出的 episode 的轮数分布（中位数和 p90）。如果 p90 达到 9 轮或更多，要特别指出来；
  - 格式错误率；
  - 总花费，以及单个 episode 的平均花费；
  - 估算一下：要在全部训练题上拿到 1000 条成功轨迹，大概要花多少钱。

### 4. 判分假阴性估计

在第 2、3 步所有没拿到奖励的 episode 里，找出满足下面条件的：

- 所有 `test` 和 `challenge` 类型的隐藏用例都过了；
- 失败的只有 `generated` 用例。

统计这类 episode 有多少。每个模型抽 5 个（不够 5 个就全列），贴出模型改动的 diff 和失败的那条用例，人工判断一下：是生成的输入本身越界了（比如 `find_Extra` 那一类），还是模型的修复确实不对。

### 5. CHECKPOINT-1D

报告里要有：

- 一张汇总表，把 5 个模型和 DeepSeek 放在一起对比，第 2、3 步各项指标都列上；
- 第 4 步的结果；
- 环境修改前后，Qwen2.5-1.5B 的对比；
- 遇到的问题。

不要写训练代码，不要跑全量校准，不要开始阶段 2。
