# 服务器和本地之间怎么同步

阶段 3 的做法：GitHub 上的仓库是私有的，服务器上没有凭据，`git pull` 和 `git push` 都用不了。所以代码用 git bundle 传，数据用 tar 打包，再用 sha256 逐个文件核对。下面的 `server` 指服务器上的普通用户（例如 `arl@<ip>`），仓库在服务器的 `~/work/agentic-rl`。

**租机时必须选驱动 580 以上的镜像**（`nvidia-smi` 里的 CUDA Version 至少 13.0）。torch 2.13.0 和 vLLM 0.31.0 没有能在 CUDA 12.8 或 12.6 上用的官方组合，`setup.sh` 遇到更低的驱动会停下。详见 PROGRESS 里修订 2C-3 第 5 条。

## 代码：本地 → 服务器（git bundle）

第一次：

```bash
git bundle create /tmp/arl.bundle main --tags
sha256sum /tmp/arl.bundle
rsync -a /tmp/arl.bundle server:/tmp/
ssh server 'sha256sum /tmp/arl.bundle && git bundle verify /tmp/arl.bundle \
  && git clone -b main /tmp/arl.bundle ~/work/agentic-rl'
```

之后只传服务器上还没有的提交。`<server-head>` 是服务器上 `git rev-parse main` 的结果：

```bash
git bundle create /tmp/arl-update.bundle <server-head>..main
rsync -a /tmp/arl-update.bundle server:/tmp/
ssh server 'cd ~/work/agentic-rl && git bundle verify /tmp/arl-update.bundle \
  && git fetch /tmp/arl-update.bundle main:refs/remotes/bundle/main && git merge --ff-only bundle/main'
```

`git bundle verify` 会检查 bundle 依赖的提交服务器上都有。只用 `--ff-only` 合并，服务器上有本地没有的提交时会停下，不会自动生成合并提交。

## 代码：服务器 → 本地

服务器上的提交（PROGRESS、报告、小脚本）也用 bundle 拉回来。`<base>` 是上一次从本地传过去的提交：

```bash
ssh server 'cd ~/work/agentic-rl && git bundle create /tmp/server.bundle <base>..main'
rsync -a server:/tmp/server.bundle /tmp/
git bundle verify /tmp/server.bundle
git fetch /tmp/server.bundle main:refs/remotes/server/main
git log --oneline main..server/main      # 先看一遍要合进来的提交
git merge --ff-only server/main
```

## 数据：本地 → 服务器

不在 git 里、要单独传的文件：

- `runs/sft-v1/adapter/`：SFT-v1 的 fp32 LoRA。服务器上用 `merge_fp16.sh` 把它合并进以 fp16 加载的原始基座。
- `data/tasks/train.jsonl`、`data/tasks/heldout.jsonl`：题面。heldout 只在预注册的那一次评测里用，训练和 bench 都不读它。题号文件 `data/tasks/grpo_v1_ids.txt`、`dev_ids.txt` 在 git 里。
- `data/sft/v2/`：只有要在服务器上重放 SFT 数据校验时才需要。正式 GRPO 不读它。

不要传：

- `.env`。训练不调用付费 API。
- `runs/sft-preview/merged/`。服务器要自己用 `merge_fp16.sh` 从原始基座合并，不用这份本地预览目录。
- `runs/grpo-calib/` 的 jsonl。题号文件已经是校准的结果。

传之前记下 sha256，传完在服务器上核对：

```bash
sha256sum runs/sft-v1/adapter/* data/tasks/train.jsonl data/tasks/heldout.jsonl > /tmp/upload.sha256
rsync -aR runs/sft-v1/adapter data/tasks/train.jsonl data/tasks/heldout.jsonl server:~/work/agentic-rl/
rsync -a /tmp/upload.sha256 server:/tmp/
ssh server 'cd ~/work/agentic-rl && sha256sum -c /tmp/upload.sha256'
```

`rsync -R` 保留相对路径，文件落在服务器仓库里的同一位置。

`configs/sft/v1.json` 里记着适配器三个文件的 sha256，`data/tasks/manifest.json` 的 `frozen` 里记着两个 jsonl 的 sha256，两边都可以对照。

## 数据：服务器 → 本地（tar + sha256）

在服务器上先生成逐文件的 sha256 清单，再打包。两个 fp16 合并模型不拉回，它们能用适配器加原始基座重新合并：

```bash
ssh server 'cd ~/work/agentic-rl && find runs data -type f \
    ! -path "runs/sft-v1/merged-fp16/*" ! -path "runs/grpo-v1/merged-fp16-step-*/*" -print0 \
  | sort -z | xargs -0 sha256sum > /tmp/phase-files.sha256 \
  && cut -c67- /tmp/phase-files.sha256 > /tmp/phase-files.list \
  && tar -cf /tmp/phase-files.tar -T /tmp/phase-files.list \
  && sha256sum /tmp/phase-files.tar'
rsync -a server:/tmp/phase-files.tar server:/tmp/phase-files.sha256 /tmp/
sha256sum /tmp/phase-files.tar           # 和服务器上打印的值比
tar -xf /tmp/phase-files.tar             # 在仓库根目录解开
sha256sum -c /tmp/phase-files.sha256     # 逐个文件核对，全部 OK 才算拉回完成
```

- 解包会覆盖本地同名文件。本地已有、不想覆盖的目录，先解到别处比对。
- 阶段 3 拉回了 165 个文件，逐个核对过 sha256（`docs/plan/PHASE-4-CLOSEOUT.zh-CN.md` 第 0 节第 2 条）。
- 不要在拉回之前改 `docs/plan/HELDOUT-PREREG.zh-CN.md`。
