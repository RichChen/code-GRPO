# 原始数据来源

两份文件都从 GitHub 的 raw 地址下载，URL 固定到具体 commit，下载后校验 sha256（`arl.taskgen.sources.RAW_FILES`，`download_raw()` 校验不一致就报错）。没有使用任何第三方镜像。

重新下载：

```bash
# 本机直连 raw.githubusercontent.com 不稳定（Connection reset），下载时经 Windows 上的代理
export HTTPS_PROXY=http://<WSL 宿主 IP，见 ip route>:7890
uv run python -m arl.taskgen.build --raw data/raw --out data/tasks --download
```

| 文件 | URL | sha256 | 大小 |
| --- | --- | --- | --- |
| `sanitized-mbpp.json` | https://raw.githubusercontent.com/google-research/google-research/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/sanitized-mbpp.json | `ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9` | 255053 B |
| `HumanEval.jsonl.gz` | https://raw.githubusercontent.com/openai/human-eval/d321ec0b6c23dec317337be99f6d0c45ca73f3d5/data/HumanEval.jsonl.gz | `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef` | 44877 B |

## 许可证

- **MBPP（sanitized）**：位于 `google-research/google-research` 仓库，仓库根目录 `LICENSE` 是 Apache License 2.0（同一 commit 下 sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`）。`mbpp/README.md`（sha256 `a648ec210c94b614b63e0065c5af3f2e8fe144a8d7dbe3a2149baead9d87ff92`）没有单独写许可证。Hugging Face 上的 MBPP 数据卡写的是 CC BY 4.0，这一点没有在 GitHub 上找到出处，未核实。论文：Austin et al., 2021, *Program Synthesis with Large Language Models*。
- **HumanEval**：`openai/human-eval` 仓库，MIT License（`LICENSE` sha256 `bcba3de214851cce46ed5af42d6698044616eeace887c3231bc7a20474ab639e`，Copyright (c) OpenAI）。LICENSE 文件是在 commit `d321ec0`（2021-07-14）加入的，所以 URL 固定到这个 commit。数据文件在更早的 `463c980`、`d321ec0` 和当前 master 上 sha256 相同。论文：Chen et al., 2021, *Evaluating Large Language Models Trained on Code*。

两份许可证都允许再分发（需保留署名和许可证声明），所以原始文件提交在仓库里。生成的 `data/tasks/*.jsonl` 由这两份文件派生，体积较大（约 27MB），不提交，用固定 seed 重新生成，sha256 记录在 `data/tasks/manifest.json`。
