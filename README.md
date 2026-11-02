# model-data

多源模型信息库，每个数值带**可信度**与**证据链**。

产出：`https://w-run.github.io/model-data/models.json`（每 2 小时重建）

---

## 解决什么问题

上游 API 给的 `max_completion_tokens` 不能直接用。provider 没上报时，
OpenRouter 会用 `floor(context × 0.9)` 填充：

```
deepseek-v4.1-flash:  context = 1048576  →  max_completion_tokens = 943718
```

`943718` 是 `1048576 × 0.9`，不是模型能力。拿它当 `max_tokens`，
输入侧只剩 10%，几万 token 就开始自动压缩 —— 这是本库的出发点。

三个源里这类脏值：models.dev 1410 条、OpenRouter 85/446、LiteLLM 若干。
本库的做法不是「删掉脏值」，而是**判断哪些值可信、可信到什么程度**。

---

## 产出

| 文件 | 内容 |
|---|---|
| `models.json` | 模型列表 |
| `vendors.json` | 厂商表（显示名、别名、官网、文档、币种、是否自有） |
| `index.html` | **搜索页**，按 id / 别名 / 厂商检索 |

搜索页默认按发布日期降序，支持多关键词 AND、快捷过滤（人民币计价 / 高可信 / 视觉 / 推理 / 工具 / 自有）。

### models.json 格式

与既有对接完全一致，仅新增 `confidence` / `release_date` / `billing` 等字段（不认识也不影响）：

```jsonc
{
  "version": "2.0.0",
  "generated_at": "2026-11-02T...",
  "count": 3099,
  "models": [{
    "id": "deepseek-v4.1-flash",
    "vendor": "deepseek",
    "name": "DeepSeek V4.1 Flash",
    "aliases": ["deepseek-v4.1-flash"],
    "context": 1024000,          // 十进制整 K，向下取整
    "max_output": 384000,
    "modalities": {"input": ["text", "image"], "output": ["text"]},
    "capabilities": {"tool_call": true, "reasoning": true, "vision": true},
    "price": {"unit": "USD/Mtok", "input": 0.3, "output": 1.2, "cache_read": 0.006},
    "kind": "chat",
    "description": "...",
    "release_date": "2026-09-10",
    "sources": ["litellm", "mimi", "models.dev", "openrouter"],

    // ↓ 新增，可选读
    "confidence": {
      "level": "verified",
      "overall": 1.0,
      "context":     {"level": "verified", "confidence": 1.0, "agreement": 0.84, "n_sources": 32},
      "max_output":  {"level": "verified", "confidence": 1.0, "agreement": 0.54, "n_sources": 30}
    }
  }]
}
```

### 容量记法

统一用**十进制整 K**，向下取整：

```
1048576 → 1024000      393216 → 384000      262144 → 256000
```

声明值本就是理论上限，tokenizer 边界、system prompt、特殊 token 都会占位，
实际永远用不满 —— 取整不损失可用空间，且只向下取整，消费方按它算不会超限。

取整后若 `max_output >= context`（把上下界拉平），会压到窗口的 3/4 并记入
`exact` 供回溯。非整 K 的值（如 943718）原样保留，不强套。

---

## 可信度怎么算

`level` 五级：`verified` ≥0.90 · `high` ≥0.70 · `medium` ≥0.50 · `low` <0.50 · `unknown`

### 源分层

| tier | 源 | 权重 |
|---|---|---|
| `manual` | 人工标称 | 1.00 |
| `first_party` | models.dev 第一方 provider（人工核对、带出处） | 0.95 |
| `official_cloud` | Azure / Bedrock / Vertex 转售 | 0.80 |
| `aggregator` | 聚合商 | 0.55 |
| `community` | LiteLLM | 0.50 |
| `openrouter` | OpenRouter（仅补覆盖） | 0.30 |

单条权重 = `tier + 出处加成(doc_url +0.10) × 新鲜度(180 天线性衰减)`。

### 三条关键规则

**1. 权威优先，不被票数稀释**

20 家聚合商一致报 `1000000`，原因常是它们都从官方抄来后做了人类友好化取整，
而非各自独立观测。所以有第一方条目时以它为准，聚合商票数不参与加权。

**2. 上下文与输出必须同源**

`context` 和 `max_output` 分开投票会拼出物理上不存在的组合：

```
unslopnemo-12b
  models.dev/aimlapi   ctx=32768    out=32768(无效)
  openrouter           ctx=1024000  out=819200
→ ctx 取 32768，out 取 819200 —— 荒谬
```

所以对 `(context, max_output)` 整体聚类，每个簇内再定两个值，天然同源。

**3. 等价容量要合并，不同容量要分开**

```
393216 (=384×1024)  与  384000 (=384×1000)   → 同一个 384K，合并
1048576 (=1024²)    与  1000000 (=1000²)      → 不同容量，分开
```

用 K 倍数归一化而非百分比容差：`384000 % 1024 == 0`（=375×1024），
只看「能否整除 1024」会把它算成 375K，所以要多基数并查集。

---

## mimi 特殊处理

`mimi.wrun.cc` 是自建聚合网关，兼有两类模型：

**自有模型**（`mimi-lite` / `mimi-free` / `mimi-flash`）
别处没有，是唯一来源 → 采信，`level: "own"`，置信度 0.70。
同一 id 多个上游条目时取**最小**（保守）。

**转售模型**（`deepseek-v4.1-flash` / `glm-5.3` …）
中转层可能滞后或降配（实测 catpaw 报 600k，官方实为 1M）。
所以**按保守值取会把能力压掉**，只作为一条普通观测参与投票。

---

## 数据源

| 源 | 形态 | 说明 |
|---|---|---|
| `sst/models.dev` 仓库 | TOML | 稀疏克隆，有出处注释与核对日期，优先 |
| LiteLLM | JSON | 社区维护，多数条目带 `source` URL |
| OpenRouter | API | 仅补覆盖，占位值判无效 |
| mimi | API | 自有模型 + 转售，见上 |

models.dev 用仓库而非 API，是因为 API 只给裸数字，无从判断真假；
TOML 里有 `# Sources (api-docs.deepseek.com, accessed 2026-10-19)`。

86% 的 models.dev 条目靠 `base_model` 继承。解析器会沿链最多 8 跳解析，
所以实际可用数据比 `[limit]` 的 51% 覆盖率高得多。

---

## 开发

```sh
pip install -e .            # 或直接用 PYTHONPATH=src
PYTHONPATH=src python -m unittest discover -s tests -v    # 单测，不联网
PYTHONPATH=src python -m model_data.cli                   # 全量构建
PYTHONPATH=src python -m model_data.cli --strict          # CI 用，有问题即失败
PYTHONPATH=src python -m model_data.cli --only mimi       # 只跑单源
PYTHONPATH=src python -m model_data.cli --offline         # 用缓存
PYTHONPATH=src python -m model_data.cli --dump-observations obs.json   # 导出原始观测
```

### 目录

```
src/model_data/
  cli.py              入口：采集 → 装配 → 校验 → 写出
  assemble.py         证据 → 条目（保持既有格式）
  identity.py         模型身份对齐（4.5/4-5、日期快照、:batch 变体）
  notation.py         容量记法（向下取整 + 钳制）
  trust/engine.py     可信度引擎：分层、聚类、投票、证据链
  sources/
    models_dev_repo.py  TOML 解析 + base_model 继承
    collect.py          各源 → 统一 Observation
    litellm.py openrouter.py mimi.py
tests/
data/vendors.toml     厂商元信息
```

---

## 已知限制

- `unknown` 归属仍有约 1700 个 —— 多为聚合商独有的小模型，无 org 信息可查。
- 价格只取 `input`/`output`/`cache_read` 三项，阶梯价、batch 折扣未建模。
- 同一模型的不同路由（`-fast` / `-flex`）被当作独立模型，未做归并。
- LiteLLM 条目的 `observed_at` 全为空，新鲜度按 0.85 折损。
