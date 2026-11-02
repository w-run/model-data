# model-data

模型信息库。目标不只是「列出模型」，而是给出**消费方直接能用**的参数。

产出（gh-pages 托管）：

| 文件 | 内容 |
|---|---|
| `models.json` | 模型列表（含 `limit` / `recommended`） |
| `vendors.json` | 厂商列表 |
| `index.json` | 别名索引，`别名 -> 模型 id` |

---

## 为什么需要 `recommended`

上游（OpenRouter 等聚合平台）给的 `max_completion_tokens` 不能直接用。

provider 没有上报这个值时，OpenRouter 会用 `floor(context * 0.9)` 填充：

```
deepseek-v4.1-flash:  context = 1048576  ->  max_completion_tokens = 943718
```

`943718` 看起来是个合法数字，实际是 `1048576 × 0.9`。把它当 `max_tokens` 传给
客户端，输入侧就只剩 `104858`（10%）—— 喂到几万 token 就开始自动压缩/截断。

本库会把这类**占位值识别出来并替换**，同时区分两个层面：

- **`limit`** —— 客观能力。只用于校验「这条请求会不会被拒」。
- **`recommended`** —— 使用预算。客户端直接拿这个值。

### 示例

```jsonc
{
  "id": "deepseek-v4.1-flash",
  "limit": {
    "context": 1000000,            // 总窗口
    "max_output": 384000,          // 真实上限（硬限制）
    "semantics": "shared",         // context 是输入+输出共享
    "confidence": "first_party",   // 这个结论有多可信，见下文
    "votes": 44,
    "checked_at": "2026-11-02",
    "sources": ["deepseek", "..."],
    "openrouter_placeholder_output": 943718   // 被丢弃的原始值，留痕
  },
  "recommended": {
    "input_budget": 868928,        // = context - output_budget
    "output_budget": 131072,       // 128K 封顶
    "output_cap": 131072
  }
}
```

**关键**：`limit.max_output` 是 384000，但 `recommended.output_budget` 只有 131072。
真实上限更高，不代表应该一次吃满 —— 输出预算越大，留给输入的窗口越小。

---

## `confidence` 的含义

| 值 | 含义 | 可信度 |
|---|---|---|
| `first_party` | 厂商官方口径（models.dev 的官方 provider） | 最高 |
| `consensus` | 多个独立 provider 交叉一致 | 高 |
| `inferred` | 无共识，回落到单一来源或继承值 | **用前请自己核对** |

消费方建议：`inferred` 的模型不要直接用于自动压缩阈值，或至少自己留 20% 余量。

---

## 数据是怎么定的

### 1. 占位值识别

```python
output >= context           # 共享窗口下不可能
output == floor(ctx * 0.9)  # OpenRouter 的填充公式
```

命中即丢弃，**绝不当成真实能力**，必要时留 `null` 也不给假值。

### 2. 跨源共识投票

models.dev 上同一模型有 200+ 个 provider 各自报数，按容量聚类投票：

```
deepseek-v4.1-flash 的 ctx 分布：
  1048576  ← 48 票
  1000000  ← 44 票        （1Mi vs 1M，都是 1M，聚类合并）
  524288   ←  1 票        （512K，真的不同，独立）
```

- 聚类按「K 倍数」而非百分比容差。`393216 (=384×1024)` 和 `384000 (=384×1000)` 是同一个 384K，必须合并；`524288` 和 `500000` 是不同容量，必须分开。
- **第一方票加权 2x**，但不给无限权重：官方孤票 vs 20 家聚合共识，还是共识赢。
- 簇内取**众数**，不取 min —— 个别 provider 少报 16 个 token 会把结果带偏。

### 3. 变体继承

`:batch` / `:free` 等变体与基础模型共用同一窗口，但 models.dev 上没有独立条目。
自身置信度不足时继承基础模型结论。`:vision` / `-exp` / `-preview` **不继承**（是不同模型）。

### 4. 校验兜底

CI 里强制校验（`--strict`，有 error 直接失败）：

- `input_budget + output_budget <= context`
- `output_budget <= max_output`
- `confidence` 为 `first_party`/`consensus` 时，`max_output` 不得是占位值
- 输入预算不得低于 context 的 50%
- 全局：占位值残留率不得超过 25%

---

## 作为消费方怎么用

```python
import json, urllib.request

data = json.load(urllib.request.urlopen("https://w-run.github.io/model-data/models.json"))
model = next(m for m in data["models"] if m["id"] == "deepseek-v4.1-flash")

rec = model["recommended"]
max_tokens = rec["output_budget"]           # 传给 API 的 max_tokens
compress_at = rec["input_budget"] * 0.8     # 自动压缩阈值，留 20% 余量

if model["limit"]["confidence"] == "inferred":
    compress_at *= 0.5                       # 数据不可靠，更保守
```

**不要**这么做：

```python
model["limit"]["max_output"]   # 别直接当 max_tokens —— 这是硬上限，不是预算
model["limit"]["context"]      # 别直接当可用输入 —— 输出要占位置
```

---

## 本地开发

```sh
python -m unittest discover -s tests -v    # 单元测试，不联网
python -m scripts.build                    # 全量构建
python -m scripts.build --strict           # 严格校验（CI 用）
python -m scripts.build -o /tmp/out        # 输出到别处
python -m scripts.build --only openrouter,manual   # 跳过 models.dev（离线调试）
```

### 目录

```
scripts/
  build.py      主流程：采集 -> 合并 -> 共识 -> 继承 -> 人工覆盖 -> 校验 -> 产出
  limits.py     占位识别 / 聚类投票 / 预算推导   ← 核心逻辑，改动请先看 tests/
  inherit.py    变体继承
  validate.py   产出校验
  sources/      models_dev.py（共识票源）
  scrapers/     厂商专属爬虫（千帆/火山等，尚未接入 build）
data/
  vendors.toml  厂商元信息 + 第一方 provider 白名单
  manual/       人工标称（最高优先级，支持深层合并）
tests/
```

### 人工标称

优先级最高，只写要改的字段即可（深层合并，其余保留自动推导）：

```jsonc
{
  "models": {
    "deepseek-v4.1-flash": {
      "limit": { "max_output": 393216, "confidence": "first_party" },
      "recommended": { "output_budget": 65536 }
    }
  }
}
```

手改 `limit` 但没给 `recommended` 时，会自动重算预算，不会出现两者打架。

---

## 已知限制

- `scrapers/` 里千帆、火山豆包等厂商专属采集器**尚未接入** `build.py`（当前是死代码）。
- `:batch` 变体的价格与基础模型不同，继承只处理窗口，不处理价格。
- 中文厂商（字节、腾讯、百度）在 models.dev 上没有第一方 provider，只能靠共识或人工。
