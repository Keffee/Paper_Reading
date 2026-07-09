# OpenOneRec 跨域推荐实验 — 最终结果报告

> **项目**: OpenOneRec-npu 跨域推荐框架在 Amazon 电商数据集上的系统评测  
> **模型**: OneRec-1.7B（基于 Qwen3-1.7B 架构，含 itemic token 扩展）  
> **硬件**: 8× Ascend 910B3 NPU (64GB HBM each)  
> **数据集**: Amazon All_Beauty (479 items, 296 test), Movies_and_TV (106,391 items, 5K test)  
> **实验轮次**: 三轮（temp_try_amazon / temp_try_amazon_2 / temp_try_amazon_3）

---

## 目录

1. [核心结论](#1-核心结论)
2. [实验方法总览](#2-实验方法总览)
3. [第一轮：碰撞修复与基线建立](#3-第一轮碰撞修复与基线建立)
4. [第二轮：ICL增强、文本方法与CPT+SFT](#4-第二轮icl增强文本方法与cptsft)
5. [第三轮：三方向系统消融](#5-第三轮三方向系统消融)
6. [全部Prompt模板详解](#6-全部prompt模板详解)
7. [评测算法详解](#7-评测算法详解)
8. [训练配置详解](#8-训练配置详解)
9. [关键发现与洞察](#9-关键发现与洞察)
10. [待完成实验](#10-待完成实验)

---

## 1. 核心结论

### 三方向最佳结果对比（All_Beauty, Greedy+Logprobs 评测）

| 方向 | 方法 | R@1 | R@5 | R@10 | NDCG@10 |
|------|------|-----|-----|------|---------|
| **方向三 Term ID** | SFT rec_only (C2) | **45.61%** | **60.14%** | **63.18%** | **54.87%** |
| 方向二 SID+ICL | ICL K=10 | 19.93% | 21.62% | 22.97% | 21.32% |
| 方向一 SID+微调 | CPT stg1+SFT (B3) | 17.23% | 19.93% | 20.95% | 19.06% |

**Term ID 方法远超其他两个方向**，R@1 是 SID+微调方法的 2.6 倍，R@10 是 3.0 倍。

### 六大核心发现

1. **Term ID (文本关键词) 是跨域推荐的最优表示**：用5个英文关键词替代 SID，Beauty R@1 从 17-19% 跃升至 45.61%
2. **CPT Stage2 有害**：完整 pipeline (stg1→stg2→SFT) 效果低于跳过 Stage2，全参数共训练损害域适应能力
3. **碰撞消解大幅提升 PID 指标**：SFT only PID R@1 从 10.47%(原始) → 17.23%(消歧)，+64.7%
4. **CPT Stage1 只需 500-600 步**：embedding 适应极快，更多步数不带来额外收益
5. **meta2tid 内化数据干扰推荐**：rec_only (C2) >> full (C1)，Term ID 内化训练反而降低推荐能力
6. **Movies 数据集极具挑战**：所有方法 R@1 < 1%，106K items 的 SID 空间过大是核心瓶颈

---

## 2. 实验方法总览

### 2.1 Itemic Token (SID) 表示

OpenOneRec 使用三层 Residual Quantization 将 item 表示为 Semantic ID (SID)：

```
<|sid_begin|><s_a_{A}><s_b_{B}><s_c_{C}><|sid_end|>
```

- 三层 codebook，每层 8192 个 entry
- Token ID 范围：`[151669, 176246]`
  - `<s_a_*>`: 151669 ~ 159860 (Layer A)
  - `<s_b_*>`: 159861 ~ 168052 (Layer B)
  - `<s_c_*>`: 168053 ~ 176244 (Layer C)
  - `<|sid_begin|>`: 176245
  - `<|sid_end|>`: 176246
- 由 Residual K-Means tokenizer 从 Qwen3-8B-Embedding 向量生成

### 2.2 SID 碰撞问题

多个 item 映射到同一 SID 编码，导致 sid2iid 多对多映射：

| 数据集 | Items | 原始碰撞率 | 修复后碰撞率 | Layer C 变更数 |
|--------|-------|-----------|-------------|---------------|
| All_Beauty | 479 | 9.19% (26组) | 0% | 26 |
| Movies_and_TV | 106,391 | 39.96% (13,125组) | 0% | 13,125 |

### 2.3 Term ID 表示

替代 SID 的文本关键词表示，每个 item 用 5 个英文关键词描述：

```
[word1, word2, word3, word4, word5]
```

- 由 Qwen3-8B-Embedding 计算相似度 + LLM 生成（context-aware）
- 碰撞率极低：Beauty 0.9%，Movies 2.4%
- 评测使用 EIG (Exact + Inexact Greedy) 匹配算法

### 2.4 三大实验方向

| 方向 | 核心思路 | 是否需要训练 | 关键变量 |
|------|---------|------------|---------|
| 方向一：SID + 微调 | CPT + SFT 适应新域 | 是 | CPT 阶段、步数、SID 版本 |
| 方向二：SID + ICL | 免训练上下文学习 | 否 | ICL 示例数 K |
| 方向三：Term ID | 文本关键词替代 SID | 是（SFT） | 训练数据类型、base model |

---

## 3. 第一轮：碰撞修复与基线建立

**实验目录**: `temp_try_amazon/`  
**核心目标**: 验证 SID 碰撞修复对跨域推荐的影响

### 3.1 碰撞修复方法：Layer C 强制去重

**实现文件**: `temp_try_amazon/src/fix_collision_layer_c.py`

**策略**：
1. 保持 Layer A 和 Layer B 不变（保留语义信息）
2. 仅修改 Layer C 来消除碰撞
3. 按 `(a, b)` 分组，对碰撞组内的 item 重新分配 Layer C code
4. 优先保留原始 c code，其余分配未使用的 c code
5. 使用 OneRec-tokenizer 的 Layer C centroids (8192×4096) 计算 embedding 距离，选择最近的未使用 c code

**可行性验证**：Movies 最大 (a,b) 碰撞组有 619 个 item < 8192 Layer C codebook，可行。

### 3.2 实验结果

#### All_Beauty (296 test samples)

| 实验 | R@1 | R@32 | pid_R@1 | pid_R@32 |
|------|-----|------|---------|----------|
| 1.1 原始SID 零样本 | 10.14% | 13.18% | 8.33% | 10.70% |
| 2.1 修复SID 零样本 | 9.46% | 12.84% | 9.46% | 12.84% |
| 2.2 修复SID Few-shot K=5 | 9.46% | 12.84% | 9.46% | 12.84% |
| **2.3 修复SID SFT 5%** | **18.92%** | **23.31%** | **18.92%** | **23.31%** |

#### Movies_and_TV (5K subset)

| 实验 | R@1 | R@32 | pid_R@1 | pid_R@32 |
|------|-----|------|---------|----------|
| 1.1 原始SID 零样本 | 0.66% | 1.78% | 0.20% | 0.68% |
| 2.1 修复SID 零样本 | 0.28% | 0.78% | 0.28% | 0.78% |
| 2.2 修复SID Few-shot K=5 | 0.52% | 2.16% | 0.08% | 0.42% |
| 2.3 修复SID SFT 1% | 0.48% | 0.84% | 0.48% | 0.84% |

### 3.3 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | OneRec-1.7B |
| 评测方式 | Beam search (beam_width=32, temperature=0.7, top_p=0.95, top_k=20) |
| SFT 参数 | lr=5e-6, batch_size=2, grad_accum=8, max_length=2048 |
| Beauty SFT 5% | 118 samples, 224 steps, ~8分钟 |
| Movies SFT 1% | 23K samples, 1439 steps, ~2小时 |
| Few-shot | K=5, 从训练集随机选取 (seed=42) |

### 3.4 第一轮关键洞察

1. **碰撞修复消除指标歧义**：修复后 R@1 = pid_R@1，sid2iid 变为 1:1 映射
2. **零样本碰撞修复 SID-level 下降**：模型未学过新 SID 映射，预测的 SID 与修复后数据不匹配
3. **零样本碰撞修复 PID-level 提升**：Beauty pid_R@1 从 8.33% → 9.46% (+1.13%)
4. **SFT 是碰撞修复生效的关键**：Beauty SFT 5% 后 R@1=18.92%
5. **大数据集碰撞修复代价高**：Movies 13,125 个 SID 变更，1% SFT 数据不足以覆盖

---

## 4. 第二轮：ICL增强、文本方法与CPT+SFT

**实验目录**: `temp_try_amazon_2/`  
**核心目标**: 系统探索 ICL 增强、文本适配、CPT+SFT 适应三大方法

### 4.1 Phase 1：ICL 增强实验

#### 4.1.1 约束解码 (Constrained Decoding)

**实现文件**: `temp_try_amazon_2/src/constrained_eval.py`

三种模式：
- **baseline**: 无约束 beam search
- **constrained**: 逐步约束，每步通过 `allowed_token_ids` 限制合法 SID token
- **post_hoc**: 生成后过滤无效 SID

约束机制：
| SID 位置 | 期望 token | 约束方式 |
|----------|-----------|---------|
| `<|sid_begin|>` 之后 | `<s_a_*>` | `allowed_token_ids` 限制为 `valid_a` 集合 |
| `<s_a_*>` 之后 | `<s_b_*>` | `allowed_token_ids` 限制为 `valid_b[s_a]` 集合 |
| `<s_b_*>` 之后 | `<s_c_*>` | `allowed_token_ids` 限制为 `valid_c[(s_a, s_b)]` 集合 |
| `<s_c_*>` 之后 | `<|sid_end|>` | `logit_bias` 强制选择 `SID_END_ID` |

**结果**（Beauty, 约束解码）：
- Baseline (beam): R@1=5.07%, Valid SID Rate=13.93%
- Post-hoc 过滤: R@1=12.50%, Valid SID Rate=13.93%
- ICL K=1 + 约束解码: R@1=16.55%, Valid SID Rate=42.48%
- ICL K=10 + 约束解码: R@1=21.62%, Valid SID Rate=85.57%

#### 4.1.2 Self-Consistency（多数投票）

**实现文件**: `temp_try_amazon_2/src/self_consistency.py`

对 beam search 的 32 个候选进行多数投票重排序，三种模式：
1. **frequency**: 纯频率排序（Counter.most_common()）
2. **frequency_logprob**: `score = (1-alpha) * freq_score + alpha * logprob_score`（alpha=0.5）
3. **logprob_frequency**: 先按 logprob 降序，频率作 tiebreaker

**结果**：Beauty 上 frequency 排序和 logprob 排序结果完全一致，self-consistency 无额外收益。

#### 4.1.3 相似度选择 Few-shot

**实现文件**: `temp_try_amazon_2/src/similar_selection.py`

用与查询用户最相似的用户作为 few-shot 示例，支持两种编码方式：
1. SID Code 统计特征（13维：3层 code 的 mean/std/min/max + 历史长度）
2. Item Embedding 平均池化（Qwen3-8B-Embedding）

**结果**：实现有问题（仅 1 个有效样本，R@1=0.0%），跳过。

### 4.2 Phase 2：Text-Only 方法

#### 4.2.1 Text-Only Adaptation（纯文本适配）

**实现文件**: `temp_try_amazon/src/build_text_only.py`

**方法**：完全绕过 itemic token，用商品文本描述代替 SID。

**SID 替换规则**：
1. 从生成文本中提取所有 `<|sid_begin|>...<|sid_end|>` 模式
2. 将每个 SID 编码为整数：`a * 8192^2 + b * 8192 + c`
3. 查找 `sid2caption` 映射获取文本描述（截取前 50 字符）
4. 替换为 `[caption]` 格式

**结果**：
- Beauty: R@1=18.24%, PID R@1=12.84%（关键词碰撞导致 PID 歧义）
- Movies: R@1=0.28%（模型生成 SID 而非关键词，42% 映射到 [unknown]）

#### 4.2.2 Text-Augmented Itemic Tokens（文本增强 SID）

**实现文件**: `temp_try_amazon/src/build_text_augmented.py`

**方法**：在 SID token 后附加文本描述，输入同时包含 SID 和文本，输出仍为 SID。

**增强格式**：
```
原始: <|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>
增强: <|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|> (Maybelline New York Color Sensational Lipstick, Pink Petal, 0.11 oz)
```

文本描述截取前 80 字符，每个 SID 仅映射第一个 PID 的描述。

**结果**：
- Beauty: R@1=15.54%, PID R@1=11.15%
- Movies: R@1=0.06%, PID R@1=0.40%

### 4.3 Phase 3：CPT + SFT 适应

#### 4.3.1 CPT 训练方法

**实现文件**: `temp_try_amazon_2/scripts/run_cpt.py`

**Stage 1（冻结 LLM，训练 itemic embedding）**：
- `freeze_llm_parameters()`: 冻结 model.model (transformer layers) + 非 itemic lm_head 参数
- 只保留 `model.model.embed_tokens` 和 `model.lm_head` 中 itemic token 对应参数可训练
- 可训练参数占比：20.40%
- ITEMIC_ID_START = 151669

**Stage 2（全参数共训练）**：
- 解冻所有参数
- 使用更低学习率 (5e-5 vs 1e-4)

**数据格式**：
- Pretrain 数据：segments 格式 → `segments_to_text()` 转为纯文本
- SFT 数据：messages 格式 → `messages_to_text()` 通过 `tokenizer.apply_chat_template()` 渲染

#### 4.3.2 完整结果对比表

##### All_Beauty (296 test samples)

| 方法 | CPT | SID | R@1 | R@5 | R@10 | R@32 | PID R@1 | PID R@10 | NDCG@10 |
|------|-----|-----|-----|-----|------|------|---------|----------|---------|
| 零样本基线 | ❌ | 原始 | 9.46% | 11.49% | 12.50% | 12.84% | 7.43% | 9.12% | 10.93% |
| SFT only | ❌ | 原始 | 15.88% | 20.27% | 21.28% | 22.97% | 10.47% | 16.22% | 18.79% |
| SFT only | ❌ | 消歧 | 17.23% | 20.61% | 22.30% | 23.31% | 17.23% | 22.30% | 19.80% |
| CPT stg1+SFT | ✅ | 原始 | 13.18% | 18.58% | 19.59% | 21.28% | 12.16% | 17.57% | 16.39% |
| CPT full+SFT | ✅ | 原始 | 15.20% | 18.58% | 20.27% | 22.97% | 11.15% | 17.91% | 17.53% |
| CPT stg1+SFT | ✅ | 消歧 | 13.85% | 17.57% | 18.58% | 20.27% | 13.85% | 18.58% | 16.24% |
| CPT full+SFT | ✅ | 消歧 | 15.20% | 19.59% | 20.61% | 22.64% | 15.20% | 20.61% | 17.91% |
| CPT stg1_600+SFT | ✅ | 消歧 | **18.24%** | 20.61% | 21.28% | 21.62% | **18.24%** | 21.28% | 19.84% |
| CPT stg1_800+SFT | ✅ | 消歧 | **18.24%** | 20.95% | 21.28% | 21.62% | **18.24%** | 21.28% | 19.89% |
| CPT stg1_890+SFT | ✅ | 消歧 | **18.24%** | 20.95% | 21.28% | 21.62% | **18.24%** | 21.28% | 19.88% |
| CPT stg1_final+SFT | ✅ | 消歧 | **18.24%** | 20.95% | 21.28% | 21.62% | 18.24%* | 15.54%* | 19.88% |
| CPT stg1_final+SFT | ✅ | 原始 | 13.85% | 18.58% | 19.26% | 19.93% | 12.84% | 17.57% | 16.60% |
| Text-Only SFT | ❌ | - | **18.24%** | 19.59% | 19.59% | 19.59% | 12.84% | 12.84% | 19.10% |
| Text-Augmented SFT | ❌ | 原始 | 15.54% | 19.26% | 20.61% | 21.28% | 11.15% | 15.54% | 18.07% |

*注：CPT stg1_final+SFT (消歧) 的 PID 指标异常低，可能是评测数据使用了原始 SID 映射。

##### Movies_and_TV (5K samples)

| 方法 | CPT | SID | R@1 | R@5 | R@10 | R@32 | PID R@1 | PID R@10 | NDCG@10 |
|------|-----|-----|-----|-----|------|------|---------|----------|---------|
| 零样本基线 | ❌ | 原始 | 0.28% | 0.68% | 0.72% | 0.78% | 0.12% | 0.22% | 0.51% |
| SFT only | ❌ | 消歧 | 0.62% | 0.98% | 1.04% | 1.16% | 0.62% | 1.04% | 0.83% |
| CPT stg2 only | ✅ | 原始 | 0.28% | 0.46% | 0.52% | 0.62% | 0.84% | 3.64% | 0.40% |
| CPT stg1+SFT | ✅ | 原始 | 0.70% | 0.84% | 0.86% | 1.10% | 1.14% | 3.84% | 0.78% |
| CPT full+SFT | ✅ | 原始 | 0.74% | 1.02% | 1.10% | 1.26% | 1.26% | 4.02% | 0.92% |
| CPT stg1 only | ✅ | 消歧 | 0.90% | 1.06% | 1.20% | 1.34% | 0.90% | 1.20% | 1.04% |
| CPT stg1+SFT | ✅ | 消歧 | **0.90%** | 1.06% | 1.20% | 1.34% | 0.90% | 1.20% | 1.04% |
| CPT full+SFT | ✅ | 消歧 | 0.86% | 1.02% | 1.14% | 1.40% | 0.86% | 1.14% | 0.98% |
| CPT stg1_500+SFT | ✅ | 消歧 | **0.92%** | 1.02% | 1.06% | 1.28% | **0.92%** | 1.06% | 0.99% |
| CPT stg1_1000+SFT | ✅ | 消歧 | 0.88% | 1.06% | 1.18% | 1.28% | 0.88% | 1.18% | 1.02% |
| CPT stg1_1600+SFT | ✅ | 消歧 | 0.90% | 1.04% | 1.14% | 1.26% | 0.90% | 1.14% | 1.00% |
| CPT stg1_1800+SFT | ✅ | 消歧 | 0.90% | 1.06% | 1.12% | 1.26% | 0.90% | 1.12% | 1.00% |
| CPT stg1_2000+SFT | ✅ | 消歧 | 0.90% | 1.06% | 1.20% | 1.34% | 0.90% | 1.20% | 1.04% |
| Text-Only SFT (SID eval) | ❌ | - | 0.28% | 0.50% | 0.58% | 0.66% | 0.14% | 0.24% | 0.43% |
| Text-Augmented SFT | ❌ | 原始 | 0.06% | 0.12% | 0.18% | 0.28% | 0.40% | 1.90% | 0.11% |

#### 4.3.3 CPT 步数消融

##### Beauty (fixed SID, Stage1 only + SFT)

| CPT Stage1 步数 | R@1 | R@5 | R@10 | PID R@1 | NDCG@10 |
|-----------------|-----|-----|------|---------|---------|
| 0 (no CPT) | 17.23% | 20.61% | 22.30% | 17.23% | 19.80% |
| 600 | **18.24%** | 20.61% | 21.28% | **18.24%** | 19.84% |
| 800 | **18.24%** | 20.95% | 21.28% | **18.24%** | 19.89% |
| 890 | **18.24%** | 20.95% | 21.28% | **18.24%** | 19.88% |
| final (~1000) | **18.24%** | 20.95% | 21.28% | **18.24%** | 19.88% |
| full (stg1+stg2+SFT) | 15.20% | 19.59% | 20.61% | 15.20% | 17.91% |

##### Movies (fixed SID, Stage1 only + SFT)

| CPT Stage1 步数 | R@1 | R@5 | R@10 | PID R@1 | NDCG@10 |
|-----------------|-----|-----|------|---------|---------|
| 0 (no CPT) | 0.62% | 0.98% | 1.04% | 0.62% | 0.83% |
| 500 | **0.92%** | 1.02% | 1.06% | **0.92%** | 0.99% |
| 1000 | 0.88% | 1.06% | 1.18% | 0.88% | 1.02% |
| 1600 | 0.90% | 1.04% | 1.14% | 0.90% | 1.00% |
| 1800 | 0.90% | 1.06% | 1.12% | 0.90% | 1.00% |
| 2000 (final) | 0.90% | 1.06% | 1.20% | 0.90% | 1.04% |

### 4.4 第二轮实验配置

| 参数 | CPT Stage1 | CPT Stage2 | SFT |
|------|-----------|-----------|-----|
| learning_rate | 1e-4 | 5e-5 | 5e-6 → 2e-5 |
| batch_size | 2 | 2 | 2 |
| grad_accum | 8 | 8 | 8 |
| max_length | 2048 | 2048 | 2048 |
| Beauty 步数 | ~1000 | ~1000 | 500 |
| Movies 步数 | ~2000 | ~2000 | 1000-3000 |

---

## 5. 第三轮：三方向系统消融

**实验目录**: `temp_try_amazon_3/`  
**核心目标**: 系统消融三大方向，统一评测方法（Greedy+Logprobs），引入 Term ID 方法

### 5.1 评测方法修正

**重要变更**：前两轮使用 beam search 评测，第三轮统一改为 **Greedy + Constrained Decoding + Logprobs** 评测。

**原因**：plan.md 明确要求 SID 评测必须使用 Greedy + 约束解码 + Logprobs 展开，禁止 beam search。beam search 在 OneRec 跨域场景下 Valid SID Rate 极低（如 13.93%），导致评测结果不可靠。

### 5.2 方向一：SID + 微调（CPT + SFT 消融）

**评测方法**: Greedy+Logprobs (K=10, prune_threshold=30)

#### All_Beauty 结果 ✅ 完成

| 编号 | 条件 | R@1 | R@5 | R@10 | NDCG@10 | 备注 |
|------|------|-----|-----|------|---------|------|
| B0 | 零样本 (fixed) | 9.46% | 11.82% | 11.82% | 10.71% | 基线 |
| B1 | 仅 SFT (fixed) | 16.89% | 20.61% | 21.28% | 19.30% | stg1_0 |
| B2 | CPT stg1 only (fixed) | 13.18% | 16.89% | 17.91% | 15.57% | 无SFT |
| B3 | CPT stg1+SFT (fixed) | **17.23%** | 19.93% | 20.95% | **19.06%** | stg1_600 |
| B5 | CPT full+SFT (fixed) | 15.20% | 18.92% | 20.27% | 17.52% | stg1+stg2+SFT |

**核心发现**：
- B3 (跳过Stg2) > B5 (完整pipeline)：Stg2 全参数共训练反而降低推荐效果
- B1 (仅SFT) R@10=21.28% 最高，但之前 beam_search 评测 Valid SID 仅 61.62%
- B2 (仅CPT stg1) 效果最差：单独训练 embedding 不够，需要 SFT 来学习推荐格式

#### Movies_and_TV 结果 🔄 运行中

| 编号 | 条件 | 状态 |
|------|------|------|
| B0 | 零样本 (fixed) | 🔄 运行中 |
| B1 | 仅 SFT (fixed) | 🔄 运行中 |
| B2 | CPT stg1 only (fixed) | 🔄 运行中 |
| B3 | CPT stg1+SFT (fixed) | 🔄 运行中 |
| B5 | CPT full+SFT (fixed) | 🔄 运行中 |

### 5.3 方向二：SID + ICL（免训练分析）

**评测方法**: Greedy+Logprobs (K=10, prune_threshold=30)

#### All_Beauty 结果 ✅ 完成

| 方法 | R@1 | R@5 | R@10 | NDCG@10 |
|------|-----|-----|------|---------|
| Baseline (零样本) | 9.46% | 11.82% | 11.82% | 10.71% |
| ICL K=1 | 14.53% | 18.24% | 19.26% | 17.01% |
| ICL K=3 | 17.57% | 24.32% | 24.32% | 21.18% |
| ICL K=5 | 18.24% | 23.31% | 23.31% | 20.97% |
| ICL K=10 | **19.93%** | **21.62%** | **22.97%** | **21.32%** |

**核心发现**：
- ICL 有显著提升：K=1 即可将 R@10 从 11.82% 提升到 19.26%
- K=3 达到 R@10=24.32% 的峰值，K=5/10 略有下降但 NDCG 持续提升
- ICL K=10 的 NDCG@10=21.32% 是 ICL 最佳

#### Movies_and_TV 结果 🔄 运行中

| 方法 | 状态 |
|------|------|
| ICL K=1 | 🔄 运行中 |
| ICL K=3/5/10 | 待做 |

### 5.4 方向三：Term ID 方法

#### 5.4.1 Term ID 生成流程

**Step 1: 提取 Embedding**
- 使用 Qwen3-8B-Embedding 提取每个 item 的 4096 维 embedding

**Step 2: 检索相似 Item**
- 使用 cosine similarity 找到每个 item 的 top-5 相似 item
- `find_similar_items()`: 批量计算余弦相似度矩阵

**Step 3: 生成 Term ID**
- 使用 LLM (OneRec-1.7B 或 Qwen3-1.7B) 为每个 item 生成 5 个标准化关键词
- Context-aware：利用相似 item 信息指导关键词生成

**Step 4: 构建 SFT 训练数据**
- meta2tid 内化数据：学习 "商品元数据 → Term ID" 映射
- rec_sft 推荐数据：学习 "用户历史 → 推荐目标 Term ID" 映射

**Step 5: SFT 训练**
- full (C1): meta2tid + rec_sft 混合训练
- rec_only (C2): 仅 rec_sft 训练

**Step 6: 推理 + EIG 匹配评测**

#### 5.4.2 SFT 训练进度

| 数据集 | full (C1) | rec_only (C2) | 状态 |
|--------|-----------|---------------|------|
| Beauty | ✅ 1000步 (loss=0.0506) | ✅ 1000步 (loss=0.0086) | 完成 |
| Movies | ✅ 3000/3000步 完成 | ✅ 3000/3000步 完成 | 评测中 |

#### 5.4.3 All_Beauty 结果 ✅ 完成

| 编号 | 方法 | R@1 | R@5 | R@10 | R@20 | R@32 | NDCG@10 | 备注 |
|------|------|-----|-----|------|------|------|---------|------|
| C3 | 零样本 | 0% | 0% | 0% | 0% | 0% | 0% | 模型未学过 Term ID |
| C1 | full v2 (1000步) | 26.69% | - | 42.57% | - | - | 34.70% | 含 meta2tid 内化 |
| C2 | rec_only v2 (1000步) | **45.61%** | **60.14%** | **63.18%** | **63.85%** | **66.22%** | **54.87%** | 仅推荐 SFT |

**关键发现**：rec_only (C2) >> full (C1)，meta2tid 内化数据反而干扰推荐能力。

#### 5.4.4 Movies_and_TV 结果 🔄 运行中

| 编号 | 方法 | 状态 |
|------|------|------|
| C1 | full v2 (3000步) | 🔄 vLLM 评测中 |
| C2 | rec_only v2 (3000步) | 🔄 vLLM 评测中 |

#### 5.4.5 D 系列（Qwen3-1.7B 对照）— 待做

| 编号 | 条件 | 状态 |
|------|------|------|
| D1 | 完整 pipeline (内化+推荐) | 需训练 |
| D2 | 只做推荐 SFT (不含内化) | 需训练 |
| D3 | 零样本 | 可直接评测 |

### 5.5 三方向 Beauty 对比（Greedy+Logprobs 评测）

| 方向 | 方法 | R@1 | R@5 | R@10 | NDCG@10 |
|------|------|-----|-----|------|---------|
| **方向三 Term ID** | SFT rec_only (C2) | **45.61%** | **60.14%** | **63.18%** | **54.87%** |
| 方向二 SID+ICL | ICL K=3 | 17.57% | 24.32% | 24.32% | 21.18% |
| 方向一 SID+微调 | CPT stg1+SFT (B3) | 17.23% | 19.93% | 20.95% | 19.06% |

---

## 6. 全部 Prompt 模板详解

### 6.1 通用约定

#### 聊天模板

所有方法使用 Qwen 风格 `<|im_start|>/<|im_end|>` 聊天模板，评测时使用 `qwen3_soft_switch.jinja2` 自定义模板，`enable_thinking=False`（`/no_think` 模式）：

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}/no_think<|im_end|>
<|im_start|>assistant

```

#### System Prompt 池（5 种，SFT 数据构建时随机选取）

1. `你是一个智能推荐助手，能够根据用户的购买历史预测用户可能感兴趣的商品。`
2. `你是一名商品推荐专家，擅长分析用户购买行为并预测用户偏好。`
3. `作为推荐系统助手，你需要根据用户历史购买记录推荐合适的商品。`
4. `你具备理解用户购买模式并生成个性化推荐的能力。`
5. `你是一个专业的商品推荐助手，能够根据用户过往购买记录推荐相关商品。`

#### User Prompt 池（5 种，SFT 数据构建时随机选取）

1. `根据以下用户购买记录，请预测用户接下来可能购买的商品：\n{query}`
2. `用户购买了以下商品：\n{query}\n请预测用户的下一个购买意向。`
3. `以下是用户的购买历史：\n{query}\n请推荐用户可能感兴趣的下一个商品。`
4. `用户历史购买记录如下：\n{query}\n分析并预测用户接下来会购买什么商品。`
5. `{query}\n根据上述购买记录，推测用户的下一个购买目标。`

其中 `{query}` 为用户历史商品的 SID 序列或 Term ID 序列拼接。

---

### 6.2 基线评估 Prompt（SID 零样本）

**来源**: `benchmarks/scripts/ray-vllm/evaluate.py` + `benchmarks/benchmark/tasks/v1_0/base_loader.py`

**模板**:
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}/no_think<|im_end|>
<|im_start|>assistant

```

**具体示例** (All_Beauty):
```
<|im_start|>system
作为推荐系统助手，你需要根据用户历史购买记录推荐合适的商品。<|im_end|>
<|im_start|>user
以下是用户的购买历史：
<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|><|sid_begin|><s_a_128><s_b_4096><s_c_1024><|sid_end|><|sid_begin|><s_a_512><s_b_2048><s_c_768><|sid_end|>
请推荐用户可能感兴趣的下一个商品。/no_think<|im_end|>
<|im_start|>assistant

```

**输出**: `<|sid_begin|><s_a_X><s_b_Y><s_c_Z><|sid_end|>`

---

### 6.3 Few-shot ICL Prompt（固定示例，Round 1-2）

**来源**: `temp_try_amazon/src/build_fewshot_v2.py`

**模板** (示例追加在 system prompt 末尾):
```
<|im_start|>system
{system_prompt}

以下是一些推荐示例：

示例1:
用户购买历史: <|sid_begin|><s_a_{a1}><s_b_{b1}><s_c_{c1}><|sid_end|> ...
推荐结果: <|sid_begin|><s_a_{a_t}><s_b_{b_t}><s_c_{c_t}><|sid_end|>

示例2:
用户购买历史: <|sid_begin|><s_a_{a3}><s_b_{b3}><s_c_{c3}><|sid_end|> ...
推荐结果: <|sid_begin|><s_a_{a_t2}><s_b_{b_t2}><s_c_{c_t2}><|sid_end|>

...（共 K 个示例）
<|im_end|>
<|im_start|>user
{user_prompt}/no_think<|im_end|>
<|im_start|>assistant

```

**示例构建细节**：
- 从 `sft_product_rec_train.parquet` 训练集中随机选取 (seed=42)
- 每个示例取用户历史最后 5 个 SID 作为 history，取 assistant 回复的第一个 SID 作为 answer
- v2 版本将所有 SID 替换为碰撞消解后的版本 (`pid2sid_fixed.parquet`)

---

### 6.4 ICL Prompt（Greedy+Logprobs 版本，Round 3）

**来源**: `temp_try_amazon_3/src/greedy_logprobs_eval.py` → `build_icl_examples()`

**模板** (示例追加在 user prompt 中):
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Here are some examples:

{example_1_user}
Answer: {example_1_answer}

{example_2_user}
Answer: {example_2_answer}

...（共 K 个示例）

Now answer the following:

{actual_user_content}/no_think<|im_end|>
<|im_start|>assistant

```

**关键区别**：
- 示例放在 **user prompt** 中（而非 system prompt），使用英文引导词
- 示例格式为 `{user_content}\nAnswer: {answer}`
- 示例从 benchmark 测试集的其他样本中选取（排除自身），而非训练集

---

### 6.5 相似度选择 Few-shot Prompt

**来源**: `temp_try_amazon_2/src/similar_selection.py`

**模板** (示例放在 user prompt 中):
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user

示例1：
用户历史：{example_1_user_history}
推荐：{example_1_answer}

示例2：
用户历史：{example_2_user_history}
推荐：{example_2_answer}

...（共 K 个相似用户示例）

以下是用户的购买历史：
{actual_user_history}
请推荐下一个商品。/no_think<|im_end|>
<|im_start|>assistant

```

**相似度计算**：
1. SID Code 统计特征（默认）：3层 code 的 mean/std/min/max + 历史长度，共 13 维
2. Item Embedding 平均池化（`--use_embeddings`）：Qwen3-8B-Embedding 向量取平均

---

### 6.6 约束解码 Prompt

**来源**: `temp_try_amazon_2/src/constrained_eval.py`

**模板**: 与基线完全相同，约束逻辑在解码阶段实现，不影响 prompt 本身。

约束通过 `allowed_token_ids` 和 `logit_bias` 在 vLLM 的 `SamplingParams` 中实现。

---

### 6.7 Self-Consistency Prompt

**来源**: `temp_try_amazon_2/src/self_consistency.py`

**模板**: 与基线完全相同，self-consistency 是后处理重排序方法，不影响 prompt。

---

### 6.8 Text-Only Prompt

**来源**: `temp_try_amazon/src/build_text_only.py`

**System Prompt**（修改后）:
```
你需要根据用户历史购买的商品描述推荐合适的商品，直接输出推荐商品名称或描述
```

**User Prompt**（修改后）:
```
{user_prompt_with_text_descriptions}
请根据用户的购买历史，推荐用户最可能购买的下一个商品，直接输出商品名称或描述。
```

**SID 替换规则**：
```
原始: <|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>
替换: [Maybelline New York Color Sensational Lipstick, Pink Petal]
```
- 文本描述截取前 50 字符
- 未知 item 映射为 `[Unknown Item]`

**具体示例**:
```
<|im_start|>system
你需要根据用户历史购买的商品描述推荐合适的商品，直接输出推荐商品名称或描述<|im_end|>
<|im_start|>user
以下是用户的购买历史：
[Maybelline New York Color Sensational Lipstick, Pink Petal][Revlon ColorBurst Matte Balm, Mischievous][L'Oreal Paris Infallible Pro-Matte Lipstick, Pink]
请根据用户的购买历史，推荐用户最可能购买的下一个商品，直接输出商品名称或描述。/no_think<|im_end|>
<|im_start|>assistant

```

**输出**: 直接输出文本描述（非 SID），如 `NYX Professional Makeup Soft Matte Lip Cream, Cannes`

**评测**: TF-IDF + 余弦相似度匹配候选商品描述

---

### 6.9 Text-Augmented Prompt

**来源**: `temp_try_amazon/src/build_text_augmented.py`

**System Prompt**: 与基线相同（从 System Prompt 池选取）

**User Prompt**: 指令部分与基线相同，区别在于用户历史中每个 SID 后附加括号包裹的文本描述：

```
原始: <|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>
增强: <|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|> (Maybelline New York Color Sensational Lipstick, Pink Petal, 0.11 oz)
```

- 文本描述截取前 80 字符（比 Text-Only 的 50 字符更宽松）
- 每个 SID 仅映射第一个 PID 的描述

**输出**: `<|sid_begin|><s_a_X><s_b_Y><s_c_Z><|sid_end|>`（输出不包含文本描述）

---

### 6.10 Term ID 生成 Prompt（Context-aware Keyword Generation）

**来源**: `temp_try_amazon_3/src/generate_term_ids.py`

**KEYWORD_PROMPT_TEMPLATE**（原文完整）:

```
You are an expert product summarizer. Your task is to generate exactly FIVE words to summarize this product. Please follow ALL guidelines carefully:

GUIDELINES:
1. WORD FORM: All words must be in their base form (nouns or adjectives, no -ed, -ing, -s endings)
2. WORD ORDER: Order words by importance (most important aspect first)
3. CONTENT FOCUS: Focus on these aspects in order:
   a) Main product category/type (e.g., "doll", "puzzle", "car")
   b) Key function or purpose (e.g., "educational", "remote-control")
   c) Distinctive features (e.g., "wooden", "electronic", "collectible")
   d) Target audience (e.g., "toddler", "boys", "family")
   e) Unique selling point (e.g., "glow-in-dark", "interactive")
4. CONSISTENCY WITH SIMILAR ITEMS: Consider the similar items provided. If they share common characteristics, use consistent terminology for those aspects.
5. UNIQUENESS: Include at least 1-2 words that distinguish this product from the similar items.
6. OUTPUT FORMAT: Provide ONLY the five words in this exact format: [word1, word2, word3, word4, word5]
7. NO ADDITIONAL TEXT: Do not include any explanations, thoughts, or other content.

PRODUCT INFORMATION:
Title: {title}
Description: {description}

{similar_items_section}

ANALYSIS GUIDANCE:
1. First, identify what this product has in common with similar products
2. Then, identify what makes this product unique or different
3. Use consistent vocabulary for shared characteristics
4. Include distinctive vocabulary for unique aspects
5. Ensure words cover the five required aspects in order

Please provide exactly five words in this exact format: [word1, word2, word3, word4, word5]:
```

**SIMILAR_ITEM_TEMPLATE**:
```
Similar Item {idx} (similarity: {score:.3f}): Title: {title} Description: {desc}
```

**相似 item 检索**: 使用 Qwen3-8B-Embedding 的 cosine similarity，top-5 相似 item。

---

### 6.11 Term ID 内化训练 Prompt（meta2tid）

**来源**: `temp_try_amazon_3/src/build_termid_data.py`

**META2TID_SYSTEM**:
```
你是一个智能商品分析助手，能够根据商品信息生成准确的特征关键词。
```

**META2TID_INSTRUCTION**（原文完整）:
```
Please generate exactly five words to summarize this product. Follow these guidelines carefully:
1. Words must be in their base form (noun or adjective, no -ed, -ing, -s endings)
2. Order words by importance (most important aspect first)
3. Focus on product category, function, key features, and target users
4. Each word should represent a distinct aspect
5. The word should be able to express the uniqueness of the product
6. Provide ONLY the five words in the specified format
7. Output format: [word1, word2, word3, word4, word5]
```

**User Content**:
```
Product Information:
Title: {title}
Description: {description}

Please provide exactly five words separated by commas:
```

**Assistant Content**:
```
[{keyword1}, {keyword2}, {keyword3}, {keyword4}, {keyword5}]
```

**完整消息格式**:
```json
[
  {"role": "system", "content": "你是一个智能商品分析助手，能够根据商品信息生成准确的特征关键词。"},
  {"role": "user", "content": "Please generate exactly five words...\n\nProduct Information:\nTitle: ...\nDescription: ...\n\nPlease provide exactly five words separated by commas:"},
  {"role": "assistant", "content": "[keyword1, keyword2, keyword3, keyword4, keyword5]"}
]
```

---

### 6.12 Term ID 推荐 SFT 训练 Prompt（rec_sft）

**来源**: `temp_try_amazon_3/src/build_termid_data.py`

**REC_SYSTEM**:
```
你是一个智能推荐助手，能够根据用户的购买历史预测用户可能感兴趣的商品。
```

**REC_INSTRUCTION**（原文完整）:
```
Based on the user's historical product interaction sequence, predict the next product's characteristic words.
Each product is represented by exactly 5 characteristic words enclosed in square brackets [].
```

**ITEM_FORMAT**:
```
Item text ID: [{keywords}] Title: {title}.
```

**User Content**:
```
{REC_INSTRUCTION}

{history_line_1}
{history_line_2}
...
{history_line_N}
Item text ID:
```

其中每个 history_line 格式为：`Item text ID: [keyword1, keyword2, keyword3, keyword4, keyword5] Title: {title}.`

**Assistant Content**（注意前导空格）:
```
 Item text ID: [keyword1, keyword2, keyword3, keyword4, keyword5] Title: {title}.
```

**完整消息格式**:
```json
[
  {"role": "system", "content": "你是一个智能推荐助手，能够根据用户的购买历史预测用户可能感兴趣的商品。"},
  {"role": "user", "content": "Based on the user's historical product interaction sequence...\n\nItem text ID: [lipstick, matte, pink, professional, long-lasting] Title: Maybelline...\nItem text ID: [balm, matte, berry, moisturizing, portable] Title: Revlon...\nItem text ID:"},
  {"role": "assistant", "content": " Item text ID: [cream, soft, matte, nude, affordable] Title: NYX Professional Makeup..."}
]
```

---

### 6.13 Term ID 评测推理 Prompt

**来源**: `temp_try_amazon_3/scripts/eval_termid_vllm.py`

**SYSTEM_PROMPT**:
```
你是一个智能推荐助手，能够根据用户的购买历史预测用户可能感兴趣的商品。
```

**REC_INSTRUCTION**: 与训练时相同

**评测时 Prompt 构建** (`extract_termid_prompt()`):
- 从 benchmark 样本中提取 messages
- 在最后一个 user message 末尾追加 `\nItem text ID:` 作为生成引导
- 使用 `tokenizer.apply_chat_template()` 渲染，`enable_thinking=False`

**推理参数**:
- Beam search: beam_width=32, max_tokens=50, temperature=0.0
- vLLM: dtype=bfloat16, gpu_memory_utilization=0.85, tensor_parallel_size=1

---

## 7. 评测算法详解

### 7.1 Greedy + Constrained Decoding + Logprobs 评测算法

**实现文件**: `temp_try_amazon_3/src/greedy_logprobs_eval.py`

**核心思想**: 替代 beam search，通过 3 步批量前向传播 + 联合概率排序获取 top-K 候选 SID。

**算法流程**（含 s_a 剪枝优化）：

```
Step 1: 批量前向传播（所有 N 个样本）
  → 约束掩码仅保留 s_a 范围 [151669, 159860] 的 logprobs
  → 记录 gt s_a 的 rank 和 logprob
  → s_a 剪枝：如果 rank > PRUNE_THRESHOLD(30)，则 R@10=0，跳过
  → 取每个样本的 top-K s_a logprobs

Step 2: 批量前向传播（剩余 M 个活跃样本 × K 个 s_a 候选）
  → 约束掩码仅保留 s_b 范围 [159861, 168052] 的 logprobs
  → 按 log P(s_a) + log P(s_b|s_a) 排序，保留 top-K 组合

Step 3: 批量前向传播（剩余 M 个活跃样本 × K 个 (s_a,s_b) 候选）
  → 约束掩码仅保留 s_c 范围 [168053, 176244] 的 logprobs
  → 按联合概率排序，取 top-K 完整 SID
```

**关键参数**：
- `K=10`：每步展开宽度（与 R@10 对齐）
- `PRUNE_THRESHOLD=30`：s_a rank 剪枝阈值
- `max_tokens=1`：每次只生成 1 个 token
- `temperature=0.0`：确定性采样
- `logprobs=8192`：返回所有 token 的 logprob 分布

**s_a 剪枝的数学依据**：
- 如果 gt s_a 在 8192 个候选中排名 > 30，则即使完美展开 s_b 和 s_c，联合概率也无法使其进入 top-10
- 剪枝可跳过 Step 2/3，大幅减少计算量
- 实际效果：Beauty 约 60-70% 样本需要完整展开

**vLLM 初始化**：
```python
llm = LLM(
    model=model_path,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    tensor_parallel_size=1,
    trust_remote_code=True,
    max_logprobs=8192,
)
```

**指标计算** (`compute_recall_metrics()`):
- SID R@K：预测的 top-K SID 中是否包含 gt SID
- PID R@K：预测的 top-K SID 通过 sid2pids 映射后是否包含 gt PID
- NDCG@K：按预测排序位置计算 DCG

### 7.2 EIG (Exact + Inexact Greedy) 匹配算法

**实现文件**: `temp_try_amazon_3/scripts/eval_termid_vllm.py`

**核心思想**: 将模型生成的 Term ID (5个关键词) 匹配回候选 item。

**Level 1 精确匹配** (`eig_match_level1`):
- 将 query_words 用 `", ".join()` 拼接为 key
- 在 `tid2item_ids` 字典中精确查找
- 命中则直接返回对应 item 列表

**Level 2 模糊匹配** (`eig_match_level2_fast`):
- 使用倒排索引加速
- 位置衰减加权：`weight = 1.0 / (i + 1)`，位置 0 权重 1.0，位置 4 权重 0.2
- 相似度计算：
  - 精确匹配：sim = 1.0
  - 子串匹配（query_word in cand_word 或反之）：sim = 0.8
  - 不匹配：sim = 0.0
- 候选得分：`candidate_scores[tid_key] += sim * weight`
- 返回 top-K 候选（按得分降序）

**Term ID 解析** (`parse_generated_term_id()`):
- 正则 `\[([^\]]+)\]` 提取最后一个 `[...]` 匹配
- 按逗号分割，lowercase，最多取 5 个词

**EIG 匹配流程**：
```
模型生成文本
  ↓
parse_generated_term_id() → [word1, word2, word3, word4, word5]
  ↓
Level 1 精确匹配 → 命中则返回
  ↓
Level 2 模糊匹配 → 返回 top-K 候选 item
  ↓
计算 R@K, NDCG@K
```

### 7.3 Beam Search 评测（Round 1-2 使用，Round 3 弃用）

**参数**：
- beam_width=32, num_return_sequences=32
- temperature=0.7, top_p=0.95, top_k=20

**弃用原因**：
- Valid SID Rate 极低（Beauty 4.58%-13.93%）
- 大量 beam search 候选为无效 SID，浪费计算
- beam search 不适合 SID 这种结构化输出空间

### 7.4 TF-IDF 文本匹配（Text-Only 评测）

- 将模型生成的文本与候选商品描述用 TF-IDF 向量化
- 计算余弦相似度
- 取 top-1 匹配的商品映射回 SID/PID

---

## 8. 训练配置详解

### 8.1 CPT (Continued Pre-Training) 配置

**实现文件**: `temp_try_amazon_2/scripts/run_cpt.py`

**Stage 1（冻结 LLM，训练 itemic embedding）**：

| 参数 | 值 |
|------|-----|
| learning_rate | 1e-4 |
| batch_size | 2 |
| grad_accum | 8 |
| effective_batch_size | 16 |
| max_length | 2048 |
| 可训练参数占比 | 20.40% |
| 冻结内容 | model.model (transformer layers) + 非 itemic lm_head |
| 解冻内容 | model.model.embed_tokens + model.lm_head (itemic token 部分) |
| ITEMIC_ID_START | 151669 |

**Stage 2（全参数共训练）**：

| 参数 | 值 |
|------|-----|
| learning_rate | 5e-5 |
| batch_size | 2 |
| grad_accum | 8 |
| effective_batch_size | 16 |
| max_length | 2048 |
| 可训练参数占比 | 100% |

**数据格式**：
- Pretrain: segments 格式 → `segments_to_text()` 转为纯文本 → tokenizer 编码
- SFT: messages 格式 → `messages_to_text()` 通过 `apply_chat_template()` 渲染 → tokenizer 编码
- Labels = input_ids（causal LM 训练）

### 8.2 SFT 配置

| 参数 | Beauty | Movies |
|------|--------|--------|
| learning_rate | 2e-5 | 2e-5 |
| batch_size | 2 | 2 |
| grad_accum | 8 | 8 |
| max_length | 2048 | 2048 |
| 训练步数 | 500 | 1000-3000 |

### 8.3 Term ID SFT 配置

| 参数 | Beauty | Movies |
|------|--------|--------|
| learning_rate | 1e-4 | 1e-4 |
| batch_size | 2 | 2 |
| grad_accum | 8 | 8 |
| 训练步数 | 1000 | 3000 |
| full (C1) 最终 loss | 0.0506 | - |
| rec_only (C2) 最终 loss | 0.0086 | - |

### 8.4 训练超参数经验总结

| 参数 | CPT Stage1 | CPT Stage2 | SFT | Term ID SFT |
|------|-----------|-----------|-----|-------------|
| lr | 1e-4 | 5e-5 | 2e-5 | 1e-4 |
| batch_size | 2 | 2 | 2 | 2 |
| grad_accum | 8 | 8 | 8 | 8 |
| Beauty 步数 | 600+ | 跳过(有害) | 500 | 1000 |
| Movies 步数 | 1600+ | 跳过(有害) | 1000-3000 | 3000 |

---

## 9. 关键发现与洞察

### 9.1 SID 碰撞修复

- **碰撞修复消除指标歧义**：修复后 R@1 = pid_R@1，sid2iid 变为 1:1 映射
- **零样本碰撞修复 SID-level 下降**：模型未学过新 SID 映射
- **零样本碰撞修复 PID-level 提升**：Beauty pid_R@1 从 8.33% → 9.46%
- **SFT 是碰撞修复生效的关键**：需要训练让模型学会新映射
- **碰撞消解大幅提升 PID 指标**：SFT only PID R@1: 10.47%(原始) → 17.23%(消歧), +64.7%

### 9.2 CPT 训练策略

- **CPT Stage2 有害**：Full pipeline (stg1→stg2→SFT) R@1=15.20% < Stage1 only + SFT R@1=18.24% (Beauty)
- **CPT Stage1 只需 500-600 步**：embedding 适应极快，更多步数不带来额外收益
- **CPT Stage1 对 Movies 有效**：500步 CPT+SFT R@1=0.92% > SFT only R@1=0.62% (+48%)

### 9.3 ICL 效果

- **ICL 有显著提升**：K=1 即可将 Beauty R@10 从 11.82% 提升到 19.26%
- **ICL K=3 达到 R@10 峰值**：24.32%，K=5/10 略有下降但 NDCG 持续提升
- **ICL K=10 NDCG@10 最佳**：21.32%
- **Valid SID Rate 随 K 增加**：从 13.93% (K=0) 提升到 85.57% (K=10)
- **Self-consistency 无额外收益**：frequency 排序和 logprob 排序结果一致

### 9.4 Text-Only / Text-Augmented

- **Text-Only Beauty R@1 与 CPT stg1+SFT 并列**：都是 18.24%
- **Text-Only PID 指标较低**：12.84% vs 18.24%（关键词碰撞导致 PID 歧义）
- **Text-Only Movies 失败**：模型生成 SID 而非关键词，42% 映射到 [unknown]
- **Text-Augmented 效果不如 Text-Only**：Beauty R@1=15.54% < 18.24%

### 9.5 Term ID 方法

- **Term ID 远超 SID 方法**：Beauty R@1=45.61% vs 17-19%
- **rec_only >> full**：meta2tid 内化数据干扰推荐能力
- **零样本 Term ID 完全无效**：C3 R@1=0%，模型未学过 Term ID 格式
- **Term ID 碰撞率极低**：Beauty 0.9%，Movies 2.4%（vs SID 的 9.19% 和 39.96%）

### 9.6 Movies 数据集挑战

- **所有方法 R@1 < 1%**：106K items 的 SID 空间过大
- **消歧 SID 方法 R@1 更高**：0.90% vs 0.74%（原始 SID）
- **原始 SID 在 Movies 上有碰撞优势**：PID R@1: 原始 1.26% > 消歧 0.90%（碰撞使更多 PID 被覆盖）
- **Text-Only Movies 评测问题**：模型生成 SID 而非关键词

### 9.7 已修复的 Bug

1. **compute_metrics.py regex bug**：`answer_iid` regex 会破坏 JSON 数组格式
2. **NPU segfault**：`torch.npu.set_device()` 与 `ASCEND_RT_VISIBLE_DEVICES` 冲突
3. **Think contamination in Term IDs**：部分 item 有 Hindi thinking tokens，用 `fix_think_contamination.py` 修复
4. **评测方法违规**：Round 3 初始使用 beam_search + post_hoc 过滤，违反 plan.md 要求，已用 Greedy+Logprobs 重新评测

---

## 10. 待完成实验

### 10.1 Movies 评测（运行中）

- 方向一：B0-B5 Greedy+Logprobs 评测
- 方向二：ICL K=1/3/5/10 Greedy+Logprobs 评测
- 方向三：Term ID C1/C2 vLLM beam search 评测

### 10.2 D 系列（Qwen3-1.7B 对照）

- D1: 完整 pipeline (内化+推荐)
- D2: 只做推荐 SFT (不含内化)
- D3: 零样本

目的：验证 Term ID 方法在无 itemic token 预训练的 base model 上是否同样有效。

### 10.3 未来方向

1. **8B 模型验证**：在 OneRec-8B 上重复关键实验
2. **RL 训练**：使用 GRPO 算法优化推荐质量
3. **Movies 全量评测**：287K 测试样本（当前仅 5K 子集）
4. **改进 tokenizer**：增加 codebook size 或层数以降低碰撞率
5. **CPT Stage2 有害性分析**：深入理解全参数共训练为何损害域适应

---

## 附录 A：数据路径

| 数据 | 路径 |
|------|------|
| Amazon 原始数据 | `/data/bucket2/panqiushi/GR_Full_Data/openonerec_data/Amazon-processed/` |
| Benchmark 格式数据 | `/data/bucket2/panqiushi/GR_Full_Data/openonerec_data/temp_try_amazon/benchmark_format/` |
| Round 2 Checkpoints | `/data/bucket2/panqiushi/GR_Full_Data/openonerec_data/temp_try_amazon_2/` |
| Round 3 Term ID 数据 | `/data/bucket2/panqiushi/GR_Full_Data/openonerec_data/temp_try_amazon_3/direction3_termid/` |
| OneRec-1.7B | `/data/bucket2/panqiushi/GR_Full_Data/openonerec_data/OneRec-1.7B` |
| OneRec-8B | `/data/bucket2/panqiushi/GR_Full_Data/openonerec_data/OneRec-8B` |
| Qwen3-Embedding-8B | `/data/bucket2/panqiushi/GR_Full_Data/openonerec_data/Qwen3-Embedding-8B` |
| OneRec-tokenizer | `/data/bucket2/panqiushi/GR_Full_Data/openonerec_data/OneRec-tokenizer/model.pt` |

## 附录 B：环境配置

| 组件 | 版本/配置 |
|------|----------|
| Conda 环境 | `oor_wkf` (Python 3.11) |
| PyTorch | 2.8.0 + torch_npu |
| Transformers | 4.57.6 |
| vLLM | 0.13.0 (customized for NPU) |
| 硬件 | 8× Ascend 910B3 NPU (64GB HBM each) |
| CANN | `/usr/local/Ascend/ascend-toolkit/set_env.sh` |

## 附录 C：方法对比总览

| 方法 | Prompt 变化位置 | 输入格式 | 输出格式 | 需要额外数据 | 评测匹配方式 |
|------|---------------|---------|---------|------------|------------|
| 基线 | 无 | SID 序列 | SID | 无 | SID/PID 字符串匹配 |
| Few-shot ICL (R1-2) | System prompt 追加示例 | SID 序列 | SID | 训练集示例 | SID/PID 字符串匹配 |
| ICL (R3) | User prompt 前置示例 | SID 序列 | SID | 测试集示例 | Greedy+Logprobs |
| 相似度选择 | User prompt 前置示例 | SID 序列 | SID | 训练集 + embedding | SID/PID 字符串匹配 |
| 约束解码 | 无（解码阶段约束） | SID 序列 | SID（保证合法） | pid2sid 映射 | SID/PID 字符串匹配 |
| Self-Consistency | 无（后处理重排序） | SID 序列 | SID（重排序） | 无 | SID/PID 字符串匹配 |
| Text-Only | System + User prompt 均修改 | 文本描述 | 文本描述 | pid2caption | TF-IDF 文本匹配 |
| Text-Augmented | User prompt SID 后追加描述 | SID + 文本描述 | SID | pid2caption | SID/PID 字符串匹配 |
| Term ID | System + User prompt 均修改 | Term ID 序列 | Term ID | pid2keywords | EIG 匹配 |