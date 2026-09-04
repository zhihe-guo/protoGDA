# Cell-Drug Cross-Attention 模型架构文档

## 一、项目概述

本项目用于预测药物分子对细胞系的半抑制浓度（IC50）。核心思路：将药物分子建模为图，将细胞系建模为探针（Probe），通过多层交叉注意力（Cross-Attention）让细胞系探针"阅读"药物分子图，提取药物-细胞相互作用特征。

---

## 二、问题公式化

### 2.1 任务定义

给定药物分子 \( d \) 和细胞系 \( c \)，预测半抑制浓度（IC50）在 \(\log_{10}\) 尺度上的值：

\[
\hat{y} = f_\theta(d, c) \approx \log_{10}(\text{IC50}_{d,c})
\]

其中 \( f_\theta \) 为待学习的神经网络。

### 2.2 输入表示

**药物图** \( G_d = (V_d, E_d) \)：

\[
\mathbf{X}_d \in \mathbb{R}^{|V_d| \times 158}, \quad
\mathbf{E}_d \in \mathbb{R}^{|E_d| \times 6}, \quad
\mathbf{p}_d \in \mathbb{R}^{|V_d| \times 3}
\]

- \( \mathbf{X}_d \)：原子特征（原子类型、杂化、电荷、芳香性等 158 维）
- \( \mathbf{E}_d \)：键特征（键类型、共轭性等 6 维）
- \( \mathbf{p}_d \)：3D 原子坐标（EGNN 使用）

**分子级指纹**（每药物一个向量）：

\[
\mathbf{m}_d \in \mathbb{R}^{1024} \quad \text{(Morgan ECFP4)}, \qquad
\mathbf{b}_d \in \mathbb{R}^{768} \quad \text{(ChemBERTa)}
\]

**细胞系特征**：

\[
\mathbf{e}_c \in \mathbb{R}^{256} \quad \text{(PCA 降维后的基因表达谱)}
\]

### 2.3 原子级门控广播（Atom-type Gated Broadcast）

将分子级指纹投影到原子空间，并用原子化学属性门控：

\[
\tilde{\mathbf{m}}_d = \operatorname{Linear}_{1024 \to 32}(\mathbf{m}_d) \in \mathbb{R}^{32}
\]
\[
\tilde{\mathbf{b}}_d = \operatorname{Linear}_{768 \to 32}(\mathbf{b}_d) \in \mathbb{R}^{32}
\]

对每个原子节点 \( i \in V_d \)：

\[
\mathbf{g}_i^{(m)} = \sigma\left( \operatorname{Linear}_{158 \to 32}(\mathbf{x}_i) \right) \in [0,1]^{32} \quad \text{(Morgan 门控)}
\]
\[
\mathbf{g}_i^{(b)} = \sigma\left( \operatorname{Linear}_{158 \to 32}(\mathbf{x}_i) \right) \in [0,1]^{32} \quad \text{(ChemBERTa 门控)}
\]

门控注入（逐通道调制）：

\[
\mathbf{x}_i' = \mathbf{x}_i \,\|\, (\tilde{\mathbf{m}}_d \odot \mathbf{g}_i^{(m)}) \,\|\, (\tilde{\mathbf{b}}_d \odot \mathbf{g}_i^{(b)})
\quad \in \mathbb{R}^{222}
\]

其中 \(\|\) 表示向量拼接，\(\odot\) 表示逐元素乘法。

### 2.4 原子投影与细胞系投影

**原子投影**到统一隐藏空间：

\[
\mathbf{h}_i^{(0)} = \operatorname{atom\_proj}(\mathbf{x}_i') \in \mathbb{R}^{64}, \quad \forall i \in V_d
\]

**细胞系探针投影**（CellQueryProjector）：

\[
\mathbf{Q} = \operatorname{CellQueryProj}(\mathbf{e}_c) \in \mathbb{R}^{M \times D}, \quad M=16, \; D=64
\]

每个探针向量 \( \mathbf{q}_k \in \mathbb{R}^{64} \)（\(k = 1,\dots,M\)）代表细胞系关心的一个查询方向。

### 2.5 训练时分支控制机制（仅训练模式）

**细胞侧**：

\[
\tilde{\mathbf{e}}_c = \mathbf{e}_c + \epsilon_c, \quad \epsilon_c \sim \mathcal{N}(0, \sigma_{cn}^2 \mathbf{I}) \quad \text{(if enable\_cell\_noise)}
\]
\[
\tilde{\mathbf{e}}_c = \tilde{\mathbf{e}}_c \cdot m_c, \quad m_c \sim \text{Bernoulli}(1 - p_{c\_drop}) \quad \text{(if enable\_cell\_drop)}
\]

**药物侧**（镜像设计）：

\[
\tilde{\mathbf{m}}_d = \tilde{\mathbf{m}}_d + \epsilon_m, \quad \epsilon_m \sim \mathcal{N}(0, \sigma_{dn}^2 \mathbf{I}) \quad \text{(if enable\_drug\_noise)}
\]
\[
\tilde{\mathbf{b}}_d = \tilde{\mathbf{b}}_d + \epsilon_b, \quad \epsilon_b \sim \mathcal{N}(0, \sigma_{dn}^2 \mathbf{I}) \quad \text{(if enable\_drug\_noise)}
\]
\[
\tilde{\mathbf{m}}_d = \tilde{\mathbf{m}}_d \cdot m_m, \quad m_m \sim \text{Bernoulli}(1 - p_{d\_drop}) \quad \text{(if enable\_drug\_drop, Morgan 独立)}
\]
\[
\tilde{\mathbf{b}}_d = \tilde{\mathbf{b}}_d \cdot m_b, \quad m_b \sim \text{Bernoulli}(1 - p_{d\_drop}) \quad \text{(if enable\_drug\_drop, ChemBERTa 独立)}
\]

其中细胞侧参数：\(\sigma_{cn}\) = `cell_noise_std`, \(p_{c\_drop}\) = `cell_drop_prob`；药物侧参数：\(\sigma_{dn}\) = `drug_noise_std`, \(p_{d\_drop}\) = `drug_drop_prob`。两侧各有独立的 boolean 开关：`enable_cell_noise` / `enable_cell_drop` / `enable_drug_noise` / `enable_drug_drop`。所有机制仅在训练时生效，验证/测试时关闭。药物侧噪声仅加在 Morgan/ChemBERTa 投影后的 32 维向量上，`graph.x` 原子化学属性保持干净。

### 2.6 多层交互（InteractionBlock × L）

设 \(L = \text{num\_layers} = 3\)。第 \(l\) 层（\(l = 1,\dots,L\)）：

**GNN 图进化**：

\[
\mathbf{H}^{(l)} = \operatorname{GNN}^{(l)}\left( \mathbf{H}^{(l-1)}, \mathbf{E}_d \right) + \mathbf{H}^{(l-1)}, \quad 
\mathbf{H}^{(l)} \in \mathbb{R}^{|V_d| \times 64}
\]

**交叉注意力**（细胞探针查询药物原子）：

\[
\mathbf{Q}' = \mathbf{Q} \mathbf{W}_Q^{(l)}, \quad
\mathbf{K}^{(l)} = \mathbf{H}^{(l)} \mathbf{W}_K^{(l)}, \quad
\mathbf{V}^{(l)} = \mathbf{H}^{(l)} \mathbf{W}_V^{(l)}
\]
\[
\mathbf{A}^{(l)} = \operatorname{softmax}\left( \frac{\mathbf{Q}' {\mathbf{K}^{(l)}}^\top}{\sqrt{64}} \right) \in \mathbb{R}^{M \times |V_d|}
\]
\[
\mathbf{E}^{(l)} = \mathbf{A}^{(l)} \mathbf{V}^{(l)} \in \mathbb{R}^{M \times 64}
\]

其中 \( \mathbf{A}^{(l)}_{k,i} \) 表示第 \(k\) 个细胞探针对药物第 \(i\) 个原子的注意力权重。\(\mathbf{E}^{(l)}\) 是第 \(l\) 层提取的交互特征。

### 2.7 交互模式

模型支持两种交互模式，由 `interaction_mode` 配置控制。

**memory 模式（门控记忆累积）**——将 \(L\) 层的提取结果按时间顺序积累，初始记忆由 `memory_init` 配置控制：

\[
\mathbf{M}^{(0)} = \begin{cases}
\mathbf{Q} & \text{(cell\_probe)} \\
\mathbf{0} & \text{(zero)} \\
\mathbf{M}^{(0)} = \mathbf{0}, \text{ then } \mathbf{M}^{(0)} \leftarrow \operatorname{Accumulator}(\mathbf{0}, \mathbf{Q}) & \text{(cell\_probe\_gated)}
\end{cases}
\]
\[
\mathbf{M}^{(l)} = \mathbf{M}^{(l-1)} + g^{(l)} \cdot \operatorname{FFN}\left( \operatorname{LayerNorm}(\mathbf{M}^{(l-1)}) + \mathbf{E}^{(l)} \right)
\]
\[
g^{(l)} = \sigma\left( \operatorname{Linear}_{64 \to 1}( [\mathbf{M}^{(l-1)} \,\|\, \mathbf{E}^{(l)}] ) \right) \in [0, 1]
\]

其中 \(\operatorname{FFN}\) 为 \(64 \to 256 \to 64\) 前馈网络。\(g^{(l)}\) 为第 \(l\) 层的门控权重。

**residual 模式（Q 自更新残差连接）**——跳过 MemoryAccumulator，每层 extraction 直接残差写回 Q：

\[
\mathbf{Q}^{(0)} = \operatorname{CellQueryProj}(\mathbf{e}_c)
\]
\[
\mathbf{Q}^{(l)} = \mathbf{Q}^{(l-1)} + \operatorname{FFN}_q\left( \operatorname{LayerNorm}(\mathbf{Q}^{(l-1)} + \mathbf{E}^{(l)}) \right), \quad l = 1..L
\]

其中 \(\operatorname{FFN}_q\) 为 \(64 \to 64 \to 64\) 前馈网络（含 GELU + Dropout）。Q 在层间逐步融入药物交互信息，最终直接 flatten 送 readout。

最终特征 \( \mathbf{M}^{(L)} \)（memory）或 \( \mathbf{Q}^{(L)} \)（residual）\( \in \mathbb{R}^{M \times 64} \) 展平为 \( \mathbb{R}^{M \cdot 64} = \mathbb{R}^{1024} \)（若 \(M=16\)）。

### 2.8 Readout 与预测

\[
\mathbf{m} = \operatorname{Flatten}(\mathbf{M}^{(L)}) \in \mathbb{R}^{1024}
\]
\[
\hat{y} = \operatorname{MLP}_{\text{readout}}(\mathbf{m}) \in \mathbb{R}
\]

Readout MLP 结构：\(1024 \to 256 \to 128 \to 1\)，每层含 LayerNorm + GELU + Dropout。

### 2.9 损失函数

均方误差（MSE）回归：

\[
\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{(d,c) \in \mathcal{B}} \left( \hat{y}_{d,c} - y_{d,c} \right)^2
\]

其中 \(y_{d,c} = \log_{10}(\text{IC50}_{d,c})\) 为真实标签，\(\mathcal{B}\) 为小批量。

### 2.10 模型总览

\[
\boxed{
\begin{aligned}
& \text{Input: } (d, c) \;\rightarrow\; \mathbf{X}_d, \mathbf{m}_d, \mathbf{b}_d, \mathbf{e}_c \\
& \text{Train controls: } [\text{noise}] \rightarrow [\text{drop}] \text{ (cell-side)} \\
& \qquad\qquad\;\,\, [\text{noise, drop}] \text{ on } \tilde{\mathbf{m}}_d, \tilde{\mathbf{b}}_d \text{ (drug-side)} \\
& \text{Gated Broadcast: } \mathbf{x}_i' = \mathbf{x}_i \,\|\, (\tilde{\mathbf{m}}_d \!\odot\! \mathbf{g}_i^{(m)}) \,\|\, (\tilde{\mathbf{b}}_d \!\odot\! \mathbf{g}_i^{(b)}) \\
& \text{atom\_proj: } \mathbf{h}_i^{(0)} = \operatorname{atom\_proj}(\mathbf{x}_i'), \quad \mathbf{Q} = \operatorname{CellQueryProj}(\mathbf{e}_c) \\
& \text{Interaction × } L \text{: } \mathbf{E}^{(l)} = \operatorname{CrossAttn}^{(l)}(\mathbf{Q}, \operatorname{GNN}^{(l)}(\mathbf{H}^{(l-1)})), \quad l = 1..L \\
& \text{Mode: } \begin{cases}
\text{memory: } \mathbf{M}^{(L)} = \operatorname{Accumulator}(\mathbf{M}^{(0)}, \mathbf{E}^{(1)}, \dots, \mathbf{E}^{(L)}) \\
\text{residual: } \mathbf{Q}^{(l)} = \mathbf{Q}^{(l-1)} + \operatorname{FFN}(\operatorname{LN}(\mathbf{Q}^{(l-1)} + \mathbf{E}^{(l)}))
\end{cases} \\
& \text{Output: } \hat{y} = \operatorname{MLP}(\operatorname{Flatten}(\mathbf{M}^{(L)} \text{ or } \mathbf{Q}^{(L)}))
\end{aligned}
}
\]

### 2.11 核心公式总结

| 步骤 | 公式 | 关键维度 |
|------|------|----------|
| 门控广播 | \( \mathbf{x}_i' = \mathbf{x}_i \,\|\, (\tilde{\mathbf{m}}_d \odot \mathbf{g}_i^{(m)}) \,\|\, (\tilde{\mathbf{b}}_d \odot \mathbf{g}_i^{(b)}) \) | 158→222 |
| 原子投影 | \( \mathbf{h}_i^{(0)} = \operatorname{atom\_proj}(\mathbf{x}_i') \) | 222→64 |
| 细胞投影 | \( \mathbf{Q} = \operatorname{CellQueryProj}(\mathbf{e}_c) \) | 256→16×64 |
| 第 \(l\) 层 GNN | \( \mathbf{H}^{(l)} = \operatorname{GNN}^{(l)}(\mathbf{H}^{(l-1)}) + \mathbf{H}^{(l-1)} \) | 64→64 |
| 第 \(l\) 层交叉注意力 | \( \mathbf{E}^{(l)} = \operatorname{softmax}(\mathbf{Q}{\mathbf{K}^{(l)}}^\top/\sqrt{64}) \mathbf{V}^{(l)} \) | M×64 |
| 记忆累积（memory） | \( \mathbf{M}^{(l)} = \mathbf{M}^{(l-1)} + g^{(l)} \cdot \operatorname{FFN}(\operatorname{LN}(\mathbf{M}^{(l-1)}) + \mathbf{E}^{(l)}) \) | M×64 |
| 残差更新（residual） | \( \mathbf{Q}^{(l)} = \mathbf{Q}^{(l-1)} + \operatorname{FFN}_q(\operatorname{LN}(\mathbf{Q}^{(l-1)} + \mathbf{E}^{(l)})) \) | M×64 |
| Readout | \( \hat{y} = \operatorname{MLP}(\operatorname{Flatten}(\mathbf{M}^{(L)} \text{ or } \mathbf{Q}^{(L)})) \) | 1024→1 |

### 2.12 符号速查表

| 符号 | 含义 | 典型维度 |
|------|------|----------|
| \( d \) | 药物分子索引 | — |
| \( c \) | 细胞系索引 | — |
| \( V_d \) | 药物图的原子集合 | 10~50 |
| \( E_d \) | 药物图的键集合 | ~2×|V_d| |
| \( \mathbf{x}_i \) | 第 \(i\) 个原子的原始特征 | 158 |
| \( \mathbf{m}_d \)  | Morgan 指纹 | 1024 |
| \( \mathbf{b}_d \) | ChemBERTa 嵌入 | 768 |
| \( \mathbf{e}_c \) | 细胞系基因表达 | 256 |
| \( \mathbf{g}_i^{(m)}, \mathbf{g}_i^{(b)} \) | 原子 \(i\) 的门控系数 | 32 |
| \( D \) | 隐藏维度 | 64 |
| \( M \) | 探针数量 | 16 |
| \( L \) | InteractionBlock 层数 | 3 |
| \( \mathbf{Q} \) | 细胞系探针矩阵 | \(M \times D\) |
| \( \mathbf{H}^{(l)} \) | 第 \(l\) 层的原子特征 | \(|V_d| \times D\) |
| \( \mathbf{E}^{(l)} \) | 第 \(l\) 层的提取特征 | \(M \times D\) |
| \( \mathbf{M}^{(l)} \) | 第 \(l\) 层后的累积记忆 | \(M \times D\) |
| \( g^{(l)} \) | 第 \(l\) 层的门控权重 | 标量 |
| \( \hat{y} \) | 预测值 (\(\log_{10}\) IC50) | 标量 |
| \( y \) | 真实标签 | 标量 |

---

## 三、输入数据

系统接收四种输入特征：

### 2.1 药物分子图（SMILES → RDKit → 图）

| 特征 | 维度 | 说明 |
|------|------|------|
| 原子特征 `graph.x` | 158 维/原子 | 原子类型、价态、手性、杂化、芳香性、形式电荷、原子团归属、氢原子数、环大小等 |
| 键特征 `graph.edge_attr` | 6 维/键 | 键类型、共轭性、环属性、立体构型等 |
| 3D 坐标 `graph.pos` | 3 维/原子 | 三维原子坐标（用于 EGNN），由 RDKit ETKDG 生成 |
| 边索引 `graph.edge_index` | (2, E) | 有向边列表 |

每批药物分子通过 `Batch.from_data_list` 拼成一个大的批量图，用 `graph.batch` 记录每个节点属于哪个药物。

### 2.2 Morgan 分子指纹

| 特征 | 维度 | 说明 |
|------|------|------|
| ECFP4 | 1024 位/药物 | 扩展连接指纹（Morgan radius=2），编码局部子结构信息 |

### 2.3 ChemBERTa 分子嵌入

| 特征 | 维度 | 说明 |
|------|------|------|
| Transformer 嵌入 | 768 维/药物 | 使用 ChemBERTa（在 SMILES 上预训练的 transformer）提取的全局分子语义向量 |

### 2.4 细胞系基因表达

| 特征 | 维度 | 说明 |
|------|------|------|
| 基因表达谱 | 256 维/细胞（PCA 降维后） | 原始高通量基因表达数据经 PCA 降维 |

---

## 四、全局分子语义源头广播 + 原子类型门控（Atom-type Gating）

Morgan 指纹和 ChemBERTa 嵌入在 **原子级别** 注入药物图，而非传统的 readout 前拼接。这确保 GNN 的每一层都能利用这些分子语义信息。

此外，通过**原子类型门控**，每个原子根据自身化学属性（`graph.x` 的 158 维特征）产生专属的 0~1 门控系数，逐通道调制注入的分子指纹——碳原子和氮原子从同一个 Morgan 指纹中提取的信号不同。

### 3.1 原子投影

```text
morgan_atom_proj:  Linear(1024 → 32)
chemberta_atom_proj: Linear(768 → 32)
```

### 3.2 门控网络

```text
gate_morgan: Linear(158 → 32) → Sigmoid    # 根据原子化学属性产生 Morgan 门控
gate_bert:   Linear(158 → 32) → Sigmoid    # 根据原子化学属性产生 ChemBERTa 门控
```

参数增量极小：每个门控网络约 5K 参数，两个共约 10K。

### 3.3 门控广播到原子节点

```python
node_feats = [graph.x]                          # (total_nodes, 158)
drug_to_node = graph.batch                      # (total_nodes,) — 节点 → 药物索引

if use_morgan:
    morgan_batch = _morgan_table[drug_ids]       # (B, 1024)
    morgan_32 = morgan_atom_proj(morgan_batch)   # (B, 32)
    atom_gate_m = gate_morgan(graph.x)            # (total_nodes, 32) — 每个原子有自己的门控
    node_feats.append(morgan_32[drug_to_node] * atom_gate_m)

if use_chemberta:
    bert_batch = _chemberta_table[drug_ids]      # (B, 768)
    bert_32 = chemberta_atom_proj(bert_batch)    # (B, 32)
    atom_gate_b = gate_bert(graph.x)              # (total_nodes, 32) — 每个原子有自己的门控
    node_feats.append(bert_32[drug_to_node] * atom_gate_b)

x = atom_proj(torch.cat(node_feats, dim=-1))     # (total_nodes, 64)
# 158 + 32 + 32 = 222 → 64
```

### 3.4 门控的语义含义

- `graph.x` 的 158 维编码了原子类型、杂化、电荷、芳香性等化学属性
- `gate_morgan(graph.x)` 输出 32 个 0~1 的通道权重，按位控制 Morgan 32 个通道各注入多少
- 碳原子和氮原子对同一 Morgan 指纹通道的吸收比例不同
- GNN 消息传递时，相邻原子携带的分子语义既有重叠（同一分子），又有差异（原子类型不同），丰富了图的语义多样性
- `cell_to_drug` 和 `drug_to_cell` 两条路径使用**相同的门控逻辑**（镜像对称）

---

## 五、atom_proj

将所有原子特征（包括 gated Morgan/ChemBERTa）投影到统一的 64 维隐藏空间。

```text
atom_proj = Sequential(
    Linear(222, 64),
    LayerNorm(64),
    GELU(),
    Dropout(drug_dropout if enabled else global_dropout),
)
```

`atom_proj` 内的 Dropout 由 `enable_drug_dropout` 控制：开启时使用 `drug_dropout`（通常高于全局，用于抑制药物信号），关闭时使用全局 `dropout`。

---

## 六、细胞系探针投影（CellQueryProjector）

将 256 维细胞系基因表达编码为 `M = 16` 个**探针向量**（每个 64 维），即细胞系的"问题集合"——每个探针代表细胞系关心的一个方向（如"有没有苯环？""有没有羟基？"）。

```text
CellQueryProjector:
    Linear(256, 64) → LN → GELU → Drop(0.3)
    Linear(64, 64) → LN → GELU → Drop(0.3)
    Linear(64, 1024)   # 16 × 64
    → reshape(B, 16, 64)
    → + probe_bias (可学习偏置)

输出: Q = (B, 16, 64) — 16 个探针向量
```

**细胞侧独立 Dropout**：由 `enable_cell_dropout` 控制，开启时使用 `cell_dropout`（通常高于全局，用于抑制细胞系信号），关闭时使用全局 `dropout`。

---

## 七、训练时分支控制机制

本模型支持对细胞侧和药物侧分别施加独立的训练时控制，所有机制仅训练时生效，验证/测试时自动关闭。每项控制均配有 boolean 开关（`enable_*`）供显式开启/关闭。

### 7.1 细胞侧控制

**高斯噪声注入**（`enable_cell_noise` / `cell_noise_std`）：

```python
cell_feats = cell_feats + torch.randn_like(cell_feats) * cell_noise_std
```

**样本级随机掩码**（`enable_cell_drop` / `cell_drop_prob`）：

```python
mask = (torch.rand(B, 1) > cell_drop_prob).float()
cell_feats = cell_feats * mask
```

**投影器独立 Dropout**（`enable_cell_dropout` / `cell_dropout`）：高于全局 dropout，进一步抑制细胞侧信号。

### 7.2 药物侧控制（镜像设计）

**投影层高斯噪声**（`enable_drug_noise` / `drug_noise_std`）：仅在 Morgan/ChemBERTa 投影后的 32 维向量上加噪，`graph.x` 原子化学属性保持干净。

```python
morgan_32 = morgan_32 + torch.randn_like(morgan_32) * drug_noise_std
bert_32 = bert_32 + torch.randn_like(bert_32) * drug_noise_std
```

**分子指纹独立随机掩码**（`enable_drug_drop` / `drug_drop_prob`）：Morgan 和 ChemBERTa 各自独立抛硬币（每样本、每分支独立决定是否清零）。

```python
m_mask = (torch.rand(B, 1) > drug_drop_prob).float()
morgan_32 = morgan_32 * m_mask          # Morgan 独立
b_mask = (torch.rand(B, 1) > drug_drop_prob).float()
bert_32 = bert_32 * b_mask              # ChemBERTa 独立
```

**atom_proj 独立 Dropout**（`enable_drug_dropout` / `drug_dropout`）：仅 `atom_proj` 内的 Dropout 使用此值，GNN 内部和 cross_attn 继续使用全局 `dropout`。

以上洒噪/掩码逻辑封装在 `_broadcast_drug_features()` 方法中，`cell_to_drug` 和 `drug_to_cell` 两路径共享调用，确保同 batch 内随机状态一致。

### 7.3 禁用方式

若不使用某项控制，将对应的 `enable_*` 设为 `false`，守卫条件保证相关代码不会执行。

---

## 八、InteractionBlock（多层 GNN + 交叉注意力）

这是模型的核心迭代模块，重复 `num_layers = 3` 次。

### 7.1 单层流程

```text
输入:  H_nodes (N, 64)  — 当前药物图节点特征
       Q (B, 16, 64)    — 细胞系探针（共享，不变）

┌──────────────────────────────────────────────────┐
│ 1. GNN 进化（GraphEvolver）                      │
│    H' = GNN(H) + H （残差）                      │
│    → LN → GELU → Drop                           │
│    输出: evolved (N, 64)                         │
│                                                  │
│ 2. 交叉注意力（GraphCrossAttention）              │
│    Q_proj = Linear(Q)                            │
│    K_proj = Linear(evolved)                      │
│    V_proj = Linear(evolved)                      │
│    extracted = softmax(Q·K^T/√d) · V            │
│    → Linear → LN                                │
│    输出: extraction (B, 16, 64)                  │
└──────────────────────────────────────────────────┘

返回: extraction (B, 16, 64)   — 本层提取的信息
      evolved (N, 64)         — 传给下一层作为 H_nodes
      attention_weights       — 可视化用
```

### 7.2 支持的 GNN 变体

| 类型 | 说明 | 是否使用边特征 | 是否使用 3D 坐标 |
|------|------|:---:|:---:|
| GATv2 | 注意力图卷积 | 否 | 否 |
| GCN | 标准图卷积 | 否 | 否 |
| GINE | 边特征神经网络 | 是 | 否 |
| EGConv | 多聚合子卷积 | 否 | 否 |
| EGNN | 等变图神经网络 | 是 | 是 |

**EGNN 特殊处理**：更新 3D 原子坐标后再更新节点特征，坐标更新是等变的（旋转、平移不变），仅节点特征传给下游交叉注意力。

### 7.3 三层循环

```text
atom_proj 输出 (N, 64)
    │
    ▼
┌─────── Layer 1 ───────┐
│  GNN → LN → CrossAttn  │
│  提取 E1               │
└───────┬───────────────┘
    │ evolved (N, 64)
    ▼
┌─────── Layer 2 ───────┐
│  GNN → LN → CrossAttn  │
│  提取 E2               │
└───────┬───────────────┘
    │ evolved (N, 64)
    ▼
┌─────── Layer 3 ───────┐
│  GNN → LN → CrossAttn  │
│  提取 E3               │
└───────────────────────┘

Q (B,16,64) 在所有层共享 → CrossAttn(Q, H_evolved)
E1, E2, E3 并行送入 MemoryAccumulator
```

- **前向传递**：`evolved` 节点特征在层间传递——每层 GNN 在图上做一次消息传递，节点嵌入越来越丰富
- **不传递**：Q（细胞系探针）在所有层共用——"关心什么"不随层数变化
- **交叉注意力**：每层都让 Q 查询**更新后的图**，提取新的相互作用信息

---

## 九、GraphCrossAttention（交叉注意力详解）

### 8.1 核心机制

细胞系探针 Q 作为 **Query**，药物图的原子节点作为 **Key/Value**，计算每个探针对每个原子的注意力权重。

```text
Q: (B, 16, 64)   — 细胞系想知道什么
K: (B, N_max, 64) — 每个原子"是什么"
V: (B, N_max, 64) — 每个原子"提供什么信息"

attention = softmax( Q·K^T / √64 )
extraction = attention · V   → (B, 16, 64)
```

### 8.2 批量填充策略（GPU 高效）

药物图的原子数不同（有的 10 个原子，有的 50 个），不能直接做成 (B, N, D) 的整齐张量。本实现采用 **填充批量注意力（Padded Batched Attention）**：

1. 统计每批药物的最大原子数 N_max
2. 将所有图的原子特征填充到 (B, N_max, D)
3. 生成 key_padding_mask（B, N_max），标记哪些位置是填充位
4. 在 softmax 前将填充位的注意力分数置为 `-inf`
5. 一次矩阵乘法算出所有批次的注意力输出

无需逐药物的 for 循环，完全在 GPU 上并行化。

---

## 十、交互模式：MemoryAccumulator vs Residual Update

模型支持两种交互模式，由配置 `interaction_mode` 控制。

### 10.1 memory 模式（门控记忆累积器）

#### 动机

三层的 cross-attention 产生了三个 extraction（E1, E2, E3），需要合并。简单平均会丢失时序信息，且无法区分层的重要性。

### 10.2 初始记忆

由配置 `memory_init` 控制：

- `cell_probe`（默认）：M0 = 细胞系探针 Q，记忆从细胞先验知识起步
- `zero`：M0 = 0 矩阵，记忆"从无到有"纯累积交互信息，首层 FFN 拿到纯 E[1]
- `cell_probe_gated`：M0 = 0 矩阵，但先执行 `Accumulator(0, Q)` 将探针信息通过门控写入记忆（与后续 extraction 一致的处理方式），然后再进入层循环

### 10.3 累积公式

```text
M[t] = M[t-1] + gate · FFN( LayerNorm(M[t-1]) + E[t] )

其中:
  M[0] = Q 或 0                    — 初始记忆（可配置）
  E[t]                              — 第 t 层的 extraction
  gate = sigmoid(Linear(h))         — 0~1 的门控权重
  FFN = Linear(64→256→64) + GELU    — 前馈变换
```

### 10.4 直觉

想象你是一个研究者（细胞系探针），反复阅读一篇论文（药物图）：

- 第 1 遍读到 "有苯环" → 门控判断靠谱 → 写入记忆
- 第 2 遍读到 "有羟基" → 门控判断靠谱 → 追加记忆
- 第 3 遍读到 "某某基团" → 门控判断不靠谱（可能是噪音）→ 几乎不写入

最终记忆 = M0 + g1·FFN(M0+E1) + g2·FFN(M1+E2) + g3·FFN(M2+E3)

参数：可训练的门控权重让模型自己学会"这一层的信息靠不靠谱"。

### 10.5 residual 模式（Q 自更新残差连接）

residual 模式**完全跳过 MemoryAccumulator**，每层提取的 interaction feature 直接通过残差连接写回 Q 自身。Q 在层间逐步融入药物交互信息，最终 Q 直接 flatten 送 readout。

```text
Q[0] = CellQueryProj(cell_feats)

for block in blocks:
    extracted, H, attn = block(Q, H, ...)
    Q = Q + query_update(LN(Q + extracted))
    #   └── FFN: 64→64 + GELU + Dropout

flat = Q.view(B, -1)  →  readout → pred
```

**与 memory 模式的对比**：

| 维度 | memory | residual |
|------|--------|----------|
| 信息传递 | Q 不变，E 写入独立记忆 M | Q 自身逐步更新 |
| 门控 | 可学习 sigmoid gate | 无门控（纯残差） |
| Readout 输入 | M (累积记忆) | Q (更新后的探针) |
| 参数量 | Accumulator (FFN + gate) | query_update (FFN) |
| 设计来源 | 记忆网络 / SELECTOR 风格 | 标准 Transformer 风格 |

**适用场景**：residual 模式更接近标准 Transformer 编码器，结构简洁、计算量略低，适合快速收敛和调试。memory 模式通过门控选择性积累，可能对多粒度信息融合更有利。

---

## 十一、Readout MLP

```text
Flat memory: (B, 512) → zero-pad → (B, 1024)
     ↓
Readout MLP: 1024 → 256 → 128 → 1 → IC50 (log10)
```

**注意**：readout 层不拼接任何额外的 Morgan/ChemBERTa 特征——这些已经在原子级广播中通过 GNN 深度融合了。

---

## 十二、完整数据流（cell_to_drug）

```text
输入 batch:
  drug_ids: (B,)      — 药物索引
  cell_ids: (B,)      — 细胞索引

1. 查表 → 批量图 + 细胞特征
   graph = batch_graphs(drug_ids)   → x(Batch), edge_index, edge_attr, pos, batch
   cell_feats = _cell_table[cell_ids]  → (B, 256)

2. 分支训练控制（仅训练时，各自由 enable_* 布尔开关控制）
   细胞侧: cell_feats += Gaussian_noise * cell_noise_std
           cell_feats *= mask(prob=cell_drop_prob)
   药物侧: [在 _broadcast_drug_features 内部]
           morgan_32 += Gaussian_noise * drug_noise_std
           bert_32   += Gaussian_noise * drug_noise_std
           morgan_32 *= independent_mask(prob=drug_drop_prob)
           bert_32   *= independent_mask(prob=drug_drop_prob)

3. _broadcast_drug_features (封装方法，两条路径共享调用)
   morgan_32 = morgan_atom_proj(morgan_table[drug_ids])         → (B, 32)
   bert_32   = chemberta_atom_proj(chemberta_table[drug_ids])   → (B, 32)
   → [noise] → [drop]  (if enabled, training only)
   gate_m = gate_morgan(graph.x)    → (total_nodes, 32)
   gate_b = gate_bert(graph.x)      → (total_nodes, 32)
   node_feats = [graph.x, morgan_32[drug_to_node]*gate_m, bert_32[drug_to_node]*gate_b]
   concat → (total_nodes, 222)

4. atom_proj: 222 → 64  →  (total_nodes, 64)  [独立 dropout = drug_dropout]

5. edge_proj（仅 GINE/EGNN）

6. CellQueryProjector: 256 → (B, 16, 64) = Q  [独立 dropout = cell_dropout]

7. 交互模式分支（由 interaction_mode 控制）：

   ┌─ memory 模式 ─────────────────────────────────────────┐
   │ MemoryAccumulator.init(Q, mode=memory_init)          │
   │   M0 = Q (cell_probe) 或 0 (zero)                    │
   │                                                       │
   │ for block in blocks:                                  │
   │     extracted, H, attn = block(Q, H, ...)             │
   │     memory = accumulator(memory, extracted)           │
   │                                                       │
   │ final = memory  →  flatten  →  readout               │
   └───────────────────────────────────────────────────────┘

   ┌─ residual 模式 ───────────────────────────────────────┐
   │ for block in blocks:                                  │
   │     extracted, H, attn = block(Q, H, ...)             │
   │     Q = Q + query_update(LN(Q + extracted))           │
   │                                                       │
   │ final = Q  →  flatten  →  readout                    │
   └───────────────────────────────────────────────────────┘

8. Flatten: final → (B, M*64) → 必要时 zero-pad 到 readout 输入维

9. Readout MLP: → 256 → 128 → 1

10. 输出: IC50 预测（log10 尺度）
```

---

## 十三、维度速查表

| 组件 | 输入 | 输出 |
|------|------|------|
| 原子原始特征 | 158 | — |
| Morgan 指纹（每药） | — | 1024 |
| ChemBERTa（每药） | — | 768 |
| morgan_atom_proj | 1024 | 32 |
| chemberta_atom_proj | 768 | 32 |
| gate_morgan | 158 | 32 |
| gate_bert | 158 | 32 |
| 合并原子特征 | 158+32+32 | 222 |
| atom_proj | 222 | 64 |
| GraphEvolver (GNN) | 64 | 64 |
| 细胞基因表达 | 256 | — |
| CellQueryProjector | 256 | 16×64 |
| Q = (B, 16, 64) | — | — |
| GraphCrossAttention | Q(B,16,64) + H(N,64) | (B, 16, 64) |
| MemoryAccumulator | (B, 16, 64) + (B, 16, 64) | (B, 16, 64) |
| Flatten | (B, 16, 64) | (B, 1024) |
| Readout MLP | 1024 | 1 |

---

## 十四、训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 1280 | 批量大小 |
| num_gnn_layers | 3 | GNN 层数 |
| num_probes | 16 | 细胞系探针数 |
| hidden_dim | 64 | 隐藏维度 |
| gnn_type | egnn | GNN 类型 |
| gnn_heads | 8 | GNN/GAT 注意力头数 |
| num_heads_cross_attn | 8 | 交叉注意力头数 |
| dropout | 0.1 | 全局 dropout |
| interaction_mode | memory | 交互模式：memory（门控记忆累积）/ residual（Q 自更新残差） |
| memory_init | cell_probe | 初始记忆状态：cell_probe / zero（仅 memory 模式） |
| lr | 1e-4 | 学习率 |
| weight_decay | 1e-4 | 权重衰减 |
| scheduler | plateau | 学习率调度策略 |
| warmup_epochs | 5 | 预热轮数 |
| early_stopping_patience | 10 | 早停耐心 |
| memory_init | cell_probe | MemoryAccumulator 初始状态（cell_probe / zero / cell_probe_gated） |

### 14.1 细胞侧控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| enable_cell_noise | false | 细胞侧高斯噪声开关 |
| cell_noise_std | 0.1 | 细胞噪声标准差 |
| enable_cell_drop | false | 细胞样本级掩码开关 |
| cell_drop_prob | 0.1 | 细胞样本掩码概率 |
| enable_cell_dropout | false | CellQueryProjector 独立 dropout 开关 |
| cell_dropout | 0.3 | 细胞投影器独立 dropout |

### 14.2 药物侧控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| enable_drug_noise | false | 药物投影层噪声开关 |
| drug_noise_std | 0.05 | 药物噪声标准差 |
| enable_drug_drop | false | 药物分子指纹掩码开关 |
| drug_drop_prob | 0.1 | Morgan/ChemBERTa 独立掩码概率 |
| enable_drug_dropout | false | atom_proj 独立 dropout 开关 |
| drug_dropout | 0.2 | atom_proj 独立 dropout |

### 14.3 梯度裁剪与监控

| 参数 | 默认值 | 说明 |
|------|--------|------|
| grad_clip | 1.0 | 共享参数最大梯度范数 |
| cell_grad_clip | 1.0 | 细胞分支最大梯度范数 |
| drug_grad_clip | 1.0 | 药物分支最大梯度范数 |

每步训练后，模型参数按 `cell` / `drug` / `shared` 分组分别裁剪，同时记录 L2 和 RMS 梯度模长（`gn_c/d/r` 为 L2 范数，`rms_c/d/r` 为均方根范数）用于监控分支梯度平衡。

---

## 十五、数据划分

支持三种划分模式：

| 模式 | 训练集 | 验证集 | 测试集 |
|------|--------|--------|--------|
| interpolation | 随机划分 | 随机划分 | 随机划分 |
| drug_cold | 部分药物 | 新药物 | 新药物 |
| cell_cold | 部分细胞 | 新细胞 | 新细胞 |

**药物冷启动（drug_cold）** 是主要评估场景：验证集和测试集包含训练集未见的药物分子，考察模型的泛化能力。

---

## 十六、关键设计原则

1. **原子级门控广播**：Morgan 和 ChemBERTa 在 GNN 入口注入，而非 readout 前拼接。每个原子通过自身的化学属性（`graph.x` 158 维）生成专属的 0~1 门控系数，逐通道调制分子指纹的注入量——碳原子和氮原子从同一 Morgan 指纹中提取的信号不同。GNN 消息传递时相邻原子携带的分子语义既有重叠又有差异，丰富了图的语义多样性。

2. **镜像分支控制**：细胞侧和药物侧各有一套独立的训练时控制（噪声、掩码、dropout），每项配有 `enable_*` boolean 开关。药物侧噪声仅加在 Morgan/ChemBERTa 投影后的 32 维向量上，保留原子化学属性干净。药物指纹掩码对 Morgan 和 ChemBERTa 独立抛硬币，保证精细粒度分布调节。

3. **广播逻辑封装**：`_broadcast_drug_features()` 统一封装 Morgan/ChemBERTa 的投影、噪声、掩码、门控广播全过程，`cell_to_drug` 和 `drug_to_cell` 双路径共享调用，消除代码重复并确保同 batch 内随机状态一致。

4. **纯 readout**：readout 输入仅来自 MemoryAccumulator 的展平记忆，无"后门"拼接。避免了语义重复叠加。

5. **可配置 MemoryAccumulator 起点**（仅 memory 模式）：`cell_probe` = Q 直接作为 M0；`zero` = 零矩阵；`cell_probe_gated` = 零矩阵 + 先通过 Accumulator 的门控写入 Q。三种方式对应不同的记忆初始化哲学：纯先验、纯经验、先验通过门控审核后写入。

6. **双交互模式可选**：`memory` 模式通过门控累积独立记忆 M，`residual` 模式将 extraction 直接残差写回 Q 自身。residual 更接近标准 Transformer，结构轻量；memory 的门控选择可能对多粒度融合更有利。两者通过 `interaction_mode` 一键切换。

7. **门控记忆累积**（memory 模式）：多层提取的信息通过可学习的门控权重融合，模型自动区分重要提取和噪音。

8. **分支梯度监控与裁剪**：参数按 `cell` / `drug` / `shared` 分组分别裁剪，同时记录 L2 范数（`gn_c/d/r`）和 RMS 均方根范数（`rms_c/d/r`）用于诊断梯度不均衡。RMS 范数除以标量梯度总数，与参数规模解耦，便于跨分支直接对比。

9. **批量填充注意力**：变长分子图通过填充 + 掩码实现 GPU 并行计算，无需逐药物循环。

---

## 十七、文件结构

```text
src/
├── config.py                  # 配置加载
├── data/
│   ├── dataset.py             # GDSC2 数据集加载
│   ├── external_data.py       # 外部数据集加载（纯读取预计算注册表）
│   └── preprocessing.py       # SMILES→图、Morgan、ChemBERTa 特征提取
├── models/
│   ├── model.py               # CellDrugModel 主模型
│   ├── layers.py              # GraphCrossAttention, MemoryAccumulator, GraphEvolver, InteractionBlock
│   └── cellquery.py           # CellQueryProjector
└── training/
    └── trainer.py             # Trainer 训练循环

scripts/
├── train.py                   # 训练入口
└── precompute_external.py     # 外部数据集离线预计算

config/
└── default.yaml               # 默认配置
```
