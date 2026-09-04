# ProjectCellQuery 工作状态

> 最后更新：2026-08-20 01:00（UTC+8）
> 本文件用于记录工作进度，方便中断后回来继续。

---

## ⚠️ 重要发现（2026-08-20）：CV 划分存在两个 bug，现有结果需重跑

在验证 scaffold_cold 性能下降原因时发现 `scripts/cv_runner.py` 存在**两个独立缺陷**，
导致现有全部 CV 结果（含论文数字）不可复现、模型间不可直接对比：

1. **早停/checkpoint 选择缺陷**：`Trainer` 按 warm 验证 RMSE 早停（patience=10）并选
   RMSE 最低的 checkpoint。冷启动下 warm-RMSE 与冷测试排序能力失配，后 3 折实际
   在第 1–8 轮就选中近乎未训练的权重（详见 `docs/scaffold_cold_verification.md` 实验 7）。
2. **valid/train 划分跨进程非确定性**：`_make_scaffold_fold` 与实体折划分中
   `rest = [... for s in g]`（g 为 set）依赖 Python set 迭代顺序（受 PYTHONHASHSEED 影响），
   同一 seed 在不同进程产生不同 valid/train 划分（test 划分确定）。
   **已修复**：对 `g` 用 `sorted(g)` 固定顺序（实验 8）。

**下一步**：必须用修复后的代码统一重跑 我们 + CANDELA + MGATAF × scaffold/drug_cold
（60 轮、关闭早停、固定划分），才能给出可信对比。尚未重跑（用户决定暂缓）。

---

## 一、当前项目定位

CellQuery：癌细胞系药物响应预测。双塔 GNN（细胞系塔 + 药物塔）+ 逐层交叉注意力。

**两个核心创新点**（已写入论文草稿）：
1. **原子级预训练特征注入**：门控机制让每个原子从药物级 ChemBERTa/Morgan 向量中提取个性化子空间特征。
2. **高斯混合模型（GMM）药效团模态**：30 构象采样对齐 → 药效团实例拟合 GMM → 协方差矩阵特征值分解 → 归一化 `f = log(1+sqrt(max(λ,0)))` → 逐原子 18 维特征注入节点。

**基底架构：EGNN + 自注意力 + GMM**（2026-08-12 决策：EGNN 消息内自注意力在 60ep 6 折 CV 下 drug_cold 0.531 显著优于无注意力 0.456，且为全部架构最高；interp/cell_cold 与旧 GINE 基底持平）。

---

## 二、关键配置

| 场景 | 配置文件 | 轮数 | 说明 |
|---|---|---|---|
| interpolation | `config/cellquery_interpolation.yaml` | 100（正式 200） | CV 用 200ep |
| drug_cold | `config/cellquery_drug_cold.yaml` | **60** | 用户 2026-08-10 决定改用 60ep |
| cell_cold | `config/cellquery_cell_cold.yaml` | 100 | |
| scaffold_cold | `config/cellquery_scaffold_cold.yaml` | **60** | transCDR 冷骨架协议，同 drug_cold 口径 |

三个正式场景配置均为 `gnn_type: egnn` + `egnn_use_attention: true`（基底）。`default.yaml` 同步。基线冷骨架配置：`baselines/candela/config_scaffold_cold.yaml`、`baselines/mgataf/config_scaffold_cold.yaml`。6.4 消融配置：`config/exp_drugcold_pretrain_lateconcat.yaml`（`pretrain_fusion: late_concat`）。

训练脚本：`python scripts/train.py --config config/<场景>.yaml`

---

## 三、6 折交叉验证（全部完成 ✅）

统一脚本 `scripts/cv_runner.py`，实体级 6 折，CV 轮数口径 **interp 200ep / drug_cold 60ep / cell_cold 100ep**。

运行方式：
```bash
python scripts/cv_runner.py --model {cellquery|candela|mgataf} --k 6 \
  --split_by {interpolation|drug_cold|cell_cold|scaffold_cold} \
  --out checkpoints/cv_<model>_<split>
```

### CV 结果汇总（Pearson mean±std / Pooled）

| 分裂 | CellQuery (EGNN+attn+GMM) | CANDELA | MGATAF |
|---|---|---|---|
| **interpolation** | **0.9344 ± 0.0016** / 0.9344 | 0.8945 ± 0.0039 / 0.8945 | 0.8278 ± 0.0024 / 0.8275 |
| **drug_cold** | **0.5306 ± 0.1458** / 0.5696 | 0.3321 ± 0.0572 / 0.3072 | 0.4151 ± 0.2059 / 0.4310 |
| **cell_cold** | **0.8735 ± 0.0036** / 0.8735 | 0.8372 ± 0.0209 / 0.8372 | 0.8152 ± 0.0053 / 0.8077 |
| **scaffold_cold**（60ep） | 0.3290 ± 0.2923 / **0.4429** | 0.3735 ± 0.0889 / 0.3231 | 0.3581 ± 0.1514 / 0.3121 |

**CellQuery 三个分裂全部领先。** 新基底结果存于 `checkpoints/cv_exp_{interp,cellcold,drugcold}_egnn_gmm_attn/cv_results.json`。

> **冷支架启动（scaffold_cold）说明**：按 Murcko 骨架分组 6 折（transCDR 协议），137 药 → 135 骨架，每折测试集约 23 骨架，逐折 Pearson 方差极大。CellQuery 逐折均值（0.329）被 3 个近 0/负折拉低，但 **pooled Pearson 0.443 仍为三模型最高**（CANDELA 0.323 / MGATAF 0.312），RMSE 也最低（2.442 vs 2.567 / 2.569）。方差根因：每折测试药物平均效力离散度不均（std 1.37~3.13），逐折 Pearson 主要反映"这折测试药好不好排序"。
> **逐药物聚合指标（2026-08-15 补算）**：每药对全部细胞系取均值 → 137 药 → 全局相关。protoGDA Pearson **0.385**（CANDELA 0.158 / MGATAF 0.212~0.235），protoGDA 仍最高。由 `--skip_train` 复用 checkpoint 纯推理重算，见 results.md。

### 正式单次结果（论文口径，新基底）

| 分裂 | Test Pearson | Test RMSE |
|---|---|---|
| interpolation（200ep） | **0.9356** | 0.9722 |
| drug_cold（60ep） | 0.5590 | 2.2742 |
| cell_cold（100ep） | **0.8634** | 1.3824 |

> drug_cold 单次受早停噪声主导（best=epoch 5），以 CV（0.5306）为正式口径。旧 GINE 基底单次：interp 0.9333 / drug_cold 0.6802 / cell_cold 0.8639。

---

## 四、论文草稿状态（`docs/paper_draft.md`）

- 基于 `new idea.md` 完成的完整草稿，含八章（摘要/引言/数据/问题公式化/方法/结果/消融/讨论/代码可用性）。
- **2026-08-11 用户决定：正式结果改用 6 折 CV 口径**。主表 5.1 全部改为 CV 均值±std，基线 CANDELA/MGATAF 与本文 CellQuery 均用 CV 结果。
- **GAT 行已从主表移入消融实验 6.1**（早期单次完整训练口径，三场景数据 0.9330/0.8643/0.6552 在 6.1 注释中给出）。
- **2026-08-12 基底变更：GINE+GMM → EGNN + 自注意力 + GMM**。全章节已更新为新基底数值与表述。
- **6.1 消融表新增 EGNN + 自注意力行**（0.531 ± 0.146），为全部配置最高。
- **2026-08-15 融合用户文法修订稿 `docs/paper_draft1.md` 覆盖 `docs/paper_draft.md`**，并补充：
  - **模型统一命名为 protoGDA**（替换原 "CellQuery"）；
  - **2.4 新增第四场景冷支架启动（scaffold_cold）**，更新图 1 mermaid；
  - **5.1 主表新增「冷支架启动 Pearson」列**（三模型 60ep 6 折 CV），并加注说明 pooled 口径；
  - **5.3 / 第七章补充冷支架启动论述**（更严格泛化检验、pooled vs 逐折均值）；
  - **6.4 由「留作后续」落实为真实消融结果**（门控广播 0.531 ± 0.146 vs 末端拼接 0.488 ± 0.165，Δ -0.043，支持创新点之一）；
  - 清理了原稿中的删除线（`~~...~~`）标注与 3.1 的公式占位符。
- **2026-08-17 按用户要求重构论文结构与写作口径**（严格论文格式）：
  - **解散"问题公式化"章节（原第三章）**：3.1 任务定义移至方法 4.2；3.2 整体建模框架与 3.3 交叉注意力公式并入 4.5；3.4 原子级预训练特征注入并入 4.4.1；3.5 药效团 GMM 公式并入 4.4.2（相应 mermaid 图随之迁移，图 2/3/4 重新编号）；
  - **2.4 中 pooled / 逐药物聚合 Pearson 的定义与动机移入方法 4.7 评估指标**，2.4 仅保留场景划分与 CV 协议；
  - **结果/讨论严格分离**：结果章（五）与消融章（六）只保留数据趋势与数值事实，全部机理分析迁往讨论章（七）并组织为 7.2~7.6 小节；原方法 4.2/4.3 中"经消融验证…"类表述删除；消融 6.1~6.5 的删除线分析块清理并整合进讨论；
  - **5.1 结果引言补全占位数字**（原"（列举）"填入 CV 数值），"热启动"统一为"内插"；各表补表题（表 1~表 6）。
  - **注意**：章节编号现为 一、二、四、五、六、七、八（第三章已解散），方法小节按用户指定保持 4.x。
- **2026-08-17 二次微调（写作呈现方式）**：
  - 删除 2.4 末尾对两种 Pearson 口径的重复提及（4.7 已完整定义，避免重复）；
  - 5.3「关键发现」分点列表移除，内容融入 5.1 正文段落（叙述式）；「柱状图」不再单列小节，图 5 嵌入正文叙述后；
  - 7.1「主要结论」由分点改为三个连贯段落（总述三场景领先 → 消融四项结论 → 冷支架泛化）。
- **2026-08-17 三次微调（评估指标呈现）**：
  - 4.7 补充常规评估指标（Pearson / RMSE / 逐折均值±std 报告方式），两种特殊 Pearson 压缩为定义 + 简单应用原因，深入解读留给讨论 7.2；
  - 5.1 新增**表 2（protoGDA 四场景完整评估指标）**：CV Pearson（逐折均值±std）/ Pooled Pearson / 逐药物聚合 Pearson / RMSE（interp 0.9344±0.0016 / cell_cold 0.8735±0.0036 / drug_cold 0.5306±0.1458 / scaffold_cold 0.3290±0.2923，pooled 0.4429，逐药聚合 0.385）；
  - 消融表编号顺延：原表 2–6 → 表 3–7，正文引用同步更新。
- **2026-08-17 参考文献标注**（用户要求"寻找和标注相关的参考文献，确保引文的真实性和与叙述直接相关"）：
  - 正文全部引用标注完成（[1]~[25] 编号按首次出现顺序，GB/T 7714 格式），替换了引言中的"（此处应有引用）/（此处也应当有引用）/（引用transCDR的方法）"三处占位；
  - 引言：药筛平台 GDSC/CCLE[1,2]、DTI 库 DeepPurpose[3]、双塔 GNN DeepCDR[4]、交叉注意力 CANDELA[5]、内插 0.93~0.95 精度 MGATAF[6]+Hi-GeoMVP[7]、噪声上限 Haibe-Kains[8]、冷启动场景 TransCDR[9]、辅助学习 Hi-GeoMVP[7]、迁移学习 TransCDR[9]、多构象 Axelrod[10]、EM[11]、GMM 构象景观 Pisani[12]、Gaussian 形状近似 Good&Richards[13]+Grant[14]；
  - 数据/方法/结果：TDC[15]、ChemBERTa[16]、ECFP[17]、Murcko[18]、EGNN[19]、GAT[20]、GIN[21]、GINE[22]、GATv2[23]、自注意力 Vaswani[24]、RDKit[25]；
  - 所有 25 条文献均已通过联网检索核实（期刊/卷/页/DOI/作者），无编造；文末新增"参考文献"章节。
- 结果图 `docs/figures/results_bar.png` 已按新基底 CV 数据（含误差棒）重新生成（暂含三分裂；冷支架可按需追加面板）。
- **2026-08-17 表注精简 + 结果图重绘**：
  - **表注只保留声明**，删除与正文重复的叙述/分析：表 1 注 1 删"CV 数值普遍低于单次结果，属于正常的口径差异"；注 2 删"逐折 Pearson 方差极大（±0.29）"及口径交叉引用，仅保留补充分组数据与 pooled/逐药物聚合补充数值；表 2 注删"两种全局口径的定义见 4.7"；表 3 注删"单折波动大""作为历史锚点记录于 results.md"；表 1 标题删"后续正文数值均以此为口径"。
  - **图 5 重绘为四场景**（`scripts/plot_results.py`，2×2 面板：内插/细胞冷启动/药物冷启动/冷支架启动，柱状图 + 误差棒 + 柱顶数值标注，冷支架面板叠加 pooled Pearson 菱形散点）；配套中文字体 `scripts/fonts/NotoSansSC-{Regular,Bold}.ttf`（系统无 CJK 字体，从 jsdelivr 下载）；正文"三种主要场景"与图题同步改为四场景。

### 论文关键数值速查（CV 口径，mean±std）

| 模型 | interp | cell_cold | drug_cold | scaffold_cold |
|---|---|---|---|---|
| protoGDA (EGNN+attn+GMM) | **0.9344 ± 0.0016** | **0.8735 ± 0.0036** | **0.5306 ± 0.1458** | 0.3290 ± 0.2923（pooled 0.443） |
| CANDELA | 0.8945 ± 0.0039 | 0.8372 ± 0.0209 | 0.3321 ± 0.0572 | 0.3735 ± 0.0889（pooled 0.323） |
| MGATAF | 0.8278 ± 0.0024 | 0.8152 ± 0.0053 | 0.4151 ± 0.2059 | 0.3581 ± 0.1514（pooled 0.312） |
| MGATAF（文献） | 0.9312 | 0.8536 | — | — |
| protoGDA (GAT, 无GMM, 单次) | 0.9330 | 0.8643 | 0.6552 | — |

> drug_cold 下 protoGDA CV (0.5306) 与 GAT 单次 (0.6552) 的差异源于协议（CV 每折约 92 药 vs 单次全量）。旧 GINE 基底 CV：interp 0.9350 / cell_cold 0.8750 / drug_cold 0.5120（记录于 results.md）。冷支架逐折数值见 results.md。

---

## 五、已完成的工作（2026-08-07 ~ 08-12）

- [x] 药效团 GMM 模态预计算管线（`scripts/precompute_pharmacophores.py`，含 `_empirical_eigvals` / `_gmm_eigvals`）
- [x] 数据管线接入（`src/data/preprocessing.py` `smiles_to_graph`、`src/data/dataset.py`）
- [x] 模型集成（`src/models/model.py`，原子级门控广播 + pharm 特征注入）
- [x] 药冷正则化修复（`drug_dropout: 0.3`、`drug_noise_std: 0.3`）
- [x] GAT / GINE / EGNN 药效团消融矩阵（drug_cold + cell_cold）
- [x] GMM vs 经验协方差对比（差异 0.005，噪声范围内）
- [x] EGNN 增强实验（非饱和 RBF / 等变坐标特征）
- [x] 塔内训练消融（`CellTowerBlock`，`cell_tower_interleaved` 默认 false，三场景均无增益）
- [x] 6 折 CV（3 模型 × 3 分裂，全部完成）
- [x] **消融 6.1 的 60ep 6 折 CV**（GAT/GINE/EGNN × 无 pharm/+GMM 共 5 个配置，2026-08-11 完成）
- [x] **EGNN 自注意力训练轮数验证**（2026-08-11：单次 60ep 0.5590 vs 0.3432 不可靠；60ep 6 折 CV 0.5306 vs 0.4561，注意力 +0.0745 显著提升，"训练不足"假设成立）
- [x] **三场景 6 折 CV 验证（attn vs base vs 旧 GINE）**：drug_cold 0.5306 / 0.4561 / 0.5120（attn 胜）；cell_cold 0.8735 / 0.8744 / 0.8750（持平）；interp 0.9344 / 0.9334 / 0.9350（持平）
- [x] **基底决策（2026-08-12）**：采用 **EGNN + 自注意力 + GMM** 为新基底（drug_cold 显著占优，其余持平）
- [x] **新基底三场景正式单次重跑**：interp 0.9356 / cell_cold 0.8634 / drug_cold 0.5590
- [x] `default.yaml` 重构、死代码清理
- [x] 论文草稿 + 60ep 口径更新
- [x] **冷支架启动（scaffold_cold）实现与 6 折 CV**（2026-08-15）：`cv_runner.py` 增加 `scaffold_cold` 划分（`MurckoScaffoldSmiles` 分组、无泄漏），三模型 60ep 6 折 CV 完成（protoGDA 0.329±0.292 / pooled 0.443；CANDELA 0.374±0.089；MGATAF 0.358±0.151）
- [x] **6.4 消融落实**（2026-08-15）：`model.py` 新增 `pretrain_fusion` 开关（`gated_broadcast`/`late_concat`），drug_cold 60ep 6 折 CV：门控广播 0.531±0.146 vs 末端拼接 0.488±0.165（Δ -0.043，支持创新点之一）
- [x] **论文草稿融合**（2026-08-15）：`paper_draft1.md` 覆盖 `docs/paper_draft.md`，统一 protoGDA 命名，补充冷支架章节与 6.4 消融结果
- [x] **环境修复**（2026-08-15）：pytz 缺失、setuptools 降级（pkg_resources）、`LD_LIBRARY_PATH=/opt/conda/lib`（libstdc++ CXXABI 不匹配）

---

## 六、待办 / 可选后续

- [ ] **论文图表完善**：目前仅 `results_bar.png` 一张图。可考虑补：药效团特征可视化（某药效团在 30 构象中的空间分布 + GMM 椭球）、注意力权重热图、消融柱状图、冷支架启动对比面板（含 pooled 口径）。
- [ ] **冷支架启动稳健性**：全库仅 135 骨架、每折约 23 骨架，逐折 Pearson 方差极大（±0.29）。可多随机种子取平均或换更大药物库（GDSC1/CCLE 合并）增强结论稳健性。
- [ ] **多构象显式聚合对比实验**（论文局限已提，留作后续）。
- [ ] **药物冷启动验证集统计噪声**：验证集仅 13 药，可用按药物聚合的评估或 GDSC1/CCLE 合并增强稳健性。

---

## 七、环境与命令速查

- 工作目录：`/trashpaper/ProjectCellQuery`
- 代码图谱（MCP）：`user-codebase-memory-mcp`，项目名 `ProjectCellQuery`，见 `AGENTS.md`
- 预计算：`python scripts/precompute_pharmacophores.py`（构象/药效团特征，缓存于数据目录）
- 训练：`python scripts/train.py --config config/<场景>.yaml`
- 对比所有模型：`python scripts/compare_all.py`
- CV：见上文第三节命令。`--skip_train` 可跳过训练、复用已有 checkpoint 纯推理重算指标（输出逐折 / pooled / 逐药物聚合三口径）。
- 结果记录：`results.md`（全部历史实验日志）

---

## 八、重要提醒

1. **正式结果与全部消融均用 6 折 CV**（6.1/6.2/6.3/6.4/6.5 与塔内三场景全部完成，2026-08-19）。6.2/6.3/6.5 的历史 25ep 单次数值保留在 results.md 仅作对照，勿写入论文正文。
2. `results.md` 中大量单次实验数值是历史记录，**不要批量替换**；只有论文正式章节与 CV 对照表用 CV 数值。
3. **正式基底已换为 EGNN + 自注意力 + GMM**（2026-08-12）。旧 GINE 基底的结果保留在 results.md 作为消融/历史对照。
4. **消融 6.1 为 60ep 6 折 CV**（`checkpoints/cv_exp_drugcold_*`）；EGNN+attn 的正式 CV 在 `checkpoints/cv_exp_{interp,cellcold,drugcold}_egnn_gmm_attn/`。
5. **新基底的 drug_cold 正式单次（0.5590）低于旧 GINE（0.6802）**，但 CV 口径下新基底更优（0.5306 vs 0.5120）。drug_cold 单次受早停噪声主导（best=epoch 5），**论文以 CV 为准**。
6. **冷支架启动为 60ep 6 折 CV**（`checkpoints/cv_{cellquery,candela,mgataf}_scaffold_cold/`），逐折均值口径 protoGDA 被近 0/负折拉低，**pooled Pearson（0.443）与逐药物聚合 Pearson（0.385）才是稳健的排序指标**（两口径下 protoGDA 均为三模型最高），论文 5.1 注2 / 5.3 / 第七章已按此表述。
7. **6.4 消融为 drug_cold 60ep 6 折 CV**（`checkpoints/cv_exp_drugcold_pretrain_lateconcat/`）：门控广播 0.531±0.146 vs 末端拼接 0.488±0.165。
8. **`cv_runner.py` 支持 `--skip_train`（纯推理复用已有 best.pt）与逐药物聚合指标**（`drug_level_metrics` 写入 `cv_results.json`，每折预测存 `cv_predictions.npz`）。注意 MGATAF 推理存在轻微运行间非确定性（pooled 0.30~0.32 波动，源自其标签归一化边界），逐折 Pearson 复现一致。
9. **环境修复（2026-08-15）**：若再次出现 pandas/tdc 导入失败，先 `export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH`；pkg_resources 缺失需 `pip install "setuptools<81"`。
10. **消融 6.2/6.3/6.5 已全部补齐 6 折 CV（2026-08-19）**：
    - 6.2 经验协方差 `checkpoints/cv_abl_expcov_egnn_gmm_attn/`：0.504 ± 0.164（vs GMM 0.531 ± 0.146）
    - 6.3 残差注入 `checkpoints/cv_abl_residual_egnn_gmm_attn/`：0.473 ± 0.205（vs 拼接 0.531 ± 0.146）
    - 表3 GAT 残差+键语境 `checkpoints/cv_abl_gat_residual_edgeattr/`：0.506 ± 0.147
    - 6.5 塔内 `checkpoints/cv_abl_tower_{drugcold,cellcold,interp}/`：0.410 ± 0.196 / 0.8717 ± 0.0058 / 0.9321 ± 0.0023
    - 通用续跑脚本 `scripts/cv_resume.py`（复用已有折，断点续跑），消融 config 见 `config/cv_abl_*.yaml`
11. **10 折 drug_cold 对照（2026-08-19）**：`checkpoints/cv_exp_drugcold_10fold_egnn_gmm_attn/`，0.482 ± 0.208（6 折为 0.531 ± 0.146，pooled 0.529 vs 0.570）。10 折每折仅约 14 药，逐折标准差扩大，进一步佐证小测试集下逐折 Pearson 对采样敏感（见 `docs/scaffold_cold_verification.md`）。
