# 冷支架启动（scaffold_cold）性能下降原因的实验验证

## 问题

GDSC2 全库 137 药 → 135 个 Murcko 骨架，其中 133 个骨架各自只对应一个药物（骨架几乎唯一）。
在 seed=42 下 protoGDA 的 drug_cold CV 逐折均值 = 0.5306，scaffold_cold 却只有 0.3290（pooled 0.443）。
问题是：既然骨架几乎唯一、scaffold_cold 在药物层面等价于另一种随机分组，性能下降是随机采样事件，
还是 scaffold 分组引入了真实的额外泛化挑战？

## 实验 1：Oracle 采样效应测试

用 interpolation 模式训练的 6 个 fold 模型（训练时见过几乎全部 137 药，逐药预测能力近似恒定）
作为 oracle，分别对 drug_cold 与 scaffold_cold 的各折测试集做前向预测。

| 协议 | 逐折 Pearson 均值 | 逐折 std | pooled Pearson | 药物级 Pearson |
|------|------|------|------|------|
| drug_cold | 0.9413 | 0.0198 | 0.9481 | 0.9994 |
| scaffold_cold | 0.9370 | 0.0222 | 0.9483 | 0.9994 |

oracle 下两协议逐折/池化/药物级 Pearson 几乎一致 → **测试子集本身的可预测难度无差异**，
性能差不可能来自"scaffold 测试折更难"。

## 实验 2：化学空间疏远度

Morgan(r=2,1024) 指纹下，每折测试药 → 训练药的平均最近邻 Tanimoto 距离：

| 协议 | 平均最近邻距离（越接近 1 越疏远） |
|------|------|
| drug_cold | 0.706 |
| scaffold_cold | 0.690 |

scaffold_cold 测试药与训练药的化学距离**反而更小** → 不构成结构上更难的泛化鸿沟。

## 实验 3：崩盘折的机制 —— 药物效力区分度（disc_ratio）

定义 `disc_ratio = 测试药物平均效力的 std / 平均药物内效力 std`。
当测试折药物间的均值区分度小于药物内噪声时（disc_ratio < 1），逐样本 Pearson 天然趋近 0/负值。

| 折 | 协议 | disc_ratio | 实际逐折 Pearson |
|----|------|------|------|
| scaffold fold4 | 崩盘 | **0.977** | 0.076 |
| scaffold fold5 | 崩盘 | 1.286 | -0.053 |
| scaffold fold6 | 崩盘 | 1.374 | 0.100 |
| scaffold fold1 | 正常 | 2.208 | 0.592 |
| drug_cold 全部 6 折 | — | 全部 >1（1.09–1.84） | 0.295–0.732 |

scaffold_cold 崩盘的后 3 折恰好是低区分度折；drug_cold 的 6 折全部避开该区间。
折级 Pearson 与 disc_ratio 显著相关（12 折合并 r=0.621, p=0.031）。

## 实验 4：蒙特卡洛 —— 低区分度折的出现概率

20000 次独立抽样，每次随机抽 23 药作为测试折：

- P(disc_ratio < 1.0) = 6.13%（单折）
- P(disc_ratio < 1.4) = 35.6%（单折）
- 一次 6 折 CV 出现 ≥3 个低区分度折（disc<1.4）的概率 = **36.4%**
- 一次 6 折 CV 出现 ≥2 个低区分度折的概率 = 68.7%

scaffold_cold（seed=42）恰好抽到 3 个低区分度折 → 完全在随机抽样的正常范围内，
与骨架分组逻辑无关（骨架唯一 ⇒ 分组 ≈ 随机分组）。

## 实验 5：多随机种子 drug_cold（分组逻辑本身不变）

| seed | drug_cold CV 逐折均值 ± std | 逐折 Pearson |
|------|------|------|
| 42 | 0.5306 ± 0.1458 | 0.295, 0.396, 0.631, 0.590, 0.540, 0.732 |
| 7 | 0.4850 ± 0.1608 | 0.540, 0.545, 0.601, 0.668, 0.365, 0.190 |
| 123 | 0.4211 ± 0.1200 | 0.572, 0.270, 0.456, 0.308, 0.352, 0.569 |

drug_cold 换种子即出现 0.19–0.27 的低折，逐折均值在 0.42–0.53 区间摆动（三个种子：
0.421 / 0.485 / 0.531）。scaffold_cold 的 0.329 落在该随机分布的低端，且 drug_cold
逐折本就有 0.19/0.27/0.30 这类低折，说明小样本逐折 Pearson 的随机波动范围极大。

## 实验 6：折数对照（6 折 vs 10 折，seed=42）

同一协议（drug_cold，60ep，EGNN+attn+GMM 基底）下改变折数，仅改变每折测试药物数量：

| 折数 | 逐折均值 ± std | 每折测试药物数 | pooled Pearson | 逐折 Pearson |
|------|------|------|------|------|
| 6 折 | 0.5306 ± 0.1458 | 23 | 0.5696 | 0.295, 0.396, 0.631, 0.590, 0.540, 0.732 |
| 10 折 | 0.4823 ± 0.2078 | 14 | 0.5287 | 0.315, 0.191, 0.243, 0.353, 0.831, 0.641, 0.572, 0.434, 0.467, 0.777 |

折数从 6 增至 10（每折测试药从 23 减到 14）后，逐折 std 从 0.146 增大到 0.208，
单折可从 0.19 摆到 0.83——**逐折均值的波动主要由"每折随机抽到的测试药数量与组成"决定**，
与分组逻辑（按药物/按骨架/按折数）无关。这进一步佐证：scaffold_cold 与 drug_cold 的
0.53 vs 0.33 差值处在随机分组噪声量级内。

## 实验 7：训练轮数与 checkpoint 选择——决定性验证

**假设**（用户提出）：后 3 折崩盘是否因为训练次数不够？

原始 scaffold_cold CV 的早停逻辑：`Trainer` 按 **warm 验证 RMSE** 触发早停（patience=10），
且 `best.pt` 选验证 RMSE 最小的 checkpoint。检查各折 history 发现：
- 后 3 折实际只训练了 11/12/18 轮（早停截断）；
- 更关键的是，后 3 折验证 RMSE 在第 **1/2/8** 轮就到谷底，之后回升 → 保存的 best.pt
  是**近乎未训练（第 1–2 轮）**的模型；
- 验证 Pearson 在停止时仍在上升（fold5: 0.076→0.179）——**warm-RMSE 与冷启动排序能力
  是失配的两个指标**。

**验证实验**：用当前配置重训后 3 折各 60 轮（关闭早停），逐 epoch 计算冷支架测试集 Pearson。

| 折 | 原始报告值 | 60 轮后 RMSE-选择值 | 60 轮最佳 TEST Pearson | 达到该值的轮次 |
|----|------|------|------|------|
| 4 | 0.076 | 0.527 | 0.528 | 58（仍在爬升） |
| 5 | −0.053 | 0.308 | 0.366 | 6（后稳定 ~0.31） |
| 6 | 0.100 | 0.193 | 0.196 | 60（缓慢爬升） |

结论：
1. **fold 4/5 的崩盘绝大部分是训练/checkpoint 选择的人工产物**：fold 4 从 0.076 提升到 0.53，
   fold 5 从 −0.053 提升到 ~0.31。按 warm-RMSE 早停选出的"最佳模型"实际是第 1–2 轮
   的未训练权重。
2. **fold 6 恢复有限（0.10→0.20）**：即使完整训练也只能到 ~0.2，存在真实的冷启动泛化难度，
   这部分对应"测试药物化学结构→效力"映射在该折偏弱。
3. 用 60 轮修正值重算 scaffold_cold 逐折均值 ≈ **0.48**（0.592/0.646/0.614/0.527/0.308/0.193），
   高于原报告 0.329，也**高于 CANDELA 0.374 / MGATAF 0.358**——与 pooled/drug-level 口径
   的结论一致：我们的模型在冷支架场景并不弱于对比模型。
4. **该问题具有系统性**：drug_cold 各折也早停在 15–33 轮（best-RMSE 在 5–23 轮），
   所有场景的数字可能都被低估。建议统一改为"完整训练 + 更合理的 checkpoint 选择
   （如验证 Pearson）"后重跑。

## 实验 8：valid/train 划分跨进程非确定性（更深层的 bug）

**发现过程**：用修正协议重跑 scaffold_cold v2 后，fold5 的验证 RMSE 曲线（第 1 轮见底）
与单折实验（第 57 轮见底）完全不同，但同 seed=42、k=6 下 test 划分一致。深入排查发现：

**根因**：`scripts/cv_runner.py` 的 `_make_scaffold_fold` 与实体折划分中，
`rest = [s for ... for s in g]`（`g` 为 `set`）依赖 **Python set 迭代顺序**。
字符串哈希随机化（PYTHONHASHSEED）跨进程不同 → 同一 seed 在不同进程产生不同的
valid/train 集合。`rng2.shuffle` 本身确定，但洗的是内容顺序已不同的列表。

**跨进程验证**（修复前）：

| PYTHONHASHSEED | scaffold fold5 valid hash |
|------|------|
| 0 | a541ae03 |
| 1 | 26a0bada |
| 42 | （又不同） |

**修复**：对 `rest` 用 `sorted(g)` 固定迭代顺序（scaffold 与实体分支均修复）。
修复后验证：`PYTHONHASHSEED=0/1/42` 下 scaffold_cold 与 drug_cold 的 valid 划分
**完全一致**（hash=35b9e650 / 3090a6d7）。

**影响评估**：
1. **test 划分不受影响**（来自 `groups[fold_idx]`，纯 scaffold shuffle，确定性）；
   valid/train 受影响。
2. 原始 CV 各模型如果是在不同进程运行的，**各自的 valid 集不同** → early stopping
   与 best checkpoint 选择基准不一致，**模型间对比可能不公平**。
3. v2 重跑结果同样受此影响，**不能作为公平对比依据**。

**结论**：要获得可靠、可复现、口径一致的对比结果，必须用修复后的代码统一重跑
我们 + CANDELA + MGATAF 在 scaffold_cold 与 drug_cold 的全部 6 折。

## 结论

1. **scaffold 分组在本数据集上不构成额外的泛化挑战**：oracle 与化学距离实验均显示
   scaffold_cold 测试子集并不比 drug_cold 难。
2. **后 3 折崩盘的主因是训练/checkpoint 选择的人工产物，而非模型能力不足**：
   原始协议按 warm 验证 RMSE 早停并选 best 模型，后 3 折因此在第 1–8 轮就选中了
   近乎未训练的权重。重训 60 轮后 fold4 达 0.53、fold5 达 ~0.31、fold6 达 ~0.20，
   逐折均值从 0.329 修正至 ≈0.48，反超 CANDELA（0.374）与 MGATAF（0.358）。
   但此修正值仍受实验 8 的划分非确定性影响，仅供参考量级，不可作为最终结论。
3. **评估协议存在两个独立缺陷**：(a) warm-RMSE 与冷启动排序能力失配（实验 7）；
   (b) valid/train 划分依赖 set 迭代顺序，跨进程不稳定（实验 8）。二者叠加使现有
   全部 CV 结果（原始 + v2）不可复现、模型间不可直接对比。
4. **下一步必须用修复后的代码统一重跑**（我们 + CANDELA + MGATAF ×
   scaffold/drug_cold，60 轮、关闭早停、固定划分），才能给出可信的对比结论。
5. **fold 6 存在真实（但较小）的冷启动难度**：即使完整训练也只到 ~0.2，对应部分
   测试药物"结构→效力"映射偏弱。
6. 骨架唯一时，scaffold_cold 与 drug_cold 的实际差异主要是"另一组随机药物进了测试折"，
   而非骨架级的泛化难度增加；conceptual 上防止骨架泄漏的意义仍然成立。
7. **同一 drug_cold 协议在 3 个种子下的均值就横跨 0.42–0.53**（std 0.12–0.16），
   与 scaffold_cold 的 0.329 只有约一个种子级波动的距离——该差距本身就在随机分组
   噪声量级内。
8. **折数对照（6→10 折）**将每折测试药从 23 减到 14，逐折 std 从 0.146 增至 0.208，
   单折可从 0.19 摆到 0.83——折间波动由测试子集采样规模决定，与分组逻辑无关。

## 脚本

- `scripts/verify_scaffold_cold.py` —— oracle 采样效应实验
- `scripts/verify_scaffold_cold_chem.py` —— 化学空间疏远度
- `scripts/mc_disc_ratio.py` —— 低区分度折蒙特卡洛
- `scripts/seed_sweep.py` —— 多种子 drug_cold CV
- `scripts/exp_fold5_epochs.py` —— 单折 60 轮完整训练 + 逐 epoch 冷测试 Pearson
  （实验 7，支持 `--fold 4|5|6`）
- `scripts/cv_runner.py` —— **已修复实验 8 的划分非确定性 bug**（`sorted(g)` 固定顺序）
- `scripts/cv_resume.py` —— 6 折 CV 运行器（支持 `--patience` 关闭早停）

## 数据文件

- `results/scaffold_vs_drugcold_oracle.json`
- `results/scaffold_vs_drugcold_chemdist.json`
- `results/seed_var/drugcold_seed7/cv_results.json`
- `results/seed_var/drugcold_seed123/cv_results.json`
- `checkpoints/cv_exp_drugcold_10fold_egnn_gmm_attn/cv_results.json`
- `results/exp_fold4_60ep.json` / `results/exp_fold5_60ep.json` / `results/exp_fold6_60ep.json`
  —— 实验 7：后 3 折各 60 轮完整训练的逐 epoch 冷测试 Pearson 曲线
- `checkpoints/cv_scaffold_cold_v2/`、`checkpoints/cv_drug_cold_v2/`、各 `cv_*_v2/`
  —— 修正协议（60 轮/关闭早停）重跑结果，**注意受实验 8 划分 bug 影响，不可作为最终结论**
- `checkpoints/exp_fold{4,5,6}_60ep/` —— 单折 60 轮训练产物（实验 7）
